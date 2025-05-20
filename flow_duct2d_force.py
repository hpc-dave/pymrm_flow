import numpy as np
import scipy as sp
from scipy.sparse import linalg as sla
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pymrm import interp_stagg_to_cntr, interp_cntr_to_stagg, interp_cntr_to_stagg_tvd
from pymrm import construct_grad, construct_div, minmod
from pyamg.aggregation import smoothed_aggregation_solver as amg
from centered_bc import compute_centered_bc, apply_centered_bc

# a duct with L = 20, H = 1

# Physical Parameters
dim = 2               # dimensionality of the problem
rho = 1000            # density of the fluid
mu = 1e-3                # viscosity of the fluid
p_out = 0             # pressure at the outlet
u_in, v_in = 0.1, 0.   # velocities at the inlet
box   = [1., 1.]     # size of the rectangular domain # noqa: E221
l_cell = 0.1  # 0.1   # cell size (equidistant discretization here!)
rtol = 1e-14

# Numerical Parameters
shape_p = [int(b/l_cell) for b in box]  # number of cells in each direction
Co = 0.5                                # maximum Courant number for timestep
dt = Co*(l_cell/u_in)                   # time step size
num_time_steps = 50                     # number of timesteps
dim_u, dim_v = 0, 1                     # indices for readability
beta = 1e6                              # implicit force term

Re = rho * u_in * box[1] / mu
print(f'The Reynolds number of this problem is: {Re}')

# Preallocating lists
x_cntr_c  = [None] * dim  # positions cell faces     # noqa: E221
x_cntr_f  = [None] * dim  # cell centered-positions     # noqa: E221
x_stagg_c = [None] * dim  # staggered positions for velocities     # noqa: E221
x_stagg_f = [None] * dim  # face-positions corresponding to staggered cells     # noqa: E221
shape_v   = [None] * dim  # sizes velocity ndarrays     # noqa: E221
bc_v      = [None] * dim  # velocity bc     # noqa: E221
v         = [None] * dim  # noqa: E221
v_old     = [None] * dim  # noqa: E221
v_cntr    = [None] * dim  # noqa: E221
Grad_v    = [None] * dim  # noqa: E221
grad_bc_v = [None] * dim  # noqa: E221
Div_v     = [None] * dim  # noqa: E221
Lapl_v    = [None] * dim  # noqa: E221
lapl_bc_v = [None] * dim  # noqa: E221
Jac_v     = [None] * dim  # noqa: E221
Jac_v_ilu = [None] * dim  # noqa: E221
Jac_v_pc  = [None] * dim  # noqa: E221
g_v       = [None] * dim  # noqa: E221
bc_p      = [None] * dim  # noqa: E221
Grad_p    = [None] * dim  # noqa: E221
grad_bc_p = [None] * dim  # noqa: E221
Div_p     = [None] * dim  # noqa: E221
mu_v      = [None] * dim  # noqa: E221
rho_v     = [None] * dim  # noqa: E221
beta_v    = [None] * dim  # noqa: E221
prefac_p  = [None] * dim  # noqa: E221
bc_v_pre  = [None] * dim            # noqa: E221
p         = p_out * np.ones(shape_p)  # noqa: E221         # initial guess pressure
mu_p      = np.full_like(p, fill_value=mu)  # noqa: E221 # scalar field with viscosity
rho_p     = np.full_like(p, fill_value=rho)   # noqa: E221 # scalar field with density
beta_p    = np.full_like(p, fill_value=beta)  # noqa: E221 # scalar field with implicit force terms

Lapl_p    = sp.sparse.csr_array((p.size, p.size))  # allocation of Laplacian for pressure    # noqa: E221
lapl_bc_p = sp.sparse.csr_array((p.size, 1))       # allocation of boundary array for pressure

# initialization of the various values and boundary conditions
for i in range(dim):
    shape_v[i] = shape_p.copy()
    shape_v[i][i] += 1
    x_cntr_f[i]  = np.linspace(0, box[i], shape_p[i]+1)     # noqa: E221
    x_cntr_c[i]  = 0.5*(x_cntr_f[i][:-1] + x_cntr_f[i][1:])     # noqa: E221
    x_stagg_c[i] = x_cntr_f[i]
    x_stagg_f[i] = np.concatenate((-1.1*x_cntr_c[i][0:1], x_cntr_c[i], x_cntr_c[i][-1]+x_cntr_c[i][0:1]*2), axis=0)
    bc_v[i] = [None] * dim
    bc_v_pre[i] = [None] * dim
    # shape_v[i][i] -= 1

# WEST/EAST BC
bc_v[dim_u][0] = ({'a': 0, 'b': 1, 'd': u_in}, {'a': 1, 'b': 0, 'd': 0.})
bc_v[dim_v][0] = ({'a': 0, 'b': 1, 'd': v_in}, {'a': 1, 'b': 0, 'd': 0.})
bc_p[0] = ({'a': 1, 'b': 0, 'd': 0}, {'a': 0, 'b': 1, 'd': 0})

# SOUTH/NORTH BC
bc_v[dim_u][1] = ({'a': 1, 'b': 0, 'd': 0.}, {'a': 1, 'b': 0, 'd': 0.})
bc_v[dim_v][1] = ({'a': 0, 'b': 1, 'd': 0}, {'a': 0, 'b': 1, 'd': 0.})
bc_p[1] = ({'a': 1, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0})

for i in range(dim):
    # BE AWARE: THOSE NEED TO BE UPDATED LATER FOR DYNAMIC SYSTEMS
    rho_v[i] = interp_cntr_to_stagg(rho_p, x_f=x_cntr_f[i], x_c=x_cntr_c[i], axis=i)
    mu_v[i] = interp_cntr_to_stagg(mu_p, x_f=x_cntr_f[i], x_c=x_cntr_c[i], axis=i)
    beta_v[i] = interp_cntr_to_stagg(beta_p, x_f=x_cntr_f[i], x_c=x_cntr_c[i], axis=i)
    prefac_p[i] = 1./(rho_v[i]/dt+beta_v[i])

# Preparation of the operators
for i in range(dim):
    Grad_p[i], grad_bc_p[i] = construct_grad(shape_p, x_cntr_f[i], x_cntr_c[i], bc_p[i], axis=i)
    Div_p[i] = construct_div(shape_p, x_cntr_f[i], nu=0, axis=i)
    Lapl_p = Lapl_p + Div_p[i] @ (prefac_p[i].reshape(-1, 1) * Grad_p[i])
    lapl_bc_p = lapl_bc_p + Div_p[i] @ (prefac_p[i].reshape(-1, 1) * grad_bc_p[i])

    Grad_v[i]    = [None] * dim     # noqa: E221
    grad_bc_v[i] = [None] * dim
    Div_v[i]     = [None] * dim     # noqa: E221
    v[i] = np.zeros(shape_v[i])
    Lapl_v[i] = sp.sparse.csc_array((v[i].size, v[i].size))
    lapl_bc_v[i] = sp.sparse.csc_array((v[i].size, 1))
    for j in range(dim):
        if (i == j):
            Grad_v[i][j], grad_bc_v[i][j] = construct_grad(shape_v[i], x_stagg_f[j], x_stagg_c[j], axis=j)
            Div_v[i][j] = construct_div(shape_v[i], x_stagg_f[j], nu=0, axis=j)
            # mu_v_loc = mu_v[j]
            mu_v_loc = np.asarray(mu)
        else:
            Grad_v[i][j], grad_bc_v[i][j] = construct_grad(shape_v[i], x_cntr_f[j], x_cntr_c[j], bc_v[i][j], axis=j)
            Div_v[i][j] = construct_div(shape_v[i], x_cntr_f[j], nu=0, axis=j)
            # mu_v_loc = mu_p
            mu_v_loc = np.asarray(mu)
        print('Be aware, visocsity is not yet a field, only a scalar!')
        Lapl_v[i] = Lapl_v[i] + Div_v[i][j] @ (mu_v_loc.reshape(1, -1) * Grad_v[i][j])
        lapl_bc_v[i] = lapl_bc_v[i] + Div_v[i][j] @ (mu_v_loc * grad_bc_v[i][j])
    Jac_v[i] = sp.sparse.diags(rho_v[i].reshape(-1)/dt, format='csc').dot(sp.sparse.eye(v[i].size, format='csc')) - Lapl_v[i]
    # Jac_v_ilu[i] = sla.spilu(Jac_v[i])
    # Jac_v_pc[i] = sla.LinearOperator(Jac_v_ilu[i].shape, lambda x: Jac_v_ilu[i].solve(x))

# Precompute centered boundary condition parameters
bc_p_pre = compute_centered_bc(c=p, bc=bc_p[0][1], axis=0, boundary=1)
bc_v_pre[dim_u][0] = compute_centered_bc(c=v[dim_u], bc=bc_v[dim_u][0][0], axis=0, boundary=0)
bc_v_pre[dim_u][1] = compute_centered_bc(c=v[dim_u], bc=bc_v[dim_u][0][1], axis=0, boundary=1)

bc_v_pre[dim_v][0] = compute_centered_bc(c=v[dim_v], bc=bc_v[dim_v][1][0], axis=1, boundary=0)
bc_v_pre[dim_v][1] = compute_centered_bc(c=v[dim_v], bc=bc_v[dim_v][1][1], axis=1, boundary=1)

# enforce centered boundary condition for momentum

# enforce the prescribed boundary condition
for d in range(dim):
    Jac_v[d], _ = apply_centered_bc(c=v[d], bc=bc_v[d][d], bc_param=bc_v_pre[d], A=Jac_v[d])
    Jac_v_ilu[d] = sla.spilu(Jac_v[d])
    Jac_v_pc[d] = sla.LinearOperator(Jac_v_ilu[d].shape, lambda x: Jac_v_ilu[d].solve(x))
Lapl_p, _ = apply_centered_bc(c=p, bc=bc_p[dim_u][1], bc_param=bc_p_pre, A=Lapl_p)
Lapl_p_M = amg(Lapl_p).aspreconditioner(cycle='V')

X, Y = np.meshgrid(x_cntr_c[0], x_cntr_c[1])

for k in range(num_time_steps):
    div_v = np.zeros((p.size, 1))

    # Solve Momentum
    for i in range(dim):
        v_old[i] = v[i].copy()

        # Convective contribution (explicit)
        g_conv = np.zeros((v[i].size, 1))
        for j in range(dim):
            if (j == i):        # self-convection
                v_f = interp_cntr_to_stagg(v[i], x_stagg_f[i], x_stagg_c[i], axis=i)
                vi_f, dvi_f = interp_cntr_to_stagg_tvd(rho_v[i]*v[i], x_f=x_stagg_f[i], x_c=x_stagg_c[i], bc=bc_v[i][j], v=v_f, tvd_limiter=minmod, axis=j)   # noqa: E501
                rho_f, drho_f = interp_cntr_to_stagg_tvd(rho_v[i], x_f=x_stagg_f[i], x_c=x_stagg_c[i], bc=bc_v[i][j], v=v_f, tvd_limiter=minmod, axis=j)   # noqa: E501
                # TODO: adapt rho staggered
                conv_flux = rho_f*vi_f*vi_f
            else:               # regular convection
                shape_bc_v = list(v[j].shape)
                shape_bc_v[i] = 1
                v_f = interp_stagg_to_cntr(v[j], x_f=x_cntr_f[i], axis=i)
                # append staggered boundary conditions
                v_f_bc_low = np.zeros(shape_bc_v, dtype=float)
                v_f_bc_high = np.zeros(shape_bc_v, dtype=float)
                slice_bc = tuple(slice(0, 1) if d == i else slice(None) for d in range(dim))
                if bc_v[i][i][0]['a'] == 1.:
                    # Neumann boundary at low face
                    v_f_bc_low[:] = v_f[slice_bc]
                else:
                    # Dirichlet at low face
                    # Neumann boundary at low face
                    v_f_bc_low[:] = 2 * bc_v[i][i][0]['d'] - v_f[slice_bc]
                slice_bc = list(slice_bc)
                slice_bc[i] = slice(v_f.shape[i]-1, v_f.shape[i])
                slice_bc = tuple(slice_bc)
                if bc_v[i][i][1]['a'] == 1.:
                    # Neumann boundary at low face
                    v_f_bc_high[:] = v_f[slice_bc]
                else:
                    # Dirichlet at low face
                    # Neumann boundary at low face
                    v_f_bc_high[:] = 2 * bc_v[i][i][1]['d'] - v_f[slice_bc]
                v_f = np.concatenate((v_f_bc_low, v_f, v_f_bc_high), axis=i)
                # v_f = interp_stagg_to_cntr(v[j], x_f=x_cntr_f[i], x_c=x_cntr_c[i], axis=i)
                # shape_v_f_bnd = list(v_f.shape)
                # shape_v_f_bnd[j] = 1
                # v_f = np.concatenate((np.zeros(shape_v_f_bnd), v_f, np.zeros(shape_v_f_bnd)), axis=j)
                rho_vi_f, drho_vi_f = interp_cntr_to_stagg_tvd(rho_v[i]*v[i], x_cntr_f[j], x_cntr_c[j], bc_v[i][j], v_f, minmod, axis=j)
                conv_flux = v_f * rho_vi_f
            g_conv = g_conv + Div_v[i][j] @ conv_flux.reshape(-1, 1)

        # Pressure contribution (explicit)
        shape_p_f = shape_p.copy()
        shape_p_f[i] = shape_p[i]+1
        grad_p = (Grad_p[i] @ p.reshape(-1, 1) + grad_bc_p[i]).reshape(shape_p_f)
        # idx = [slice(None)] * dim
        # idx[i] = slice(1, shape_p_f[i]-1)
        # g_grad_p = grad_p[tuple(idx)].reshape(-1, 1)
        g_grad_p = grad_p.reshape(-1, 1)

        # assemble contributions, with implicit stress contribution and Euler backward time stepping
        # Note that this here is NOT conservative, should use density of the previous timestep
        g_v[i] = Jac_v[i] @ v[i].reshape(-1, 1) - (rho_v[i].reshape(-1, 1)/dt)*v_old[i].reshape(-1, 1)\
                + g_conv + g_grad_p - lapl_bc_v[i]  # noqa: E127
        print('Warning, mu is only a scalar!')

        _, g_v[i] = apply_centered_bc(c=v[i], bc=bc_v[i][i], bc_param=bc_v_pre[i], B=g_v[i])
        # apply_centered_bc(c=v[i], bc=bc_v[i], bc_param=bc_v_pre[i][0], B=g_v)
        # apply_centered_bc(c=v[i], bc=bc_v[i], bc_param=bc_v_pre[i][1], B=g_v)
        # solve and update velocity
        dv, exit_code = sla.bicgstab(Jac_v[i], g_v[i], M=Jac_v_pc[i], rtol=rtol)
        v[i] -= dv.reshape(shape_v[i])

        v_cntr[i] = interp_stagg_to_cntr(v[i], x_cntr_f[i], x_cntr_c[i], axis=i)               # for outputting
        div_v += Div_p[i] @ v[i].reshape(-1, 1)                                                # add contribution to defect  # noqa: E501

    # Apply Implicit Force Term
    _, defect_continuity = apply_centered_bc(c=p, bc=bc_p, bc_param=bc_p_pre, B=div_v)
    dp, exit_code = sla.bicgstab(A=Lapl_p, b=-defect_continuity, M=Lapl_p_M, rtol=rtol)
    p += dp.reshape(shape_p)                            # update pressure
    # update velocities
    div_v.fill(0)
    for i in range(dim):
        shape_p_f = shape_p.copy()
        shape_p_f[i] = shape_p[i]+1
        grad_dp = (Grad_p[i] @ dp.reshape(-1, 1)).reshape(shape_p_f)
        idx = [slice(None)] * dim
        idx[i] = slice(0, shape_p_f[i])
        v[i] = prefac_p[i][tuple(idx)] * (v[i] - (dt/rho_v[i][tuple(idx)]) * grad_dp[tuple(idx)])
        v_cntr[i] = interp_stagg_to_cntr(v[i], x_cntr_f[i], x_cntr_c[i], axis=i)
        div_v += Div_p[i] @ v[i].reshape(-1, 1)

    # Enforce continuity
    _, defect_continuity = apply_centered_bc(c=p, bc=bc_p, bc_param=bc_p_pre, B=div_v)
    dp, exit_code = sla.bicgstab(A=Lapl_p, b=-defect_continuity, M=Lapl_p_M, rtol=rtol)
    p += dp.reshape(shape_p)                            # update pressure
    # update velocities
    for i in range(dim):
        shape_p_f = shape_p.copy()
        shape_p_f[i] = shape_p[i]+1
        grad_dp = (Grad_p[i] @ dp.reshape(-1, 1)).reshape(shape_p_f)
        idx = [slice(None)] * dim
        idx[i] = slice(0, shape_p_f[i])
        v[i] -= (dt/rho_v[i][tuple(idx)]) * grad_dp[tuple(idx)]
        v_cntr[i] = interp_stagg_to_cntr(v[i], x_cntr_f[i], x_cntr_c[i], axis=i)

    clear_output(wait=True)
    fig, axes = plt.subplots(1, 2)

    p_center = p[:, int(p.shape[1]/2)]
    u_center = v_cntr[0][int(v_cntr[0].shape[0]/2), :]

    axes[0].plot(x_cntr_c[0], p_center)
    axes[1].plot(x_cntr_c[1], v_cntr[0][int(v_cntr[0].shape[0]/2), :])
    axes[1].plot(x_cntr_c[1], v_cntr[0][-3, :])
    fig.suptitle(f"Time step: {k}")
    plt.show()
