import numpy as np
import scipy as sp
from scipy.sparse import linalg as sla
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pymrm import interp_stagg_to_cntr, interp_cntr_to_stagg, interp_cntr_to_stagg_tvd
from pymrm import construct_grad, construct_div, minmod

# Physical Parameters
dim = 2
rho = 1
mu = 1
p_top = 0
v_top = 50
box   = [1.0] * dim     # noqa: E221

# Numerical Parameters
shape_p = [50] * dim
Co = 0.5
dt = Co*(box[0]/v_top)/shape_p[0]
num_time_steps = 50

x_cntr_c  = [None] * dim  # positions cell faces     # noqa: E221
x_cntr_f  = [None] * dim  # cell centered-positions     # noqa: E221
x_stagg_c = [None] * dim  # staggered positions for velocities     # noqa: E221
x_stagg_f = [None] * dim  # face-positions corresponding to staggered cells     # noqa: E221
shape_v   = [None] * dim  # sizes velocity ndarrays     # noqa: E221
bc_v      = [None] * dim  # velocity bc     # noqa: E221
v         = [None] * dim     # noqa: E221
v_old     = [None] * dim     # noqa: E221
v_cntr    = [None] * dim     # noqa: E221
Grad_v    = [None] * dim     # noqa: E221
grad_bc_v = [None] * dim     # noqa: E221
Div_v     = [None] * dim     # noqa: E221
Lapl_v    = [None] * dim     # noqa: E221
lapl_bc_v = [None] * dim     # noqa: E221
Jac_v     = [None] * dim     # noqa: E221
Jac_v_pc  = [None] * dim     # noqa: E221
g_v       = [None] * dim     # noqa: E221
bc_p      = [None] * dim     # noqa: E221
Grad_p    = [None] * dim     # noqa: E221
grad_bc_p = [None] * dim     # noqa: E221
Div_p     = [None] * dim     # noqa: E221
p = p_top*np.ones(shape_p)
Lapl_p    = sp.sparse.csc_array((p.size, p.size))     # noqa: E221
lapl_bc_p = sp.sparse.csc_array((p.size, 1))

# initialization of the various values and boundary conditions
for i in range(dim):
    shape_v[i] = shape_p.copy()
    x_cntr_f[i]  = np.linspace(0, box[i], shape_p[i]+1)     # noqa: E221
    x_cntr_c[i]  = 0.5*(x_cntr_f[i][:-1] + x_cntr_f[i][1:])     # noqa: E221
    x_stagg_c[i] = x_cntr_f[i][1:-1]
    x_stagg_f[i] = np.concatenate((x_cntr_f[i][0:1], x_cntr_c[i][1:-1], x_cntr_f[i][-1:]), axis=0)
    # Neumann condition for the pressure at the walls
    bc_p[i] = ({'a': 1, 'b': 0, 'd': 0},)*2
    bc_v[i] = [None] * dim
    # Dirichlet condition for the velocities at the wall
    for j in range(dim):
        shape_v[i][j] = shape_p[j]
        bc_v[i][j] = ({'a': 0, 'b': 1, 'd': 0}, {'a': 0, 'b': 1, 'd': 0})
    shape_v[i][i] -= 1
# Boundary condition at the top wall
bc_p[-1] = ({'a': 1, 'b': 0, 'd': 0}, {'a': 0, 'b': 1, 'd': p_top})     # Setting pressure at the top wall to 1?!
bc_v[0][-1][1]['d'] = v_top # x-velocity at the top wall to velocity
if (dim == 3):  # in the 3D case just Neumann on the top and bottom wall -> Symmetry condition
    bc_v[1][-1][1]['a'] = 1
    bc_v[1][-1][1]['b'] = 0
    bc_v[1][-1][1]['d'] = 0

# Preparation of the operators
for i in range(dim):
    Grad_p[i], grad_bc_p[i] = construct_grad(shape_p, x_cntr_f[i], x_cntr_c[i], bc_p[i], axis=i)
    Div_p[i] = construct_div(shape_p, x_cntr_f[i], nu=0, axis=i)
    Lapl_p = Lapl_p + Div_p[i] @ Grad_p[i]
    lapl_bc_p = lapl_bc_p + Div_p[i] @ grad_bc_p[i]

    Grad_v[i]    = [None] * dim     # noqa: E221
    grad_bc_v[i] = [None] * dim
    Div_v[i]     = [None] * dim     # noqa: E221
    v[i] = np.zeros(shape_v[i])
    Lapl_v[i] = sp.sparse.csc_array((v[i].size, v[i].size))
    lapl_bc_v[i] = sp.sparse.csc_array((v[i].size, 1))
    for j in range(dim):
        if (i == j):
            Grad_v[i][j], grad_bc_v[i][j] = construct_grad(shape_v[i], x_stagg_f[j], x_stagg_c[j], bc_v[i][j], axis=j)
            Div_v[i][j] = construct_div(shape_v[i], x_stagg_f[j], nu=0, axis=j)
        else:
            Grad_v[i][j], grad_bc_v[i][j] = construct_grad(shape_v[i], x_cntr_f[j], x_cntr_c[j], bc_v[i][j], axis=j)
            Div_v[i][j] = construct_div(shape_v[i], x_cntr_f[j], nu=0, axis=j)
        Lapl_v[i] = Lapl_v[i] + Div_v[i][j] @ Grad_v[i][j]
        lapl_bc_v[i] = lapl_bc_v[i] + Div_v[i][j] @ grad_bc_v[i][j]
    Jac_v[i] = (rho/dt) * sp.sparse.eye(v[i].size, format='csc') - mu * Lapl_v[i]
    Jac_v_ilu = sla.spilu(Jac_v[i])
    Jac_v_pc[i] = sla.LinearOperator(Jac_v_ilu.shape, lambda x: Jac_v_ilu.solve(x))
Lapl_p_ilu = sla.spilu(Lapl_p)
Lapl_p_lu = sla.splu(Lapl_p)
Lapl_p_pc = sla.LinearOperator(Lapl_p_ilu.shape, lambda x: Lapl_p_ilu.solve(x))

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
                vi_f, dvi_f = interp_cntr_to_stagg_tvd(v[i], x_stagg_f[i], x_stagg_c[i], bc_v[i][j], v_f, minmod, axis=j)   # noqa: E501
                conv_flux = rho*vi_f*vi_f
            else:               # regular convection
                v_f = interp_stagg_to_cntr(v[j], x_cntr_c[i], x_stagg_c[i], axis=i)
                shape_v_f_bnd = list(v_f.shape)
                shape_v_f_bnd[j] = 1
                v_f = np.concatenate((np.zeros(shape_v_f_bnd), v_f, np.zeros(shape_v_f_bnd)), axis=j)
                vi_f, dvi_f = interp_cntr_to_stagg_tvd(v[i], x_cntr_f[j], x_cntr_c[j], bc_v[i][j], v_f, minmod, axis=j)
                conv_flux = rho*v_f*vi_f
            g_conv = g_conv + Div_v[i][j] @ conv_flux.reshape(-1, 1)

        # Pressure contribution (explicit)
        shape_p_f = shape_p.copy()
        shape_p_f[i] = shape_p[i]+1
        grad_p = (Grad_p[i] @ p.reshape(-1, 1) + grad_bc_p[i]).reshape(shape_p_f)
        idx = [slice(None)] * dim
        idx[i] = slice(1, shape_p_f[i]-1)
        g_grad_p = grad_p[tuple(idx)].reshape(-1, 1)

        # assemble contributions, with implicit stress contribution and Euler backward time stepping
        g_v[i] = Jac_v[i] @ v[i].reshape(-1, 1) - (rho/dt)*v_old[i].reshape(-1, 1) + g_conv + g_grad_p - mu*lapl_bc_v[i]

        # solve and update velocity
        dv, exit_code = sla.bicgstab(A=Jac_v[i], b=g_v[i], M=Jac_v_pc[i])
        # Jac_lu = sla.splu(Jac_v[i])
        # dv = Jac_lu.solve(g_v[i])
        v[i] -= dv.reshape(shape_v[i])

        shape_v_bnd = list(v[i].shape)
        shape_v_bnd[i] = 1
        v_incl_wall = np.concatenate((np.zeros(shape_v_bnd), v[i], np.zeros(shape_v_bnd)), axis=i)  # velocity field with enforced wall velocity
        v_cntr[i] = interp_stagg_to_cntr(v_incl_wall, x_cntr_f[i], x_cntr_c[i], axis=i)     # for outputting
        div_v += Div_p[i] @ v_incl_wall.reshape(-1, 1)      # add contribution to defect

    # Enforce Continuity
    # dp, exit_code = sla.bicgstab(Lapl_p, (rho/dt)*div_v, M=Lapl_p_pc)
    dp = Lapl_p_lu.solve((rho/dt)*div_v)    # solve defect
    p += dp.reshape(shape_p)
    for i in range(dim):
        shape_p_f = shape_p.copy()
        shape_p_f[i] = shape_p[i]+1
        grad_dp = (Grad_p[i] @ dp.reshape(-1, 1)).reshape(shape_p_f)
        idx = [slice(None)] * dim
        idx[i] = slice(1, shape_p_f[i]-1)
        v[i] -= (dt/rho) * grad_dp[tuple(idx)]
        shape_v_bnd = list(v[i].shape)
        shape_v_bnd[i] = 1
        v_incl_wall = np.concatenate((np.zeros(shape_v_bnd), v[i], np.zeros(shape_v_bnd)), axis=i)
        v_cntr[i] = interp_stagg_to_cntr(v_incl_wall, x_cntr_f[i], x_cntr_c[i], axis=i)

    clear_output(wait=True)
    plt.figure()
    contour = plt.pcolormesh(x_cntr_f[0], x_cntr_f[1], p.T, shading='flat', cmap='viridis')
    plt.streamplot(X, Y, v_cntr[0].T, v_cntr[1].T, color='white')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(contour, label='Pressure')
    plt.title(f"Time step: {k}")
    plt.show()
