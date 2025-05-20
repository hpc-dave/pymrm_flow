import numpy as np
import scipy.sparse as sp


def compute_centered_bc(c: np.ndarray,
                        bc: dict,
                        axis: int,
                        boundary: int,
                        format: str = 'csr') -> dict[sp.spmatrix, np.ndarray]:
    r"""
    Precomputes parameters for a prescribed boundary condition

    Paramters
    ---------
    c: np.ndarray
        input array with field values which determines the size of the matrix
    bc: dict
        boundary condition dictionary in 'abd' form
    axis: int
        the direction in which the boundary is applied
    boundary: int
        0->WEST/SOUTH/BOTTOM boundary, 1->EAST/NORTH/TOP
    format: str
        format of the resulting matrix

    Returns
    -------
    dictionary with a sparse matrix and a mask for further processing

    """
    c_shape = c.shape
    bc_id = 0 if boundary == 0 else c_shape[axis]
    mask_centered = np.zeros(shape=c_shape, dtype=bool)
    idx = tuple(slice(bc_id, bc_id + 1) if d == axis else slice(None) for d in range(c.ndim))
    mask_centered[idx] = True
    mask_centered = mask_centered.reshape(-1)
    diag = (mask_centered).astype(float)
    if bc['a'] != 0 and bc['b'] != 0:
        raise ValueError(f'Mixed boundaries are not supported, either a or b have to be zero! received: {bc}')
    elif bc['a'] == 1:
        k = [0, (1 - boundary * 2) * c_shape[1]]
        diag_centered = sp.spdiags([diag, -diag], k, format=format)
    elif bc['b'] == 1:
        diag_centered = sp.spdiags([diag], [0], format=format)
    else:
        raise ValueError(f'Something went wrong here with bc = {bc}')
    diag_centered.eliminate_zeros()    # freeing some memory
    return {'mask': mask_centered, 'diag': diag_centered}


def apply_centered_bc(c: np.ndarray,
                      bc: dict,
                      bc_param: dict,
                      A: sp.spmatrix | None = None,
                      B: np.ndarray | None = None,
                      alg: str = 'Jacobian',
                      format: str = 'csr'):
    r"""
    Application of centered boundary condition to Matrix and/or RHS

    Parameters
    ----------
    c: np.ndarray
        array with field values
    bc: dict
        boundary condition in 'abd' format
    bc_param:
        dictionary with precomputed parameters
    A: spmatrix | None
        sparse matrix which needs to be manipulated
    B: np.ndarray | None
        RHS array which needs to be manipulated
    alg: str
        type of solving algorithm

    Returns
    -------
    manipulated A and/or B

    """
    if A is None and B is None:
        raise ValueError('Minimally A or G have to be specified')

    if isinstance(bc_param, list):
        for i in range(len(bc_param)):
            A, B = apply_centered_bc(c=c, bc=bc[i], bc_param=bc_param[i], A=A, B=B, alg=alg, format=format)
    else:
        mask = bc_param['mask']
        if A is not None:
            A = A.tolil()          # need to convert to lil here for efficiency, otherwise we are directly manipulating the sparse matrix structure, which is expensive! # noqa: E501
            A[mask, :] = 0         # setting the complete rows with prescribed pressure to 0
            A += bc_param['diag'].tolil()  # setting the main diagonal of the affected rows to 1
            match format:
                case 'csr':
                    A = sp.csr_matrix(A)
                case 'csc':
                    A = sp.csc_matrix(A)
                case 'coo':
                    A = sp.coo_matrix(A)
                case 'lil':
                    # do nothing
                    A = A
                case _:
                    raise ValueError(f'Unknown sparse matrix type: {format}')

        if B is not None:
            match alg:
                case 'Jacobian':
                    B.ravel()[mask] = c.ravel()[mask] - bc['d']
                case 'Direct':
                    B.ravel()[mask] = bc['d']
                case _:
                    raise ValueError(f'Unknown algorithm type: {alg}')

    return A, B


# c = np.linspace(0, 5, 6).reshape((2, -1))

# bc = {'a': 0, 'b': 1, 'd': 0}
# axis = 1
# boundary = 0

# bc_presc = compute_centered_bc(c, bc, axis, boundary)

# A = sp.csr_matrix((c.size, c.size))

# apply_centered_bc(bc, bc_presc, A, c.reshape(-1))
