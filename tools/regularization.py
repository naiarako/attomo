import numpy as np
from scipy.sparse import diags
from scipy.linalg import inv, norm

try:
    import cupyx
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
except ModuleNotFoundError:
    pass


def regularization(order, nx, nz, reg_parameter=0, hessian=None, gpu=False, model=None):
    """
    Function to compute the posterior covariance matrix [model=None] or the regularization term(s) [model~=None]
    :param order: order of Tikhonov regularization. [int]
    :param nx: number of parameters in x-direction [int]
    :param nz: number of parameters in z-direction [int]
    :param reg_parameter: regularization parameter [float or list of 2 floats (x- and z-direction)]
    :param hessian: Hessian operator [GPU: sparse csr_gpu array, CPU: numpy array: nx*nz x nx*nz]
    :param gpu: computations in gpu? [boolean]
    :param model: medium properties, solution to the inverse problem [numpy array nx*nz x 1]
    :return: post: posterior covariance matrix [GPU: cupy array, CPU: numpy array: nx*nz x nx*nz]
             reg: value of the regularization term [if reg_parameter isotropic,
             then it contains reg in x and z; else only x]
             reg2: if regularization anisotropic, this is the regularization term in z
    """

    npoints = nx * nz  # number of model parameters

    if order == 0:  # zero-order Tikhonov regularization

        if model is None:
            assert (isinstance(reg_parameter, float)), "Regularization parameter must be a scalar."

        if model is not None:  # compute regularization term
            reg = norm(model)  # value regularization term

        else:  # compute posterior

            if gpu:  # in GPU

                R = cupyx.scipy.sparse.eye(hessian.shape[0])  # regularization matrix
                H = (hessian + reg_parameter * R).toarray()  # regularized hessian operator

                post = cp.linalg.inv(H)  # posterior covariance matrix

            else:  # in CPU

                H = hessian + reg_parameter * np.eye(hessian.shape[0])  # regularized Hessian operator

                post = inv(H)  # posterior covariance matrix

    elif order == 1:  # (an)isotropic first-order Tikhonov regularization (Dirichlet boundary conditions)

        # verify that regularization parameter is a list
        if model is None:
            assert (isinstance(reg_parameter, list)), "Regularization parameter must be " \
                                                      "a list of length 2, " \
                                                      "the regularization in x- and z-direction."

        if gpu:  # in GPU

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator in x

            diag[np.arange(1, nz + 1) * nx - 1] = 0  # 0 when the pixel is at the boundary of domain

            # finite-difference operator in x-direction (sparse matrix)
            fd_op_x = cupyx.scipy.sparse.diags([diag, -diag], [0, 1],
                                                    shape=(npoints, npoints))

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator in z

            diag[-nx:] = 0  # 0 when the pixel is at the boundary of domain

            # finite-difference operator in z-direction (sparse matrix)
            fd_op_z = cupyx.scipy.sparse.diags([diag, -diag], [0, nx], shape=(npoints, npoints))

            if model is not None:  # convert reg operators to numpy array for later
                fd_op_x = cp.asnumpy(fd_op_x)
                fd_op_z = cp.asnumpy(fd_op_z)

            else:  # compute posterior

                H = (hessian + reg_parameter[0] * (fd_op_x.transpose() * fd_op_x) +
                     reg_parameter[1] * (fd_op_z.transpose() * fd_op_z)).toarray()  # regularized hessian operator

                post = cp.linalg.inv(H)  # posterior covariance matrix

        else:  # in CPU

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator in x

            diag[np.arange(1, nz + 1) * nx - 1] = 0  # 0 when the pixel is at the boundary of domain

            # finite-difference operator in x-direction (sparse matrix)
            fd_op_x = diags([diag, -diag], [0, 1])

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator in z

            diag[-nx:] = 0  # 0 when the pixel is at the boundary of domain

            # finite-difference operator in z-direction (sparse matrix)
            fd_op_z = diags([diag, -diag], [0, nx])

            if model is None:  # compute posterior

                post = inv(hessian + reg_parameter[0] * (fd_op_x.transpose() * fd_op_x)
                           + reg_parameter[1] * (fd_op_z.transpose() * fd_op_z))  # posterior covariance matrix

        if model is not None:  # compute regularization term

            reg = norm(fd_op_x.toarray() * model)  # value of regularization term in x
            reg2 = norm(fd_op_z.toarray() * model)  # value of regularization term in z

    elif order == 2:  # (an)isotropic second-order Tikhonov regularization (Dirichlet boundary conditions)

        # verify that regularization parameter is a list
        if model is None:
            assert (isinstance(reg_parameter, list)), "Regularization parameter must be " \
                                                      "an list of length 2, " \
                                                      "the regularization in x- and z-direction."

        if gpu:  # in GPU

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator in x
            diag1 = np.ones((npoints,))

            diag[np.arange(1, nz + 1) * nx - 1] = 0  # 0 when the pixel is at the boundary of domain
            diag[np.arange(0, nz) * nx] = 0

            diag1[:-1] = diag[1:]  # 0 when the pixel is at the boundary of domain
            diag1[-1] = 0

            # finite-difference operator in x-direction (sparse matrix)
            fd2_op_x = cupyx.scipy.sparse.diags([2 * diag, -diag, -diag1], [0, 1, -1],
                                                     shape=(npoints, npoints))

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator in z
            diag1 = np.ones((npoints,))

            diag[:nx] = 0  # 0 when the pixel is at the boundary of domain
            diag[-nx:] = 0
            diag1[-2 * nx:] = 0

            # finite-difference operator in z-direction (sparse matrix)
            fd2_op_z = cupyx.scipy.sparse.diags([2 * diag, -diag, -diag1], [0, nx, -nx],
                                                     shape=(npoints, npoints))

            if model is not None:  # convert reg operators to numpy array for later
                fd2_op_x = cp.asnumpy(fd2_op_x)
                fd2_op_z = cp.asnumpy(fd2_op_z)

            else:  # compute posterior

                H = (hessian + reg_parameter[0] * (fd2_op_x.transpose() * fd2_op_x)
                     + reg_parameter[1] * (fd2_op_z.transpose() * fd2_op_z)).toarray()  # regularized hessian operator

                post = cp.linalg.inv(H)  # posterior covariance matrix

        else:  # in CPU

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator in x
            diag1 = np.ones((npoints,))

            diag[np.arange(1, nz + 1) * nx - 1] = 0  # 0 when the pixel is at the boundary of domain
            diag[np.arange(0, nz) * nx] = 0

            diag1[:-1] = diag[1:]  # 0 when the pixel is at the boundary of domain
            diag1[-1] = 0

            # finite-difference operator in x-direction (sparse matrix)
            fd2_op_x = diags([2 * diag, -diag, -diag1], [0, 1, -1])

            diag = np.ones((npoints,))  # diagonal elements of finite-difference operator in z
            diag1 = np.ones((npoints,))

            diag[:nx] = 0  # 0 when the pixel is at the boundary of domain
            diag[-nx:] = 0
            diag1[-2 * nx:] = 0

            # finite-difference operator in z-direction (sparse matrix)
            fd2_op_z = diags([2 * diag, -diag, -diag1], [0, nx, -nx])

            # compute posterior
            if model is None:
                post = inv(hessian + reg_parameter[0] * (fd2_op_x.transpose() * fd2_op_x)
                           + reg_parameter[1] * (fd2_op_z.transpose() * fd2_op_z))  # posterior covariance matrix

        if model is not None:  # compute regularization term

            reg = norm(fd2_op_x.toarray() * model)  # value of regularization term in x
            reg2 = norm(fd2_op_z.toarray() * model)  # value of regularization term in z

    # return posterior if required, else return regularization terms
    if model is None:
        return post
    else:
        if np.isscalar(reg_parameter):  # if order == 0
            return reg
        else:
            return reg, reg2





