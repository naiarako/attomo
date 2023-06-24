# -*- coding: utf-8 -*-
"""
:authors:
    Naiara Korta Martiartu (naiara.korta@iap.unibe.ch)
    Michael Jaeger

    University of Bern
    2021 - 2023

:license:
    TODO
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import timeit
from .ray_tracer import ray_tracer
from scipy.interpolate import RectBivariateSpline
from scipy.linalg import norm
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm_notebook
from .regularization import regularization

try:
    import cupyx
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
except ModuleNotFoundError:
    print('Cupy python package not found. Disable GPU options.')
    pass


class ATTomo:

    def __init__(self, bmode, spacing=None, gpu=False, range_rec=0):

        """
        Initialize class and define attributes
        :param bmode: Bmode object after synthetic focusing is performed
        :param spacing: grid spacing for attenuation reconstruction [list, [dx, dz]]
        :param gpu: if True, run computations in GPU whenever possible [boolean]
        :param: range_rec: max depth for reconstruction [scalar]
        """

        self.system = bmode.system  # Ultrasound system being used

        self.recons = bmode.sa_recons  # crf-images after synthetic focusing
        self.angles = bmode.sa_angles  # angles corresponding to crf-images
        self.c = bmode.c  # assumed SoS for reconstruction
        self.range = bmode.range if range_rec == 0 else range_rec  # max depth of the reconstructed image
        self.x_rec = bmode.x  # reconstruction grid points in x-axis
        self.z_rec = bmode.z  # reconstruction grid points in z-axis
        self.freq = bmode.freq  # center frequency

        spacing = [1e-3, 1e-3] if spacing is None else spacing  # grid spacing for tomography

        self.nx = round((bmode.elems[-1 - 0] - bmode.elems[0 + 0]) / spacing[0])  # number of grid points in x
        self.nz = round(self.range / spacing[1])  # number of grid points in z

        # tomography grid points in x-axis
        self.tomo_x = np.arange(0, self.nx) * spacing[0] - 0.5 * (self.nx - 1) * spacing[0]

        # tomography grid points in z-axis
        self.tomo_z = np.arange(0, self.nz) * spacing[1] + 0.5 * spacing[1]

        self.gpu = gpu  # run in gpu?

    def logamp_measurements(self, kernel_size=None):

        """
        Function for computing normalized cross-correlation log-amplitudes (i.e., the observed data for tomography)
        :param kernel_size: kernel size for ensemble averaging [list, default=[2e-3, 2e-3]]
        Created attributes:
        - self.amp_map: echo-amplitude change between successive emissions [numpy array, nz x nx x n_angles]
        - self.ccs_map: cross-correlations between successive emissions [numpy array, nz x nx x n_angles]
        - self.auto_map: auto-correlations first emissions [numpy array, nz x nx x n_angles]
        - self.auto2_map: auto-correlations second emissions [numpy array, nz x nx x n_angles]
        """

        start = timeit.default_timer()  # start counting run time

        print(50 * '-')
        print('Computing log-amplitude measurements...')

        kernel_size = [2e-3, 2e-3] if kernel_size is None else kernel_size  # assign default value to kernel

        dx = self.x_rec[1] - self.x_rec[0]  # grid spacing
        dz = self.z_rec[1] - self.z_rec[0]

        size_ker = [round(kernel_size[0] / dz), round(kernel_size[1] / dx)]  # kernel size in pixels

        if self.gpu:

            from cupyx.scipy.signal import convolve2d as conv2

            # Initialize cross(auto)-correlation maps
            self.ccs_map = cp.zeros((self.recons.shape[0], self.recons.shape[1],
                                            self.recons.shape[2] - 1), dtype=cp.complex)
            self.auto_map = cp.zeros((self.recons.shape[0], self.recons.shape[1],
                                      self.recons.shape[2] - 1), dtype=cp.complex)
            self.auto2_map = cp.zeros((self.recons.shape[0], self.recons.shape[1],
                                       self.recons.shape[2] - 1), dtype=cp.complex)

            # Define kernel:
            kernel = cp.ones((size_ker[0], size_ker[1]), dtype=cp.complex)
            kernel /= kernel.sum()

            recons_gpu = cp.array(self.recons)

            for i in range(self.recons.shape[2] - 1):  # Loop over angles

                # Zero-lag cross(auto)-correlation of analytical signals
                crosscor = recons_gpu[:, :, i] * cp.conj(recons_gpu[:, :, i + 1])  # cross-correlation
                autocc = recons_gpu[:, :, i] * cp.conj(recons_gpu[:, :, i])  # auto-correlations
                autocc2 = recons_gpu[:, :, i + 1] * cp.conj(recons_gpu[:, :, i + 1])

                # Ensemble averaging via kernel
                cc = conv2(crosscor, kernel, mode='same', boundary='symm')
                auto = conv2(autocc, kernel, mode='same', boundary='symm')
                auto2 = conv2(autocc2, kernel, mode='same', boundary='symm')

                self.ccs_map[:, :, i] = cc
                self.auto_map[:, :, i] = auto
                self.auto2_map[:, :, i] = auto2

            self.ccs_map = cp.asnumpy(self.ccs_map)
            self.auto_map = cp.asnumpy(self.auto_map)
            self.auto2_map = cp.asnumpy(self.auto2_map)

        else:

            # Initialize cross(auto)-correlation maps
            self.ccs_map = np.zeros((self.recons.shape[0], self.recons.shape[1],
                                     self.recons.shape[2] - 1), dtype=complex)  # Initialize output
            self.auto_map = np.zeros((self.recons.shape[0], self.recons.shape[1],
                                      self.recons.shape[2] - 1), dtype=complex)  # Initialize output
            self.auto2_map = np.zeros((self.recons.shape[0], self.recons.shape[1],
                                       self.recons.shape[2] - 1), dtype=complex)  # Initialize output

            # Define kernel:
            kernel = np.ones((size_ker[0], size_ker[1]))
            kernel /= kernel.sum()

            for i in range(self.recons.shape[2] - 1):  # Loop over angles

                # # Zero-lag cross(auto)-correlation of analytical signals
                crosscor = self.recons[:, :, i] * np.conj(self.recons[:, :, i + 1])  # cross-correlation
                autocc = self.recons[:, :, i] * np.conj(self.recons[:, :, i])  # auto-correlations
                autocc2 = self.recons[:, :, i + 1] * np.conj(self.recons[:, :, i + 1])

                # Ensemble averaging via kernel
                cc = convolve2d(crosscor, kernel, mode='same', boundary='symm')
                auto = convolve2d(autocc, kernel, mode='same', boundary='symm')
                auto2 = convolve2d(autocc2, kernel, mode='same', boundary='symm')

                self.ccs_map[:, :, i] = cc
                self.auto_map[:, :, i] = auto
                self.auto2_map[:, :, i] = auto2

        self.amp_map = 0.5 * (np.log(np.abs(self.ccs_map) / np.abs(self.auto_map),
                                         where=np.abs(self.ccs_map) / np.abs(self.auto_map) > 0) -
                                  np.log(np.abs(self.ccs_map) / np.abs(self.auto2_map),
                                         where=np.abs(self.ccs_map) / np.abs(self.auto2_map) > 0))

        end = timeit.default_timer()  # end counting run time

        print('...done!')
        print(f'Run time: {(end - start):.2f} sec')
        print(50 * '-')

    def define_masks(self, angles, resolution=1e-3, tomo=False):

        """
        Define masks for non-illuminated areas of the image
        :param angles: angle list for masks [np.array, num_angles, radians]
        :param resolution: It defines how sharp are the edges of the masks [default: 1 mm]
        :param tomo: if True, consider grid and angles for tomography; if False, take the ones for beamforming [boolean]
        :return masks: maps to mask non-illuminated areas per transmit angle [numpy array, size: nz x nx x n_angles]
        """

        if tomo:

            # Coordinates of grid points:
            x = self.tomo_x
            z = self.tomo_z

            dx = x[1] - x[0]  # grid spacing in x direction

            delta = 2 * np.pi / 180   # to reduce edge waves

        else:

            # Coordinates of grid points:
            x = self.x_rec
            z = self.z_rec

            dx = x[1] - x[0]  # grid spacing in x direction

            delta = 0 * np.pi / 180  # to reduce edge waves

        xx, zz = np.meshgrid(x - x[0], z)  # mesh grid points

        # Initialize output array
        masks = np.zeros((len(z), len(x), len(angles)))

        for i_angle, angle in enumerate(angles):  # Loop through all angles

            # tanh function is used to define a smooth transition from 0 to 1 values

            # Left side of the mask
            mask1 = (0.5 + 0.5 * np.tanh((xx - zz * np.tan(angle + delta) - 0.5 * dx) / resolution))

            # Right side
            mask2 = (0.5 - 0.5 * np.tanh(
                (xx - np.max(xx) + 0.5 * dx - zz * np.tan(angle - delta)) / resolution))

            masks[:, :, i_angle] = mask1 * mask2  # Multiply left and right sides

        return masks

    def add_data(self, tomo_angles):

        """
        Add data to compute the data corresponding to larger angle steps (if needed) and interpolate them to the
        tomography grid (coarser).
        :param tomo_angles: angles for which we compute data [numpy array, 1 x n_angles] [rad]
        Created attribute:
        - self.tomo_logamp_map: log amplitude maps defined in tomographic grid [numpy array, nz x nx x n_angles]
        """

        start = timeit.default_timer()  # start counting run time

        print(50 * '-')
        print('Adding and interpolating log-amplitudes...')

        self.tomo_angles = tomo_angles  # create attribute with angles considered for tomography
        self.tomo_nangles = len(tomo_angles)  # number of angles

        # Define not illuminated areas (masks)
        masks = self.define_masks(self.angles, resolution=1e-6)

        # Initialize log amplitude maps defined in tomographic grid (coarser)
        self.tomo_logamp_map = np.zeros((self.nz, self.nx, self.tomo_nangles - 1))

        for i in range(self.tomo_nangles - 1):  # loop over angles

            # indexes of first and last data maps for the summation
            ind_0 = np.where(self.angles == self.tomo_angles[i])[0][0]
            ind_1 = np.where(self.angles == self.tomo_angles[i + 1])[0][0]

            # sum maps
            tmp_logamp = np.zeros_like(self.amp_map[:, :, 0])
            for j in np.arange(ind_0, ind_1):
                tmp_logamp += self.amp_map[:, :, j]

            # interpolate to tomography grid and mask non-illuminated areas
            self.tomo_logamp_map[:, :, i] = self.interpolate_map(tmp_logamp * masks[:, :, ind_0] * masks[:, :, ind_1])

        end = timeit.default_timer()  # end counting run time

        print('...done!')
        print(f'Run time: {(end - start):.2f} sec')
        print(50 * '-')

    def interpolate_map(self, data_map, x_new=None, x_orig=None):

        """
        Function to interpolate data maps to the coarser rectilinear grid used for tomography
        :param data_map: original data map defined in x_orig grid [numpy array, nz x nx]
        :param x_new: new grid [list, z_new x x_new]
        :param x_orig: original grid [list, z x x]
        :return: map_new: interpolated maps in x_new grid [numpy array, nz_new x nx_new]
        """
        
        x_orig = [self.z_rec, self.x_rec] if x_orig is None else x_orig
        x_new = [self.tomo_z, self.tomo_x] if x_new is None else x_new

        # Define interpolator for bivariate spline approximation over rectangular grid
        interpolator = RectBivariateSpline(x_orig[0], x_orig[1], data_map)

        # interpolate to new grid
        map_new = interpolator(x_new[0], x_new[1])

        return map_new
        
    def compute_forward_op(self):

        """
        Function to compute the forward operator for our linear tomographic problem. It assumes straight rays.
        It computes forward operator for angle and -angle simultaneously.
        Valid for angles that are centered around zero, angle=0 included.
        Created attributes:
        - self.forward_op: submatrices of the forward operator. For each angle, the submatrix is composed by line
                                  integrals from z=0 to each pixel position, following the transmit angle direction.
                                  [compressed Sparse Row matrix, nx*nz x nx*nz x tomo_nangles]
        """

        print(50 * '-')
        print('Computing forward operator...')

        self.forward_op = [0] * self.tomo_nangles  # initialize output

        xx, zz = np.meshgrid(self.tomo_x, self.tomo_z)  # coordinate of grid points

        xx = np.reshape(xx, -1)  # reshape arrays to vector
        zz = np.reshape(zz, -1)

        dx = self.tomo_x[1] - self.tomo_x[0]  # grid spacing
        dz = self.tomo_z[1] - self.tomo_z[0]  # grid spacing

        if self.tomo_nangles % 2 == 0:  # if number of angles is even (there is no angle = 0)

            ind = int(self.tomo_nangles / 2)  # take half of the angles

            for i, angle in enumerate(tqdm_notebook(self.tomo_angles[0: ind])):  # loop over these angles

                # compute sparse matrices with ray paths for angle and -angle
                vals, ind_row, ind_col, ind_row_neg_angle, ind_col_neg_angle = ray_tracer(angle, xx, zz,
                                                                                      self.tomo_x[[0, -1]], dx,
                                                                                      dz,
                                                                                      self.nx, self.nz)

                self.forward_op[i] = csr_matrix((vals, (ind_row, ind_col)),
                                                shape=(self.nx * self.nz,
                                                       self.nx * self.nz))  # (angle)

                self.forward_op[-i - 1] = csr_matrix((vals, (ind_row_neg_angle, ind_col_neg_angle)),
                                                     shape=(self.nx * self.nz,
                                                            self.nx * self.nz))  # (-angle)

        else:  # if number of angles is odd (there is angle = 0)

            ind = int((self.tomo_nangles - 1) / 2)  # take half of the angles except angle = 0

            for i, angle in enumerate(tqdm_notebook(self.tomo_angles[0: ind])):  # loop over these angles

                # compute sparse matrices with ray paths for angle and -angle
                vals, ind_row, ind_col, ind_row_neg_angle, ind_col_neg_angle = ray_tracer(angle, xx, zz,
                                                                                      self.tomo_x[[0, -1]], dx,
                                                                                      dz,
                                                                                      self.nx, self.nz)

                self.forward_op[i] = csr_matrix((vals, (ind_row, ind_col)),
                                                shape=(self.nx * self.nz,
                                                       self.nx * self.nz))  # (angle)

                self.forward_op[-i - 1] = csr_matrix((vals, (ind_row_neg_angle, ind_col_neg_angle)),
                                                     shape=(self.nx * self.nz,
                                                            self.nx * self.nz))  # (-angle)

            # compute ray paths for angle = 0
            vals, ind_row, ind_col, _, _ = ray_tracer(self.tomo_angles[ind], xx, zz,
                                                  self.tomo_x[[0, -1]], dx, dz,
                                                  self.nx, self.nz, neg=False)

            self.forward_op[ind] = csr_matrix((vals, (ind_row, ind_col)),
                                              shape=(self.nx * self.nz,
                                                     self.nx * self.nz))

        print('...done!')
        print(50 * '-')

    def plot_ray_density(self, savefig=False, filename='Ray_density'):

        """
        Function to plot ray density from forward operator (sum of all ray paths in each pixel).
        :param savefig: if True, save figrue with ray density [boolean]
        :param filename: if savefig==True, the filename in which the figure is saved [string]
        """

        ray_density = np.zeros((1, self.nz * self.nx))  # initialize ray density array [nz x nx]

        for i in range(self.tomo_nangles-1):  # loop over all angles
            # for each angle (submatrix of the forward operator) sum rows (ray paths). Note that our forward operator
            # relates attenuation to data changes between every pair of succesive angles.
            a = self.forward_op[i+1] > 0  # value 1 for non-zero elements, 0 otherwise
            b = self.forward_op[i] > 0
            ray_density += np.sum(a + b, axis=0)

        # Define extent of imaging domain and plot
        extent = 100 * np.min(self.tomo_x), 100 * np.max(self.tomo_x), 100 * np.max(self.tomo_z), 100 * np.min(self.tomo_z)

        plt.imshow(np.reshape(ray_density/np.max(ray_density), (self.nz, self.nx)),
                   aspect='equal', extent=extent, cmap='plasma')
        plt.colorbar().set_label('Normalized\nnumber of rays')
        plt.xlabel('Lateral direction (cm)')
        plt.ylabel('Axial direction (cm)')
        plt.title('Ray density')

        if savefig:  # save the image if required

            plt.savefig(filename + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(filename + '.png', dpi=300, bbox_inches='tight', facecolor='white')

    def inversion(self, reg_parameter=1e-3, order=0, att_exponent=1.0, toprint=True):

        """
        Function to solve the Tikhonov-regularized linear inverse problem
        :param reg_parameter: regularization parameter [positive scalar]
        :param order: order of Tikhonov regularization [integer, supported: 0, 1, or 2]
        :param att_exponent: power-law exponent for transforming results to attenuation coefficient [integer]
        :param toprint: if True, print run times [boolean]
        Created attributes:
        - self.model_rec: reconstructed model [numpy array, nz*nx x 1]
        - self.att: computed attenuation coefficient from model_rec [numpy array, nz*nx x 1]
        - self.post: posterior covariance operator [numpy array, nz*nx x nz*nx]
        - self.hessian: hessian operator [numpy array, nz*nx x nz*nx]
        """

        start = timeit.default_timer()  # start counting run time

        if toprint:
            print(50 * '-')
            print('Start inversion...')

        npoints = self.nx * self.nz  # number of grid points (number of model parameters)

        data = -self.tomo_logamp_map

        if self.gpu:  # in GPU

            # Define not illuminated areas (masks)
            masks = cp.array(self.define_masks(self.tomo_angles, resolution=1e-6, tomo=True))

            dobs_gpu = cp.array(data)

            self.hessian = csr_gpu((npoints, npoints))  # initialize hessian operator and gradient
            grad = cp.zeros((npoints, self.tomo_nangles - 1))

            for i in range(self.tomo_nangles - 1):  # loop over pair of angles

                mask = cp.reshape(masks[:, :, i + 1] * masks[:, :, i], (-1, 1))

                # submatrix of the forward operator for angles i, i+1
                A = (csr_gpu(self.forward_op[i + 1]) - csr_gpu(self.forward_op[i])).multiply(mask)

                # save gradients for later
                grad[:, i] = A.transpose().dot(cp.reshape(dobs_gpu[:, :, i], -1))

                # Hessian operator: compute hessian for submatrices and add
                self.hessian += A.transpose() * A

            # compute posterior covariance operator:
            self.post = regularization(order=order, reg_parameter=reg_parameter, nx=self.nx,
                                       nz=self.nz, hessian=self.hessian, gpu=self.gpu)

            # Initialize model parameters:
            self.model_rec = cp.zeros((npoints,))

            for i in range(self.tomo_nangles - 1):  # loop over pair of angles

                # invert model parameters
                self.model_rec += cp.dot(self.post, grad[:, i])

            self.model_rec = cp.asnumpy(self.model_rec)
            self.post = cp.asnumpy(self.post)
            self.hessian = cp.asnumpy(self.hessian.toarray())

            # transform to dB/cm/MHz^y 
            self.att = self.model_rec * 8.6886 / 100 / (self.freq / 1e+6) ** att_exponent

        else:  # in CPU

            # Define not illuminated areas (masks)
            masks = self.define_masks(self.tomo_angles, resolution=1e-6, tomo=True)

            self.hessian = np.zeros((npoints, npoints))  # initialize hessian operator and gradient.
            grad = np.zeros((npoints, self.tomo_nangles - 1))

            for i in range(self.tomo_nangles - 1):  # loop over pair of angles

                mask = np.reshape(masks[:, :, i + 1] * masks[:, :, i], (-1, 1))  # current masks

                # submatrix of the forward operator for angles i, i+1
                A = (self.forward_op[i + 1] - self.forward_op[i]).multiply(mask)

                # save gradients for later
                grad[:, i] = A.transpose().dot(np.reshape(data[:, :, i], -1))

                # Hessian operator: compute hessian for submatrices and add
                self.hessian += A.transpose() * A

            # compute posterior covariance operator:
            self.post = regularization(order=order, reg_parameter=reg_parameter, nx=self.nx,
                                       nz=self.nz, hessian=self.hessian, gpu=self.gpu)

            # Initialize model parameters:
            self.model_rec = np.zeros((npoints,))

            for i in range(self.tomo_nangles - 1):  # loop over pair of angles

                # invert model parameters
                self.model_rec += np.dot(self.post, grad[:, i])

            # transform to dB/cm/MHz^y 
            self.att = self.model_rec * 8.6886 / 100 / (self.freq / 1e+6) ** att_exponent

        end = timeit.default_timer()  # end counting run time

        if toprint:
            print('...done!')
            print(f'Run time: {(end - start):.2f} sec')
            print(50 * '-')

    def plot_lcurve(self, order=0, reg_params=None,
                    reg_anis_ratio=1.0, savefig=False, filename='L-curve_tikhonov_order'):

        """
        Function to plot the L-curve useful to optimally choose the regularization parameter.
        :param order: order of Tikhonov regularization [integer, supported: 0, 1, or 2]
        :param reg_params: list with regularization parameter values. If None, take default ones. [list floats]
        :param reg_anis_ratio: Ratio between reg. parameter values in x- and z-direction. The value in z is
                                multiplied by this value [float]
        :param savefig: if True, save the L-curve figure [boolean]
        :param filename: if savefig==True, filename in which the figure will be saved [string]
        """

        npoints = self.nx * self.nz  # number of grid points (number of model parameters)

        data = -self.tomo_logamp_map

        # selected regularization parameters
        if reg_params is None:
            reg_params = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

        start = timeit.default_timer()  # start counting run time

        # Define not illuminated areas (masks)
        masks = self.define_masks(self.tomo_angles, resolution=1e-6, tomo=True)

        # initialize vectors with all misfits and regularization terms
        misfit = np.zeros((len(reg_params),))
        reg = np.zeros_like(misfit)
        reg2 = np.zeros_like(misfit)

        print(50 * '-')
        print(f'Computing {len(misfit)} inversions to plot the L-curve...')

        for i in range(len(misfit)):  # loop over different regularization parameters

            # run inversion for the given reg_parameter
            if order == 0:
                reg_parameter = reg_params[i]
            else:
                reg_parameter = [reg_params[i], reg_anis_ratio * reg_params[i]]

            print(50 * '-')
            print(f'Start inversion #{i + 1}...')

            self.inversion(reg_parameter=reg_parameter, order=order, toprint=False)

            print('...done!')
            print(50 * '-')

            residual = np.zeros(((self.tomo_nangles - 1) * npoints,))  # initialize vector of residuals

            for angle in range(self.tomo_nangles - 1):  # loop over pair of angles

                mask = np.reshape(masks[:, :, angle + 1] * masks[:, :, angle], (-1, 1))  # current masks

                # submatrix of the forward operator for angles i, i+1
                A = (self.forward_op[angle + 1] - self.forward_op[angle]).multiply(mask)

                # observed data for the specific angle
                dobs = np.reshape(data[:, :, angle], -1)

                # compute residuals for the specific angle
                residual[angle * npoints:(angle + 1) * npoints] = dobs - A.dot(self.model_rec)

            misfit[i] = norm(residual)  # store corresponding misfit term

            # compute regularization term

            if order == 0:
                reg[i] = regularization(order=order, reg_parameter=reg_parameter, nx=self.nx,
                                        nz=self.nz, model=self.model_rec)
            else:
                reg[i], reg2[i] = regularization(order=order, reg_parameter=reg_parameter,
                                                 nx=self.nx, nz=self.nz, model=self.model_rec)

        # plot L-curve (loglog scale) and save it if required
        plt.loglog(misfit, reg + reg_anis_ratio * reg2, '*-', color='k')
        plt.xlabel('Misfit')
        plt.ylabel('Regularization')
        plt.title('L-curve')

        if savefig:  # save the image if required

            plt.savefig(filename + f'{order}.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(filename + f'{order}.png', dpi=300, bbox_inches='tight', facecolor='white')

        end = timeit.default_timer()  # end counting run time

        print('...done L-curve computation!')
        print(f'Total run time: {(end - start) / 60:.2f} min')
        print(50 * '-')

    def misfit_reduction(self):

        """
        Function to compute misfit reduction achieved by the solution of the inverse problem
        """

        npoints = len(self.model_rec)  # number of model parameters

        data = -self.tomo_logamp_map

        masks = self.define_masks(self.tomo_angles, resolution=1e-6, tomo=True)  # Define not illuminated areas (masks)

        residual = np.zeros(((self.tomo_nangles - 1) * npoints,))  # initialize useful vectors
        dobs = np.zeros_like(residual)

        for i in range(self.tomo_nangles - 1):  # loop over pair of angles

            mask = np.reshape(masks[:, :, i + 1] * masks[:, :, i], (-1, 1))  # current masks

            # submatrix of the forward operator for angles i, i+1
            A = (self.forward_op[i + 1] - self.forward_op[i]).multiply(mask)

            # observed data
            dobs[i * npoints:(i + 1) * npoints] = np.reshape(data[:, :, i], -1)

            # residual between observed data and predicted data
            residual[i * npoints:(i + 1) * npoints] = dobs[i * npoints:(i + 1) * npoints] - A.dot(self.model_rec)

        initial_misfit = norm(dobs)  # initial misfit (equal to norm of observed data)
        final_misfit = norm(residual)  # final misfit

        # Print misfit reduction in %:
        print(f'Misfit reduction: \
            {(initial_misfit ** 2 - final_misfit ** 2) / (initial_misfit ** 2) * 100:.2f} %')

    def compute_obs_data(self, model):

        """
        Function to compute predicted data from reconstructed attenuation model
        :param model: reconstructed attenuation of the medium [Np/m]. Can be scalar or numpy_array of size nx*nz x 1.
        :return: predicted data [numpy_array, nz x nx x n_angles - 1]
        """

        medium = model * np.ones((self.nz * self.nx,))

        masks = self.define_masks(self.tomo_angles, resolution=1e-6, tomo=True)  # Define not illuminated areas (masks)

        dobs = np.zeros_like(self.tomo_logamp_map)  # initialize observed data

        for i in range(self.tomo_nangles - 1):  # loop over pair of angles

            mask = np.reshape(masks[:, :, i + 1] * masks[:, :, i], (-1, 1))  # current masks

            # submatrix of the forward operator for angles i, i+1
            A = (self.forward_op[i + 1] - self.forward_op[i]).multiply(mask)

            # observed data
            dobs[:, :, i] = np.reshape(A.dot(medium), (self.nz, self.nx))

        return dobs
