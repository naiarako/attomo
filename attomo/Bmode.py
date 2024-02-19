# -*- coding: utf-8 -*-
"""
:authors:
    Naiara Korta Martiartu (naiara.korta@unibe.ch)
    Michael Jaeger

    University of Bern
    2021 - 2023

:license:
    TODO
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.io
from scipy.signal import hilbert
from tqdm.notebook import tqdm_notebook
from scipy import signal
from numba import njit, cuda

try:
    import cupyx
    import cupy as cp
except ModuleNotFoundError:
    print('Cupy python package not found. Please, disable GPU options.')
    pass


class Bmode:

    def __init__(self, system):

        """
        Initialize Bmode object
        :param system: name of the ultrasound system [string, supported: kWave]
        """
        # update this when other systems are also included in load_data function
        assert (system == 'kWave'), "Please enter a supported system (kWave) "

        self.system = system

    def load_data(self, path, filename, use_filter=False, filter_type='high', cutoff_freq=0.2e7):

        """
        Load matlab file with acquisition info and data and define object attributes
        :param path: path to the folder containing .mat file [string]
        :param filename: name of the matlab file with .mat extension [string]
        :param use_filter: use band-pass filter in the data [boolean]
        :param filter_type: string indicating type of filter: 'bandpass', 'high', or 'low'
        :param cutoff_freq: float (for 'high' and 'low') or list of two-floats (for 'bandpass') indicating the
        cute-off frequencies.
        """

        start = timeit.default_timer()  # start counting run time

        print(50 * '-')
        print('Loading data...')

        if self.system == 'kWave':

            mat = scipy.io.loadmat(path + filename, struct_as_record=True)

            self.freq = float(mat['properties']['freq'][0, 0][0, 0])  # center frequency [Hz]
            self.pitch = float(mat['properties']['pitch'][0, 0][0, 0])  # distance between elements [m]
            self.c_system = float(mat['properties']['c0'][0, 0][0, 0])  # assumed SoS by the system [m]
            self.n_elem = int(mat['properties']['num_elements'][0, 0][0, 0])  # number of elements in the probe

            # acquisition delay [s] (time at which the US pulse is emitted)
            self.acq_delay = float(mat['properties']['t0'][0, 0][0, 0])

            self.lens_thickness = 0.0  # thickness of the acoustic lens [m]
            self.lens_c = 1000.0  # speed of sound of the acoustic lens [m/s]

            self.apodtan = np.tan(35 * np.pi / 180)  # element apodization

            self.fsampling = 1/(3*float(mat['properties']['dt'][0, 0][0, 0]))  # sampling frequency [Hz]

            # rf-signals [time, elem, angle]
            data = mat['scan_lines'][::3, :, :]

            # Filter:
            if use_filter:
                nyq = 0.5 * self.fsampling
                ba = signal.butter(8, cutoff_freq / nyq, btype=filter_type, output='ba')
                data_f = signal.filtfilt(ba[0], ba[1], np.real(data), axis=0)
                data = data_f

            # Hilbert transform data (crf-signals, [time, elem, angle]):
            self.data = hilbert(data, axis=0)
            self.nt = data.shape[0]  # recorded number of time samples

            # plane wave steering angles used for acquisition
            self.angles_sys = mat['properties']['angles'][0, 0][0] * np.pi/180

            # Time delays for the center of the aperture
            self.tx_delays = np.zeros_like(self.angles_sys)  # dummy to keep format

        end = timeit.default_timer()  # end counting run time

        print('...done!')
        print(f'Run time: {(end - start):.2f} sec')
        print(50 * '-')

    def grid(self, c=1540, rec_range=50e-3):

        """
        This function defines the reconstruction grid. It considers only a rectilinear mesh.
        :param c: Assumed speed of sound for reconstruction [scalar, m/s]
        :param rec_range: maximum depth for the reconstruction [scalar, m]
        """

        self.range = rec_range  # maximum reconstruction depth
        self.c = c  # assumed SoS for reconstruction

        dt = 1 / self.fsampling  # time step

        dx = self.pitch  # resolution in x axis
        dz = 0.5 * self.c * dt * 2  # resolution in z axis [last multiplication included to speed-up reconstructions]

        nx = round(((self.n_elem - 1) * self.pitch) / dx)  # Number of grid points in x direction
        nz = round(self.range / dz) + 1  # Number of grid points in z direction

        # Define the location of grid points
        self.x = np.arange(0, nx) * dx - 0.5 * (nx - 1) * dx
        self.z = np.arange(0, nz) * dz + 0.5 * dz

        self.elems = (np.arange(0, self.n_elem) - 0.5 * (
                self.n_elem - 1)) * self.pitch  # x-position of transducer elements

    def das(self, c=1540, rec_range=50e-3, gpu=False):

        """
         Delay-and-sum beamforming for individual plane-wave emissions. Array with crf-images is created as object
         attribute.
         :param c: Assumed speed of sound for reconstruction [scalar, m/s]
         :param rec_range: maximum depth for the reconstruction [scalar, m]
         :param gpu: run in GPU [boolean]
         """

        # define grid
        self.grid(c, rec_range)

        # Transform angles to our assumed SoS (Snell's law):
        self.angles = np.arcsin(np.sin(self.angles_sys) * self.c / self.c_system)

        # Coordinates of grid points:
        xx, zz = np.meshgrid(self.x, self.z)

        xx = np.reshape(xx, -1)  # reshape arrays to vector
        zz = np.reshape(zz, -1)

        print(50 * '-')
        print('Reconstruction in progress...')
        print(50 * '-')
        
        # take real and imaginary part of the data and convert to single precision
        data_re = np.real(self.data).astype(np.float32)
        data_im = np.imag(self.data).astype(np.float32)
        fsampling = self.fsampling

        if not gpu:

            # Initialize output image variable:
            self.recons = np.empty((len(self.z) * len(self.x), len(self.angles)), dtype=complex)

            # Loop through steering angles (transmit plane waves)
            for i_angle, angle in enumerate(tqdm_notebook(self.angles)):
                tx_delay = self.tx_delays[i_angle]  # delay in transmission

                recons_angle_re = np.zeros_like(xx, dtype=np.float32)  # initialize output single angles
                recons_angle_im = np.zeros_like(xx, dtype=np.float32)

                loop_das(angle, i_angle, xx, zz, self.lens_c,
                            self.lens_thickness, self.c, self.elems,
                            data_re, data_im, tx_delay, self.acq_delay, fsampling,
                            self.apodtan, self.freq, recons_angle_re, recons_angle_im)

                self.recons[:, i_angle].real = recons_angle_re  # assign to general output
                self.recons[:, i_angle].imag = recons_angle_im

            # reshape first dimension to 2d-array (mesh)
            self.recons = self.recons.reshape((len(self.z), len(self.x), len(self.angles)))

        else:

            # import DAS solver
            from .das_gpu import das_gpu

            # define inputs for DAS algorithm
            nt = data_re.shape[0] 
            nx = len(self.x)
            nz = len(self.z)
            dx = self.x[1] - self.x[0]
            dz = self.z[1] - self.z[0]
            nangles = len(self.angles)

            # Initialize output image variable:
            self.recons = np.empty((nz, nx, nangles), dtype=np.csingle)

            # inputs for CUDA kernel launch
            threadsperblock = 128
            blockspergrid = (nz * nx * self.n_elem + (threadsperblock - 1)) // threadsperblock

            # send data to gpu device as 1D array
            data_re_device = cuda.to_device(data_re.ravel('F'))
            data_im_device = cuda.to_device(data_im.ravel('F'))

            # send angles and delays to gpu device
            delays_device = cuda.to_device(self.tx_delays.astype(np.float32))
            angles_device = cuda.to_device(self.angles.astype(np.float32))

            # initialize output kernels in device as 1D array
            recons_re = cuda.to_device(np.zeros((nz * nx * nangles,), dtype=np.float32))
            recons_im = cuda.to_device(np.zeros((nz * nx * nangles,), dtype=np.float32))

            start = timeit.default_timer()  # start count run time

            # Launch CUDA kernels
            das_gpu[blockspergrid, threadsperblock](nx, dx, nz, dz, nangles, angles_device, delays_device,
                                                         self.acq_delay, self.c, self.lens_c, self.lens_thickness,
                                                         data_re_device, data_im_device, recons_re, recons_im, nt,
                                                         fsampling, self.freq, self.apodtan)

            # reshape output and assign to attributes
            self.recons.real = recons_re.copy_to_host().reshape((nz, nx, nangles), order='F')
            self.recons.imag = recons_im.copy_to_host().reshape((nz, nx, nangles), order='F')

            end = timeit.default_timer()  # end counting run time

            print(f'Run time: {(end - start)} sec')  # print run time

        print('...done!')
        print(50 * '-')

    def define_masks(self, resolution=1e-3, option_rec=0):

        """
        Define masks, i.e., non-illuminated areas of the image
        :param resolution: It defines how sharp are the edges of the masks [default: 1 mm]
        :param option_rec: 0: use acquisition angles; 1: use synthetic aperture angles
        :return masks: maps to mask non-illuminated areas per transmit angle [numpy array, size: nz x nx x n_angles]
        """

        # Coordinates of grid points:
        xx, zz = np.meshgrid(self.x - self.x[0], self.z)

        dx = self.x[1] - self.x[0]  # grid spacing in x direction

        angles = self.angles
        if option_rec == 1:
            angles = self.sa_angles

        # Initialize output array
        masks = np.zeros((len(self.z), len(self.x), len(angles)))

        for i_angle, angle in enumerate(angles):  # Loop through all angles

            # tanh function is used to define a smooth transition from 0 to 1 values

            mask1 = (0.5 + 0.5 * np.tanh((xx - zz * np.tan(angle) - 0.5 * dx) / resolution))  # Left side of the mask

            mask2 = (0.5 - 0.5 * np.tanh((xx - np.max(xx) + 0.5 * dx - zz * np.tan(angle)) / resolution))  # Right side

            masks[:, :, i_angle] = mask1 * mask2  # Multiply left and right sides

        return masks

    def plot_bmode(self, option_rec=0, dym_range=60, normlevel=-60, option_comp=1,
                   savefig=True, filename='Bmode', output=False):

        """
        Function to incoherently sum images from all steering angles and plot the result
        :param option_rec: 0: use acquisition angles; 1: use synthetic aperture angles
        :param dym_range: dynamic range of B-mode image [scalar] [dB]
        :param normlevel: reference value [scalar] [dB]
        :param option_comp: log compression option [integer] [valid values: 0, 1, 2]
        :param savefig: True: save B-mode image as .png and .pdf file [boolean]
        :param filename: Filename [without extension and including path] [string]
        :param output: True: returns reconstructed matrix. False: returns None [boolean]
        :return: if output True: reconstructed matrix [numpy array, nz x nx]; else: None
        """

        n_angles = len(self.angles)  # number of angles
        rec2abs = np.abs(self.recons) ** 2  # Intensity

        if option_rec == 1:
            n_angles = len(self.sa_angles)
            rec2abs = np.abs(self.sa_recons) ** 2  # Intensity

        rec2abs[rec2abs < 1e-20] = 1e-20  # To avoid errors with log

        rec = np.zeros_like(self.recons[:, :, 0], dtype=float)  # Initialize output array

        masks = self.define_masks(option_rec=option_rec)

        # Loop through all angles and log compress the intensity for each before summing all

        if option_comp == 0:

            for i in range(n_angles):
                tmp = 10 * np.log10(rec2abs[:, :, i]) - normlevel  # Log compression option 0

                tmp[tmp > dym_range] = dym_range  # Limit values for outliers
                tmp[tmp < 0] = 0

                rec += tmp * masks[:, :, i]  # mask not illuminated areas and add

        elif option_comp == 1:

            for i in range(n_angles):
                tmp = 10 * np.log10(rec2abs[:, :, i]) - np.max(
                    10 * np.log10(rec2abs[:, :, i])) - normlevel  # Log compression option 1

                tmp[tmp > dym_range] = dym_range  # Limit values for outliers
                tmp[tmp < 0] = 0

                rec += tmp * masks[:, :, i]  # mask not illuminated areas and add

        elif option_comp == 2:

            for i in range(n_angles):
                tmp = 10 * np.log10(rec2abs[:, :, i] / np.mean(
                    rec2abs[:, :, i])) - normlevel  # Log compression option 2

                tmp[tmp > dym_range] = dym_range  # Limit values for outliers
                tmp[tmp < 0] = 0

                rec += tmp * masks[:, :, i]  # mask not illuminated areas and add

        else:

            print('Please enter a valid option [0, 1, or 2]')

        # Take the mean and plot

        extent = 100 * np.min(self.x), 100 * np.max(self.x), 100 * np.max(self.z), 100 * np.min(self.z)

        plt.imshow(rec / n_angles, cmap='gray', aspect='equal', extent=extent, vmin=0, vmax=dym_range)

        plt.colorbar().set_label('Intensity (dB)')
        plt.xlabel('x (cm)')
        plt.ylabel('z (cm)')

        if savefig:  # save the image if required

            plt.savefig(filename + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(filename + '.png', dpi=300, bbox_inches='tight', facecolor='white')

        if output:  # return reconstructed matrix if required
            return rec / n_angles

    def coherent_compounding(self, sa_angles, sa_sigma, gpu=False):

        """
        Function to compute synthetically focused crf-images in transmission to reduced clutter
        :param sa_angles: synthetic aperture angles [numpy array, n_angles x 1] [rad]
        :param sa_sigma: standard deviation for gaussian weighting [scalar] [rad]
        :param gpu: if True, run in GPU [boolean]
        :return: sa_recons: reconstructed image per synthetic angle [numpy array, nz x nx x n_sa_angles]
        """

        start = timeit.default_timer()  # start counting run time

        print(50 * '-')
        print('Start coherent compounding...')

        self.sa_angles = sa_angles  # synthetic aperture angles

        if gpu:

            sa_angles_device = cp.array(sa_angles)
            angles_device = cp.array(self.angles)
            recons_device = cp.array(self.recons)

            # Initialize output
            self.sa_recons = cp.zeros((self.recons.shape[0], self.recons.shape[1], sa_angles.shape[0]), dtype=complex)

            for i_angle, sa_angle in enumerate(sa_angles_device):  # Loop over synthetic angles

                # Gaussian weights
                ang_w = cp.exp(-0.5 * ((sa_angle - angles_device) / sa_sigma) ** 2)

                # weighted sum over original images
                self.sa_recons[:, :, i_angle] = np.sum(recons_device * ang_w, axis=2)

            self.sa_recons = cp.asnumpy(self.sa_recons)

        else:

            # Initialize output
            self.sa_recons = np.zeros((self.recons.shape[0], self.recons.shape[1], sa_angles.shape[0]), dtype=complex)

            for i_angle, sa_angle in enumerate(sa_angles):  # Loop over synthetic angles

                # Gaussian weights
                ang_w = np.exp(-0.5 * ((sa_angle - self.angles) / sa_sigma) ** 2)

                # weighted sum over original images
                self.sa_recons[:, :, i_angle] = np.sum(self.recons * ang_w, axis=2)

        end = timeit.default_timer()  # end counting run time

        print('...done!')
        print(f'Run time: {(end - start):.2f} sec')
        print(50 * '-')


# Delay-and-sum CPU version:
@njit(parallel=True, fastmath=True)
def loop_das(angle, i_angle, xx, zz, lens_c, lens_thickness, c, elems,
             data_re, data_im, tx_delay, acq_delay, fsampling, apodtan, freq, recons_angle_re, recons_angle_im):
    """
    Parallelized and compiled delay-and-sum beamforming function. It computes the crf-image corresponding
     to each transmit angle.
    :param angle: current transmit angle [float]
    :param i_angle: index transmit angle [int]
    :param xx: x-position of grid points [numpy array, nx*nz x 1]
    :param zz: z-position of grid points [numpy array, nx*nz x 1]
    :param lens_c: speed of sound inside the lens [float]
    :param lens_thickness: lens thickness [float]
    :param c: assumed speed of sound [float]
    :param elems: x-position of elements/transducers [numpy array, nelem x 1]
    :param data_re: real part of recorded crf-signals [numpy array, nt x nelem x nangles]
    :param data_im: imaginary part of recorded crf-signals [numpy array, nt x nelem x nangles]
    :param tx_delay: delay in emission of current plane wave [float]
    :param acq_delay: delay in acquisition or emission.  [float]
    :param fsampling: sampling frequency [float]
    :param apodtan: apodization [float]
    :param freq: center frequency [float]
    :param recons_angle_re: (output) Real part of crf-image for current plane wave [numpy array, nx*nz x 1]
    :param recons_angle_im: (output) Imaginary part of crf-image for current plane wave [numpy array, nx*nz x 1]
    """

    t_tx = (xx * np.sin(angle) + zz * np.cos(angle)) / c  # travel time from plane wave to pixel

    # Lens correction in transmission (only depends on transmit angle):
    lens_cos = np.sqrt(
        1 - (np.sin(angle) * lens_c / c) ** 2)  # cosine of transmit angle in lens (Snell's law)
    t_tx_corr = lens_thickness * lens_cos / lens_c  # correction travel time inside lens (plane waves)

    for iel, elem in enumerate(elems):  # Loop through elements

        # Take recorded rf-signal by elem for steering angle:
        signal_re = data_re[:, iel, i_angle]
        signal_im = data_im[:, iel, i_angle]
        signal_re[-1] = 0.0  # Value assigned to invalid time indices
        signal_im[-1] = 0.0

        t_rx = np.sqrt((xx - elem) ** 2 + zz ** 2) / c  # travel time from pixel to receive element

        # Lens correction reception (plane wave approximation):
        angle_rx_sin = (xx - elem) / (c * t_rx)  # sine of the incident angle from pixel
        lens_cos = np.sqrt(1 - (angle_rx_sin * lens_c / c) ** 2)  # cosine receive angle in lens (Snell's law)
        t_rx_corr = lens_thickness * lens_cos / lens_c  # lens correction

        # Total arrival time:
        t = t_rx + t_tx + t_tx_corr + t_rx_corr + tx_delay + acq_delay

        t_ind = np.floor(t * fsampling).astype(np.int64)  # the nearest index of arrival time, t_ind <= t/dt
        mask = (t_ind >= 0) & (t_ind < data_re.shape[0])  # valid time indexes

        t_ind[~mask] = -1  # assign index -1 to invalid time indices

        # Apodization (weight received signals for each element)
        if apodtan != 0:
            elem_weight = np.exp(-1.0 * ((xx - elem) / zz / apodtan) ** 2)
        else:
            elem_weight = np.ones_like(xx)

        # phase for phase correction
        phase = 2 * np.pi * (t - t_ind / fsampling) * freq
        s_re = signal_re[t_ind]  # real part of c-rf signals
        s_im = signal_im[t_ind]  # imaginary part of c-rf signals

        # add signals from same pixels
        recons_angle_re += elem_weight * (s_re * np.cos(phase) - s_im * np.sin(phase))
        recons_angle_im += elem_weight * (s_im * np.cos(phase) + s_re * np.sin(phase))


