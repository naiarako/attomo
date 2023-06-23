from numba import cuda
import math


@cuda.jit('void(int16, float32, int16, float32, int16, float32[:], float32[:], \
float32, float32, float32, float32, float32[:], float32[:], float32[:], \
float32[:], int16, float32, float32, float32)')
def das_gpu(nx, dx, nz, dz, nangles, angles, tx_delays, acq_delay, c, lens_c,
            lens_thickness, data_re, data_im, recons_re, recons_im, nt, fsampling, freq, apodtan):

    """
    Delay-and-sum (DAS) function to run in GPU
    :param nx: number of grid point in x-direction [int]
    :param dx: grid spacing in x-direction [float]
    :param nz: number of grid point in z-direction [int]
    :param dz: grid spacing in z-direction [float]
    :param nangles: number of total transmit angles [int]
    :param angles: transmit angles [numpy array, nangles x 1]
    :param tx_delays: delays in emission of plane waves [numpy array, nangles x 1]
    :param acq_delay: delay in acquisition or emission.  [float]
    :param c: assumed speed of sound [float]
    :param lens_c: speed of sound inside the lens [float]
    :param lens_thickness: lens thickness [float]
    :param data_re: real part of recorded crf-signals [numpy array, nt*nelem*nangles x 1]
    :param data_im: imaginary part of recorded crf-signals [numpy array, nt*nelem*nangles x 1]
    :param recons_re: (output) Real part of crf-images [numpy array, nx*nz*nangles x 1]
    :param recons_im: (output) Imaginary part of crf-images [numpy array, nx*nz*nangles x 1]
    :param nt: number of time samples in the signals [int]
    :param fsampling: sampling frequency [float]
    :param freq: center frequency [float]
    :param apodtan: apodization [float]
    """

    width = nx * nz * nangles  # Total number of threads

    startx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x  # index current thread
    gridx = cuda.gridDim.x * cuda.blockDim.x

    for ind in range(startx, width, gridx):

        # find indices angle and pixel in x and z
        i_angle = int(math.floor(ind / nx / nz))
        iz = ind % nz
        ix = int(math.floor((ind - i_angle * nx * nz) / nz))

        angle = angles[i_angle]  # current angle
        x = - 0.5 * (nx - 1) * dx + ix * dx  # current x-position
        z = 0.5 * dz + iz * dz  # current z-position

        t_tx = (x * math.sin(angle) + z * math.cos(angle)) / c  # travel time for plane wave to reach pixel

        tx_delay = tx_delays[i_angle]  # delay in transmission

        lens_cos = math.sqrt(1 - (math.sin(angle) * lens_c / c) ** 2)  # cosine of transmit angle in lens (Snell's law)
        t_tx_corr = lens_thickness * lens_cos / lens_c  # correction travel time inside lens (plane waves)

        for iel in range(nx + 1):  # loop through all elements

            xel = (-0.5 * nx + iel) * dx  # dx is equal to pitch and nx = n_elem - 1

            t_rx = math.sqrt((x - xel) ** 2 + z ** 2) / c

            # Lens correction reception (plane-wave approximation):
            angle_rx_sin = (x - xel) / (c * t_rx)  # sine of the incident angle from pixel
            lens_cos = math.sqrt(1 - (angle_rx_sin * lens_c / c) ** 2)  # cosine receive angle in lens (Snell's law)
            t_rx_corr = lens_thickness * lens_cos / lens_c  # lens correction

            # Total arrival time:
            t = t_rx + t_tx + t_tx_corr + t_rx_corr + tx_delay + acq_delay

            t_ind = int(math.floor(t * fsampling))

            # Apodization in reception
            if apodtan != 0:
                elem_weight = math.exp(-1.0 * ((x - xel) / z / apodtan) ** 2)
            else:
                elem_weight = 1.0

            if (t_ind >= 0) & (t_ind < nt):  # is the time index valid?

                ind_data = i_angle * nt * (nx + 1) + nt * iel + t_ind  # index of signal time sample in 1D data array

                phase = 2 * math.pi * (t - t_ind / fsampling) * freq  # for phase correction
                s_re = data_re[ind_data]  # real part of c-rf signals
                s_im = data_im[ind_data]  # imaginary part of c-rf signals

                # add signals from same pixels
                recons_re[ind] += elem_weight * (s_re * math.cos(phase) - s_im * math.sin(phase))
                recons_im[ind] += elem_weight * (s_im * math.cos(phase) + s_re * math.sin(phase))

