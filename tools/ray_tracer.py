"""
:authors:
    Naiara Korta Martiartu (naiara.korta@iap.unibe.ch)
    Michael Jaeger

    University of Bern
    2021

:license:
    TODO
"""

import numpy as np
from numba import njit


@njit(cache=True)
def ray_tracer(angle, xx, zz, x, dx, dz, nx, nz, neg=True):

    """
    Straight-ray tracer from a plane wave with angle 'angle' to virtual receivers at all gridpoints specified
    by 'x' and 'z'.
    :param angle: transmit plane wave angle. Convention: positive from z-axis counterclockwise [scalar; radians]
    :param xx: mesh grid locations in x direction [numpy array, nx*nz x 1]
    :param zz: mesh grid locations in z direction [numpy array, nz*nx x 1]
    :param x: limits of grid locations in x direction [numpy array, 2 x 1]
    :param dx: grid size in x direction [scalar]
    :param dz: grid size in z direction [scalar]
    :param nx: number of grid points in x direction [scalar]
    :param nz: number of grid points in z direction [scalar]
    :param neg: if True, it gives the output for planes waves with angles 'angle' and '-angle'. [boolean]
                if False, it only gives the output for the plane wave with 'angle'.
    :return: vals: numpy array for non-zero values of the forward operator relating traveltimes with the
    slowness of the medium. [numpy array]
             ind_row: numpy array indicating the row number of non-zero values of the forward operator [numpy array]
             ind_col: numpy array indicating the column number of non-zero values of the forward operator [numpy array]
            The forward operator has: Rows: number of rays; Columns: number of gridpoints. Each row is a ray path,
            specifying the length of the ray in each pixel. This function returns the inputs for Compressed Sparse Row
            matrix of dimension nx*nz x nx*nz.
    """

    # Compute number of intersections with pixel edges.
    niz_all = (np.floor(zz / dz)).astype(np.int64) + 1  # number of intersections in z. From zz to z=0 (included)
    nix_all = (np.floor((xx - (x[0] - dx / 2)) / dx)).astype(np.int64)  # From left edge of origin x[0] to pixel xx

    ind_col = np.zeros((nx * nz * 2 * nz,), dtype=np.int64)  # initialize output. Column indices
    ind_row = np.zeros_like(ind_col)  # row indices
    vals = np.zeros((nx * nz * 2 * nz,))  # corresponding values of the matrix

    if neg:  # if True, initialize also the output for the plane wave with negative angle
        ind_col_neg_angle = np.zeros((nx * nz * 2 * nz,), dtype=np.int64)
        ind_row_neg_angle = np.zeros_like(ind_col)

    pointer = 0  # the pointer indicates the number of nonzero values in the sparse matrix (output) at each iteration

    if angle == 0:  # if angle==0, trivial length of rays in each pixel

        for i in range(nx * nz):  # loop through all virtual receivers

            if niz_all[i] == 1:  # if we are in the first pixel row, length is dz/2
                ind_col[pointer] = nix_all[i]
                ind_row[pointer] = i
                vals[pointer] = 0.5 * dz

                pointer += 1

            else:  # else, length is dz in each pixel expect for the one where the virtual receiver is located (dz/2)
                ind_col[pointer:pointer + niz_all[i]] = nix_all[i] + np.arange(niz_all[i]) * nx
                ind_row[pointer:pointer + niz_all[i]] = i
                vals[pointer:pointer + niz_all[i] - 1] = dz
                vals[pointer + niz_all[i] - 1] = 0.5 * dz  # pixel where the virtual receiver is located

                pointer += niz_all[i]

    else:  # if angle ~= 0

        for i in range(nx * nz):  # loop through all virtual receivers

            x_surf = xx[i] + zz[i] * np.tan(-angle)  # x-position where the ray crosses the surface (z = 0)

            if (x_surf <= x[0] - dx / 2) or (x_surf >= x[-1] + dx / 2):  # if x_surf is outside aperture, no ray
                continue

            # number of intersections in x and z direction from (x_surf,0) to (xx, zz)
            nix = int(abs(nix_all[i] - np.floor((x_surf - (x[0] - dx / 2)) / dx)))
            niz = niz_all[i]

            npoints = nix + niz + 1  # number of intersections with pixel edges + 1 (virtual receiver location)

            inter_points = np.ones((npoints, 2))  # initialize array with locations of intersecting points (x,z)

            inter_points[0, 0] = xx[i]  # first value is the virtual receiver location
            inter_points[0, 1] = zz[i]  # first value is the virtual receiver location
            inter_points[-1, 0] = x_surf  # last intersection is with surface
            inter_points[-1, 1] = 0  # last intersection is with surface

            direction = [np.sin(-angle), -np.cos(angle)]  # unitary vector indicating ray direction

            # intersecting points with pixel edges in z-direction
            if niz == 1:

                inter_points[1, 1] = 0
                step_z = - zz[i] / direction[1]
                inter_points[1, 0] = xx[i] + step_z * direction[0]

            else:

                inter_points[1:niz + 1, 1] = np.arange(niz) * dz
                step_z = (np.arange(niz) * dz - zz[i]) / direction[1]
                inter_points[1:niz + 1, 0] = xx[i] + step_z * direction[0]

            # intersecting points with pixel edges in x-direction
            if nix == 1:

                sign = (x_surf - xx[i]) / abs(x_surf - xx[i])
                inter_points[niz + 1, 0] = xx[i] + sign * 0.5 * dx
                step_x = sign * 0.5 * dx / direction[0]
                inter_points[niz + 1, 1] = zz[i] + step_x * direction[1]

            elif nix > 1:

                sign = (x_surf - xx[i]) / abs(x_surf - xx[i])
                inter_points[niz + 1: niz + nix + 1, 0] = \
                    xx[i] + sign * (0.5 * dx + np.arange(nix) * dx)
                step_x = sign * (0.5 * dx + np.arange(nix) * dx) / direction[0]
                inter_points[niz + 1: niz + nix + 1, 1] = zz[i] + step_x * direction[1]

            # sort the array before computing differences
            ind_sort = np.argsort(np.sqrt((inter_points[:, 0] - inter_points[0, 0]) ** 2 +
                                          (inter_points[:, 1] - inter_points[0, 1]) ** 2)) # indices of sorted array elements

            inter_points_sorted = inter_points[ind_sort, :]  # sorted intersecting points

            # mid points between intersecting points
            mid_points = 0.5 * (inter_points_sorted[:-1, :] + inter_points_sorted[1:, :])

            # differences between intersecting points
            diff_points = inter_points_sorted[:-1, :] - inter_points_sorted[1:, :]

            is_dup = np.sqrt(diff_points[:, 0] ** 2 + diff_points[:, 1] ** 2) < 1e-10  # is duplicated?

            mid_points_unq = mid_points[~is_dup, :]  # remove duplicates

            # mid points indices:
            ix = np.floor((mid_points_unq[:, 0] - (x[0] - 0.5 * dx)) / dx).astype(np.int64)
            iz = np.floor(mid_points_unq[:, 1] / dz).astype(np.int64)

            ind_glob = nx * iz + ix  # corresponding indices in xx- or zz-like array

            # update row, colm, and values vectors for output sparse matrix
            ind_row[pointer: pointer + len(ind_glob)] = i
            ind_col[pointer: pointer + len(ind_glob)] = ind_glob
            vals[pointer: pointer + len(ind_glob)] = np.sqrt(diff_points[~is_dup, 0] ** 2
                                                             + diff_points[~is_dup, 1] ** 2)  # ray length in pixel

            if neg:  # to compute output for -angle, reorder the row and column indices
                ind_glob_neg_angle = nx * iz - ix + nx - 1  # corresponding indices in xx- or zz-like array
                ind_col_neg_angle[pointer: pointer + len(ind_glob)] = ind_glob_neg_angle
                ind_row_neg_angle[pointer: pointer + len(ind_glob)] = 2 * int(np.floor(i / nx)) * nx + (nx - 1) - i

            pointer += len(ind_glob)  # update pointer

    if neg:  # build sparse forward operator for -angle
        return vals, ind_row, ind_col, ind_row_neg_angle, ind_col_neg_angle
    else:
        return vals, ind_row, ind_col, ind_row, ind_col  # dummy output for last two variables


@njit(cache=True)
def ray_tracer_div(x_rec, xx, zz, x, dx, dz, nx, nz):

    """
    Straight-ray tracer from a plane wave with angle 'angle' to virtual receivers at all gridpoints specified
    by 'x' and 'z'.
    :param angle: transmit plane wave angle. Convention: positive from z-axis counterclockwise [scalar; radians]
    :param xx: mesh grid locations in x direction [numpy array, nx*nz x 1]
    :param zz: mesh grid locations in z direction [numpy array, nz*nx x 1]
    :param x: limits of grid locations in x direction [numpy array, 2 x 1]
    :param dx: grid size in x direction [scalar]
    :param dz: grid size in z direction [scalar]
    :param nx: number of grid points in x direction [scalar]
    :param nz: number of grid points in z direction [scalar]
    :param neg: if True, it gives the output for planes waves with angles 'angle' and '-angle'. [boolean]
                if False, it only gives the output for the plane wave with 'angle'.
    :return: vals: numpy array for non-zero values of the forward operator relating traveltimes with the
    slowness of the medium. [numpy array]
             ind_row: numpy array indicating the row number of non-zero values of the forward operator [numpy array]
             ind_col: numpy array indicating the column number of non-zero values of the forward operator [numpy array]
            The forward operator has: Rows: number of rays; Columns: number of gridpoints. Each row is a ray path,
            specifying the length of the ray in each pixel. This function returns the inputs for Compressed Sparse Row
            matrix of dimension nx*nz x nx*nz.
    """

    # Compute number of intersections with pixel edges.
    niz_all = (np.floor(zz / dz)).astype(np.int64) + 1  # number of intersections in z. From zz to z=0 (included)
    nix_all = (np.floor((xx - (x[0] - dx / 2)) / dx)).astype(np.int64)  # From left edge of origin x[0] to pixel xx

    ind_col = np.zeros((nx * nz * 2 * nz,), dtype=np.int64)  # initialize output. Column indices
    ind_row = np.zeros_like(ind_col)  # row indices
    vals = np.zeros((nx * nz * 2 * nz,))  # corresponding values of the matrix

    pointer = 0  # the pointer indicates the number of nonzero values in the sparse matrix (output) at each iteration

    for i in range(nx * nz):  # loop through all virtual receivers

        x_surf = x_rec  # x-position where the ray crosses the surface (z = 0)

        # number of intersections in x and z direction from (x_surf,0) to (xx, zz)
        nix = int(abs(nix_all[i] - np.floor((x_surf - (x[0] - dx / 2)) / dx)))
        niz = niz_all[i]

        npoints = nix + niz + 2  # number of intersections with pixel edges + 1 (virtual receiver location)

        inter_points = np.ones((npoints, 2))  # initialize array with locations of intersecting points (x,z)

        inter_points[0, 0] = xx[i]  # first value is the virtual receiver location
        inter_points[0, 1] = zz[i]  # first value is the virtual receiver location
        inter_points[-1, 0] = x_surf  # last intersection is with surface
        inter_points[-1, 1] = 0  # last intersection is with surface

        angle = np.arctan((xx[i] - x_surf)/zz[i])

        direction = [np.sin(-angle), -np.cos(angle)]  # unitary vector indicating ray direction

        # intersecting points with pixel edges in z-direction
        if niz == 1:

            inter_points[1, 1] = 0
            step_z = - zz[i] / direction[1]
            inter_points[1, 0] = xx[i] + step_z * direction[0]

        else:

            inter_points[1:niz + 1, 1] = np.arange(niz) * dz
            step_z = (np.arange(niz) * dz - zz[i]) / direction[1]
            inter_points[1:niz + 1, 0] = xx[i] + step_z * direction[0]

        # intersecting points with pixel edges in x-direction
        if nix == 1:

            sign = (x_surf - xx[i]) / abs(x_surf - xx[i])
            inter_points[niz + 1, 0] = xx[i] + sign * 0.5 * dx
            step_x = sign * 0.5 * dx / direction[0]
            inter_points[niz + 1, 1] = zz[i] + step_x * direction[1]

        elif nix > 1:

            sign = (x_surf - xx[i]) / abs(x_surf - xx[i])
            inter_points[niz + 1: niz + nix + 1, 0] = \
                xx[i] + sign * (0.5 * dx + np.arange(nix) * dx)
            step_x = sign * (0.5 * dx + np.arange(nix) * dx) / direction[0]
            inter_points[niz + 1: niz + nix + 1, 1] = zz[i] + step_x * direction[1]

        # sort the array before computing differences
        ind_sort = np.argsort(np.sqrt((inter_points[:, 0] - inter_points[0, 0]) ** 2 +
                                      (inter_points[:, 1] - inter_points[0, 1]) ** 2)) # indices of sorted array elements

        inter_points_sorted = inter_points[ind_sort, :]  # sorted intersecting points

        # mid points between intersecting points
        mid_points = 0.5 * (inter_points_sorted[:-1, :] + inter_points_sorted[1:, :])

        # differences between intersecting points
        diff_points = inter_points_sorted[:-1, :] - inter_points_sorted[1:, :]

        is_dup = np.sqrt(diff_points[:, 0] ** 2 + diff_points[:, 1] ** 2) < 1e-10  # is duplicated?

        mid_points_unq = mid_points[~is_dup, :]  # remove duplicates

        # mid points indices:
        ix = np.floor((mid_points_unq[:, 0] - (x[0] - 0.5 * dx)) / dx).astype(np.int64)
        iz = np.floor(mid_points_unq[:, 1] / dz).astype(np.int64)
#         print(ix, iz)

        ind_glob = nx * iz + ix  # corresponding indices in xx- or zz-like array

        # update row, colm, and values vectors for output sparse matrix
        ind_row[pointer: pointer + len(ind_glob)] = i
        ind_col[pointer: pointer + len(ind_glob)] = ind_glob
        vals[pointer: pointer + len(ind_glob)] = np.sqrt(diff_points[~is_dup, 0] ** 2
                                                         + diff_points[~is_dup, 1] ** 2)  # ray length in pixel


        pointer += len(ind_glob)  # update pointer

    return vals, ind_row, ind_col
