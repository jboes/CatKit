import numpy as np


def expand_cell(
        coordinates,
        cell,
        pbc=True,
        centers=None,
        cutoff=6.0):
    """Return Cartesian coordinates atoms within a supercell
    which contains spheres of a given cutoff radius. One sphere
    for each atom inside of the given unit cell.

    Parameters
    ----------
    coordinates : ndarray (N, 3)
        Cartesian coordinates of atomic positions.
    cell : ndarray (3, 3)
        Unit cell dimensions
    pbc : bool | array_like (3,)
        Periodic boundary conditions.
    cutoff : float
        Radius of the spheres to use for expanding the unit cell.

    Returns
    -------
    indices : ndarray (M,)
        Indices associated with the original unit cell atom index.
    positions : ndarray (M, 3)
        Cartesian coordinates associated with positions in the
        supercell.
    offsets : ndarray (M, 3)
        Integer offsets of each unit cell.
    """
    pbc = np.asarray(pbc, dtype=bool)
    if centers is None:
        centers = np.linalg.solve(cell.T, coordinates.T).T

    basis_lengths = np.linalg.norm(cell, axis=1)
    reciprocal_cell = np.linalg.inv(cell).T
    reciprocal_lengths = np.linalg.norm(reciprocal_cell, axis=1)

    lengths = pbc * cutoff * reciprocal_lengths + 0.01
    min_pad = np.floor(centers - lengths).min(axis=0)
    max_pad = np.ceil(centers + lengths).max(axis=0)
    padding = np.array([min_pad.astype(int), max_pad.astype(int)]).T

    offsets = np.mgrid[[slice(*_) for _ in padding]].T

    ncell = np.prod(offsets.shape[:-1])
    indices = np.tile(np.arange(len(coordinates)), ncell)

    cartesian_cells = np.dot(offsets, cell)
    positions = cartesian_cells[:, :, :, None, :] + \
        coordinates[None, None, None, :, :]
    positions = positions.reshape(-1, 3)

    offsets = np.tile(offsets, len(coordinates)).reshape(-1, 3)

    return indices, positions, offsets


def trilaterate(centers, r, zvector=None):
    """Find the intersection of two or three spheres. In the case
    of two sphere intersection, the z-coordinate is assumed to be
    an intersection of a plane whose normal is aligned with the
    points and perpendicular to the positive z-coordinate.

    If more than three spheres are supplied, the centroid of the
    points is returned (no radii used).

    Parameters
    ----------
    centers : list | ndarray (n, 3)
        Cartesian coordinates representing the center of each sphere
    r : list | ndarray (n,)
        The radii of the spheres.
    zvector : ndarray (3,)
        The vector associated with the upward direction for under-specified
        coordinations (1 and 2).

    Returns
    -------
    intersection : ndarray (3,)
        The point where all spheres/planes intersect.
    """
    if zvector is None:
        zvector = np.array([0, 0, 1])

    if len(r) == 1:
        return centers[0] + r[0] * zvector
    elif len(r) > 3:
        centroid = np.sum(centers, axis=0) / len(centers)
        centroid += np.mean(r) / 2 * zvector
        return centroid

    vec1 = centers[1] - centers[0]
    uvec1 = vec1 / np.linalg.norm(vec1)
    d = np.linalg.norm(vec1)

    if len(r) == 2:
        x0 = (d**2 - r[0]**2 + r[1]**2)
        x = d - x0 / (2 * d)
        a = np.sqrt(4 * d**2 * r[1]**2 - x0**2)
        z = 0.5 * (1 / d) * a
        if np.isnan(z):
            z = 0.01
        h = z * zvector
        intersection = centers[0] + uvec1 * x + h

    elif len(r) == 3:
        vec2 = centers[2] - centers[0]
        i = np.dot(uvec1, vec2)
        vec2 = vec2 - i * uvec1

        uvec2 = vec2 / np.linalg.norm(vec2)
        uvec3 = np.cross(uvec1, uvec2)
        j = np.dot(uvec2, vec2)

        x = (r[0]**2 - r[1]**2 + d**2) / (2 * d)
        y = (r[0]**2 - r[2]**2 - 2 * i * x + i**2 + j**2) / (2 * j)
        z = np.sqrt(r[0]**2 - x**2 - y**2)
        if np.isnan(z):
            z = 0.01
        intersection = centers[0] + x * uvec1 + y * uvec2 + z * uvec3

    return intersection


def get_matching_positions(fractional, tol=1e-8):
    """Get the indices of all points in a position list that are
    equal (with a tolerance), with periodic boundary conditions.
    This will only accept a fractional coordinate scheme.

    Parameters
    ----------
    fractional : ndarray (N, 3)
        Fractional coordinates to test for uniqueness.
    tol : float
        Float point precision tolerance.

    Returns
    -------
    match : list (N,)
        Indices of matches.
    """
    match = np.arange(fractional.shape[0])
    for i, j in enumerate(match):
        if i != j:
            continue

        diff = fractional[:, None] - fractional[i]
        diff -= np.floor(diff + tol)
        matched = np.all(diff < tol, -1).flatten()
        match[matched] = i

    return match


def get_integer_enumeration(N=3, span=[0, 2]):
    """Return the enumerated array of a span of integer values.
    These enumerations are limited to the length N.

    For the default span of [0, 2], the enumeration equates to
    the corners of an N-dimensional hypercube.

    Parameters
    ----------
    N : int
        Length of enumerated lists to produce.
    span : list | slice
        The range of integers to be considered for enumeration.

    Returns
    -------
    enumeration : ndarray (M, N)
        Enumeration of the requested integers.
    """
    if not isinstance(span, slice):
        span = slice(*span)
    enumeration = np.mgrid[[span] * N].reshape(N, -1).T

    return enumeration
