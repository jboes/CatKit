from .. import Gratoms
from . import utils
import numpy as np
import scipy
import ase
try:
    from math import gcd
except ImportError:
    from fractions import gcd


def align_crystal(bulk, miller_index, tol=1e-5):
    """Return a bulk unit cell with its ab-plane parallel to the
    miller plane of the requested miller index.

    The a-basis is always the longer of a and b, and aligned with
    the x dimension. The b- and c-basis are aligned with the positive
    surface normal which they correspond to.

    Parameters
    ----------
    bulk : Atoms object
        Bulk system to be aligned.
    miller_index : array_like (3,)
        Miller indices to align with the basis vectors.
    tol : float
        Float point precision tolerance.

    Returns
    -------
    new_bulk : Gratoms object
        Aligned bulk unit cell.
    """
    if len(np.nonzero(miller_index)[0]) == 1:
        mi = max(np.abs(miller_index))
        new_lattice = scipy.linalg.circulant(
            miller_index[::-1] / mi).astype(int)
    else:
        h, k, l = miller_index
        p, q = utils.ext_gcd(k, l)
        a1, a2, a3 = bulk.cell

        k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3),
                    l * a2 - k * a3)
        k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3),
                    l * a2 - k * a3)

        if abs(k2) > tol:
            i = -int(np.round(k1 / k2))
            p, q = p + i * l, q - i * k

        a, b = utils.ext_gcd(p * k + q * l, h)

        c1 = (p * k + q * l, -p * h, -q * h)
        c2 = np.array((0, l, -k)) // abs(gcd(l, k))
        c3 = (b, a * p, a * q)
        new_lattice = [c1, c2, c3]

    new_cell = np.dot(new_lattice, bulk.cell)

    # Ensure the a-basis is longer than the b-basis
    basis_lengths = np.linalg.norm(new_cell[:2], axis=1)
    new_cell[:2] = new_cell[np.argsort(basis_lengths)[::-1]]

    # Align the a-basis with x and the reset with their
    # corresponding plane normals
    ab_norm = np.cross(*new_cell[:2])
    R = ase.build.tools.rotation_matrix(
        new_cell[0], [1, 0, 0], ab_norm, [0, 0, 1])
    new_cell = np.dot(new_cell, R.T).round(-int(np.log10(tol)))
    new_cell *= np.sign(new_cell)

    positions = np.dot(bulk.positions, R.T)
    positions = ase.geometry.wrap_positions(
        positions, new_cell, eps=tol)
    positions = positions.round(-int(np.log10(tol)))

    new_bulk = bulk.copy()
    new_bulk.cell = new_cell
    new_bulk.positions = positions

    return new_bulk


def get_unique_terminations(bulk, tol):
    """Determine the fractional coordinate shift that will result in
    a unique surface termination. This is not required if bulk
    standardization has been performed, since all available z shifts will
    result in a unique termination for a primitive cell.

    Parameters
    ----------
    bulk : Atoms object
        Bulk system to determine terminations for.
    tol : float
        Float point precision tolerance.

    Returns
    -------
    unique_shift : array (n,)
        Fractional coordinate shifts which will result in unique
        terminations.
    """
    zcoords = utils.get_unique_coordinates(bulk)

    if len(zcoords) > 1:
        itol = tol ** -1
        zdiff = np.cumsum(np.diff(zcoords))
        zdiff = np.floor(zdiff * itol) / itol

        sym = symmetry.Symmetry(bulk, tol)
        rotations, translations = sym.get_symmetry_operations(affine=False)

        # Find all symmetries which are rotations about the z-axis
        zsym = np.abs(rotations)
        zsym[:, 2, 2] -= 1
        zsym = zsym[:, [0, 1, 2, 2, 2], [2, 2, 2, 0, 1]]
        zsym = np.argwhere(zsym.sum(axis=1) == 0)

        ztranslations = np.floor(translations[zsym, -1] * itol) / itol
        z_symmetry = np.unique(ztranslations)

        if len(z_symmetry) > 1:
            unique_shift = np.argwhere(zdiff < z_symmetry[1]) + 1
            unique_shift = np.append(0, zcoords[unique_shift])
        else:
            unique_shift = zcoords
    else:
        unique_shift = zcoords

    return unique_shift
