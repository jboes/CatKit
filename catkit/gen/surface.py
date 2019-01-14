from __future__ import division
from .. import Gratoms
from catkit import gen
from . import symmetry
from . import utils
from . import adsorption
from . import defaults
import numpy as np
import itertools
import warnings
import scipy
import ase
import re


class SlabGenerator():
    """Class for generation of slab unit cells from bulk unit cells.

    Many surface operations rely upon / are made easier through the
    bulk basis cell they are created from. The SlabGenerator class
    is designed to house these operations.

    Return the miller indices associated with the users requested
    values. Follows the following steps:

    - Convert Miller-Bravais notation into standard Miller index.
    - (optional) Ensure the bulk cell is in its standard form.
    - Convert the indices to the cell for the primitive lattice.
    - Reduce the indices by their greatest common divisor.

    Parameters
    ----------
    bulk : Atoms object
        Bulk system to be converted into slab.
    miller_index : list (3,) or (4,)
        Miller index to construct surface from. If length 4, Miller-Bravais
        notation is assumed.
    layers : int | str
        Number of layers to include in the slab. A slab layer is defined
        as a unique z-coordinate. If a string is passed, the type of layering
        will be altered.

        'A': Layers denotes the thickness of the slab in Angstroms.
        'S' : Constraints any slab generated to have the same
        stoichiometric ratio as the provided bulk.
        'I' : Return a slab which is inversion symmetric.
    tol : float
        Tolerance for floating point rounding errors.
    """

    def __init__(
            self,
            bulk,
            miller_index,
            termination=0,
            layers=1,
            tol=1e-8):
        self.tol = tol

        self.layer_type = re.findall('[a-zA-Z]', str(layers))
        self.layers = float(re.sub('[a-zA-Z]', '', str(layers)))

        self._bulk = gen.bulk.align_crystal(
            bulk, miller_index, tol)

        self.miller = bravis_miller_to_miller(miller_index)

        zpos = np.sort(self._bulk.positions[:, 2])
        if 'A' in self.layer_type:
            layers = self.layers / self._bulk.cell[2][2]
            self.repetitions = np.ceil(layers)
        else:
            zlayers = utils.get_matching_positions(zpos)
            layers = self.layers / len(np.unique(zlayers))
            self.repetitions = int(np.ceil(layers))

        terminations = gen.bulk.get_unique_terminations(
            self._bulk, tol)

        self.basis = self._bulk.copy()

        generator = gen.bulk.GraphGenerator(self.basis)
        positions, connectivity = generator.get_adsorption_sites()
        self.basis *= [1, 1, self.repetitions]
        self.basis.set_surface_atoms(generator.surface_atoms)

        fractional= np.linalg.solve(self.basis.cell.T, positions.T).T
        sym = gen.adsorption.symmetry_equivalent_points(
            fractional, self.basis)
        con = np.array([len(_) for _ in connectivity])
        self.basis.set_site_properties(
            positions, symmetry=sym, connectivity=con)

    def get_slab(self, size=1, vacuum=0):
        """Generate a slab from the bulk structure. This function is meant
        specifically for selection of an individual termination or enumeration
        through surfaces of various size.

        This function will orthogonalize the c basis vector and align it
        with the z-axis which breaks bulk symmetry along the z-axis.

        Parameters
        ----------
        size : int | array_like (2, 2)
            Size of the unit cell to create as described in :meth:`set_size`.
        termination : int
            A termination index in reference to the list of possible
            terminations.

        Returns
        -------
        slab : Gratoms object
            The modified basis slab produced based on the layer specifications
            given.
        """
        slab = self.basis.copy()
        slab = set_size(slab, size)

        width = np.ptp(slab.positions[:, 2])
        slab.cell[2] = [0, 0, width]
        slab.set_pbc([1, 1, 0])

        sort = np.lexsort(slab.positions.T)
        slab = slab[sort]

        zpos = slab.positions[:, 2]
        zlayers = utils.get_matching_positions(zpos)
        _, tags = np.unique(zlayers, return_inverse=True)
        slab.set_tags(tags + 1)

        R = np.diag([1, 1, -1])
        cell = np.dot(slab.cell, R.T)
        cell[2] *= -1
        cell[2][2] += 2 * vacuum

        slab.cell = cell
        slab.positions = np.dot(slab.positions, R.T)
        slab.positions[:, 2] += width + vacuum

        if not 'S' in self.layer_type:
            if 'A' in self.layer_type:
                del slab[zpos > self.layers]
            else:
                del slab[(tags + 1) > self.layers]

            if 'I' in self.layer_type:
                slab = self.make_symmetric(slab)

        slab.wrap(eps=self.tol)

        sites = slab._sites
        if sites:
            sites.cell = cell
            sites.positions = np.dot(sites.positions, R.T)
            sites.positions[:, 2] += width + vacuum
            fpos = np.linalg.solve(cell.T, sites.positions.T).T
            fpos -= np.floor(fpos + self.tol)
            sites.positions = np.dot(fpos - self.tol, cell)
            sort = np.hstack([
                sites.positions,
                sites.arrays['connectivity'][:, None]])

            sort = np.lexsort(sort.T)
            slab._sites = sites[sort]

        return slab

    def make_symmetric(self, slab):
        """Returns a symmetric slab. Note, this will trim the slab potentially
        resulting in loss of stoichiometry.
        """
        sym = symmetry.Symmetry(slab)
        inversion_symmetric = sym.get_point_group(check_laue=True)[1]

        # Trim the cell until it is symmetric
        while not inversion_symmetric:
            tags = slab.get_tags()
            bottom_layer = np.max(tags)
            del slab[tags == bottom_layer]

            sym = symmetry.Symmetry(slab)
            inversion_symmetric = sym.get_point_group(check_laue=True)[1]

            if len(slab) <= len(self._bulk):
                warnings.warn('Too many sites removed, please use a larger '
                              'slab size.')
                break

        return slab

def set_size(slab, size):
    """Set the size of a slab based one of two methods.

    1. An integer value performs a search of valid matrix operations
    to perform on the ab-basis vectors to return a set which with
    a minimal sum of distances and an angle closest to 90 degrees.

    2. An array of shape (2, 2) will be interpreted as matrix
    notation to multiply the ab-basis vectors by.

    Parameters
    ----------
    slab : Atoms object
        Slab to be made into the requested size.
    size : int | array_like  (2, 2)
        Size of the unit cell to create as described above.

    Returns
    -------
    slab : Gratoms object
        Supercell of the requested size.
    """
    size = np.asarray(size, dtype=int)
    if size.shape == (2, 2):
        matrix = size
    else:
        transforms = generate_transforms(size)

        targets = np.empty((len(transforms), 2))
        for i, M in enumerate(transforms):
            cell = np.dot(M.T, slab.cell[:2, :2])
            dis = np.linalg.norm(cell, axis=1)
            angle = np.round(np.dot(*cell) / np.prod(dis), 2)

            targets[i] = [np.abs(angle), dis.sum().round(2)]

        if defaults.get('orthogonal'):
            targets = targets.swapaxes(1, 2)
        srt = np.lexsort(targets.T)
        matrix = transforms[srt[0]]

    slab = gen.bulk.resize_cell(slab, matrix)

    return slab


def convert_miller_index(miller_index, atoms1, atoms2):
    """Return a converted miller index between two atoms objects."""
    recip1 = utils.get_reciprocal_vectors(atoms1)
    recip2 = utils.get_reciprocal_vectors(atoms2)

    converted_index = np.dot(
        miller_index, np.dot(recip1, np.linalg.inv(recip2)))
    converted_index = np.round(converted_index)
    converted_index = (converted_index /
                       utils.list_gcd(converted_index)).astype(int)
    if converted_index[0] < 0:
        converted_index *= -1

    return converted_index


def generate_indices(max_index):
    """Return an array of miller indices enumerated up to values
    plus or minus some maximum. Filters out lists with greatest
    common divisors greater than one. Only positive values need to
    be considered for the first index.

    Parameters
    ----------
    max_index : int
        Maximum number that will be considered for a given surface.

    Returns
    -------
    unique_index : ndarray (n, 3)
        Unique miller indices
    """
    grid = np.mgrid[max_index:-1:-1,
                    max_index:-max_index-1:-1,
                    max_index:-max_index-1:-1]
    index = grid.reshape(3, -1)
    gcd = utils.list_gcd(index)
    unique_index = index.T[np.where(gcd == 1)]

    return unique_index


def get_unique_indices(bulk, max_index):
    """Returns an array of miller indices which will produce unique
    surface terminations based on a provided bulk structure.

    Parameters
    ----------
    bulk : Atoms object
        Bulk structure to get the unique miller indices.
    max_index : int
        Maximum number that will be considered for a given surface.

    Returns
    -------
    unique_millers : ndarray (n, 3)
        Symmetrically distinct miller indices for a given bulk.
    """
    sym = symmetry.Symmetry(bulk)
    operations = sym.get_symmetry_operations()
    unique_index = generate_indices(max_index)

    unique_millers = []
    for i, miller in enumerate(unique_index):
        affine_point = np.insert(miller, 3, 1)

        symmetric = False
        for affine_matrix in operations:
            operation = np.dot(affine_matrix, affine_point)[:3]

            # TODO: This function is replaced with get_matching_positions
            match = utils.matching_coordinates(operation, unique_millers)
            if len(match) > 0:
                symmetric = True
                break

        if not symmetric:
            unique_millers += [miller]

    unique_millers = np.flip(unique_millers, axis=0)

    return unique_millers


def get_degenerate_indices(bulk, miller_index):
    """Return the miller indices which are degenerate to a
    given miller index for a particular bulk structure.

    Parameters
    ----------
    bulk : Atoms object
        Bulk structure to get the degenerate miller indices.
    miller_index : array_like (3,)
        Miller index to get the degenerate indices for.

    Returns
    -------
    degenerate_indices : array (N, 3)
        Degenerate miller indices to the provided index.
    """
    miller_index = np.asarray(miller_index)
    miller_index = np.divide(miller_index, utils.list_gcd(miller_index))

    sym = symmetry.Symmetry(bulk)
    affine_matrix = sym.get_symmetry_operations()
    affine_point = np.insert(miller_index, 3, 1)
    symmetric_indices = np.dot(affine_point, affine_matrix)[:, :3]

    degenerate_indices = np.unique(symmetric_indices, axis=0)
    degenerate_indices = np.flip(degenerate_indices, axis=0).astype(int)

    return degenerate_indices


def generate_transforms(volume):
    """Return an enumerated list of possibly relevant 2D
    surface transformations. This is intended to be used on a
    normalized bulk structure, i.e. cell with upper diagonals
    set to zero and all non-zero entries positive.

    For transformation matrix: [[A, a], [b, B]], the required
    space can be explored by considering all value ranges
    which produce a determinant equal to the desired volume
    multiple.

    Parameters
    ----------
    volume : int
        Desired volume multiple of the unit cell, must be >= 1.

    Returns
    -------
    transforms : ndarray (N, 2, 2)
        2D transformation matrices. Specifically the upper two
        rows and columns.
    """
    N = volume
    cd = np.mgrid[slice(-N, 0), slice(0, 1)].reshape(2, -1).T
    cd = np.vstack([[0, 0], cd, cd[:, ::-1]])
    AB = utils.get_integer_enumeration(
        N=2, span=[0, int(N**0.5) + 1])[1:]
    products = np.prod(AB, 1) - N

    transforms = []
    for i, v in enumerate(products):
        if v == 0:
            ab = cd
        else:
            ab = utils.get_whole_factors(abs(v))
            sign = np.sign(v)
            ab[:, 0] *= sign
            ab = np.vstack([ab, ab[:, ::-1]])

        diag = np.tile(AB[i], len(ab))
        M = np.c_[diag, ab.flatten()]
        M[1::2] = M[1::2][:, ::-1]
        transforms += [M.reshape(-1, 2, 2)]
    transforms = np.vstack(transforms)

    return transforms


def bravis_miller_to_miller(bravis_miller):
    """Bravis Miller index conversion to standard miller index.

    Parameters:
    -----------
    bravis_miller : array_like (4,)
        Four integer miller index.

    Returns:
    --------
    miller_index : ndarray (3,)
        Converted miller index.
    """
    miller_index = np.array(bravis_miller)
    if len(miller_index) == 4:
        miller_index[[0, 1]] -= miller_index[2]
        miller_index = np.delete(miller_index, 2)
    miller_index = (miller_index /
                    utils.list_gcd(miller_index)).astype(int)
    miller_index = miller_index

    return miller_index
