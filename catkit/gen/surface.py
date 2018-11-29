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


class SlabGenerator(object):
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
    layers : int
        Number of layers to include in the slab. A slab layer is defined
        as a unique z-coordinate.
    vacuum : float
        Angstroms of vacuum to apply to the slab.
    fixed : int
        Number of slab layers to constrain.
    layer_type : 'angs', 'trim', 'stoich', or 'sym'
        Determines how to perform slab layering.

        'angs': Layers denotes the thickness of the slab in Angstroms.
        'trim': The slab will be trimmed to a number of layers equal to the
        exact number of unique z-coordinates. Useful for precision control.
        'stoich' : Constraints any slab generated to have the same
        stoichiometric ratio as the provided bulk.
        'sym' : Return a slab which is inversion symmetric. i.e. The
        same on both sides.
    attach_graph : bool
        Attach the connectivity graph generated from the bulk structure.
        This is only necessary for fingerprinting and setting it to False
        can save time. Surface atoms will be found regardless.
    standardize_bulk : bool
        Covert the bulk input to its standard form before and
        produce the cleave from it. This is highly recommended as
        Miller indices are not defined for non-standard cells.
    tol : float
        Tolerance for floating point rounding errors.
    """

    def __init__(self,
                 bulk,
                 miller_index,
                 layers,
                 vacuum=None,
                 fixed=None,
                 layer_type='ang',
                 attach_graph=True,
                 standardize_bulk=False,
                 primitive=True,
                 tol=1e-8):
        self.layers = layers
        self.vacuum = vacuum
        self.fixed = fixed
        self.tol = tol
        self.layer_type = layer_type
        self.standardized = standardize_bulk
        self.primitive = primitive
        self.attach_graph = attach_graph
        self.unique_terminations = None
        self.slab_basis = None
        self.slab = None

        miller_index = np.array(miller_index)
        if len(miller_index) == 4:
            miller_index[[0, 1]] -= miller_index[2]
            miller_index = np.delete(miller_index, 2)
        miller_index = (miller_index /
                        utils.list_gcd(miller_index)).astype(int)

        # Store the Miller indices associated with the standard cell.
        self.miller_index = miller_index

        if standardize_bulk:
            bulk = symmetry.get_standardized_cell(bulk, tol=1e-2)
        else:
            warnings.warn(
                ("Not using a standardized bulk will result in an arbitrary "
                 "Miller index. To get ensure you are using the correct "
                 "miller index, use standardize_bulk=True"))

        if primitive:
            primitive_bulk = symmetry.get_standardized_cell(
                bulk, primitive=True, tol=1e-2)
            miller_index = convert_miller_index(
                miller_index, bulk, primitive_bulk)
        else:
            primitive_bulk = bulk

        self._bulk = gen.bulk.align_crystal(
            primitive_bulk, miller_index, self.tol)

        self.unique_terminations = gen.bulk.get_unique_terminations(
            self._bulk, self.tol)
        self.slab_basis = [None] * len(self.unique_terminations)

    def get_slab_basis(self, iterm=0, maxn=20):
        """Return a list of all terminations which have been properly shifted
        and with an appropriate number of layers added. This function is mainly
        for performance, to prevent looping over other operations which are not
        related the size of the slab.

        This step also contains periodically constrained orthogonalization of
        the c basis. This implementation only works if the a and b basis
        vectors are properly aligned with the x and y axis. This is strictly to
        assist the correct identification of surface atoms.

        Only produces the terminations requested as a lazy evaluator.

        Parameters
        ----------
        iterm : int
            Index of the slab termination to return.
        maxn : int
            The maximum integer component to search for a more orthogonal bulk.

        Returns
        -------
        ibasis : Gratoms object
            Prepared, ith basis.
        """
        if self.slab_basis[iterm] is not None:
            ibasis = self.slab_basis[iterm].copy()
            return ibasis

        _basis = self._bulk.copy()
        ibasis = _basis.copy()
        if iterm > 0:
            zshift = terminations[iterm]
            scaled_positions = ibasis.get_scaled_positions()
            scaled_positions[:, 2] -= zshift - self.tol
            ibasis.set_scaled_positions(scaled_positions)
            ibasis.wrap(pbc=True)

        bulk_layers = utils.get_unique_coordinates(_basis)

        if self.layer_type != 'trim':
            height = np.abs(self._bulk.cell[2][2])
            minimum_repetitions = np.ceil(self.layers / height)
        else:
            minimum_repetitions = np.ceil(self.layers / len(bulk_layers))

        ibasis *= (1, 1, int(minimum_repetitions))

        # Get difference in the 2nd components of the b and c
        # basis, this is a good starting guess for the needed
        # value of the 2nd component (+/- 2 for safety)
        div = ibasis.cell[2][1] / ibasis.cell[1][1]
        sign = -np.sign(div)
        if sign == 0:
            sign = 1
        m = np.ceil(np.abs(div)) + 1

        # Try to be smart and only search a limited space
        search = np.mgrid[maxn:-maxn-1:-1, sign*m:m-4*sign:-sign, 1:2]
        search = search.T.reshape(-1, 3)

        # Need the reciprocal unit cell.
        recp = np.linalg.inv(ibasis.cell).T
        normal = np.dot(self.miller_index, recp)
        normal /= np.linalg.norm(normal)

        # Compute the lengths of the possible transformed vectors
        # and the cosine of the vector normal to the miller plane.
        vectors = np.dot(search, ibasis.cell)
        length = np.linalg.norm(vectors, axis=1)
        angles = np.abs(np.dot(vectors, normal) / length)

        # Find the cosine closest to 1 and the smallest lengths
        sort = np.lexsort([angles, length])
        scale = np.eye(3)
        scale[2] = search[sort[0]]

        newcell = np.dot(scale, ibasis.cell)
        ibasis.set_cell(newcell)
        ibasis.wrap()

        exbasis = ibasis * (1, 1, 2)
        connectivity = utils.get_voronoi_neighbors(exbasis)

        n = len(ibasis)
        diff = connectivity[:n, n:].sum(axis=1)
        surf_atoms = diff != 0

        if np.all(diff):
            warnings.warn(
                ("Your slab has no bulk atoms and may be too thin "
                 "to identify surface atoms correctly. This may cause "
                 "surface adsorption site identification to fail."))

        # TODO: Graph generation needs to go here once handling of
        # unit cell repetitions is implemented.
        scaled_zpositions = ibasis.get_scaled_positions()[:, 2] + self.tol
        scaled_zpositions = np.round(scaled_zpositions % 1 + self.tol, 4)

        indices = np.argwhere(surf_atoms).flatten()
        zcoords = scaled_zpositions - np.mean(scaled_zpositions)
        top = indices[zcoords[indices] >= 0]
        bottom = indices[zcoords[indices] < 0]
        ibasis.set_surface_atoms(top=top, bottom=bottom)

        self.slab_basis[iterm] = ibasis

        return ibasis

    def get_slab(self, size=1, iterm=0):
        """Generate a slab from the bulk structure. This function is meant
        specifically for selection of an individual termination or enumeration
        through surfaces of various size.

        This function will orthogonalize the c basis vector and align it
        with the z-axis which breaks bulk symmetry along the z-axis.

        Parameters
        ----------
        size : int, array_like (2,) or (2, 2)
            Size of the unit cell to create as described in :meth:`set_size`.
        iterm : int
            A termination index in reference to the list of possible
            terminations.

        Returns
        -------
        slab : Gratoms object
            The modified basis slab produced based on the layer specifications
            given.
        """
        slab = self.get_slab_basis(iterm).copy()
        slab = self.set_size(slab, size)

        # Orthogonalize the z-coordinate
        # Breaks bulk periodicity in the c-basis
        slab.cell[2] = [0, 0, slab.cell[2][2]]
        slab.set_pbc([1, 1, 0])

        if slab.cell[1][0] < 0:
            slab = transform_ab(slab, [[-1, 0], [0, 1]])

        # Trim the bottom of the cell, bulk symmetry may be lost
        if self.layer_type == 'trim':
            zlayers = utils.get_unique_coordinates(slab)
            reverse_sort = np.sort(zlayers)[::-1]
            ncut = reverse_sort[:self.layers][-1] * slab.cell[2][2]

            zpos = slab.positions[:, 2]
            index = np.arange(len(slab))
            del slab[index[zpos - ncut < -self.tol]]

            slab.cell[2][2] -= ncut
            slab.translate([0, 0, -ncut])

        tl = np.argmax(slab.get_scaled_positions()[:, 2])
        translation = slab[tl].position.copy()
        translation[2] = 0
        slab.translate(-translation)
        slab.wrap()

        if self.vacuum:
            slab.center(vacuum=self.vacuum, axis=2)

        utils.get_unique_coordinates(slab, tag=True)
        if self.layer_type == 'sym':
            slab = self.make_symmetric(slab)

        roundoff = np.isclose(slab.cell, 0)
        slab.cell[roundoff] = 0

        ind = np.lexsort(slab.positions.T)
        slab = slab[ind]

        if self.fixed:
            tags = slab.get_tags()
            constraints = ase.constraints.FixAtoms(
                mask=tags > (tags.max() - self.fixed))
            slab.set_constraint(constraints)

        self.slab = slab

        return slab

    def adsorption_sites(self, slab, **kwargs):
        """Helper function to return the adsorption sites of the provided slab.

        Parameters
        ----------
        slab : atoms object
            The slab to find adsorption sites for. Assumes you are using
            the same basis.

        Returns
        -------
        output : tuple (n, n) | (n, n, n)
            Coordinates and connectivity of the adsorption sites.
            The symmetry indices can also be returned.
        """
        output = adsorption.get_adsorption_sites(
            slab=slab, **kwargs)

        return output

    def set_size(self, slab, size):
        """Set the size of a slab based one of three methods.

        1. An integer value performs a search of valid matrix operations
        to perform on the ab-basis vectors to return a set which with
        a minimal sum of distances and an angle closest to 90 degrees.

        2. An array_like of length 2 will multiply the existing basis
        vectors by that amount.

        3. An array of shape (2, 2) will be interpreted as matrix
        notation to multiply the ab-basis vectors by.

        Parameters
        ----------
        slab : Atoms object
            Slab to be made into the requested size.
        size : int, array_like (2,) or (2, 2)
            Size of the unit cell to create as described above.

        Returns
        -------
        supercell : Gratoms object
            Supercell of the requested size.
        """
        supercell = slab

        if isinstance(size, (int, np.integer)):
            a = max(int(size / 2), 1) + size % 2 + 1
            T = np.mgrid[-a:a + 1, -a:a + 1].reshape(2, -1).T

            metrics = []
            search_space = itertools.product(T, repeat=2)
            for i, M in enumerate(search_space):
                M = np.array(M)
                if ~np.isclose(abs(np.linalg.det(M)), size):
                    continue

                vector = np.dot(M.T, slab.cell[:2, :2])
                d = np.linalg.norm(vector, axis=1)

                angle = np.dot(vector[0], vector[1]) / np.prod(d)
                diff = np.diff(d)[0]

                # obtuse angle
                if angle < 0 or diff < 0:
                    continue

                metrics += [[d.sum(), angle, M]]

            if metrics:
                order = [0, 1]
                if defaults.get('orthogonal'):
                    order = [1, 0]

                matrix = sorted(metrics,
                                key=lambda x: (
                                    x[order[0]], x[order[1]]))[0][-1]
                supercell = transform_ab(supercell, matrix)

        elif isinstance(size, (list, tuple, np.ndarray)):
            size = np.array(size, dtype=int)

            if size.shape == (2,):
                supercell *= (size[0], size[1], 1)
            elif size.shape == (2, 2):
                supercell = transform_ab(supercell, size)

        if self.attach_graph:
            # TODO: Creating the graph at this point is not ideal.
            # Need to be able to handle expansion of the unit cell
            # before this can be moved back to basis creation
            n = len(supercell)
            exsupercell = supercell * (1, 1, 2)

            # Look into making bulk more orthogonal
            exsupercell.wrap()
            connectivity = utils.get_voronoi_neighbors(exsupercell)
            edges = utils.connectivity_to_edges(connectivity[:n, :n])
            supercell.graph.add_weighted_edges_from(edges, weight='bonds')

        return supercell

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

def transform_ab(slab, matrix, tol=1e-5):
    """Transform the slab basis vectors parallel to the z-plane
    by matrix notation. This can result in changing the slabs
    cell size. This can also result in very unusual slab dimensions,
    use with caution.

    Parameters
    ----------
    slab : Atoms object
        The slab to be transformed.
    matrix : array_like (2, 2)
        The matrix notation transformation of the a and b basis vectors.
    tol : float
        Float point precision tolerance.

    Returns
    -------
    slab : Atoms object
        Slab after transformation.
    """
    M = np.eye(3)
    M[:2, :2] = np.array(matrix).T
    newcell = np.dot(M, slab.cell)

    scorners_newcell = np.array([
        [0, 0], [1, 0],
        [0, 1], [1, 1]])

    corners = np.dot(scorners_newcell, newcell[:2, :2])
    scorners = np.linalg.solve(slab.cell[:2, :2].T, corners.T).T
    rep = np.ceil(scorners.ptp(axis=0)).astype(int)

    slab *= (rep[0], rep[1], 1)
    slab.set_cell(newcell)

    coords = slab.get_scaled_positions()
    original_index = np.arange(coords.shape[0])
    periodic_match = original_index.copy()
    for i, j in enumerate(periodic_match):
        if i != j:
            continue

        matched = utils.matching_sites(coords[i], coords)
        periodic_match[matched] = i

    repeated = np.where(periodic_match != original_index)
    del slab[repeated]

    # Align the first basis vector with x
    sign = np.sign(slab.cell[2][2])
    slab.rotate(slab.cell[0], 'x', rotate_cell=True)
    if sign != np.sign(slab.cell[2][2]):
        slab.arrays['surface_atoms'] *= -1

    if slab.cell[1][1] < 0:
        slab.cell[1] *= -1
    if slab.cell[2][2] < 0:
        slab.translate([0, 0, -slab.cell[2][2]])
        slab.cell[2][2] = -slab.cell[2][2]

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
