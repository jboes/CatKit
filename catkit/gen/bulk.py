from .. import Gratoms
from . import utils
from . import symmetry
import networkx as nx
import numpy as np
import scipy
import ase
try:
    from math import gcd
except ImportError:
    from fractions import gcd


def align_crystal(bulk, miller_index, tol=1e-5):
    """Return a bulk unit cell with its ab-plane parallel to the
    miller plane of the requested miller index. This will also
    normalize the cell.

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
    new_bulk = normalize_cell(bulk, new_cell)

    return new_bulk


def normalize_cell(atoms, new_cell=None, tol=1e-5):
    """Return a normalized cell. The a-basis is always the longer of a 
    and b, and aligned with the x dimension. The b- and c-basis are 
    aligned with the positive surface normal which they correspond to.
 
    Parameters
    ----------
    atoms : Atoms object
        Structure to be normalized.
    new_cell : ndarray (3, 3)
        A unit cell to normalize and apply to the atoms object.
    tol : float
        Float point precision tolerance.

    Returns
    -------
    new_atoms : Atoms object
        The normalized structure.
    """
    if new_cell is None:
        new_cell = atoms.get_cell()

    # Ensure the b-basis is longer than the a-basis
    basis_lengths = np.linalg.norm(new_cell[:2], axis=1)
    new_cell[:2] = new_cell[np.argsort(basis_lengths)[::-1]]

    # Align the a-basis with x and the reset with their
    # corresponding plane normals.
    ab_norm = np.cross(*new_cell[:2])
    R = ase.build.tools.rotation_matrix(
        new_cell[0], [1, 0, 0], ab_norm, [0, 0, 1])
    new_cell = np.dot(new_cell, R.T)


    new_atoms = atoms.copy()
    new_atoms.cell = new_cell
    new_atoms.positions = np.dot(atoms.positions, R.T)

    sites = new_atoms._sites
    if sites:
        sites.positions = np.dot(sites.positions, R.T)
        sites.cell = new_cell

    # Set all components to be positive.
    det = np.sign(np.linalg.det(new_cell))
    # new_cell *= np.sign(new_cell)
    new_cell[2] *= -1

    new_atoms.set_cell(new_cell)
    new_atoms.wrap(eps=tol)

    # if sites:
    #     sites.set_cell(new_cell, scale_atoms=scale)
    #     sites.wrap(eps=tol, pbc=True)

    return new_atoms


def get_unique_terminations(bulk, tol=1e-5):
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
    unique_shift : array (N,)
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


class GraphGenerator():
    """Class for producing a 3D graph for a 3D periodic
    system. Currently, only the Voronoi graph construction
    is supported.

    The expanded unit cell is required for all following
    operations. Using a class for these operations can avoid the
    need for multiple cell expansions.

    Parameters
    ----------
    atoms : Atoms object
        Bulk system to find voronoi 3D graph and subsequent
        graph properties.
    tol : float
        Float point precision tolerance.
    """
    def __init__(self, atoms, tol=1e-5):
        index, positions, offsets = utils.expand_cell(
            atoms.positions, atoms.cell)
        voronoi = scipy.spatial.Voronoi(positions)

        self.index = index
        self.positions = positions
        self.offsets = offsets
        self.voronoi = voronoi
        self.atoms = atoms
        self.tol = tol

    def get_voronoi_graph(self):
        """Return the Voronoi graph of a 3D periodic structure.
        This is a 3D representation of the graph which includes
        vectors for unique edges.

        Returns
        -------
        graph : MultiDiGraph object
            Voronoi based multi-directional graph.
        """
        ridge_points = self.voronoi.ridge_points
        zero_offset = (self.offsets == 0).all(1)
        origional_indices = np.where(zero_offset)[0]

        # Ensures that an atom in the cell comes first.
        sel = np.in1d(ridge_points.flatten(), origional_indices)
        ridge_points[~sel[::2]] = ridge_points[~sel[::2], ::-1]
        sel = sel.reshape(-1, 2)

        # Get bonds from both directions.
        bonds_out = ridge_points[sel.any(1)]
        bonds_in = np.roll(ridge_points[sel.all(1)], 1, 1)
        all_bonds = np.vstack([bonds_out, bonds_in])

        positions = self.positions[all_bonds].swapaxes(1, 2)
        vectors = np.diff(positions, 1).reshape(-1, 3)
        bonds = self.index[all_bonds]

        edges = [(b[0], b[1], {'vector': vectors[i]})
                 for i, b in enumerate(bonds)]
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        return graph

    def get_adsorption_sites(self):
        """

        Returns
        -------
        positions : ndarray (N, 3)

        connectivity : list of lists ()
            
        """
        points = self.voronoi.ridge_points
        zridge = self.positions[points][:, :, 2] > -self.tol
        sel = np.count_nonzero(zridge , 1) == 1
        surface_indices = points[sel][zridge[sel]]

        inside_xy = np.all(self.offsets[:, :2] == 0, 1)
        xy_points = np.where(inside_xy)[0]
        sel = np.in1d(xy_points, surface_indices)
        basis_indices = xy_points[sel]

        sel = np.in1d(
            points.flatten(), surface_indices).reshape(-1, 2)
        surface_points = points[sel.all(1)]

        # Top sites
        sites = self.positions[basis_indices].tolist()
        connectivity = basis_indices[:, None].tolist()

        # Bridge sites
        sites += self.positions[surface_points].mean(1).tolist()
        connectivity += surface_points.tolist()

        zpos = self.positions[surface_indices][:, :2]
        dt = scipy.spatial.Delaunay(zpos)
        simplices = dt.simplices

        # Reindex to the Voronoi values
        for i, v in enumerate(surface_indices):
            simplices[simplices == i] = v

        # Compute the angle of all simplex corners
        edges = np.tile(simplices[:, ::-1], 2).reshape(-1, 3, 2)
        vec = self.positions[edges.swapaxes(1, 2)] - \
            self.positions[simplices[:, None]]
        norm = np.linalg.norm(vec, axis=3).swapaxes(1, 2)
        uvec = (vec.swapaxes(1, 3) / norm[:, None]).swapaxes(1, 3)
        angles = np.sum(uvec[:, 0] * uvec[:, 1], 2)
        right = np.isclose(angles, 0)
        obtuse = angles < -self.tol

        # 1. No obtuse or right angle, its a 3-fold hollow
        sel_3fold = ~right.any(1) & ~obtuse.any(1)
        hollow3 = simplices[sel_3fold]

        sites += self.positions[hollow3].mean(1).tolist() 
        connectivity += hollow3.tolist()

        # Degenerate edges are 4-fold hollows
        if right.any():
            right_edges = np.sort(edges[right], 1)
            uright_edges, inv = np.unique(
                right_edges, axis=0, return_inverse=True)
            right_corners = simplices[right]

            hollow4 = []
            for i, edge in enumerate(uright_edges):
                opposites = right_corners[i == inv]
                if len(opposites) != 2:
                    continue
                hollow4 += [np.append(edge, opposites)]

            pos4 = self.positions[np.array(hollow4)]
            sites += pos4.mean(1).tolist()
            connectivity += hollow4

        sites = np.array(sites)
        cell = self.atoms.cell

        fractional = np.linalg.solve(cell.T, sites.T).T
        screen = np.all((fractional[:, :2] > -self.tol) &
                        (fractional[:, :2] < 1 - self.tol), 1)

        positions = sites[screen]
        connectivity = [self.index[_].tolist() for _ in
                        np.array(connectivity)[screen]]

        return positions, connectivity


def resize_cell(atoms, matrix):
    """Return a resized unit cell based on the matrix notation
    provided for a given atoms object. If a two dimensional
    matrix is passed, the 3rd basis is left unchanged.

    Parameters
    ----------
    atoms : Atoms object
        Structure to be resized.
    matrix : int ndarray (2, 2) | (3, 3)
        Transformation matrix to resize the unit cell with.
        2D matrices will leave the final vector unchanged.

    Returns
    -------
    resized_atoms : Atoms oject
        Resized strucutre.
    """
    dim = matrix.shape[0]
    M = np.eye(3)
    M[:dim, :dim] = np.array(matrix).T

    newcell = np.dot(M, atoms.cell)

    corners = utils.get_integer_enumeration(dim)
    corners = np.dot(corners, newcell[:dim, :dim])
    corners = np.linalg.solve(
        atoms.cell[:dim, :dim].T, corners.T).T

    repeat = np.ones(3, dtype=int)
    repeat[:dim] = np.ceil(corners.ptp(axis=0))

    atoms = atoms.copy()
    atoms *= repeat
    atoms.set_cell(newcell)
    atoms._sites.cell = atoms.cell

    fractional = atoms.get_scaled_positions()
    match = utils.get_matching_positions(fractional)
    del atoms[match]

    # Adsorption sites
    sites = atoms._sites
    if sites:
        sfrac = sites.get_scaled_positions()
        match = utils.get_matching_positions(sfrac)
        del sites[match]

    atoms = normalize_cell(atoms)

    return atoms
