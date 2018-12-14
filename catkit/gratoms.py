import networkx as nx
import numpy as np
import ase
import copy
import warnings
try:
    from builtins import super
except(ImportError):
    from __builtin__ import super

sym = np.array(ase.data.chemical_symbols)
em = nx.algorithms.isomorphism.numerical_edge_match('bonds', 1)
nm = nx.algorithms.isomorphism.numerical_node_match('number', 1)


class Gratoms(ase.Atoms):
    """Graph based atoms object.

    An Integrated class for an ASE atoms object with a corresponding
    Networkx Graph.
    """

    def __init__(self,
                 symbols=None,
                 positions=None,
                 numbers=None,
                 tags=None,
                 momenta=None,
                 masses=None,
                 magmoms=None,
                 charges=None,
                 scaled_positions=None,
                 cell=None,
                 pbc=None,
                 celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None,
                 edges=None):
        super().__init__(
            symbols, positions, numbers, tags, momenta,
            masses, magmoms, charges, scaled_positions, cell,
            pbc, celldisp, constraint, calculator, info)

        if self.pbc.any():
            self._graph = nx.MultiDiGraph()
        else:
            self._graph = nx.Graph(edges)

        nodes = [[i, {'number': n}]
                 for i, n in enumerate(self.arrays['numbers'])]
        self._graph.add_nodes_from(nodes)

        if isinstance(edges, (list, np.ndarray)):
            self._graph.add_edges_from(edges)

        self._sites = None

    @property
    def graph(self):
        return self._graph

    @property
    def nodes(self):
        return self._graph.nodes

    @property
    def edges(self):
        return self._graph.edges

    @property
    def adj(self):
        return self._graph.adj

    @property
    def degree(self):
        degree = self._graph.degree
        return np.array([_[1] for _ in degree])

    @property
    def connectivity(self):
        connectivity = nx.to_numpy_matrix(self._graph).astype(int)
        return connectivity

    def get_surface_atoms(self):
        """Return surface atoms."""
        surf_atoms = np.where(self.get_array('surface_atoms') > 0)[0]
        return surf_atoms

    def set_surface_atoms(self, top):
        """Assign surface atoms."""
        n = np.zeros(len(self))
        n[top] = 1
        self.set_array('surface_atoms', n)

    def set_site_properties(
            self,
            positions,
            connectivity=None,
            symmetry=None):
        """Set the adsorption site properties"""
        if symmetry is None:
            symmetry = np.arange(len(positions))
        if self._sites is None:
            self._sites = SiteGraph(
                positions=positions,
                cell=self._cell,
                tags=symmetry)

    def get_site_property(self, name):
        """Get adsorption site property."""
        if self._sites is None:
            value = None
        else:
            value = getattr(self._sites, name)
        return value

    def get_neighbor_symbols(self, u):
        """Get chemical symbols for neighboring atoms of u."""
        neighbors = list(self._graph[u])
        return sym[self.arrays['numbers'][neighbors]]

    def is_isomorph(self, other):
        """Check if isomorphic by bond count and atomic number."""
        isomorphic = nx.is_isomorphic(
            self._graph, other._graph, edge_match=em, node_match=nm)

        return isomorphic

    def get_chemical_tags(self, rank=2):
        """Generate a hash descriptive of the chemical formula (rank 0)
        or include bonding (rank 1).
        """
        cnt = np.bincount(self.arrays['numbers'])
        composition = ','.join(cnt.astype(str))

        if rank == 1:
            return composition[2:]

        for adj in self.adj.items():

            num = self.arrays['numbers'][list(adj[1].keys())]
            cnt += np.bincount(num, minlength=len(cnt))

        bonding = ','.join(cnt.astype(str))

        return composition[2:], bonding[2:]

    def get_unsaturated_nodes(self, screen=None):

        unsaturated = []
        for node, data in self.nodes(data=True):
            radicals = data['valence']

            if screen in data:
                continue

            if radicals > 0:
                unsaturated += [node]

        return np.array(unsaturated)

    def copy(self):
        """Return a copy."""
        atoms = self.__class__(cell=self._cell, pbc=self._pbc, info=self.info)

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a.copy()
        atoms.constraints = copy.deepcopy(self.constraints)
        if hasattr(self, '_graph'):
            atoms._graph = self._graph.copy()
        if getattr(self, '_sites'):
            atoms._sites = self._sites.copy()

        return atoms

    def __getitem__(self, i):
        """Return a subset of the atoms.

        i -- scalar integer, list of integers, or slice object
        describing which atoms to return.

        If i is a scalar, return an Atom object. If i is a list or a
        slice, return an Atoms object with the same cell, pbc, and
        other associated info as the original Atoms object. The
        indices of the constraints will be shuffled so that they match
        the indexing in the subset returned.
        """

        if isinstance(i, (int, np.int64)):
            natoms = len(self)
            if i < -natoms or i >= natoms:
                raise IndexError('Index out of range.')

            return ase.Atom(atoms=self, index=i)
        elif isinstance(i, list) and len(i) > 0:
            # Make sure a list of booleans will work correctly and not be
            # interpreted at 0 and 1 indices.
            i = np.array(i)
        elif isinstance(i, slice):
            if i.start is None:
                istart = 0
            elif i.start < 0:
                istart = i.start + len(self)
            else:
                istart = i.start

            if i.stop is None:
                istop = len(self)
            elif i.stop < 0:
                istop = i.stop + len(self)
            else:
                istop = i.stop

            istep = i.step if i.step is not None else 1
            i = np.array([i for i in range(istart, istop, istep)])
        conadd = []
        # Constraints need to be deepcopied, but only the relevant ones.
        for con in copy.deepcopy(self.constraints):
            if isinstance(con, (
                    ase.constraints.FixConstraint,
                    ase.constraints.FixBondLengths)):
                try:
                    con.index_shuffle(self, i)
                    conadd.append(con)
                except IndexError:
                    pass

        atoms = self.__class__(cell=self._cell, pbc=self._pbc, info=self.info,
                               celldisp=self._celldisp)

        atoms.arrays = {}
        for name, a in self.arrays.items():
            atoms.arrays[name] = a[i].copy()

        # Copy the graph, conserving correct indexing
        if self.nodes:
            nodes = [[_, {'number': n}]
                     for _, n in enumerate(self.arrays['numbers'])]
            atoms.graph.add_nodes_from(nodes)

            j = i.tolist()
            for u, v in self.graph.edges():
                if u not in i or v not in i:
                    continue
                atoms.graph.add_edge(j.index(u), j.index(v))

        atoms.constraints = conadd
        return atoms

    def __iadd__(self, other):
        """Extend atoms object by appending atoms from *other*."""
        if isinstance(other, ase.Atom):
            other = self.__class__([other])

        n1 = len(self)
        n2 = len(other)

        for name, a1 in self.arrays.items():
            a = np.zeros((n1 + n2, ) + a1.shape[1:], a1.dtype)
            a[:n1] = a1
            if name == 'masses':
                a2 = other.get_masses()
            else:
                a2 = other.arrays.get(name)
            if a2 is not None:
                a[n1:] = a2
            self.arrays[name] = a

        for name, a2 in other.arrays.items():
            if name in self.arrays:
                continue
            a = np.empty((n1 + n2, ) + a2.shape[1:], a2.dtype)
            a[n1:] = a2
            if name == 'masses':
                a[:n1] = self.get_masses()[:n1]
            else:
                a[:n1] = 0

            self.set_array(name, a)

        if isinstance(other, Gratoms):
            tg = isinstance(self._graph, nx.Graph)
            if tg & isinstance(other._graph, nx.Graph):
                other._graph = nx.MultiDiGraph(other._graph)

            self._graph = nx.disjoint_union(self._graph, other._graph)

        return self

    def __delitem__(self, i):
        from ase.constraints import FixAtoms
        for c in self._constraints:
            if not isinstance(c, FixAtoms):
                raise RuntimeError('Remove constraint using set_constraint() '
                                   'before deleting atoms.')

        if isinstance(i, (list, int)):
            # Make sure a list of booleans will work correctly and not be
            # interpreted at 0 and 1 indices.
            i = np.atleast_1d(i)

        n = len(self)
        i = np.arange(n)[i]

        if len(self._constraints) > 0:
            if isinstance(i, int):
                i = [i]
            constraints = []
            for c in self._constraints:
                c = c.delete_atoms(i, n)
                if c is not None:
                    constraints.append(c)
            self.constraints = constraints

        mask = np.ones(len(self), bool)
        mask[i] = False

        for name, a in self.arrays.items():
            self.arrays[name] = a[mask]

        if self.nodes:
            self._graph.remove_nodes_from(i)
            mapping = dict(zip(np.where(mask)[0], np.arange(len(self))))
            nx.relabel_nodes(self._graph, mapping, copy=False)

    def __imul__(self, m):
        """In-place repeat of atoms."""
        if isinstance(m, int):
            m = [m, m, m]
        if np.all(m == 1):
            return self

        for x, vec in zip(m, self._cell):
            if x != 1 and not vec.any():
                raise ValueError(
                    'Cannot repeat along undefined lattice vector')

        n = len(self)
        M = np.product(m)

        for name, a in self.arrays.items():
            repeat = np.ones(a.ndim, dtype=int)
            repeat[0] = M
            self.arrays[name] = np.tile(a, repeat)

        s = [slice(0, _) for _ in m]
        offset = np.mgrid[s].reshape(3, -1).T
        offset = np.tile(offset, n).reshape(-1, 3)

        positions = self.arrays['positions']
        positions += np.dot(offset, self._cell)

        # For adsorption sites
        sites = self._sites
        if sites:
            sites *= [m[0], m[1], 1]
 
        self._cell *= np.asarray(m)[:, None]

        if self.constraints is not None:
            self.constraints = [c.repeat(m, n) for c in self.constraints]

        self._graph = nx.MultiDiGraph()
        nodes = [[_, {'number': n}]
                 for _, n in enumerate(self.arrays['numbers'])]
        self._graph.add_nodes_from(nodes)

        # This part is for the 3D graph
        if not self.edges:
            return self

        G = nx.get_edge_attributes(self._graph, 'vector')
        nodes = np.tile(np.array(list(G.keys()))[:, 0], (M, 1))
        nodes += np.arange(M)[:, None] * n
        nodes = nodes.flatten()

        vectors = np.tile(np.array(list(G.values())), (M, 1))
        edges = vectors + positions[nodes]

        tol = 1e-5
        edges = np.linalg.solve(self._cell.T, edges.T).T
        edges = ((edges + tol) % 1) - tol
        frac = np.linalg.solve(self._cell.T, positions.T).T

        fdist = edges[None, :, :] - frac[:, None]
        match = np.where((np.abs(fdist) < tol).all(2))
        bonds = np.vstack([match[0], nodes[match[1]]]).T

        edges = [(b[0], b[1], {'vector': vectors[i]})
                 for i, b in enumerate(bonds)]
        self._graph.add_edges_from(edges)

        return self


class SiteGraph(Gratoms):

    def set_connectivity(self, connectivity=None):
        """Set connectivity of the sites"""
        if connectivity is None:
            n = len(self.positions)
            connectivity = [[] for _ in range(n)]
        self.connectivity = np.asarray(connectivity)
