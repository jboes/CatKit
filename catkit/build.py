import catkit
import ase.build
import ase


def bulk(name, crystalstructure=None, primitive=False, **kwargs):
    """Return the standard conventional cell of a bulk structure
    created using ASE. Accepts all keyword arguments for the ase
    bulk generator.

    Parameters
    ----------
    name : Atoms object | str
        Chemical symbol or symbols as in 'MgO' or 'NaCl'.
    crystalstructure : str
        Must be one of sc, fcc, bcc, hcp, diamond, zincblende,
        rocksalt, cesiumchloride, fluorite or wurtzite.
    primitive : bool
        Return the primitive unit cell instead of the conventional
        standard cell.

    Returns
    -------
    standardized_bulk : Gratoms object
        The conventional standard or primitive bulk structure.
    """
    if isinstance(name, str):
        atoms = ase.build.bulk(name, crystalstructure, **kwargs)
    else:
        atoms = name
    standardized_bulk = catkit.gen.symmetry.get_standardized_cell(
        atoms, primitive=primitive)

    return standardized_bulk


def surface(
        elements,
        size,
        miller=(1, 1, 1),
        termination=0,
        vacuum=6,
        fixed=0,
        **kwargs):
    """A helper function to return the surface associated with a
    given set of input parameters to the general surface generator.

    Parameters
    ----------
    elements : str or object
        The atomic symbol to be passed to the as bulk builder function
        or an atoms object representing the bulk structure to use.
    size : array_like (2,)
        Volume multiple of the top cell and the number of layers.
    miller : list (3,) | (4,)
        The miller index to cleave the surface structure from.
        If 4 values are used, assume Miller-Bravis convention.
    termination : int
        The index associated with a specific slab termination.
    fixed : int
        Number of layers to constrain.
    vacuum : float
        Angstroms of vacuum to add to the unit cell.

    Returns
    -------
    slab : Gratoms object
        Return a slab generated from the specified bulk structure.
    """
    if isinstance(elements, ase.Atoms):
        atoms = elements
    else:
        bkwargs = kwargs.copy()
        keys = ['crystalstructure', 'a', 'c', 'covera',
                'u', 'orthorhombic', 'cubic']
        for key in kwargs:
            if key not in keys:
                del bkwargs[key]
        atoms = ase.build.bulk(elements, **bkwargs)

    miller = catkit.gen.surface.bravis_miller_to_miller(miller)

    satoms = catkit.gen.symmetry.get_standardized_cell(atoms)
    atoms = catkit.gen.symmetry.get_standardized_cell(
        atoms, primitive=True)

    miller = catkit.gen.surface.convert_miller_index(
        miller, satoms, atoms)

    generator = catkit.gen.surface.SlabGenerator(
        bulk=atoms,
        miller_index=miller,
        layers=size[-1],
        termination=termination,
        tol=kwargs.get('tol', 1e-8)
    )

    slab = generator.get_slab(size=size[0], vacuum=vacuum)

    if fixed:
        tags = slab.get_tags()
        constraints = ase.constraints.FixAtoms(
            mask=tags > (tags.max() - fixed))
        slab.set_constraint(constraints)

    return slab


def molecule(species, bond_index=None, vacuum=0):
    """Return list of enumerated gas-phase molecule structures based
    on species and topology.

    Parameters
    ----------
    species : str
        The chemical symbols to construct a molecule from.
    bond_index : int
        Construct the molecule as though it were adsorbed to a surface
        parallel to the z-axis. Will bond by the atom index given.
    vacuum : float
        Angstroms of vacuum to pad the molecules with.

    Returns
    -------
    images : list of Gratoms objects
        3D structures of the requested chemical species and topologies.
    """
    molecule_graphs = catkit.gen.molecules.get_topologies(species)

    images = []
    for atoms in molecule_graphs:
        atoms = catkit.gen.molecules.get_3D_positions(atoms, bond_index)
        images += [atoms]

    return images
