"""
Bemis-Murcko scaffold decomposition algorithm.

Implementation of the Bemis-Murcko decomposition based on:

    Bemis & Murcko, "The Properties of Known Drugs. 1. Molecular 
    Frameworks" J. Med. Chem. 1996, 39, 2887-2893.
    DOI: 10.1021/jm9602928

This module provides functions for extracting molecular scaffolds,
frameworks, and side chains from molecules.

Key concepts:
- Scaffold: Ring systems connected by linkers (preserves atom types and bond orders)
- Framework: Generic scaffold with all atoms as carbon and all bonds as single
- Side chains: Atoms not part of the scaffold
- Ring systems: Fused/spiro ring assemblies
- Linkers: Chains connecting ring systems
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterator

from chirpy.parser import parse
from chirpy.rings import find_sssr, find_ring_systems, get_ring_bonds, _find_ring_atoms_and_bonds_fast
from chirpy.writer import to_smiles
from chirpy.types import Atom, Bond, Molecule


@dataclass
class MurckoDecomposition:
    """Result of Bemis-Murcko decomposition.
    
    Attributes:
        scaffold: The molecular scaffold (rings + linkers).
        framework: Generic framework (all carbons, single bonds).
        side_chains: List of side chain fragments.
        ring_systems: List of individual ring systems.
    """
    scaffold: Molecule | None
    framework: Molecule | None
    side_chains: list[Molecule]
    ring_systems: list[Molecule]


def _get_ring_atoms(mol: Molecule) -> set[int]:
    """Get all atoms that are part of any ring.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        Set of atom indices in rings.
    """
    ring_atoms, _ = _find_ring_atoms_and_bonds_fast(mol)
    return ring_atoms


def _find_linker_atoms(
    mol: Molecule,
    ring_atoms: set[int],
) -> set[int]:
    """Find atoms that are linkers between ring systems.
    
    Linker atoms are non-ring atoms that lie on the shortest path
    between two ring atoms.
    
    Args:
        mol: Molecule to analyze.
        ring_atoms: Set of ring atom indices.
    
    Returns:
        Set of linker atom indices.
    """
    if len(ring_atoms) < 2:
        return set()
    
    # Build adjacency list
    adj: dict[int, set[int]] = {i: set() for i in range(mol.num_atoms)}
    for bond in mol.bonds:
        adj[bond.atom1_idx].add(bond.atom2_idx)
        adj[bond.atom2_idx].add(bond.atom1_idx)
    
    linker_atoms: set[int] = set()
    
    # For each non-ring atom, check if it connects two different ring atoms
    # via paths that don't go through other ring atoms
    non_ring_atoms = set(range(mol.num_atoms)) - ring_atoms
    
    for atom_idx in non_ring_atoms:
        # Find all ring atoms reachable from this atom through non-ring paths
        # plus direct ring neighbors
        ring_neighbors: set[int] = set()
        
        # Direct ring neighbors
        for neighbor in adj[atom_idx]:
            if neighbor in ring_atoms:
                ring_neighbors.add(neighbor)
        
        # If this atom connects to ring atoms, it might be a linker
        if len(ring_neighbors) >= 1:
            # Check if this atom is on a path between ring systems
            # by doing BFS from this atom through non-ring atoms
            visited: set[int] = {atom_idx}
            queue = [atom_idx]
            connected_ring_atoms: set[int] = set(ring_neighbors)
            
            while queue:
                current = queue.pop(0)
                for neighbor in adj[current]:
                    if neighbor in ring_atoms:
                        connected_ring_atoms.add(neighbor)
                    elif neighbor not in visited and neighbor in non_ring_atoms:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # This is a linker if it connects ring atoms from different ring systems
            # For simplicity, we check if the connected ring atoms belong to different rings
            if len(connected_ring_atoms) >= 2:
                linker_atoms.add(atom_idx)
    
    # Now expand to include all atoms on linker chains
    # A linker chain is a path of non-ring atoms connecting two ring atoms
    expanded_linkers: set[int] = set()
    
    # Find all chains between ring atoms
    for start_ring_atom in ring_atoms:
        for first_neighbor in adj[start_ring_atom]:
            if first_neighbor not in ring_atoms:
                # BFS to find if this leads to another ring atom
                visited: set[int] = {first_neighbor}
                parent: dict[int, int] = {first_neighbor: start_ring_atom}
                queue = [first_neighbor]
                
                while queue:
                    current = queue.pop(0)
                    for neighbor in adj[current]:
                        if neighbor in ring_atoms and neighbor != start_ring_atom:
                            # Found a path to another ring atom
                            # Add all atoms in the path as linkers
                            path_atom = current
                            while path_atom not in ring_atoms:
                                expanded_linkers.add(path_atom)
                                path_atom = parent[path_atom]
                        elif neighbor not in visited and neighbor not in ring_atoms:
                            visited.add(neighbor)
                            parent[neighbor] = current
                            queue.append(neighbor)
    
    return expanded_linkers


def _find_scaffold_atoms(mol: Molecule) -> tuple[set[int], set[int]]:
    """Find atoms belonging to the molecular scaffold.
    
    The scaffold consists of ring atoms and linker atoms.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        Tuple of (scaffold_atoms, ring_atoms).
    """
    ring_atoms = _get_ring_atoms(mol)
    
    if not ring_atoms:
        return set(), set()
    
    linker_atoms = _find_linker_atoms(mol, ring_atoms)
    scaffold_atoms = ring_atoms | linker_atoms
    
    return scaffold_atoms, ring_atoms


def _extract_substructure(
    mol: Molecule,
    atom_indices: set[int],
    cap_with_hydrogen: bool = False,
) -> Molecule | None:
    """Extract a substructure containing only specified atoms.
    
    Args:
        mol: Original molecule.
        atom_indices: Indices of atoms to include.
        cap_with_hydrogen: If True, add hydrogens where bonds were cut.
    
    Returns:
        New molecule containing only the specified atoms, or None if empty.
    """
    if not atom_indices:
        return None
    
    # Create mapping from old to new indices
    sorted_indices = sorted(atom_indices)
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(sorted_indices)}
    
    new_mol = Molecule()
    
    # Add atoms
    for old_idx in sorted_indices:
        old_atom = mol.atoms[old_idx]
        new_atom = Atom(
            idx=old_to_new[old_idx],
            symbol=old_atom.symbol,
            charge=old_atom.charge,
            explicit_hydrogens=old_atom.explicit_hydrogens,
            is_aromatic=old_atom.is_aromatic,
            isotope=old_atom.isotope,
            chirality=old_atom.chirality,
            atom_class=old_atom.atom_class,
        )
        new_atom.bond_indices = []
        new_mol.atoms.append(new_atom)
    
    # Add bonds (only between included atoms)
    for bond in mol.bonds:
        if bond.atom1_idx in atom_indices and bond.atom2_idx in atom_indices:
            new_bond = Bond(
                idx=len(new_mol.bonds),
                atom1_idx=old_to_new[bond.atom1_idx],
                atom2_idx=old_to_new[bond.atom2_idx],
                order=bond.order,
                is_aromatic=bond.is_aromatic,
                stereo=bond.stereo,
            )
            new_mol.bonds.append(new_bond)
            new_mol.atoms[new_bond.atom1_idx].bond_indices.append(new_bond.idx)
            new_mol.atoms[new_bond.atom2_idx].bond_indices.append(new_bond.idx)
    
    return new_mol


def _make_generic(mol: Molecule) -> Molecule:
    """Convert molecule to generic framework.
    
    All atoms become carbon (or nitrogen for charged atoms),
    all bonds become single bonds, aromaticity is removed.
    
    Args:
        mol: Molecule to convert.
    
    Returns:
        Generic framework molecule.
    """
    new_mol = Molecule()
    
    for atom in mol.atoms:
        # Keep nitrogen if it's charged or in a position that requires it
        # Otherwise convert to carbon
        new_symbol = 'C'
        if atom.charge != 0:
            new_symbol = atom.symbol  # Keep original if charged
        
        new_atom = Atom(
            idx=atom.idx,
            symbol=new_symbol,
            charge=0,  # Remove charges for generic framework
            explicit_hydrogens=0,
            is_aromatic=False,
            isotope=None,
            chirality=None,
            atom_class=None,
        )
        new_atom.bond_indices = []
        new_mol.atoms.append(new_atom)
    
    for bond in mol.bonds:
        new_bond = Bond(
            idx=bond.idx,
            atom1_idx=bond.atom1_idx,
            atom2_idx=bond.atom2_idx,
            order=1,  # All single bonds
            is_aromatic=False,
            stereo=None,
        )
        new_mol.bonds.append(new_bond)
        new_mol.atoms[new_bond.atom1_idx].bond_indices.append(new_bond.idx)
        new_mol.atoms[new_bond.atom2_idx].bond_indices.append(new_bond.idx)
    
    return new_mol


def _get_connected_components(mol: Molecule) -> list[set[int]]:
    """Find connected components in a molecule.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        List of atom index sets, one per component.
    """
    if mol.num_atoms == 0:
        return []
    
    visited: set[int] = set()
    components: list[set[int]] = []
    
    # Build adjacency
    adj: dict[int, set[int]] = {i: set() for i in range(mol.num_atoms)}
    for bond in mol.bonds:
        adj[bond.atom1_idx].add(bond.atom2_idx)
        adj[bond.atom2_idx].add(bond.atom1_idx)
    
    for start in range(mol.num_atoms):
        if start in visited:
            continue
        
        component: set[int] = set()
        stack = [start]
        
        while stack:
            atom = stack.pop()
            if atom in visited:
                continue
            visited.add(atom)
            component.add(atom)
            
            for neighbor in adj[atom]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        components.append(component)
    
    return components


def get_scaffold(mol: Molecule) -> Molecule | None:
    """Extract the Bemis-Murcko scaffold from a molecule.
    
    The scaffold consists of all ring systems and the linker chains
    connecting them, preserving atom types and bond orders.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        Scaffold molecule, or None if molecule has no rings.
    
    Example:
        >>> mol = parse("c1ccc(CC(=O)Nc2ccccc2)cc1")
        >>> scaffold = get_scaffold(mol)
        >>> to_smiles(scaffold)  # rings and linker preserved
        'O=C(Cc1ccccc1)Nc1ccccc1'
    """
    scaffold_atoms, _ = _find_scaffold_atoms(mol)
    return _extract_substructure(mol, scaffold_atoms)


def get_framework(mol: Molecule) -> Molecule | None:
    """Extract the generic Bemis-Murcko framework from a molecule.
    
    The framework is the scaffold with all atoms converted to carbon
    and all bonds converted to single bonds.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        Framework molecule, or None if molecule has no rings.
    
    Example:
        >>> mol = parse("c1ccc(CC(=O)Nc2ccccc2)cc1")
        >>> framework = get_framework(mol)
        >>> to_smiles(framework)  # generic carbons
        'C1CCC(CCC2CCCCC2)CC1'
    """
    scaffold = get_scaffold(mol)
    if scaffold is None:
        return None
    return _make_generic(scaffold)


def get_side_chains(mol: Molecule) -> list[Molecule]:
    """Extract side chains from a molecule.
    
    Side chains are the parts of the molecule that are not part of
    the scaffold (rings and linkers).
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        List of side chain molecules.
    
    Example:
        >>> mol = parse("CCCc1ccc(OC)cc1")
        >>> chains = get_side_chains(mol)
        >>> [to_smiles(c) for c in chains]
        ['CCC', 'OC']
    """
    scaffold_atoms, _ = _find_scaffold_atoms(mol)
    
    if not scaffold_atoms:
        # No scaffold means the whole molecule is "side chain"
        # But if the molecule is empty, return empty list
        if mol.num_atoms == 0:
            return []
        return [deepcopy(mol)]
    
    side_chain_atoms = set(range(mol.num_atoms)) - scaffold_atoms
    
    if not side_chain_atoms:
        return []
    
    # Find connected components of side chain atoms
    # Build adjacency for side chain atoms only
    adj: dict[int, set[int]] = {i: set() for i in side_chain_atoms}
    for bond in mol.bonds:
        if bond.atom1_idx in side_chain_atoms and bond.atom2_idx in side_chain_atoms:
            adj[bond.atom1_idx].add(bond.atom2_idx)
            adj[bond.atom2_idx].add(bond.atom1_idx)
    
    visited: set[int] = set()
    components: list[set[int]] = []
    
    for start in side_chain_atoms:
        if start in visited:
            continue
        
        component: set[int] = set()
        stack = [start]
        
        while stack:
            atom = stack.pop()
            if atom in visited:
                continue
            visited.add(atom)
            component.add(atom)
            
            for neighbor in adj.get(atom, set()):
                if neighbor not in visited:
                    stack.append(neighbor)
        
        components.append(component)
    
    # Extract each component as a molecule
    side_chains: list[Molecule] = []
    for component in components:
        chain = _extract_substructure(mol, component)
        if chain is not None:
            side_chains.append(chain)
    
    return side_chains


def get_ring_systems(mol: Molecule) -> list[Molecule]:
    """Extract individual ring systems from a molecule.
    
    Each ring system is a set of fused or spiro-connected rings.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        List of ring system molecules.
    
    Example:
        >>> mol = parse("c1ccc2ccccc2c1CCc3ccccc3")
        >>> systems = get_ring_systems(mol)
        >>> len(systems)  # naphthalene and benzene
        2
    """
    rings = find_sssr(mol)
    if not rings:
        return []
    
    systems = find_ring_systems(rings)
    
    result: list[Molecule] = []
    for system in systems:
        # Collect all atoms in this ring system
        system_atoms: set[int] = set()
        for ring in system:
            system_atoms.update(ring)
        
        ring_mol = _extract_substructure(mol, system_atoms)
        if ring_mol is not None:
            result.append(ring_mol)
    
    return result


def murcko_decompose(
    mol: Molecule,
    return_mols: bool = False,
) -> MurckoDecomposition | dict[str, str | list[str] | None]:
    """Perform complete Bemis-Murcko decomposition.
    
    Returns the scaffold, framework, side chains, and ring systems.
    
    Args:
        mol: Molecule to decompose.
        return_mols: If True, return Molecule objects. If False, return SMILES.
    
    Returns:
        MurckoDecomposition object if return_mols=True, otherwise dict with SMILES.
    
    Example:
        >>> mol = parse("CCCc1ccc(CC(=O)Nc2ccccc2)cc1")
        >>> result = murcko_decompose(mol)
        >>> result['scaffold']
        'O=C(Cc1ccc(CCC)cc1)Nc1ccccc1'
    """
    scaffold = get_scaffold(mol)
    framework = get_framework(mol)
    side_chains = get_side_chains(mol)
    ring_systems = get_ring_systems(mol)
    
    if return_mols:
        return MurckoDecomposition(
            scaffold=scaffold,
            framework=framework,
            side_chains=side_chains,
            ring_systems=ring_systems,
        )
    
    return {
        'scaffold': to_smiles(scaffold) if scaffold else None,
        'framework': to_smiles(framework) if framework else None,
        'side_chains': [to_smiles(c) for c in side_chains],
        'ring_systems': [to_smiles(r) for r in ring_systems],
    }


def get_scaffold_smiles(smiles: str) -> str | None:
    """Convenience function to get scaffold SMILES from input SMILES.
    
    Args:
        smiles: Input SMILES string.
    
    Returns:
        Scaffold SMILES, or None if no rings.
    
    Example:
        >>> get_scaffold_smiles("CCCc1ccccc1OC")
        'c1ccccc1'
    """
    mol = parse(smiles)
    scaffold = get_scaffold(mol)
    return to_smiles(scaffold) if scaffold else None


def get_framework_smiles(smiles: str) -> str | None:
    """Convenience function to get framework SMILES from input SMILES.
    
    Args:
        smiles: Input SMILES string.
    
    Returns:
        Framework SMILES, or None if no rings.
    
    Example:
        >>> get_framework_smiles("c1ccc(N)cc1")
        'C1CCCCC1'
    """
    mol = parse(smiles)
    framework = get_framework(mol)
    return to_smiles(framework) if framework else None
