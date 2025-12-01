"""
SMARTS substructure matching.

This module provides functionality for finding SMARTS pattern matches
within molecules, similar to RDKit's GetSubstructMatches().

The implementation uses a VF2-like backtracking algorithm for subgraph
isomorphism with SMARTS-specific atom and bond matching.

Example:
    >>> from chirpy import parse
    >>> from chirpy.match import substructure_search, has_substructure
    >>> 
    >>> mol = parse("c1ccccc1CCO")
    >>> pattern = parse("[cR]")
    >>> matches = substructure_search(mol, pattern)
    >>> print(matches)  # [(0,), (1,), (2,), (3,), (4,), (5,)]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rings import get_ring_info

if TYPE_CHECKING:
    from .types import Atom, Bond, Molecule


def _atom_matches(
    mol_atom: "Atom",
    pattern_atom: "Atom",
    mol: "Molecule",
    ring_count: dict[int, int],
    ring_sizes: dict[int, set[int]],
) -> bool:
    """Check if a molecule atom matches a SMARTS pattern atom.
    
    Args:
        mol_atom: Atom from the molecule being searched.
        pattern_atom: Atom from the SMARTS pattern.
        mol: The molecule (for context).
        ring_count: Ring membership count for each atom.
        ring_sizes: Ring sizes for each atom.
    
    Returns:
        True if the molecule atom satisfies all pattern constraints.
    """
    # Wildcard matches anything
    if pattern_atom.is_wildcard and not pattern_atom.atom_list:
        # But still check SMARTS queries if present
        pass
    elif pattern_atom.atom_list:
        # Atom list: check if mol_atom symbol is in list
        mol_symbol = mol_atom.symbol.upper()
        pattern_symbols = [s.upper() for s in pattern_atom.atom_list]
        
        if pattern_atom.atom_list_negated:
            if mol_symbol in pattern_symbols:
                return False
        else:
            if mol_symbol not in pattern_symbols:
                return False
    elif pattern_atom.atom_list_negated:
        # Negated single atom [!C]
        if mol_atom.symbol.upper() == pattern_atom.symbol.upper():
            return False
    elif not pattern_atom.is_wildcard:
        # Regular element match
        if mol_atom.symbol.upper() != pattern_atom.symbol.upper():
            return False
    
    # Aromaticity
    if pattern_atom.is_aromatic and not mol_atom.is_aromatic:
        return False
    if not pattern_atom.is_aromatic and not pattern_atom.is_wildcard:
        # Non-aromatic pattern atom - check if mol atom is also non-aromatic
        # Only enforce if pattern explicitly uses uppercase
        if pattern_atom.symbol and pattern_atom.symbol[0].isupper():
            if mol_atom.is_aromatic:
                return False
    
    # Charge
    if pattern_atom.charge != 0:
        if mol_atom.charge != pattern_atom.charge:
            return False
    
    # Ring membership (R, R0, R1, R2, ...)
    if pattern_atom.ring_count is not None:
        atom_ring_count = ring_count.get(mol_atom.idx, 0)
        if pattern_atom.ring_count == -1:
            # [R] means in any ring
            if atom_ring_count == 0:
                return False
        elif pattern_atom.ring_count == 0:
            # [R0] means not in any ring
            if atom_ring_count > 0:
                return False
        else:
            # [R2] means in exactly 2 rings
            if atom_ring_count != pattern_atom.ring_count:
                return False
    
    # Ring size (r5, r6, ...)
    if pattern_atom.ring_size is not None:
        atom_ring_sizes = ring_sizes.get(mol_atom.idx, set())
        if pattern_atom.ring_size == -1:
            # [r] means in any ring (same as R)
            if not atom_ring_sizes:
                return False
        else:
            # [r6] means in a 6-membered ring
            if pattern_atom.ring_size not in atom_ring_sizes:
                return False
    
    # Calculate total hydrogens for the molecule atom (needed for multiple queries)
    mol_total_h = mol_atom.total_hydrogens(mol)
    
    # Degree (D, D2, D3, ...) - number of explicit connections (not H)
    if pattern_atom.degree_query is not None:
        degree = len(mol_atom.bond_indices)
        if pattern_atom.degree_query == -1:
            # [D] means any degree > 0
            if degree == 0:
                return False
        else:
            if degree != pattern_atom.degree_query:
                return False
    
    # Connectivity (X, X2, ...) - total connections including implicit H
    if pattern_atom.connectivity_query is not None:
        # Connectivity = explicit bonds + total hydrogens
        explicit_bonds = len(mol_atom.bond_indices)
        connectivity = explicit_bonds + mol_total_h
        
        if pattern_atom.connectivity_query == -1:
            if connectivity == 0:
                return False
        else:
            if connectivity != pattern_atom.connectivity_query:
                return False
    
    # Valence (v, v4, ...)
    if pattern_atom.valence_query is not None:
        # Valence = sum of bond orders + total H
        valence = 0
        for bond_idx in mol_atom.bond_indices:
            bond = mol.bonds[bond_idx]
            if bond.is_aromatic:
                valence += 1.5  # Aromatic bonds count as 1.5
            else:
                valence += bond.order
        valence = int(valence + 0.5)  # Round
        valence += mol_total_h
        
        if pattern_atom.valence_query == -1:
            if valence == 0:
                return False
        else:
            if valence != pattern_atom.valence_query:
                return False
    
    # Hydrogen count (H, H2, ...) - matches total hydrogens, not just explicit
    if pattern_atom.explicit_hydrogens > 0:
        if mol_total_h != pattern_atom.explicit_hydrogens:
            return False
    
    # Isotope
    if pattern_atom.isotope is not None:
        if mol_atom.isotope != pattern_atom.isotope:
            return False
    
    return True


def _bond_matches(
    mol_bond: "Bond",
    pattern_bond: "Bond",
) -> bool:
    """Check if a molecule bond matches a SMARTS pattern bond.
    
    Args:
        mol_bond: Bond from the molecule being searched.
        pattern_bond: Bond from the SMARTS pattern.
    
    Returns:
        True if the molecule bond satisfies the pattern constraints.
    """
    # Any bond (~) matches everything
    if pattern_bond.is_any:
        return True
    
    # Aromatic bond
    if pattern_bond.is_aromatic:
        return mol_bond.is_aromatic
    
    # Specific bond order
    if pattern_bond.order == mol_bond.order:
        return True
    
    # Single bond pattern can match aromatic (implicit in aromatic systems)
    if pattern_bond.order == 1 and mol_bond.is_aromatic:
        return True
    
    return False


def _get_bond_between(mol: "Molecule", atom1_idx: int, atom2_idx: int) -> "Bond | None":
    """Get the bond between two atoms, or None if not bonded."""
    for bond_idx in mol.atoms[atom1_idx].bond_indices:
        bond = mol.bonds[bond_idx]
        if bond.other_atom(atom1_idx) == atom2_idx:
            return bond
    return None


def substructure_search(
    mol: "Molecule",
    pattern: "Molecule",
    uniquify: bool = True,
) -> list[tuple[int, ...]]:
    """Find all matches of a SMARTS pattern in a molecule.
    
    Uses a VF2-like backtracking algorithm for subgraph isomorphism.
    
    Args:
        mol: The molecule to search in.
        pattern: The SMARTS pattern to search for.
        uniquify: If True (default), return only one match per unique set
            of molecule atoms (like RDKit's default). If False, return
            all permutations of matches.
    
    Returns:
        List of tuples, where each tuple contains the molecule atom
        indices that match the pattern atoms (in pattern atom order).
    
    Example:
        >>> mol = parse("c1ccccc1CCO")
        >>> pattern = parse("[cR]")
        >>> matches = substructure_search(mol, pattern)
        >>> print(matches)  # [(0,), (1,), (2,), (3,), (4,), (5,)]
    """
    if pattern.num_atoms == 0:
        return [()]
    
    if mol.num_atoms == 0:
        return []
    
    # Precompute ring info for the molecule
    ring_count, ring_sizes = get_ring_info(mol)
    
    # Build pattern adjacency
    pattern_adj: dict[int, list[tuple[int, int]]] = {
        i: [] for i in range(pattern.num_atoms)
    }
    for bond in pattern.bonds:
        pattern_adj[bond.atom1_idx].append((bond.atom2_idx, bond.idx))
        pattern_adj[bond.atom2_idx].append((bond.atom1_idx, bond.idx))
    
    matches: list[tuple[int, ...]] = []
    seen_atom_sets: set[frozenset[int]] = set()  # For uniquify
    
    def backtrack(
        mapping: dict[int, int],  # pattern_idx -> mol_idx
        pattern_idx: int,
    ) -> None:
        """Recursively try to extend the mapping."""
        if pattern_idx == pattern.num_atoms:
            # Complete match found
            match = tuple(mapping[i] for i in range(pattern.num_atoms))
            
            if uniquify:
                # Only keep one match per unique set of molecule atoms
                atom_set = frozenset(match)
                if atom_set in seen_atom_sets:
                    return
                seen_atom_sets.add(atom_set)
            
            matches.append(match)
            return
        
        pattern_atom = pattern.atoms[pattern_idx]
        
        # Find candidate molecule atoms
        candidates: list[int]
        
        if pattern_idx == 0:
            # First atom - try all molecule atoms
            candidates = list(range(mol.num_atoms))
        else:
            # Find neighbors in pattern that are already mapped
            candidates_set: set[int] | None = None
            
            for nbr_idx, bond_idx in pattern_adj[pattern_idx]:
                if nbr_idx in mapping:
                    # This pattern neighbor is mapped to mol atom
                    mol_nbr_idx = mapping[nbr_idx]
                    mol_atom = mol.atoms[mol_nbr_idx]
                    
                    # Get neighbors of the mapped mol atom
                    nbr_candidates: set[int] = set()
                    for mol_bond_idx in mol_atom.bond_indices:
                        mol_bond = mol.bonds[mol_bond_idx]
                        mol_other = mol_bond.other_atom(mol_nbr_idx)
                        
                        # Check bond compatibility
                        pattern_bond = pattern.bonds[bond_idx]
                        if _bond_matches(mol_bond, pattern_bond):
                            nbr_candidates.add(mol_other)
                    
                    if candidates_set is None:
                        candidates_set = nbr_candidates
                    else:
                        candidates_set &= nbr_candidates
            
            if candidates_set is None:
                # No mapped neighbors - disconnected pattern component
                # Try all unmatched atoms
                used = set(mapping.values())
                candidates = [i for i in range(mol.num_atoms) if i not in used]
            else:
                candidates = list(candidates_set)
        
        # Filter out already-used atoms
        used = set(mapping.values())
        
        for mol_idx in candidates:
            if mol_idx in used:
                continue
            
            mol_atom = mol.atoms[mol_idx]
            
            # Check atom compatibility
            if not _atom_matches(mol_atom, pattern_atom, mol, ring_count, ring_sizes):
                continue
            
            # Check bond compatibility with all mapped neighbors
            bonds_ok = True
            for nbr_idx, bond_idx in pattern_adj[pattern_idx]:
                if nbr_idx in mapping:
                    mol_nbr_idx = mapping[nbr_idx]
                    mol_bond = _get_bond_between(mol, mol_idx, mol_nbr_idx)
                    
                    if mol_bond is None:
                        bonds_ok = False
                        break
                    
                    pattern_bond = pattern.bonds[bond_idx]
                    if not _bond_matches(mol_bond, pattern_bond):
                        bonds_ok = False
                        break
            
            if not bonds_ok:
                continue
            
            # Extend mapping and recurse
            mapping[pattern_idx] = mol_idx
            backtrack(mapping, pattern_idx + 1)
            del mapping[pattern_idx]
    
    backtrack({}, 0)
    return matches


def has_substructure(mol: "Molecule", pattern: "Molecule") -> bool:
    """Check if a molecule contains a SMARTS pattern.
    
    This is more efficient than substructure_search when you only
    need to know if a match exists.
    
    Args:
        mol: The molecule to search in.
        pattern: The SMARTS pattern to search for.
    
    Returns:
        True if the pattern is found in the molecule.
    
    Example:
        >>> mol = parse("c1ccccc1CCO")
        >>> pattern = parse("[OH]")
        >>> has_substructure(mol, pattern)
        True
    """
    if pattern.num_atoms == 0:
        return True
    
    if mol.num_atoms == 0:
        return False
    
    # Precompute ring info
    ring_count, ring_sizes = get_ring_info(mol)
    
    # Build pattern adjacency
    pattern_adj: dict[int, list[tuple[int, int]]] = {
        i: [] for i in range(pattern.num_atoms)
    }
    for bond in pattern.bonds:
        pattern_adj[bond.atom1_idx].append((bond.atom2_idx, bond.idx))
        pattern_adj[bond.atom2_idx].append((bond.atom1_idx, bond.idx))
    
    def backtrack(mapping: dict[int, int], pattern_idx: int) -> bool:
        if pattern_idx == pattern.num_atoms:
            return True
        
        pattern_atom = pattern.atoms[pattern_idx]
        
        if pattern_idx == 0:
            candidates = list(range(mol.num_atoms))
        else:
            candidates_set: set[int] | None = None
            
            for nbr_idx, bond_idx in pattern_adj[pattern_idx]:
                if nbr_idx in mapping:
                    mol_nbr_idx = mapping[nbr_idx]
                    mol_atom = mol.atoms[mol_nbr_idx]
                    
                    nbr_candidates: set[int] = set()
                    for mol_bond_idx in mol_atom.bond_indices:
                        mol_bond = mol.bonds[mol_bond_idx]
                        mol_other = mol_bond.other_atom(mol_nbr_idx)
                        
                        pattern_bond = pattern.bonds[bond_idx]
                        if _bond_matches(mol_bond, pattern_bond):
                            nbr_candidates.add(mol_other)
                    
                    if candidates_set is None:
                        candidates_set = nbr_candidates
                    else:
                        candidates_set &= nbr_candidates
            
            if candidates_set is None:
                used = set(mapping.values())
                candidates = [i for i in range(mol.num_atoms) if i not in used]
            else:
                candidates = list(candidates_set)
        
        used = set(mapping.values())
        
        for mol_idx in candidates:
            if mol_idx in used:
                continue
            
            mol_atom = mol.atoms[mol_idx]
            
            if not _atom_matches(mol_atom, pattern_atom, mol, ring_count, ring_sizes):
                continue
            
            bonds_ok = True
            for nbr_idx, bond_idx in pattern_adj[pattern_idx]:
                if nbr_idx in mapping:
                    mol_nbr_idx = mapping[nbr_idx]
                    mol_bond = _get_bond_between(mol, mol_idx, mol_nbr_idx)
                    
                    if mol_bond is None:
                        bonds_ok = False
                        break
                    
                    pattern_bond = pattern.bonds[bond_idx]
                    if not _bond_matches(mol_bond, pattern_bond):
                        bonds_ok = False
                        break
            
            if not bonds_ok:
                continue
            
            mapping[pattern_idx] = mol_idx
            if backtrack(mapping, pattern_idx + 1):
                return True
            del mapping[pattern_idx]
        
        return False
    
    return backtrack({}, 0)


def count_matches(mol: "Molecule", pattern: "Molecule") -> int:
    """Count the number of SMARTS pattern matches in a molecule.
    
    Args:
        mol: The molecule to search in.
        pattern: The SMARTS pattern to search for.
    
    Returns:
        Number of unique matches found.
    """
    return len(substructure_search(mol, pattern))
