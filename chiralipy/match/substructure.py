"""
SMARTS substructure matching.

This module provides functionality for finding SMARTS pattern matches
within molecules using subgraph isomorphism.

The implementation uses a VF2-like backtracking algorithm with
SMARTS-specific atom and bond matching constraints.

Example:
    >>> from chiralipy import parse
    >>> from chiralipy.match import substructure_search, has_substructure
    >>> 
    >>> mol = parse("c1ccccc1CCO")
    >>> pattern = parse("[cR]")
    >>> matches = substructure_search(mol, pattern)
    >>> print(matches)  # [(0,), (1,), (2,), (3,), (4,), (5,)]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chiralipy.rings import get_ring_info, get_ring_bonds, _find_ring_atoms_and_bonds_fast

if TYPE_CHECKING:
    from chiralipy.types import Atom, Bond, Molecule


@dataclass(slots=True)
class RingInfo:
    """Precomputed ring information for a molecule.
    
    This allows efficient reuse of ring data across multiple 
    substructure searches on the same molecule.
    
    Attributes:
        ring_count: Dict mapping atom index to number of rings it's in.
        ring_sizes: Dict mapping atom index to set of ring sizes it's in.
        ring_bonds: Set of (min_idx, max_idx) tuples for bonds in rings.
    """
    ring_count: dict[int, int]
    ring_sizes: dict[int, set[int]]
    ring_bonds: set[tuple[int, int]]
    
    @classmethod
    def from_molecule(cls, mol: "Molecule", fast: bool = False) -> "RingInfo":
        """Compute ring info for a molecule.
        
        Args:
            mol: Molecule to analyze.
            fast: If True, use fast detection without accurate ring counts.
                Only suitable for queries that check ring membership (R/R0),
                not specific ring counts (R2) or sizes (r5).
        
        Optimized to call the fast ring detection algorithm only once.
        """
        # Compute ring atoms and bonds in one pass
        ring_atoms, ring_bonds = _find_ring_atoms_and_bonds_fast(mol)
        
        if fast:
            # Fast path: just ring membership, no counts/sizes
            ring_count = {i: (1 if i in ring_atoms else 0) for i in range(mol.num_atoms)}
            ring_sizes: dict[int, set[int]] = {i: set() for i in range(mol.num_atoms)}
        else:
            # Full SSSR-based computation
            ring_count, ring_sizes = get_ring_info(mol, _ring_atoms=ring_atoms)
        
        return cls(ring_count=ring_count, ring_sizes=ring_sizes, ring_bonds=ring_bonds)


# Cache for parsed recursive SMARTS patterns
_RECURSIVE_PATTERN_CACHE: dict[str, "Molecule"] = {}
_RECURSIVE_PATTERN_ADJ_CACHE: dict[str, dict[int, list[tuple[int, int]]]] = {}


def _build_pattern_adj(pattern: "Molecule") -> dict[int, list[tuple[int, int]]]:
    """Build pattern adjacency list."""
    adj: dict[int, list[tuple[int, int]]] = {
        i: [] for i in range(pattern.num_atoms)
    }
    for bond in pattern.bonds:
        adj[bond.atom1_idx].append((bond.atom2_idx, bond.idx))
        adj[bond.atom2_idx].append((bond.atom1_idx, bond.idx))
    return adj


def _matches_recursive_smarts(
    mol: "Molecule",
    atom_idx: int,
    recursive_smarts: str,
    ring_count: dict[int, int],
    ring_sizes: dict[int, set[int]],
    mol_ring_bonds: set[tuple[int, int]] | None = None,
) -> bool:
    """Check if an atom matches a recursive SMARTS pattern $(...).
    
    The recursive SMARTS describes the local environment around an atom.
    The atom must be the first atom in the recursive pattern.
    
    Args:
        mol: The molecule being searched.
        atom_idx: Index of the atom to check.
        recursive_smarts: The SMARTS pattern inside $(...).
        ring_count: Ring membership count for each atom.
        ring_sizes: Ring sizes for each atom.
        mol_ring_bonds: Set of ring bond tuples.
    
    Returns:
        True if the atom matches the recursive pattern.
    """
    from chiralipy.parser import parse
    
    # Get or parse the recursive pattern (cached)
    # Use perceive_aromaticity=False because SMARTS patterns use lowercase
    # letters to explicitly indicate aromaticity requirements
    if recursive_smarts not in _RECURSIVE_PATTERN_CACHE:
        try:
            pattern = parse(f"[{recursive_smarts}]", perceive_aromaticity=False)
            # If pattern has just one atom, we might need to re-parse as regular SMARTS
            if pattern.num_atoms == 1:
                pattern = parse(recursive_smarts, perceive_aromaticity=False)
            _RECURSIVE_PATTERN_CACHE[recursive_smarts] = pattern
        except Exception:
            # If parsing fails, try wrapping differently
            try:
                pattern = parse(recursive_smarts, perceive_aromaticity=False)
                _RECURSIVE_PATTERN_CACHE[recursive_smarts] = pattern
            except Exception:
                return False
    
    pattern = _RECURSIVE_PATTERN_CACHE[recursive_smarts]
    
    if pattern.num_atoms == 0:
        return True
    
    # The atom at atom_idx must match the first atom of the recursive pattern
    pattern_atom = pattern.atoms[0]
    mol_atom = mol.atoms[atom_idx]
    
    # First check basic atom properties (fast rejection)
    if pattern_atom.symbol != '*':
        if mol_atom.symbol.upper() != pattern_atom.symbol.upper():
            return False
    
    # Check aromaticity
    if pattern_atom.is_aromatic and not mol_atom.is_aromatic:
        return False
    if not pattern_atom.is_aromatic and not pattern_atom.is_wildcard:
        if pattern_atom.symbol and pattern_atom.symbol[0].isupper():
            if mol_atom.is_aromatic:
                return False
    
    # Check degree if specified
    if pattern_atom.degree_query is not None:
        degree = len(mol_atom.bond_indices)
        if pattern_atom.degree_query != -1 and degree != pattern_atom.degree_query:
            return False
    
    # Check negated degree [!D1]
    if pattern_atom.negated_degree_query is not None:
        degree = len(mol_atom.bond_indices)
        if pattern_atom.negated_degree_query == -1:
            return False  # !D means "not any degree" - always fails
        elif degree == pattern_atom.negated_degree_query:
            return False
    
    # Check ring membership
    if pattern_atom.ring_count is not None:
        atom_ring_count = ring_count.get(mol_atom.idx, 0)
        if pattern_atom.ring_count == -1:
            if atom_ring_count == 0:
                return False
        elif pattern_atom.ring_count == 0:
            if atom_ring_count > 0:
                return False
        elif atom_ring_count != pattern_atom.ring_count:
            return False
    
    # If pattern has only one atom, we're done
    if pattern.num_atoms == 1:
        return True
    
    # Build pattern adjacency (cached)
    if recursive_smarts not in _RECURSIVE_PATTERN_ADJ_CACHE:
        _RECURSIVE_PATTERN_ADJ_CACHE[recursive_smarts] = _build_pattern_adj(pattern)
    pattern_adj = _RECURSIVE_PATTERN_ADJ_CACHE[recursive_smarts]
    
    # Initialize ring bonds if not provided
    if mol_ring_bonds is None:
        mol_ring_bonds = get_ring_bonds(mol)
    
    # Cache pattern size and molecule references
    pattern_n = pattern.num_atoms
    mol_atoms = mol.atoms
    mol_bonds = mol.bonds
    pattern_atoms = pattern.atoms
    pattern_bonds = pattern.bonds
    mol_num_atoms = mol.num_atoms
    
    # Use list-based mapping for faster access
    mapping: list[int] = [-1] * pattern_n
    mapping[0] = atom_idx
    used: list[bool] = [False] * mol_num_atoms
    used[atom_idx] = True
    
    def try_match(pattern_idx: int) -> bool:
        """Try to extend the mapping from pattern atoms to molecule atoms."""
        if pattern_idx == pattern_n:
            return True
        
        p_atom = pattern_atoms[pattern_idx]
        
        # Find candidates from mapped neighbors
        candidates: set[int] = set()
        for nbr_idx, bond_idx in pattern_adj[pattern_idx]:
            mol_nbr_idx = mapping[nbr_idx]
            if mol_nbr_idx >= 0:  # Neighbor is mapped
                mol_nbr = mol_atoms[mol_nbr_idx]
                
                for mol_bond_idx in mol_nbr.bond_indices:
                    mol_bond = mol_bonds[mol_bond_idx]
                    mol_other = mol_bond.other_atom(mol_nbr_idx)
                    
                    # Check bond matches
                    pattern_bond = pattern_bonds[bond_idx]
                    if _bond_matches(mol_bond, pattern_bond, mol_ring_bonds):
                        if not used[mol_other]:
                            candidates.add(mol_other)
        
        if not candidates:
            return False
        
        for mol_idx in candidates:
            m_atom = mol_atoms[mol_idx]
            
            # Check atom matches
            if not _atom_matches_basic(m_atom, p_atom, mol, ring_count, ring_sizes):
                continue
            
            # Check all bonds to mapped neighbors
            ok = True
            for nbr_idx, bond_idx in pattern_adj[pattern_idx]:
                mol_nbr_idx = mapping[nbr_idx]
                if mol_nbr_idx >= 0:
                    mol_bond = _get_bond_between(mol, mol_idx, mol_nbr_idx)
                    if mol_bond is None:
                        ok = False
                        break
                    pattern_bond = pattern_bonds[bond_idx]
                    if not _bond_matches(mol_bond, pattern_bond, mol_ring_bonds):
                        ok = False
                        break
            
            if not ok:
                continue
            
            mapping[pattern_idx] = mol_idx
            used[mol_idx] = True
            if try_match(pattern_idx + 1):
                return True
            mapping[pattern_idx] = -1
            used[mol_idx] = False
        
        return False
    
    return try_match(1)  # Start from 1 since 0 is already mapped


def _atom_matches_basic(
    mol_atom: "Atom",
    pattern_atom: "Atom",
    _mol: "Molecule",
    ring_count: dict[int, int],
    _ring_sizes: dict[int, set[int]],
) -> bool:
    """Basic atom matching without recursive SMARTS handling.
    
    Optimized for the common case of simple SMARTS patterns.
    """
    # Fast path: check symbol first (most common rejection)
    mol_symbol_upper = mol_atom.symbol.upper()
    
    # Negated atomic number list [!#6;!#16;!#0;!#1] - atom must NOT be any of these
    if pattern_atom.negated_atomic_number_list:
        if mol_atom.atomic_number in pattern_atom.negated_atomic_number_list:
            return False
    
    # Atomic number list [#6] or [#0,#6,#7] - matches by atomic number, ignores aromaticity
    if pattern_atom.atomic_number_list:
        if mol_atom.atomic_number not in pattern_atom.atomic_number_list:
            return False
        # When matching by atomic number, skip symbol and aromaticity checks
    elif pattern_atom.negated_atomic_number_list:
        # When matching by negated atomic numbers only, skip symbol check
        pass
    elif pattern_atom.atom_list:
        # Check atom list
        pattern_symbols = [s.upper() for s in pattern_atom.atom_list]
        if pattern_atom.atom_list_negated:
            if mol_symbol_upper in pattern_symbols:
                return False
        else:
            if mol_symbol_upper not in pattern_symbols:
                return False
    elif pattern_atom.atom_list_negated:
        if mol_symbol_upper == pattern_atom.symbol.upper():
            return False
    elif not pattern_atom.is_wildcard and pattern_atom.symbol != '*':
        if mol_symbol_upper != pattern_atom.symbol.upper():
            return False
    
    # Aromaticity - skip if matching by atomic number list or negated atomic number list
    if not pattern_atom.atomic_number_list and not pattern_atom.negated_atomic_number_list:
        if pattern_atom.is_aromatic and not mol_atom.is_aromatic:
            return False
        if not pattern_atom.is_aromatic and not pattern_atom.is_wildcard:
            if pattern_atom.symbol and pattern_atom.symbol[0].isupper():
                if mol_atom.is_aromatic:
                    return False
    
    # Ring membership - use direct dict access
    pattern_ring_count = pattern_atom.ring_count
    if pattern_ring_count is not None:
        atom_ring_count = ring_count.get(mol_atom.idx, 0)
        if pattern_ring_count == -1:
            if atom_ring_count == 0:
                return False
        elif pattern_ring_count == 0:
            if atom_ring_count > 0:
                return False
        elif atom_ring_count != pattern_ring_count:
            return False
    
    # Degree - if degree_query_list is present, use it (OR'd degrees like [D2,D3])
    # Otherwise check degree_query
    degree = len(mol_atom.bond_indices)
    if pattern_atom.degree_query_list:
        if degree not in pattern_atom.degree_query_list:
            return False
    elif pattern_atom.degree_query is not None:
        if pattern_atom.degree_query == -1:
            if degree == 0:
                return False
        elif degree != pattern_atom.degree_query:
            return False
    
    # Negated degree query [!D1] - atom must NOT have this degree
    if pattern_atom.negated_degree_query is not None:
        if pattern_atom.negated_degree_query == -1:
            # !D means "not any degree" - this would always fail
            return False
        elif degree == pattern_atom.negated_degree_query:
            return False
    
    return True


def _get_bond_between(mol: "Molecule", atom1_idx: int, atom2_idx: int) -> "Bond | None":
    """Get the bond between two atoms, or None if not bonded."""
    for bond_idx in mol.atoms[atom1_idx].bond_indices:
        bond = mol.bonds[bond_idx]
        if bond.other_atom(atom1_idx) == atom2_idx:
            return bond
    return None


def _atom_matches(
    mol_atom: "Atom",
    pattern_atom: "Atom",
    mol: "Molecule",
    ring_count: dict[int, int],
    ring_sizes: dict[int, set[int]],
    mol_ring_bonds: set[tuple[int, int]] | None = None,
) -> bool:
    """Check if a molecule atom matches a SMARTS pattern atom.
    
    This is the full atom matching function that handles recursive SMARTS
    and all advanced SMARTS features. For simple matching without recursive
    SMARTS, use _atom_matches_basic.
    
    Args:
        mol_atom: Atom from the molecule being searched.
        pattern_atom: Atom from the SMARTS pattern.
        mol: The molecule (for context).
        ring_count: Ring membership count for each atom.
        ring_sizes: Ring sizes for each atom.
        mol_ring_bonds: Set of ring bond tuples for recursive SMARTS.
    
    Returns:
        True if the molecule atom satisfies all pattern constraints.
    """
    # Use basic matching for common atom properties
    if not _atom_matches_basic(mol_atom, pattern_atom, mol, ring_count, ring_sizes):
        return False

    # Handle recursive SMARTS $(...)
    if pattern_atom.is_recursive and pattern_atom.recursive_smarts:
        if not _matches_recursive_smarts(mol, mol_atom.idx, pattern_atom.recursive_smarts,
                                         ring_count, ring_sizes, mol_ring_bonds):
            return False
        # If it's just a recursive SMARTS with no other constraints, we're done
        if pattern_atom.symbol == '*' and not pattern_atom.atom_list:
            return True
    
    # Handle negated recursive SMARTS !$(...)
    if pattern_atom.negated_recursive_smarts:
        for neg_smarts in pattern_atom.negated_recursive_smarts:
            if _matches_recursive_smarts(mol, mol_atom.idx, neg_smarts,
                                         ring_count, ring_sizes, mol_ring_bonds):
                return False
    
    # Additional checks not in _atom_matches_basic:
    
    # Charge
    if pattern_atom.charge != 0:
        if mol_atom.charge != pattern_atom.charge:
            return False
    
    # Explicit charge query (e.g., [+0] means charge must be exactly 0)
    if pattern_atom.charge_query is not None:
        if mol_atom.charge != pattern_atom.charge_query:
            return False
    
    # Ring size (r5, r6, ...)
    if pattern_atom.ring_size is not None:
        atom_ring_sizes = ring_sizes.get(mol_atom.idx, set())
        if pattern_atom.ring_size == -1:
            if not atom_ring_sizes:
                return False
        else:
            if pattern_atom.ring_size not in atom_ring_sizes:
                return False
    
    # Lazy calculation of total hydrogens (only when needed)
    _mol_total_h: int | None = None
    def get_mol_total_h() -> int:
        nonlocal _mol_total_h
        if _mol_total_h is None:
            _mol_total_h = mol_atom.total_hydrogens(mol)
        return _mol_total_h
    
    # Connectivity (X, X2, ...) - total connections including implicit H
    if pattern_atom.connectivity_query is not None:
        explicit_bonds = len(mol_atom.bond_indices)
        connectivity = explicit_bonds + get_mol_total_h()
        
        if pattern_atom.connectivity_query == -1:
            if connectivity == 0:
                return False
        else:
            if connectivity != pattern_atom.connectivity_query:
                return False
    
    # Valence (v, v4, ...)
    if pattern_atom.valence_query is not None:
        valence = 0
        for bond_idx in mol_atom.bond_indices:
            bond = mol.bonds[bond_idx]
            if bond.is_aromatic:
                valence += 1.5
            else:
                valence += bond.order
        valence = int(valence + 0.5)
        valence += get_mol_total_h()
        
        if pattern_atom.valence_query == -1:
            if valence == 0:
                return False
        else:
            if valence != pattern_atom.valence_query:
                return False
    
    # Hydrogen count (H, H2, ...) - matches total hydrogens
    if pattern_atom.explicit_hydrogens > 0:
        if get_mol_total_h() != pattern_atom.explicit_hydrogens:
            return False
    
    # Isotope
    if pattern_atom.isotope is not None:
        if mol_atom.isotope != pattern_atom.isotope:
            return False
    
    return True


def _bond_matches(
    mol_bond: "Bond",
    pattern_bond: "Bond",
    mol_ring_bonds: set[tuple[int, int]] | None = None,
) -> bool:
    """Check if a molecule bond matches a SMARTS pattern bond.
    
    Args:
        mol_bond: Bond from the molecule being searched.
        pattern_bond: Bond from the SMARTS pattern.
        mol_ring_bonds: Set of ring bond tuples for ring bond checking.
    
    Returns:
        True if the molecule bond satisfies the pattern constraints.
    
    Optimized for common cases (single/aromatic bonds with ring constraints).
    """
    # Fast path: Any bond (~) matches everything
    if pattern_bond.is_any:
        return True
    
    # Cache commonly accessed attributes
    mol_order = mol_bond.order
    mol_aromatic = mol_bond.is_aromatic
    pattern_order = pattern_bond.order
    pattern_aromatic = pattern_bond.is_aromatic
    
    # Check ring bond constraint (e.g., -;@ or -;!@) - common in BRICS
    if pattern_bond.is_not_ring_bond or pattern_bond.is_ring_bond:
        if mol_ring_bonds is not None:
            a1, a2 = mol_bond.atom1_idx, mol_bond.atom2_idx
            bond_key = (a1, a2) if a1 < a2 else (a2, a1)
            is_ring = bond_key in mol_ring_bonds
            
            # Check !@ constraint (must NOT be ring bond)
            if pattern_bond.is_not_ring_bond and is_ring:
                return False
            
            # Check @ constraint (must be ring bond)
            if pattern_bond.is_ring_bond and not is_ring:
                return False
    
    # Handle negated bonds (!-, !=, !#)
    if pattern_bond.is_negated:
        # Negated bond: must NOT match the specified order
        if pattern_aromatic:
            return not mol_aromatic
        # For negated single bond !-, any non-single bond matches
        # (including double, triple, aromatic)
        if mol_order == pattern_order:
            return False
        if pattern_order == 1 and mol_aromatic:
            return False  # Aromatic bonds are considered single-ish for matching
        return True
    
    # Aromatic bond pattern
    if pattern_aromatic:
        return mol_aromatic
    
    # Specific bond order match
    if pattern_order == mol_order:
        return True
    
    # Single bond pattern can match aromatic (implicit in aromatic systems)
    if pattern_order == 1 and mol_aromatic:
        return True
    
    return False


def substructure_search(
    mol: "Molecule",
    pattern: "Molecule",
    uniquify: bool = True,
    ring_info: RingInfo | None = None,
) -> list[tuple[int, ...]]:
    """Find all matches of a SMARTS pattern in a molecule.
    
    Uses a VF2-like backtracking algorithm for subgraph isomorphism.
    
    Args:
        mol: The molecule to search in.
        pattern: The SMARTS pattern to search for.
        uniquify: If True (default), return only one match per unique set
            of molecule atoms. If False, return all permutations of matches.
        ring_info: Optional precomputed ring info. If None, will be computed.
            Pass this when doing multiple searches on the same molecule
            for better performance.
    
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
    
    # Use precomputed ring info or compute it
    if ring_info is None:
        ring_info = RingInfo.from_molecule(mol)
    
    ring_count = ring_info.ring_count
    ring_sizes = ring_info.ring_sizes
    mol_ring_bonds = ring_info.ring_bonds
    
    # Build pattern adjacency
    pattern_adj = _build_pattern_adj(pattern)
    
    # Cache sizes to avoid repeated property access
    pattern_n = pattern.num_atoms
    mol_n = mol.num_atoms
    
    matches: list[tuple[int, ...]] = []
    seen_atom_sets: set[frozenset[int]] = set()  # For uniquify
    
    def backtrack(
        mapping: dict[int, int],  # pattern_idx -> mol_idx
        pattern_idx: int,
    ) -> None:
        """Recursively try to extend the mapping."""
        if pattern_idx == pattern_n:
            # Complete match found
            match = tuple(mapping[i] for i in range(pattern_n))
            
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
            candidates = list(range(mol_n))
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
                        if _bond_matches(mol_bond, pattern_bond, mol_ring_bonds):
                            nbr_candidates.add(mol_other)
                    
                    if candidates_set is None:
                        candidates_set = nbr_candidates
                    else:
                        candidates_set &= nbr_candidates
            
            if candidates_set is None:
                # No mapped neighbors - disconnected pattern component
                # Try all unmatched atoms
                used = set(mapping.values())
                candidates = [i for i in range(mol_n) if i not in used]
            else:
                candidates = list(candidates_set)
        
        # Filter out already-used atoms
        used = set(mapping.values())
        
        for mol_idx in candidates:
            if mol_idx in used:
                continue
            
            mol_atom = mol.atoms[mol_idx]
            
            # Check atom compatibility
            if not _atom_matches(mol_atom, pattern_atom, mol, ring_count, ring_sizes, mol_ring_bonds):
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
                    if not _bond_matches(mol_bond, pattern_bond, mol_ring_bonds):
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


def has_substructure(
    mol: "Molecule",
    pattern: "Molecule",
    ring_info: RingInfo | None = None,
) -> bool:
    """Check if a molecule contains a SMARTS pattern.
    
    This is more efficient than substructure_search when you only
    need to know if a match exists.
    
    Args:
        mol: The molecule to search in.
        pattern: The SMARTS pattern to search for.
        ring_info: Optional precomputed ring info for performance.
    
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
    
    # Use precomputed ring info or compute it
    if ring_info is None:
        ring_info = RingInfo.from_molecule(mol)
    
    ring_count = ring_info.ring_count
    ring_sizes = ring_info.ring_sizes
    mol_ring_bonds = ring_info.ring_bonds
    
    # Build pattern adjacency
    pattern_adj = _build_pattern_adj(pattern)
    
    # Cache sizes to avoid repeated property access
    pattern_n = pattern.num_atoms
    mol_n = mol.num_atoms
    
    def backtrack(mapping: dict[int, int], pattern_idx: int) -> bool:
        if pattern_idx == pattern_n:
            return True
        
        pattern_atom = pattern.atoms[pattern_idx]
        
        if pattern_idx == 0:
            candidates = list(range(mol_n))
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
                        if _bond_matches(mol_bond, pattern_bond, mol_ring_bonds):
                            nbr_candidates.add(mol_other)
                    
                    if candidates_set is None:
                        candidates_set = nbr_candidates
                    else:
                        candidates_set &= nbr_candidates
            
            if candidates_set is None:
                used = set(mapping.values())
                candidates = [i for i in range(mol_n) if i not in used]
            else:
                candidates = list(candidates_set)
        
        used = set(mapping.values())
        
        for mol_idx in candidates:
            if mol_idx in used:
                continue
            
            mol_atom = mol.atoms[mol_idx]
            
            if not _atom_matches(mol_atom, pattern_atom, mol, ring_count, ring_sizes, mol_ring_bonds):
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
                    if not _bond_matches(mol_bond, pattern_bond, mol_ring_bonds):
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


def match_at_root(
    mol: "Molecule",
    atom_idx: int,
    pattern: "Molecule",
    ring_info: RingInfo | None = None,
    pattern_adj: dict[int, list[tuple[int, int]]] | None = None,
) -> bool:
    """Check if a pattern matches starting at a specific molecule atom.
    
    The first atom of the pattern (index 0) is anchored to the specified
    molecule atom.
    
    Args:
        mol: The molecule to search in.
        atom_idx: The index of the molecule atom to anchor to.
        pattern: The SMARTS pattern to search for.
        ring_info: Optional precomputed ring info.
        pattern_adj: Optional precomputed pattern adjacency list.
    
    Returns:
        True if the pattern matches anchored at atom_idx.
    """
    if pattern.num_atoms == 0:
        return True
    
    if atom_idx >= mol.num_atoms:
        return False
        
    # Use precomputed ring info or compute it
    if ring_info is None:
        ring_info = RingInfo.from_molecule(mol)
    
    ring_count = ring_info.ring_count
    ring_sizes = ring_info.ring_sizes
    mol_ring_bonds = ring_info.ring_bonds
    
    # Check root atom first (fast fail)
    if not _atom_matches(mol.atoms[atom_idx], pattern.atoms[0], mol, ring_count, ring_sizes, mol_ring_bonds):
        return False
        
    if pattern.num_atoms == 1:
        return True
    
    # Build pattern adjacency if not provided
    if pattern_adj is None:
        pattern_adj = _build_pattern_adj(pattern)
    
    pattern_n = pattern.num_atoms
    mol_atoms = mol.atoms  # Local reference for faster access
    mol_bonds = mol.bonds
    pattern_atoms = pattern.atoms
    pattern_bonds = pattern.bonds
    
    # Pre-allocate mapping array (faster than dict for small patterns)
    # Use -1 to indicate unmapped
    mapping: list[int] = [-1] * pattern_n
    mapping[0] = atom_idx
    used: list[bool] = [False] * mol.num_atoms
    used[atom_idx] = True
    
    def backtrack(pattern_idx: int) -> bool:
        if pattern_idx == pattern_n:
            return True
        
        pattern_atom = pattern_atoms[pattern_idx]
        
        # Find candidates from mapped neighbors
        candidates_set: set[int] | None = None
        
        for nbr_idx, bond_idx in pattern_adj[pattern_idx]:
            mol_nbr_idx = mapping[nbr_idx]
            if mol_nbr_idx >= 0:  # Neighbor is mapped
                mol_atom = mol_atoms[mol_nbr_idx]
                
                nbr_candidates: set[int] = set()
                for mol_bond_idx in mol_atom.bond_indices:
                    mol_bond = mol_bonds[mol_bond_idx]
                    mol_other = mol_bond.other_atom(mol_nbr_idx)
                    
                    pattern_bond = pattern_bonds[bond_idx]
                    if _bond_matches(mol_bond, pattern_bond, mol_ring_bonds):
                        nbr_candidates.add(mol_other)
                
                if candidates_set is None:
                    candidates_set = nbr_candidates
                else:
                    candidates_set &= nbr_candidates
        
        if candidates_set is None:
            # Disconnected component in pattern (unlikely for BRICS but possible)
            candidates_set = {i for i in range(mol.num_atoms) if not used[i]}
        
        for mol_idx in candidates_set:
            if used[mol_idx]:
                continue
            
            mol_atom = mol_atoms[mol_idx]
            
            if not _atom_matches(mol_atom, pattern_atom, mol, ring_count, ring_sizes, mol_ring_bonds):
                continue
            
            bonds_ok = True
            for nbr_idx, bond_idx in pattern_adj[pattern_idx]:
                mol_nbr_idx = mapping[nbr_idx]
                if mol_nbr_idx >= 0:
                    mol_bond = _get_bond_between(mol, mol_idx, mol_nbr_idx)
                    
                    if mol_bond is None:
                        bonds_ok = False
                        break
                    
                    pattern_bond = pattern_bonds[bond_idx]
                    if not _bond_matches(mol_bond, pattern_bond, mol_ring_bonds):
                        bonds_ok = False
                        break
            
            if not bonds_ok:
                continue
            
            mapping[pattern_idx] = mol_idx
            used[mol_idx] = True
            if backtrack(pattern_idx + 1):
                return True
            mapping[pattern_idx] = -1
            used[mol_idx] = False
        
        return False
    
    return backtrack(1)  # Start from 1 since 0 is already mapped


def count_matches(mol: "Molecule", pattern: "Molecule") -> int:
    """Count the number of SMARTS pattern matches in a molecule.
    
    Args:
        mol: The molecule to search in.
        pattern: The SMARTS pattern to search for.
    
    Returns:
        Number of unique matches found.
    """
    return len(substructure_search(mol, pattern))
