"""
SMILES string writer.

This module provides functionality for converting Molecule objects back
to SMILES strings, with support for canonical ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from chirpy.transform.aromaticity import perceive_aromaticity
from chirpy.canon import canonical_ranks
from chirpy.elements import AROMATIC_SUBSET, ORGANIC_SUBSET
from chirpy.rings import _find_ring_atoms_and_bonds_fast

if TYPE_CHECKING:
    from chirpy.types import Atom, Bond, Molecule


# Constants for canonical traversal
_MAX_NATOMS: Final[int] = 1_000_000
_MAX_BONDTYPE: Final[int] = 4


def count_swaps_to_interconvert(ref: list[int], probe: list[int]) -> int:
    """Count swaps needed to convert probe to match ref.
    
    Uses bubble-sort swap counting algorithm.
    
    Args:
        ref: Reference ordering.
        probe: Ordering to transform.
    
    Returns:
        Number of swaps needed.
    
    Raises:
        ValueError: If lists have different elements.
    """
    if len(ref) != len(probe):
        raise ValueError("Size mismatch")
    
    probe = list(probe)
    n_swaps = 0
    
    for i, ref_val in enumerate(ref):
        if probe[i] != ref_val:
            j = i + 1
            while j < len(probe) and probe[j] != ref_val:
                j += 1
            
            if j >= len(probe):
                raise ValueError(f"Element {ref_val} not found in probe")
            
            probe[i], probe[j] = probe[j], probe[i]
            n_swaps += 1
    
    return n_swaps


class SmilesWriter:
    """SMILES string writer with canonical traversal.
    
    Converts a Molecule object to a SMILES string, optionally using
    canonical atom ordering for reproducible output.
    
    The traversal algorithm:
    1. Start from lowest-ranked atom in each component
    2. DFS with ring closures processed first
    3. Ring digits assigned in order of first encounter
    
    Example:
        >>> from chirpy import parse
        >>> mol = parse("C(C)CC")
        >>> writer = SmilesWriter(mol)
        >>> writer.to_smiles()
        'CCCC'
    """
    
    def __init__(
        self,
        mol: Molecule,
        ranks: list[int] | None = None,
    ) -> None:
        """Initialize writer.
        
        Args:
            mol: Molecule to write.
            ranks: Pre-computed canonical ranks (computed if None).
        """
        self._mol = mol
        self._ranks = ranks if ranks is not None else canonical_ranks(mol)
        self._ring_bonds: set[int] | None = None
    
    def to_smiles(self) -> str:
        """Generate canonical SMILES string.
        
        Returns:
            Canonical SMILES string.
        """
        components = self._mol.connected_components()
        parts: list[str] = []
        
        for comp in components:
            comp_set = set(comp)
            smiles = self._write_component(comp_set)
            parts.append(smiles)
        
        # Sort components lexicographically
        parts.sort()
        return ".".join(parts)
    
    def _write_component(self, comp_atoms: set[int]) -> str:
        """Write SMILES for a single connected component."""
        if not comp_atoms:
            return ""
        
        # Find starting atom (lowest rank)
        start = min(comp_atoms, key=lambda a: (self._ranks[a], a))
        
        # Find ring bonds
        ring_bonds = self._find_ring_bonds(comp_atoms)
        
        # Phase 1: Find ring closures
        WHITE, GREY, BLACK = 0, 1, 2
        colors: dict[int, int] = {a: WHITE for a in comp_atoms}
        atom_ring_closures: dict[int, list[int]] = {a: [] for a in comp_atoms}
        atom_traversal_order: dict[int, list[int]] = {a: [] for a in comp_atoms}
        
        def dfs_find_cycles(atom_idx: int, in_bond_idx: int | None) -> None:
            colors[atom_idx] = GREY
            atom = self._mol.atoms[atom_idx]
            
            possibles: list[tuple[int, int, int]] = []
            for bond_idx in atom.bond_indices:
                if bond_idx == in_bond_idx:
                    continue
                bond = self._mol.bonds[bond_idx]
                nbr = bond.other_atom(atom_idx)
                if nbr not in comp_atoms:
                    continue
                
                rank = self._ranks[nbr]
                
                if colors[nbr] == GREY:
                    # Ring closure
                    rank -= (_MAX_BONDTYPE + 1) * _MAX_NATOMS * _MAX_NATOMS
                    rank += (_MAX_BONDTYPE - bond.order) * _MAX_NATOMS
                elif bond_idx in ring_bonds:
                    rank += (_MAX_BONDTYPE - bond.order) * _MAX_NATOMS * _MAX_NATOMS
                
                possibles.append((rank, nbr, bond_idx))
            
            possibles.sort(key=lambda x: (x[0], x[1]))
            
            for _, nbr, bond_idx in possibles:
                if colors[nbr] == WHITE:
                    dfs_find_cycles(nbr, bond_idx)
                elif colors[nbr] == GREY:
                    atom_ring_closures[nbr].append(bond_idx)
                    atom_ring_closures[atom_idx].append(bond_idx)
            
            colors[atom_idx] = BLACK
        
        dfs_find_cycles(start, None)
        
        # Phase 2: Build SMILES
        colors = {a: WHITE for a in comp_atoms}
        ring_digit_map: dict[int, int] = {}
        available_digits: list[int] = list(range(1, 100))
        out: list[str] = []
        chirality_inversion: dict[int, bool] = {}
        
        def dfs_build(atom_idx: int, in_bond_idx: int | None) -> None:
            nonlocal available_digits
            colors[atom_idx] = GREY
            atom = self._mol.atoms[atom_idx]
            seen_from_here: set[int] = {atom_idx}
            
            traversal_bonds: list[int] = []
            if in_bond_idx is not None:
                traversal_bonds.append(in_bond_idx)
            
            ring_closure_bonds: list[int] = []
            other_bonds: list[tuple[int, int, int]] = []
            
            for bond_idx in atom_ring_closures[atom_idx]:
                bond = self._mol.bonds[bond_idx]
                nbr = bond.other_atom(atom_idx)
                ring_closure_bonds.append(bond_idx)
                seen_from_here.add(nbr)
            
            for bond_idx in atom.bond_indices:
                if bond_idx == in_bond_idx:
                    continue
                bond = self._mol.bonds[bond_idx]
                nbr = bond.other_atom(atom_idx)
                if nbr not in comp_atoms:
                    continue
                if colors[nbr] != WHITE or nbr in seen_from_here:
                    continue
                
                rank = self._ranks[nbr]
                if bond_idx in ring_bonds:
                    rank += (_MAX_BONDTYPE - bond.order) * _MAX_NATOMS * _MAX_NATOMS
                other_bonds.append((rank, nbr, bond_idx))
            
            other_bonds.sort(key=lambda x: (x[0], x[1]))
            
            for bond_idx in ring_closure_bonds:
                traversal_bonds.append(bond_idx)
            for _, _, bond_idx in other_bonds:
                traversal_bonds.append(bond_idx)
            
            atom_traversal_order[atom_idx] = traversal_bonds
            
            # Calculate chirality inversion
            if atom.chirality:
                original_bonds = list(atom.bond_indices)
                is_first_in_output = (in_bond_idx is None)
                was_first_in_input = atom._was_first_in_component
                
                h_position_changed = (
                    atom.explicit_hydrogens >= 1 and
                    len(atom.bond_indices) == 3 and
                    was_first_in_input and
                    not is_first_in_output
                )
                
                n_swaps = 0
                
                if h_position_changed:
                    original_with_h = [-1] + original_bonds
                    traversal_with_h = [traversal_bonds[0], -1] + traversal_bonds[1:]
                    try:
                        n_swaps = count_swaps_to_interconvert(traversal_with_h, original_with_h)
                    except ValueError:
                        n_swaps = 0
                else:
                    if len(traversal_bonds) >= 3 and len(original_bonds) >= 3:
                        try:
                            n_swaps = count_swaps_to_interconvert(traversal_bonds, original_bonds)
                        except ValueError:
                            n_swaps = 0
                
                chirality_inversion[atom_idx] = (n_swaps % 2 == 1)
            
            # Write atom
            invert = chirality_inversion.get(atom_idx, False)
            out.append(self._atom_to_smiles(atom, invert_chirality=invert))
            
            # Process ring closures
            digits_to_release: list[int] = []
            for bond_idx in ring_closure_bonds:
                bond = self._mol.bonds[bond_idx]
                
                if bond_idx in ring_digit_map:
                    digit = ring_digit_map[bond_idx]
                    out.append(self._bond_to_smiles(bond, is_ring_closure=True))
                    out.append(self._ring_number_to_smiles(digit))
                    digits_to_release.append(digit)
                    del ring_digit_map[bond_idx]
                else:
                    digit = available_digits.pop(0)
                    ring_digit_map[bond_idx] = digit
                    out.append(self._ring_number_to_smiles(digit))
            
            for digit in digits_to_release:
                available_digits.append(digit)
                available_digits.sort()
            
            # Process children
            for i, (_, nbr, bond_idx) in enumerate(other_bonds):
                if colors[nbr] != WHITE:
                    continue
                
                bond = self._mol.bonds[bond_idx]
                
                if i + 1 < len(other_bonds):
                    out.append("(")
                
                out.append(self._bond_to_smiles(bond))
                dfs_build(nbr, bond_idx)
                
                if i + 1 < len(other_bonds):
                    out.append(")")
            
            colors[atom_idx] = BLACK
        
        dfs_build(start, None)
        return "".join(out)
    
    def _find_ring_bonds(self, comp_atoms: set[int]) -> set[int]:
        """Find bonds that are part of a ring using Tarjan's bridge algorithm.
        
        Uses O(V+E) algorithm instead of O(E^2) per-bond BFS.
        """
        mol = self._mol
        
        # Use the fast algorithm - get ring bond tuples
        _, ring_bond_tuples = _find_ring_atoms_and_bonds_fast(mol)
        
        # Convert to bond indices, filtering to this component
        ring_bonds: set[int] = set()
        for bond in mol.bonds:
            if bond.atom1_idx not in comp_atoms or bond.atom2_idx not in comp_atoms:
                continue
            key = (min(bond.atom1_idx, bond.atom2_idx), max(bond.atom1_idx, bond.atom2_idx))
            if key in ring_bond_tuples:
                ring_bonds.add(bond.idx)
        
        return ring_bonds
    
    def _atom_to_smiles(self, atom: Atom, invert_chirality: bool = False) -> str:
        """Convert atom to SMILES string.
        
        Supports SMARTS features: wildcard (*), atom lists, negation,
        ring queries, degree, valence, connectivity, recursive SMARTS.
        """
        chirality = atom.chirality
        if invert_chirality and chirality:
            chirality = "@@" if chirality == "@" else "@"
        
        # Handle recursive SMARTS
        if atom.is_recursive and atom.recursive_smarts:
            return f"[$({atom.recursive_smarts})]"
        
        # Check if atom has SMARTS query attributes
        has_smarts_queries = (
            atom.ring_count is not None or
            atom.ring_size is not None or
            atom.degree_query is not None or
            atom.valence_query is not None or
            atom.connectivity_query is not None
        )
        
        # Handle simple wildcard atoms (no queries)
        if atom.is_wildcard and not atom.is_recursive and not atom.atom_list and not has_smarts_queries:
            return "*"
        
        if self._needs_brackets(atom) or has_smarts_queries:
            parts = ["["]
            
            # Negation
            if atom.atom_list_negated:
                parts.append("!")
            
            if atom.isotope is not None:
                parts.append(str(atom.isotope))
            
            # Atom list or single symbol
            if atom.atom_list:
                parts.append(",".join(atom.atom_list))
            elif atom.is_wildcard:
                parts.append("*")
            elif atom.is_aromatic and atom.symbol[0].isupper():
                parts.append(atom.symbol.lower())
            else:
                parts.append(atom.symbol)
            
            if chirality:
                parts.append(chirality)
            
            # SMARTS queries
            if atom.ring_count is not None:
                if atom.ring_count == -1:
                    parts.append("R")
                else:
                    parts.append(f"R{atom.ring_count}")
            
            if atom.ring_size is not None:
                if atom.ring_size == -1:
                    parts.append("r")
                else:
                    parts.append(f"r{atom.ring_size}")
            
            if atom.degree_query is not None:
                if atom.degree_query == -1:
                    parts.append("D")
                else:
                    parts.append(f"D{atom.degree_query}")
            
            if atom.valence_query is not None:
                if atom.valence_query == -1:
                    parts.append("v")
                else:
                    parts.append(f"v{atom.valence_query}")
            
            if atom.connectivity_query is not None:
                if atom.connectivity_query == -1:
                    parts.append("X")
                else:
                    parts.append(f"X{atom.connectivity_query}")
            
            if atom.explicit_hydrogens > 0:
                parts.append("H")
                if atom.explicit_hydrogens > 1:
                    parts.append(str(atom.explicit_hydrogens))
            
            if atom.charge > 0:
                parts.append("+")
                if atom.charge > 1:
                    parts.append(str(atom.charge))
            elif atom.charge < 0:
                parts.append("-")
                if atom.charge < -1:
                    parts.append(str(-atom.charge))
            
            if atom.atom_class is not None:
                parts.append(":")
                parts.append(str(atom.atom_class))
            
            parts.append("]")
            return "".join(parts)
        else:
            return atom.symbol.lower() if atom.is_aromatic else atom.symbol
    
    def _needs_brackets(self, atom: Atom) -> bool:
        """Check if atom needs bracket notation.
        
        Atoms are simplified to non-bracketed form when possible.
        An atom needs brackets if:
        - It's not in the organic subset
        - It has a charge
        - It has an isotope
        - It has chirality
        - It has an atom class
        - Its explicit hydrogens differ from what would be implicit
        - It's an aromatic N with an explicit H (like pyrrole [nH])
        - It has any SMARTS query features
        """
        # SMARTS features always need brackets
        if atom.is_wildcard:
            return False  # Bare * doesn't need brackets
        
        if atom.is_recursive:
            return True
        
        if atom.atom_list:
            return True
        
        if atom.atom_list_negated:
            return True
        
        if atom.ring_count is not None:
            return True
        
        if atom.ring_size is not None:
            return True
        
        if atom.degree_query is not None:
            return True
        
        if atom.valence_query is not None:
            return True
        
        if atom.connectivity_query is not None:
            return True
        
        symbol = atom.symbol
        sym_check = symbol.capitalize() if len(symbol) == 1 else symbol
        
        # Check if in organic subset
        if sym_check not in ORGANIC_SUBSET and symbol.lower() not in AROMATIC_SUBSET:
            return True
        
        if atom.charge != 0:
            return True
        
        if atom.isotope is not None:
            return True
        
        if atom.chirality:
            return True
        
        if atom.atom_class is not None:
            return True
        
        # Aromatic nitrogen with explicit H needs brackets (e.g., pyrrole [nH])
        # This H is essential for aromaticity - it contributes to the pi system
        # Must check this BEFORE the general explicit hydrogen check below
        if atom.is_aromatic and sym_check == "N" and atom.explicit_hydrogens > 0:
            return True
        
        # Aromatic oxygen or sulfur with explicit H also needs brackets
        if atom.is_aromatic and sym_check in {"O", "S"} and atom.explicit_hydrogens > 0:
            return True
        
        # Check if explicit hydrogens differ from expected implicit
        if atom.explicit_hydrogens > 0:
            from chirpy.elements import get_default_valence, get_atomic_number
            
            # Get atomic number and default valence
            atomic_num = get_atomic_number(symbol)
            default_val = get_default_valence(atomic_num)
            
            if default_val is not None:
                # Calculate bond order sum (bonds to non-H atoms)
                bond_order_sum = 0
                for bond_idx in atom.bond_indices:
                    bond = self._mol.bonds[bond_idx]
                    bond_order_sum += bond.order if not bond.is_aromatic else 1
                
                expected_implicit = max(0, default_val - bond_order_sum)
                
                # If explicit Hs match expected implicit, no brackets needed
                if atom.explicit_hydrogens == expected_implicit:
                    return False
            
            return True
        
        return False
    
    def _bond_to_smiles(self, bond: Bond, is_ring_closure: bool = False) -> str:
        """Convert bond to SMILES string.
        
        Single bonds between aromatic atoms that are not themselves aromatic
        need an explicit '-' to distinguish from implicit aromatic bonds.
        Supports SMARTS bond types: dative (->/<-), any (~), quadruple ($).
        """
        # Handle SMARTS any bond
        if bond.is_any:
            return "~"
        
        # Handle dative bonds
        if bond.is_dative:
            if bond.dative_direction == 1:
                return "->"
            elif bond.dative_direction == -1:
                return "<-"
            return "->"  # Default forward
        
        if bond.is_aromatic:
            return ""
        
        if bond.order == 1:
            # Check if both atoms are aromatic
            a1 = self._mol.atoms[bond.atom1_idx]
            a2 = self._mol.atoms[bond.atom2_idx]
            
            if a1.is_aromatic and a2.is_aromatic:
                # Single bond between aromatics needs explicit '-'
                return "-"
            
            return bond.stereo or ""
        
        if bond.order == 2:
            return "="
        
        if bond.order == 3:
            return "#"
        
        # Quadruple bond (BondOrder.QUADRUPLE = 5)
        if bond.order == 5:
            return "$"
        
        return ""
    
    def _ring_number_to_smiles(self, n: int) -> str:
        """Format ring closure digit."""
        if 1 <= n <= 9:
            return str(n)
        if 10 <= n <= 99:
            return f"%{n}"
        return f"%({n})"


def to_smiles(mol: Molecule, ranks: list[int] | None = None) -> str:
    """Convert a Molecule to a SMILES string.
    
    Args:
        mol: Molecule to convert.
        ranks: Pre-computed canonical ranks (computed if None).
    
    Returns:
        SMILES string.
    
    Example:
        >>> mol = parse("C(C)CC")
        >>> to_smiles(mol)
        'CCCC'
    """
    return SmilesWriter(mol, ranks).to_smiles()


def canonical_smiles(smiles_or_mol: str | "Molecule") -> str:
    """Get canonical SMILES for a molecule.
    
    This is the main convenience function for canonicalizing SMILES.
    It handles parsing (if needed), aromaticity perception, and
    canonical output generation.
    
    Args:
        smiles_or_mol: SMILES string or Molecule object.
    
    Returns:
        Canonical SMILES string.
    
    Example:
        >>> canonical_smiles("C(C)CC")
        'CCCC'
        >>> canonical_smiles("c1ccccc1")
        'c1ccccc1'
    """
    from chirpy.parser import parse
    
    if isinstance(smiles_or_mol, str):
        mol = parse(smiles_or_mol)
    else:
        mol = smiles_or_mol
    
    # Perceive aromaticity
    perceive_aromaticity(mol)
    
    # Compute ranks and generate SMILES
    ranks = canonical_ranks(mol)
    return SmilesWriter(mol, ranks).to_smiles()
