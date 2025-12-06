"""
Kekulization - convert aromatic representation to explicit double bonds.

This module provides functions for converting aromatic molecules
to their Kekulé form with explicit alternating single/double bonds.

The kekulization algorithm uses principled electronic structure calculations:
- Atoms contribute π electrons based on their periodic table group
- Whether an atom needs a double bond depends on whether it contributes
  1 electron (needs double bond) or 2 electrons (lone pair, no double bond)
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from chiralipy.elements import (
    AROMATIC_CAPABLE_ELEMENTS,
    get_atomic_number,
    get_outer_electrons,
)

if TYPE_CHECKING:
    from chiralipy.types import Molecule


class KekulizationError(Exception):
    """Error during Kekulization."""
    pass


# Alias for backward compatibility
KekulizeError = KekulizationError


def _atom_needs_double_bond(
    atom,
    mol: "Molecule",
    aromatic_bonds_set: set[int],
) -> tuple[bool, bool]:
    """Determine if an atom needs a double bond in the aromatic ring.
    
    Uses electronic structure to determine π electron contribution:
    - Atoms contributing 1 π electron need a double bond
    - Atoms contributing 2 π electrons (lone pair) don't need one
    - Atoms contributing 0 π electrons (empty p orbital) don't need one
    
    Args:
        atom: The atom to check.
        mol: The parent molecule.
        aromatic_bonds_set: Set of aromatic bond indices.
    
    Returns:
        Tuple of (needs_double, is_ambiguous).
        is_ambiguous is True if the atom could go either way.
    """
    symbol = atom.symbol.upper() if atom.symbol else ""
    atomic_num = get_atomic_number(symbol)
    
    if atomic_num == 0:
        # Unknown element - assume needs double bond
        return True, False
    
    if atomic_num not in AROMATIC_CAPABLE_ELEMENTS:
        # Non-aromatic element - shouldn't happen but assume needs double
        return True, False
    
    outer_e = get_outer_electrons(atomic_num)
    
    if outer_e == 0:
        return True, False
    
    # Get charge
    charge = getattr(atom, 'charge', 0) or 0
    
    # Count existing non-aromatic double bonds (exocyclic)
    exo_double = 0
    for bond_idx in atom.bond_indices:
        bond = mol.bonds[bond_idx]
        if bond_idx not in aromatic_bonds_set and bond.order == 2:
            exo_double += 1
    
    # If already has exocyclic double bond, doesn't need another in ring
    if exo_double > 0:
        return False, False
    
    # Count connections (σ bonds) - excluding hydrogens implicit in SMILES
    total_connections = len(atom.bond_indices)
    explicit_h = getattr(atom, 'explicit_hydrogens', 0) or 0
    total_connections += explicit_h
    
    # Check if has explicit hydrogen
    has_h = explicit_h > 0
    
    # Adjust effective outer electrons for charge
    # Positive charge: lose electron, negative: gain electron
    effective_outer_e = outer_e - charge
    
    # Elements with 3 outer electrons (Group 13: B)
    # All electrons in σ bonds, empty p orbital
    # Contributes 0 π electrons → doesn't need double bond
    if outer_e == 3:
        return False, False
    
    # Elements with 4 outer electrons (Group 14: C, Si, Ge)
    if outer_e == 4:
        if charge == 1:
            # Carbocation: 3 outer e⁻, empty p orbital → 0 π electrons
            return False, False
        elif charge == -1:
            # Carbanion: 5 outer e⁻, lone pair in p orbital → 2 π electrons
            return False, False
        else:
            # Neutral: 4 outer e⁻, 3 in σ bonds, 1 in p orbital
            # Needs exactly 1 double bond to complete π system
            return True, False
    
    # Elements with 5 outer electrons (Group 15: N, P, As)
    if outer_e == 5:
        if charge == 1:
            # N⁺: 4 outer electrons (like carbon)
            # In aromatic ring, behaves like carbon → needs double bond
            return True, False
        elif charge == -1:
            # N⁻: 6 outer electrons
            # Lone pair in p orbital → 2 π electrons, no double bond
            return False, False
        else:
            # Neutral N: can be pyridine-like or pyrrole-like
            # Pyridine-like: 2 ring connections, lone pair in sp2 orbital
            #   → 1 π electron in p orbital → needs double bond
            # Pyrrole-like: 2 ring connections + H, lone pair in p orbital
            #   → 2 π electrons from lone pair → no double bond
            
            if has_h:
                # Has hydrogen attached - pyrrole-like
                return False, False
            elif total_connections == 2:
                # Only 2 bonds, no H - pyridine-like
                return True, False
            elif total_connections >= 3:
                # Tertiary N (3+ bonds)
                # If 3 bonds in ring: pyridine-like → needs double
                # This case is ambiguous - could be either type
                return True, True
            else:
                # Default to pyrrole-like
                return False, False
    
    # Elements with 6 outer electrons (Group 16: O, S, Se, Te)
    # Furan/thiophene-like: 2 σ bonds, 2 lone pairs
    # One lone pair contributes to π system → 2 π electrons
    # → doesn't need double bond
    if outer_e == 6:
        if charge == 1:
            # O⁺/S⁺: 5 outer electrons, can participate in double bond
            return True, False
        elif charge == -1:
            # O⁻/S⁻: 7 outer electrons, definitely has lone pair
            return False, False
        else:
            # Neutral: furan/thiophene-like, lone pair contributes
            return False, False
    
    # Default: assume needs double bond
    return True, False


def kekulize(mol: Molecule, clear_aromatic_flags: bool = True, in_place: bool = False) -> Molecule:
    """Convert aromatic bonds to alternating single/double bonds (Kekulé form).
    
    Args:
        mol: Molecule to kekulize.
        clear_aromatic_flags: If True, clear is_aromatic flags on atoms and bonds.
        in_place: If True, modify molecule in-place; otherwise return a copy.
    
    Returns:
        Kekulized molecule (copy if in_place=False, otherwise the same object).
    
    Raises:
        KekulizationError: If kekulization fails (e.g., odd number of aromatic atoms in a ring).
    
    Example:
        >>> mol = parse("c1ccccc1")
        >>> kek_mol = kekulize(mol)
        >>> # kek_mol now has alternating C=C and C-C bonds
    """
    if not in_place:
        mol = deepcopy(mol)
    
    # Find aromatic atoms
    aromatic_atoms: set[int] = set()
    for atom in mol.atoms:
        if atom.is_aromatic:
            aromatic_atoms.add(atom.idx)
    
    if not aromatic_atoms:
        return mol  # Nothing to kekulize
    
    # Find aromatic bonds
    aromatic_bonds: list[int] = []
    aromatic_bonds_set: set[int] = set()
    for bond in mol.bonds:
        if bond.is_aromatic:
            aromatic_bonds.append(bond.idx)
            aromatic_bonds_set.add(bond.idx)
    
    if not aromatic_bonds:
        # Atoms are aromatic but no aromatic bonds - nothing to do
        return mol
    
    # Build aromatic subgraph adjacency
    aromatic_adj: dict[int, list[tuple[int, int]]] = {i: [] for i in aromatic_atoms}
    for bond_idx in aromatic_bonds:
        bond = mol.bonds[bond_idx]
        if bond.atom1_idx in aromatic_atoms and bond.atom2_idx in aromatic_atoms:
            aromatic_adj[bond.atom1_idx].append((bond.atom2_idx, bond_idx))
            aromatic_adj[bond.atom2_idx].append((bond.atom1_idx, bond_idx))
    
    # Determine which atoms need a double bond using element-based logic
    needs_double: dict[int, bool] = {}
    ambiguous_atoms: set[int] = set()
    
    for atom_idx in aromatic_atoms:
        atom = mol.atoms[atom_idx]
        needs, is_ambiguous = _atom_needs_double_bond(atom, mol, aromatic_bonds_set)
        needs_double[atom_idx] = needs
        if is_ambiguous:
            ambiguous_atoms.add(atom_idx)
    
    # Now find a perfect matching using the Hopcroft-Karp-like approach
    # We need to match atoms that need double bonds
    
    def try_kekulize_with_config(needs_config: dict[int, bool]) -> tuple[bool, set[int]]:
        """Try to find a valid kekulization with given needs_double configuration.
        
        Returns:
            Tuple of (success, double_bonds set)
        """
        atoms_needing = [idx for idx in aromatic_atoms if needs_config.get(idx, False)]
        
        # Use backtracking to find a valid assignment
        # Each atom needing a double bond must be matched with exactly one neighbor
        
        matching: dict[int, int] = {}  # atom_idx -> matched_atom_idx
        result_double_bonds: set[int] = set()
        
        def try_match(atoms_left: list[int]) -> bool:
            """Try to find a perfect matching for remaining atoms."""
            if not atoms_left:
                return True
            
            atom_idx = atoms_left[0]
            
            # This atom needs to be matched with one neighbor
            for nbr_idx, bond_idx in aromatic_adj[atom_idx]:
                # Can only match with a neighbor that also needs matching
                # and isn't already matched
                if nbr_idx in matching:
                    continue  # Neighbor already matched
                
                if not needs_config.get(nbr_idx, False):
                    continue  # Neighbor doesn't need a double bond
                
                # Try this matching
                matching[atom_idx] = nbr_idx
                matching[nbr_idx] = atom_idx
                result_double_bonds.add(bond_idx)
                
                # Recurse with remaining atoms (excluding matched ones)
                remaining = [a for a in atoms_left[1:] if a not in matching]
                if try_match(remaining):
                    return True
                
                # Backtrack
                del matching[atom_idx]
                del matching[nbr_idx]
                result_double_bonds.discard(bond_idx)
            
            return False
        
        # Sort by degree (most constrained first)
        atoms_needing.sort(key=lambda x: len(aromatic_adj[x]))
        
        if not atoms_needing:
            return True, set()
        
        if try_match(atoms_needing):
            return True, result_double_bonds
        return False, set()
    
    # First try with the initial needs_double assignments
    success, double_bonds = try_kekulize_with_config(needs_double)
    
    if not success and ambiguous_atoms:
        # Try flipping ambiguous atoms to find a valid assignment
        # The ambiguous atoms can go either way (need double or not)
        from itertools import combinations
        
        ambiguous_list = list(ambiguous_atoms)
        
        # Try flipping different subsets of ambiguous atoms
        # Start by flipping each one individually, then pairs, etc.
        for num_flips in range(1, len(ambiguous_list) + 1):
            for atoms_to_flip in combinations(ambiguous_list, num_flips):
                # Create modified needs_double config
                modified_needs = needs_double.copy()
                for atom_idx in atoms_to_flip:
                    modified_needs[atom_idx] = not modified_needs[atom_idx]
                
                success, double_bonds = try_kekulize_with_config(modified_needs)
                if success:
                    break
            if success:
                break
    
    if not success:
        raise KekulizationError("Cannot kekulize molecule - no valid assignment found")
    
    # Apply the assignment
    for bond_idx in aromatic_bonds:
        bond = mol.bonds[bond_idx]
        if bond_idx in double_bonds:
            bond.order = 2
        else:
            bond.order = 1
        
        if clear_aromatic_flags:
            bond.is_aromatic = False
    
    if clear_aromatic_flags:
        for atom_idx in aromatic_atoms:
            atom = mol.atoms[atom_idx]
            atom.is_aromatic = False
            # Capitalize the symbol
            if atom.symbol and atom.symbol[0].islower():
                atom.symbol = atom.symbol[0].upper() + atom.symbol[1:]
    
    return mol


def kekulize_smiles(mol: Molecule) -> str:
    """Get the Kekulé SMILES representation of a molecule.
    
    This creates a copy of the molecule, kekulizes it, and returns
    the SMILES string.
    
    Args:
        mol: Molecule to convert.
    
    Returns:
        Kekulé SMILES string.
    
    Example:
        >>> mol = parse("c1ccccc1")
        >>> kekulize_smiles(mol)
        'C1=CC=CC=C1'
    """
    from chiralipy.writer import to_smiles
    
    kek_mol = kekulize(mol)  # Returns a copy by default
    return to_smiles(kek_mol)
