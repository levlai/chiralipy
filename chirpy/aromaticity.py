"""
Aromaticity perception.

This module provides algorithms for detecting and assigning aromaticity
in molecular structures based on Hückel's rule.

The algorithm uses Hückel's rule (4n+2 π electrons) applied to ring
systems, with proper handling of:
- Fused ring systems
- Heteroatoms (N, O, S, etc.)
- Exocyclic double bonds (C=O in rings)
- Charged atoms
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .elements import can_be_aromatic, get_pi_contribution, get_atomic_number
from .rings import find_sssr, find_ring_systems

if TYPE_CHECKING:
    from .types import Atom, Bond, Molecule


@runtime_checkable
class AromaticityModel(Protocol):
    """Protocol for aromaticity perception models."""
    
    def perceive(self, mol: Molecule) -> None:
        """Perceive and assign aromaticity to atoms and bonds."""
        ...


class AromaticityPerceiver:
    """Hückel-based aromaticity perception.
    
    This perceiver implements aromaticity detection using Hückel's rule:
    
    1. Find all rings and fused ring systems (SSSR - Smallest Set of Smallest Rings)
    2. For each ring system, check if it can be aromatic:
       - All atoms must be sp2 hybridized (or capable of being)
       - The system must satisfy Hückel's 4n+2 rule
    3. Handle special cases:
       - Heteroatoms contributing 1 or 2 π electrons
       - Exocyclic double bonds (atoms still contribute to ring aromaticity)
       - Charged atoms
    
    π electron contribution rules (based on periodic table groups):
    - Group 14 (C, Si) with pi bond: 1 electron
    - Group 14 cation: 0 electrons (empty p orbital)
    - Group 14 anion: 2 electrons (lone pair)
    - Group 15 (N, P) pyridine-like (=N-): 1 electron
    - Group 15 (N, P) pyrrole-like (-NH-): 2 electrons
    - Group 16 (O, S) furan-like: 2 electrons
    - Group 13 (B): 0 electrons (empty p orbital)
    """
    
    def __init__(self, max_ring_size: int | None = None) -> None:
        """Initialize perceiver.
        
        Args:
            max_ring_size: Maximum ring size to consider for aromaticity
                (default None = no limit).
        """
        self._max_ring_size = max_ring_size
    
    def perceive(self, mol: Molecule) -> None:
        """Perceive and assign aromaticity.
        
        Args:
            mol: Molecule to analyze (modified in-place).
        """
        # Save aromatic info from parsing (lowercase letters in SMILES)
        # This is needed because we'll use it for pi electron counting
        # Note: Aromatic atoms can be indicated by:
        #   1. a.is_aromatic being True/truthy
        #   2. lowercase symbol (e.g., 'c' for aromatic carbon)
        parsed_aromatic_atoms = {
            i for i, a in enumerate(mol.atoms) 
            if a.is_aromatic or (a.symbol and a.symbol[0].islower())
        }
        parsed_aromatic_bonds = {i for i, b in enumerate(mol.bonds) if b.is_aromatic}
        
        # Reset all aromaticity flags (we'll recompute)
        for atom in mol.atoms:
            atom.is_aromatic = False
        for bond in mol.bonds:
            bond.is_aromatic = False
        
        # Find all rings using shared ring detection
        rings = find_sssr(mol, self._max_ring_size)
        
        if not rings:
            return
        
        # Group rings into fused systems
        ring_systems = find_ring_systems(rings)
        
        # Process each ring system
        for system in ring_systems:
            self._process_ring_system(mol, system, parsed_aromatic_atoms, parsed_aromatic_bonds)
    
    def _process_ring_system(
        self, 
        mol: Molecule, 
        rings: list[set[int]],
        parsed_aromatic_atoms: set[int] | None = None,
        parsed_aromatic_bonds: set[int] | None = None,
    ) -> None:
        """Process a ring system for aromaticity.
        
        For fused systems, we use an approach that tries different
        electron assignments for ambiguous atoms (like tertiary N).
        
        The algorithm:
        1. Identify atoms with fixed π electron contributions
        2. Identify atoms with ambiguous contributions (e.g., N with 3 bonds)
        3. Try different assignments to find one where ALL rings satisfy Hückel
        4. If found, mark the entire system as aromatic
        
        Args:
            mol: Parent molecule.
            rings: List of rings in this fused system.
            parsed_aromatic_atoms: Atoms that were aromatic in the parsed input.
            parsed_aromatic_bonds: Bonds that were aromatic in the parsed input.
        """
        if parsed_aromatic_atoms is None:
            parsed_aromatic_atoms = set()
        if parsed_aromatic_bonds is None:
            parsed_aromatic_bonds = set()
        
        # Get all atoms in the system
        system_atoms: set[int] = set()
        for ring in rings:
            system_atoms |= ring
        
        # Get pi contribution info for all atoms
        atom_info = self._get_atom_pi_info(mol, system_atoms, parsed_aromatic_atoms, parsed_aromatic_bonds)
        
        # Try to find a valid electron assignment for the whole system
        aromatic_atoms = self._try_find_aromatic_assignment(
            mol, rings, system_atoms, atom_info
        )
        
        # Mark aromatic atoms and bonds
        if aromatic_atoms:
            self._mark_aromatic(mol, aromatic_atoms)
    
    def _get_atom_pi_info(
        self, 
        mol: Molecule, 
        ring_atoms: set[int],
        parsed_aromatic_atoms: set[int] | None = None,
        parsed_aromatic_bonds: set[int] | None = None,
    ) -> dict[int, tuple[int, int | None]]:
        """Get π electron contribution info for atoms.
        
        Returns a dict mapping atom index to (fixed_contribution, optional_contribution).
        If optional_contribution is not None, the atom can contribute either value.
        
        Args:
            mol: Parent molecule.
            ring_atoms: Set of atom indices in the ring system.
            parsed_aromatic_atoms: Atoms that were aromatic in the parsed input.
            parsed_aromatic_bonds: Bonds that were aromatic in the parsed input.
        
        Returns:
            Dict of atom index -> (fixed, optional) contributions.
        """
        if parsed_aromatic_atoms is None:
            parsed_aromatic_atoms = set()
        if parsed_aromatic_bonds is None:
            parsed_aromatic_bonds = set()
            
        result = {}
        
        for idx in ring_atoms:
            atom = mol.atoms[idx]
            atomic_num = get_atomic_number(atom.symbol)
            charge = atom.charge
            
            # Count bonds
            ring_double = 0
            ring_aromatic = 0
            exo_double = 0
            
            for bond_idx in atom.bond_indices:
                bond = mol.bonds[bond_idx]
                other = bond.other_atom(idx)
                order = bond.order
                
                if other in ring_atoms:
                    # Check if bond was aromatic in parsed input
                    if bond_idx in parsed_aromatic_bonds:
                        ring_aromatic += 1
                    elif order == 2:
                        ring_double += 1
                else:
                    if order == 2:
                        exo_double += 1
            
            total_connections = len(atom.bond_indices)
            explicit_h = atom.explicit_hydrogens
            
            # Check if atom was aromatic in parsed input
            was_aromatic = idx in parsed_aromatic_atoms
            
            # has_pi_bond: Does this atom have a double/aromatic bond IN THE RING?
            # This is different from being in an aromatic ring for Group 16 elements
            # - For C: being in aromatic ring counts (participates in delocalized pi)
            # - For N: depends on bonding pattern (pyridine vs pyrrole)
            # - For O/S: only true if actual double bond (very rare in aromatics)
            has_ring_pi = ring_double >= 1 or ring_aromatic >= 1
            has_exo_double = exo_double >= 1
            
            # Use element-based pi contribution calculation
            # For Group 14 (C), being in an aromatic ring counts as having pi bond
            # For Group 16 (O, S), only explicit double bonds count
            from .elements import get_outer_electrons
            outer_e = get_outer_electrons(atomic_num)
            
            # Elements with 6 valence electrons (Group 16: O, S, Se) typically contribute
            # 2 electrons (lone pair) to the aromatic system, not 1 (pi bond).
            # They only have a "pi bond" contribution if explicitly double bonded.
            if outer_e == 6:
                has_pi_bond = ring_double >= 1 or has_exo_double
            else:
                # Elements with < 6 valence electrons (C, N) typically contribute 1 electron
                # via a pi bond, so aromatic membership implies a pi bond exists.
                has_pi_bond = has_ring_pi or was_aromatic or has_exo_double
            
            contribution = get_pi_contribution(
                atomic_num=atomic_num,
                has_pi_bond=has_pi_bond,
                has_exo_double=has_exo_double,
                has_h=explicit_h > 0,
                total_connections=total_connections,
                charge=charge,
            )
            
            result[idx] = contribution
        
        return result
    
    def _try_find_aromatic_assignment(
        self,
        mol: Molecule,
        rings: list[set[int]],
        system_atoms: set[int],
        atom_info: dict[int, tuple[int, int | None]],
    ) -> set[int]:
        """Try to find an electron assignment that makes the system aromatic.
        
        Args:
            mol: Parent molecule.
            rings: List of rings in the system.
            system_atoms: All atoms in the system.
            atom_info: Dict of atom contributions.
        
        Returns:
            Set of aromatic atoms, or empty set if not aromatic.
        """
        # Find atoms with ambiguous contributions
        ambiguous_atoms = [idx for idx, (_, opt) in atom_info.items() if opt is not None]
        
        # Generate all possible assignments
        from itertools import product
        
        if not ambiguous_atoms:
            # No ambiguous atoms, just check if any ring satisfies Hückel
            choices = [{}]
        else:
            # Generate all combinations of choices for ambiguous atoms
            options = []
            for idx in ambiguous_atoms:
                fixed, opt = atom_info[idx]
                options.append([(idx, fixed), (idx, opt)])
            
            choices = []
            for combo in product(*options):
                choice = {idx: val for idx, val in combo}
                choices.append(choice)
        
        # Find the best assignment (most rings satisfied)
        best_aromatic = set()
        best_ring_count = 0
        
        for choice in choices:
            # Calculate pi electrons for each ring with this assignment
            valid_rings = []
            
            for ring in rings:
                pi = 0
                valid = True
                for idx in ring:
                    fixed, opt = atom_info[idx]
                    if idx in choice:
                        contrib = choice[idx]
                    else:
                        contrib = fixed
                    
                    if contrib is None:
                        # This atom cannot participate in aromaticity
                        valid = False
                        break
                    pi += contrib
                
                if valid and self._satisfies_huckel(pi):
                    valid_rings.append(ring)
            
            if not valid_rings:
                continue
            
            # Calculate union of valid rings
            aromatic = set()
            for ring in valid_rings:
                aromatic |= ring
            
            # Prefer assignments that make more rings valid and cover more atoms
            if len(valid_rings) > best_ring_count or (
                len(valid_rings) == best_ring_count and len(aromatic) > len(best_aromatic)
            ):
                best_ring_count = len(valid_rings)
                best_aromatic = aromatic
        
        return best_aromatic
    
    def _satisfies_huckel(self, pi_electrons: int) -> bool:
        """Check if π electron count satisfies Hückel's 4n+2 rule.
        
        Args:
            pi_electrons: Total π electrons in the system.
        
        Returns:
            True if 4n+2 for some non-negative integer n.
        """
        # 4n + 2 means (electrons - 2) must be divisible by 4
        if pi_electrons < 2:
            return False
        return (pi_electrons - 2) % 4 == 0
    
    def _mark_aromatic(self, mol: Molecule, atoms: set[int]) -> None:
        """Mark atoms and their connecting bonds as aromatic.
        
        Args:
            mol: Parent molecule.
            atoms: Set of aromatic atom indices.
        """
        # Mark atoms
        for idx in atoms:
            mol.atoms[idx].is_aromatic = True
        
        # Mark bonds between aromatic atoms
        for bond in mol.bonds:
            if bond.atom1_idx in atoms and bond.atom2_idx in atoms:
                bond.is_aromatic = True
                # Note: bond order stays as-is for now; writer handles display


def perceive_aromaticity(
    mol: Molecule,
    model: AromaticityModel | None = None,
) -> None:
    """Perceive aromaticity in a molecule.
    
    This is a convenience function that applies aromaticity perception
    to a molecule using the specified model (or default).
    
    Args:
        mol: Molecule to analyze (modified in-place).
        model: Aromaticity model to use (default: AromaticityPerceiver).
    
    Example:
        >>> mol = parse("C1=CC=CC=C1")  # Benzene with explicit double bonds
        >>> perceive_aromaticity(mol)
        >>> mol.atoms[0].is_aromatic
        True
    """
    if model is None:
        model = AromaticityPerceiver()
    
    model.perceive(mol)
