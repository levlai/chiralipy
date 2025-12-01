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
    
    def __init__(self, max_ring_size: int = 20) -> None:
        """Initialize perceiver.
        
        Args:
            max_ring_size: Maximum ring size to consider for aromaticity.
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
            from .elements import ELEMENT_GROUPS
            group = ELEMENT_GROUPS.get(atomic_num)
            
            if group == 16:
                # O, S, Se: only count explicit double bonds, not aromatic membership
                has_pi_bond = ring_double >= 1 or has_exo_double
            else:
                # C, N, etc: aromatic ring membership counts
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
    
    def _extend_aromaticity(
        self,
        mol: Molecule,
        all_rings: list[set[int]],
        aromatic_rings: list[set[int]],
        atom_info: dict[int, tuple[int, int | None]],
        choice: dict[int, int],
    ) -> set[int]:
        """Try to extend aromaticity from known aromatic rings to adjacent rings.
        
        In fused systems, if one ring is aromatic, adjacent rings that share
        atoms often participate in the aromatic system too.
        
        Args:
            mol: Parent molecule.
            all_rings: All rings in the system.
            aromatic_rings: Rings already determined to be aromatic.
            atom_info: Atom contribution info.
            choice: Current electron assignment choices.
        
        Returns:
            Set of all aromatic atoms, or empty set if extension fails.
        """
        aromatic_atoms = set()
        for ring in aromatic_rings:
            aromatic_atoms |= ring
        
        # Try to add adjacent rings
        changed = True
        while changed:
            changed = False
            for ring in all_rings:
                if ring <= aromatic_atoms:
                    continue  # Already covered
                
                # Check if this ring shares atoms with aromatic rings
                shared = ring & aromatic_atoms
                if len(shared) >= 2:  # Shares a bond (fused)
                    # Count pi electrons for non-shared atoms
                    # Shared aromatic atoms contribute 1 each
                    pi = 0
                    for idx in ring:
                        if idx in aromatic_atoms:
                            pi += 1  # Aromatic atoms contribute 1
                        elif idx in choice:
                            pi += choice[idx]
                        else:
                            pi += atom_info[idx][0]
                    
                    if self._satisfies_huckel(pi):
                        aromatic_atoms |= ring
                        changed = True
        
        # Check if we covered all atoms in the system
        system_atoms = set()
        for ring in all_rings:
            system_atoms |= ring
        
        if aromatic_atoms == system_atoms:
            return aromatic_atoms
        
        # Partial coverage - return what we have
        return aromatic_atoms
    
    def _is_potentially_aromatic_ring(self, mol: Molecule, ring: set[int]) -> bool:
        """Check if all atoms in a ring can be sp2 hybridized.
        
        A ring can be aromatic only if:
        1. All atoms are capable of being sp2 (C, N, O, S, etc.)
        2. The ring contains at least some pi bonds (double bonds or conjugated system)
        3. Carbon atoms have appropriate degree for sp2
        
        Args:
            mol: Parent molecule.
            ring: Set of atom indices.
        
        Returns:
            True if all atoms could be sp2.
        """
        has_pi_bonds = False
        
        for idx in ring:
            atom = mol.atoms[idx]
            atomic_num = get_atomic_number(atom.symbol)
            
            # Check if atom can participate in aromaticity
            if not can_be_aromatic(atomic_num):
                return False
            
            # Check degree (number of connections)
            degree = len(atom.bond_indices)
            
            # For sp2, max degree depends on element group:
            # Group 14 (C, Si): max 3
            # Group 15 (N, P, As): max 3
            # Group 16 (O, S, Se): max 2
            # Group 13 (B): max 3
            from .elements import ELEMENT_GROUPS
            group = ELEMENT_GROUPS.get(atomic_num)
            
            if group == 16:  # O, S, Se
                if degree > 2:
                    return False
            else:  # Groups 13, 14, 15
                if degree > 3:
                    return False
            
            # Check for pi bonds
            for bond_idx in atom.bond_indices:
                bond = mol.bonds[bond_idx]
                if bond.order == 2 or bond.is_aromatic:
                    has_pi_bonds = True
                    break
        
        # Ring must have some pi bonds to be aromatic
        return has_pi_bonds
    
    def _count_pi_electrons(
        self, 
        mol: Molecule, 
        ring: set[int],
        system_atoms: set[int],
    ) -> int | None:
        """Count π electrons contributed by atoms in a ring.
        
        Args:
            mol: Parent molecule.
            ring: Set of atom indices in the ring.
            system_atoms: All atoms in the fused ring system.
        
        Returns:
            Total π electrons, or None if not aromatic.
        """
        return self._count_pi_electrons_v2(mol, ring, system_atoms, set())
    
    def _count_pi_electrons_v2(
        self, 
        mol: Molecule, 
        ring: set[int],
        system_atoms: set[int],
        already_aromatic: set[int],
    ) -> int | None:
        """Count π electrons contributed by atoms in a ring.
        
        This version considers atoms that are already marked as aromatic,
        which is important for fused ring systems where aromaticity
        spreads from one ring to adjacent rings.
        
        Rules for π electron contribution:
        - C in double bond within ring: 1 electron
        - C with exocyclic double bond (C=O): 1 electron
        - C+ (cation): 0 electrons (empty p orbital, still aromatic)
        - C- (anion): 2 electrons
        - N (pyridine-like, in double bond): 1 electron
        - N (pyrrole-like, with H or single bonds only): 2 electrons
        - O, S (furan-like, 2 single bonds): 2 electrons
        - O, S (in double bond): 1 electron
        - Atoms already aromatic: count based on their known contribution
        
        Args:
            mol: Parent molecule.
            ring: Set of atom indices in the ring.
            system_atoms: All atoms in the fused ring system.
            already_aromatic: Set of atoms already determined to be aromatic.
        
        Returns:
            Total π electrons, or None if not aromatic.
        """
        total = 0
        
        for idx in ring:
            atom = mol.atoms[idx]
            symbol = atom.symbol.upper()
            charge = atom.charge
            
            # Count bonds within ring and to outside
            ring_double_bonds = 0
            ring_single_bonds = 0
            exo_double_bonds = 0
            aromatic_bonds_in_ring = 0
            
            for bond_idx in atom.bond_indices:
                bond = mol.bonds[bond_idx]
                other = bond.other_atom(idx)
                order = bond.order
                
                if other in ring:
                    # Check if bond is to an already aromatic atom
                    if other in already_aromatic and idx in already_aromatic:
                        aromatic_bonds_in_ring += 1
                    elif order == 2:
                        ring_double_bonds += 1
                    else:
                        ring_single_bonds += 1
                else:
                    # Exocyclic bond
                    if order == 2:
                        exo_double_bonds += 1
            
            # Count total connections (for determining pyrrole vs pyridine-like N)
            total_connections = len(atom.bond_indices)
            
            # If atom is in an aromatic bond within this ring, treat it specially
            if aromatic_bonds_in_ring > 0:
                # Atom already participating in aromatic system
                contribution = self._get_aromatic_pi_contribution(symbol, charge)
            else:
                # Determine π contribution based on atom type and bonding
                contribution = self._get_pi_contribution(
                    symbol, charge, atom.explicit_hydrogens,
                    ring_double_bonds, ring_single_bonds, exo_double_bonds,
                    total_connections
                )
            
            if contribution is None:
                return None
            
            total += contribution
        
        return total
    
    def _get_aromatic_pi_contribution(self, symbol: str, charge: int) -> int:
        """Get π contribution for an atom already in an aromatic system.
        
        Args:
            symbol: Atom symbol (uppercase).
            charge: Formal charge.
        
        Returns:
            π electron contribution.
        """
        if symbol == 'C':
            if charge == 1:
                return 0
            elif charge == -1:
                return 2
            return 1
        elif symbol == 'N':
            if charge == 1:
                return 1
            elif charge == -1:
                return 2
            # For neutral N in aromatic system, depends on context
            # Pyridine-like contributes 1, pyrrole-like contributes 2
            # Default to 1 for atoms already aromatic
            return 1
        elif symbol in {'O', 'S', 'SE'}:
            if charge == 1:
                return 1
            elif charge == -1:
                return 2
            return 1  # Default for already aromatic
        elif symbol == 'B':
            return 0
        else:
            return 1
    
    def _get_pi_contribution(
        self,
        symbol: str,
        charge: int,
        explicit_h: int,
        ring_double: int,
        ring_single: int,
        exo_double: int,
        total_connections: int,
    ) -> int | None:
        """Get π electron contribution for an atom.
        
        Args:
            symbol: Atom symbol (uppercase).
            charge: Formal charge.
            explicit_h: Number of explicit hydrogens.
            ring_double: Number of double bonds within the ring.
            ring_single: Number of single bonds within the ring.
            exo_double: Number of exocyclic double bonds.
            total_connections: Total number of bonds (connections) to this atom.
        
        Returns None if atom cannot participate in aromaticity.
        """
        if symbol == 'C':
            if charge == 1:
                # Carbocation: empty p orbital, contributes 0
                return 0
            elif charge == -1:
                # Carbanion: lone pair in p orbital, contributes 2
                return 2
            else:
                # Neutral carbon
                if ring_double >= 1 or exo_double >= 1:
                    # Double bond (in ring or exocyclic): contributes 1
                    return 1
                elif ring_single >= 2:
                    # Only single bonds in ring - could be part of conjugated system
                    # This happens in fused systems where aromaticity "flows through"
                    return 1
                else:
                    return None
        
        elif symbol == 'N':
            if charge == 1:
                # N+ (like pyridinium)
                if ring_double >= 1:
                    return 1
                else:
                    return 2
            elif charge == -1:
                # N- contributes 2
                return 2
            else:
                # Neutral nitrogen
                if ring_double >= 1:
                    # Pyridine-like (=N-): contributes 1
                    return 1
                else:
                    # Nitrogen with only single bonds in ring
                    # Key distinction:
                    # - Pyrrole-like: 2 ring bonds + H (explicit or implicit), lone pair in pi system → 2
                    # - Pyridine-like: 3 connections total (no H), lone pair not in pi system → 1
                    #
                    # If N has 3 connections and no explicit H, it's pyridine-like in aromatic context
                    # (like the N-methyl nitrogens in caffeine's 6-membered ring)
                    if total_connections >= 3 and explicit_h == 0:
                        # Pyridine-like (tertiary N): contributes 1
                        return 1
                    else:
                        # Pyrrole-like (-NH- or -N< with H): contributes 2 (lone pair)
                        return 2
        
        elif symbol in {'O', 'S', 'SE'}:
            if charge == 1:
                return 1
            elif charge == -1:
                return 2
            else:
                if ring_double >= 1 or exo_double >= 1:
                    # Part of double bond
                    return 1
                else:
                    # Furan-like: contributes 2 (lone pair)
                    return 2
        
        elif symbol == 'P':
            if ring_double >= 1:
                return 1
            else:
                return 2
        
        elif symbol == 'B':
            # Boron: empty p orbital
            return 0
        
        elif symbol in {'AS', 'SI'}:
            if ring_double >= 1:
                return 1
            else:
                return 2
        
        return None
    
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
