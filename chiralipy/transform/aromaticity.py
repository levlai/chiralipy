"""
Aromaticity perception.

This module provides algorithms for detecting and assigning aromaticity
in molecular structures based on Hückel's rule.

The algorithm uses Hückel's rule (4n+2 π electrons) applied to ring
systems, with proper handling of:
- Fused ring systems (checks combinations of rings)
- Heteroatoms (N, O, S, etc.)
- Exocyclic double bonds (electron stealing by electronegative atoms)
- Charged atoms (cations, anions)
- ElectronDonorType classification
"""

from __future__ import annotations

from enum import IntEnum
from itertools import combinations
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from chiralipy.elements import (
    get_atomic_number, 
    get_default_valence,
    OUTER_ELECTRONS,
)
from chiralipy.rings import find_sssr

if TYPE_CHECKING:
    from chiralipy.types import Atom, Molecule


@runtime_checkable
class AromaticityModel(Protocol):
    """Protocol for aromaticity perception models."""
    
    def perceive(self, mol: Molecule) -> None:
        """Perceive and assign aromaticity to atoms and bonds."""
        ...


class ElectronDonorType(IntEnum):
    """Electron donor type for aromaticity classification.
    
    Each atom in a potential aromatic ring is classified by how many
    electrons it can donate to the π system.
    """
    VACANT = 0      # 0 electrons (empty p orbital)
    ONE = 1         # contributes 1 electron
    TWO = 2         # contributes 2 electrons
    ONE_OR_TWO = 3  # ambiguous 1 or 2
    ANY = 4         # can be any (dummy atom)
    NONE = 5        # cannot participate


# Electronegativity table (Pauling scale)
# Used for determining if exocyclic atom "steals" electrons
ELECTRONEGATIVITY: dict[int, float] = {
    1: 2.20,   # H
    5: 2.04,   # B
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    14: 1.90,  # Si
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    33: 2.18,  # As
    34: 2.55,  # Se
    35: 2.96,  # Br
    52: 2.10,  # Te
    53: 2.66,  # I
}


def is_more_electronegative(atom1_num: int, atom2_num: int) -> bool:
    """Check if atom1 is more electronegative than atom2.
    
    Args:
        atom1_num: Atomic number of first atom.
        atom2_num: Atomic number of second atom.
    
    Returns:
        True if atom1 is more electronegative than atom2.
    """
    en1 = ELECTRONEGATIVITY.get(atom1_num, 2.0)
    en2 = ELECTRONEGATIVITY.get(atom2_num, 2.0)
    return en1 > en2


def _get_atom_degree(atom: Atom, mol: Molecule) -> int:
    """Get total degree of atom including implicit H.
    
    Args:
        atom: Atom to check.
        mol: Parent molecule.
    
    Returns:
        Total degree (explicit bonds + total H count).
    """
    return len(atom.bond_indices) + atom.total_hydrogens(mol)


def _get_atom_valence(atom: Atom, mol: Molecule) -> int:
    """Get total valence of atom.
    
    Sum of bond orders for all bonds. Aromatic bonds contribute 1
    (not 1.5) because the pi electron contribution is handled
    separately in the aromaticity perception.
    
    Args:
        atom: Atom to check.
        mol: Parent molecule.
    
    Returns:
        Total valence (sum of bond orders).
    """
    valence = 0
    for bond_idx in atom.bond_indices:
        bond = mol.bonds[bond_idx]
        # Aromatic bonds contribute 1 to valence (sigma bond)
        # The pi contribution is handled in electron counting
        if bond.is_aromatic:
            valence += 1
        else:
            valence += bond.order
    return valence


def _has_incident_noncyclic_multiple_bond(
    atom_idx: int,
    mol: Molecule,
    ring_bonds: set[int],
) -> tuple[bool, int]:
    """Check if atom has a non-cyclic multiple bond.
    
    Args:
        atom_idx: Atom index.
        mol: Parent molecule.
        ring_bonds: Set of ring bond indices.
    
    Returns:
        Tuple of (has_multiple, other_atom_idx). other_atom_idx is -1 if not found.
    """
    atom = mol.atoms[atom_idx]
    for bond_idx in atom.bond_indices:
        bond = mol.bonds[bond_idx]
        if bond_idx not in ring_bonds:
            # Non-ring bond
            if bond.order >= 2 or (bond.is_aromatic and bond.order >= 2):
                other_idx = bond.other_atom(atom_idx)
                return True, other_idx
    return False, -1


def _has_incident_cyclic_multiple_bond(
    atom_idx: int,
    mol: Molecule,
    ring_bonds: set[int],
    parsed_aromatic_bonds: set[int] | None = None,
) -> bool:
    """Check if atom has a cyclic multiple bond.
    
    Args:
        atom_idx: Atom index.
        mol: Parent molecule.
        ring_bonds: Set of ring bond indices.
        parsed_aromatic_bonds: Bonds that were aromatic in parsed input.
    
    Returns:
        True if atom has a cyclic double/aromatic bond.
    """
    if parsed_aromatic_bonds is None:
        parsed_aromatic_bonds = set()
        
    atom = mol.atoms[atom_idx]
    for bond_idx in atom.bond_indices:
        bond = mol.bonds[bond_idx]
        if bond_idx in ring_bonds:
            # Check if bond is multiple or was aromatic
            if bond.order >= 2 or bond.is_aromatic or bond_idx in parsed_aromatic_bonds:
                return True
    return False


def _has_incident_multiple_bond(
    atom_idx: int, 
    mol: Molecule,
    parsed_aromatic_bonds: set[int] | None = None,
) -> bool:
    """Check if atom has any multiple bond (cyclic or not).
    
    Args:
        atom_idx: Atom index.
        mol: Parent molecule.
        parsed_aromatic_bonds: Bonds that were aromatic in parsed input.
    
    Returns:
        True if atom has any double/triple/aromatic bond.
    """
    if parsed_aromatic_bonds is None:
        parsed_aromatic_bonds = set()
        
    atom = mol.atoms[atom_idx]
    for bond_idx in atom.bond_indices:
        bond = mol.bonds[bond_idx]
        if bond.order >= 2 or bond.is_aromatic or bond_idx in parsed_aromatic_bonds:
            return True
    return False


def count_atom_elec(
    atom_idx: int, 
    mol: Molecule,
    parsed_aromatic_bonds: set[int] | None = None,
) -> int:
    """Count electrons available for pi system.
    
    Formula: electrons = (default_valence - degree) + lone_pairs - radicals
    
    Where:
    - lone_pairs = outer_electrons - default_valence - charge
    
    Args:
        atom_idx: Atom index.
        mol: Parent molecule.
        parsed_aromatic_bonds: Bonds that were aromatic in parsed input.
    
    Returns:
        Number of electrons available, or -1 if cannot be aromatic.
    """
    if parsed_aromatic_bonds is None:
        parsed_aromatic_bonds = set()
        
    atom = mol.atoms[atom_idx]
    atomic_num = get_atomic_number(atom.symbol)
    
    # Default valence
    dv = get_default_valence(atomic_num)
    if dv is None or dv <= 1:
        # Univalent or unknown elements can't be aromatic
        return -1
    
    # Calculate bond order sum considering aromatic bonds
    bond_order_sum = 0.0
    for bond_idx in atom.bond_indices:
        bond = mol.bonds[bond_idx]
        # Use 1.5 for aromatic bonds (either currently marked or from parsed input)
        if bond.is_aromatic or bond_idx in parsed_aromatic_bonds:
            bond_order_sum += 1.5
        else:
            bond_order_sum += bond.order
    
    # Total degree including H
    # For aromatic atoms: implicit H = default_valence - bond_order_sum + charge
    implicit_h = max(0, dv - int(round(bond_order_sum)) + atom.charge - atom.explicit_hydrogens)
    degree = len(atom.bond_indices) + atom.explicit_hydrogens + implicit_h
    
    # If more than 3-coordinated, cannot be aromatic
    if degree > 3:
        return -1
    
    # Lone pair electrons = outer_electrons - default_valence
    outer_e = OUTER_ELECTRONS.get(atomic_num, 0)
    if outer_e == 0:
        return -1
    
    nlp = outer_e - dv
    
    # Subtract charge to get true lone pair count
    nlp = max(nlp - atom.charge, 0)
    
    # TODO: Handle radicals (atom.num_radicals)
    n_radicals = 0
    
    # Electrons available for pi system
    res = (dv - degree) + nlp - n_radicals
    
    if res > 1:
        # Check for multiple unsaturations (e.g., C in C=C=C)
        # If atom has bond order > 2 (triple), only contribute 1 electron
        valence = int(round(bond_order_sum))
        n_unsaturations = valence - len(atom.bond_indices)
        if n_unsaturations > 1:
            res = 1
    
    return res


def get_atom_donor_type(
    atom_idx: int,
    mol: Molecule,
    ring_bonds: set[int],
    exocyclic_bonds_steal_electrons: bool = True,
    parsed_aromatic_bonds: set[int] | None = None,
) -> ElectronDonorType:
    """Get electron donor type for an atom.
    
    Args:
        atom_idx: Atom index.
        mol: Parent molecule.
        ring_bonds: Set of ring bond indices.
        exocyclic_bonds_steal_electrons: If True, exocyclic bonds to more
            electronegative atoms reduce electron count.
        parsed_aromatic_bonds: Bonds that were aromatic in parsed input.
    
    Returns:
        ElectronDonorType classification.
    """
    if parsed_aromatic_bonds is None:
        parsed_aromatic_bonds = set()
        
    atom = mol.atoms[atom_idx]
    atomic_num = get_atomic_number(atom.symbol)
    
    # Dummy atoms (atomic num 0) are special
    if atomic_num == 0:
        if _has_incident_cyclic_multiple_bond(atom_idx, mol, ring_bonds, parsed_aromatic_bonds):
            return ElectronDonorType.ONE
        return ElectronDonorType.ANY
    
    nelec = count_atom_elec(atom_idx, mol, parsed_aromatic_bonds)
    
    if nelec < 0:
        return ElectronDonorType.NONE
    
    if nelec == 0:
        has_exo, who = _has_incident_noncyclic_multiple_bond(atom_idx, mol, ring_bonds)
        if has_exo:
            # No electrons but has exocyclic multiple bond
            # May have empty p orbital
            return ElectronDonorType.VACANT
        elif _has_incident_cyclic_multiple_bond(atom_idx, mol, ring_bonds, parsed_aromatic_bonds):
            # No spare electrons but has cyclic multiple bond
            return ElectronDonorType.ONE
        else:
            return ElectronDonorType.NONE
    
    if nelec == 1:
        has_exo, who = _has_incident_noncyclic_multiple_bond(atom_idx, mol, ring_bonds)
        if has_exo:
            # The only electron is from exocyclic bond
            # Not available if bonded to more electronegative atom
            other_atom = mol.atoms[who]
            other_num = get_atomic_number(other_atom.symbol)
            if exocyclic_bonds_steal_electrons and is_more_electronegative(other_num, atomic_num):
                return ElectronDonorType.VACANT
            return ElectronDonorType.ONE
        else:
            # Require at least one multiple bond
            if _has_incident_multiple_bond(atom_idx, mol, parsed_aromatic_bonds):
                return ElectronDonorType.ONE
            # Tropylium/cyclopropenyl cation case
            elif atom.charge == 1:
                return ElectronDonorType.VACANT
            return ElectronDonorType.NONE
    
    # nelec >= 2
    has_exo, who = _has_incident_noncyclic_multiple_bond(atom_idx, mol, ring_bonds)
    if has_exo:
        # More electronegative atom steals one electron
        other_atom = mol.atoms[who]
        other_num = get_atomic_number(other_atom.symbol)
        if exocyclic_bonds_steal_electrons and is_more_electronegative(other_num, atomic_num):
            nelec -= 1
    
    if nelec % 2 == 1:
        return ElectronDonorType.ONE
    return ElectronDonorType.TWO


def is_atom_cand_for_arom(
    atom_idx: int,
    mol: Molecule,
    edon: ElectronDonorType,
    ring_bonds: set[int],
    allow_third_row: bool = True,
    allow_triple_bonds: bool = True,
    allow_higher_exceptions: bool = True,
    only_c_or_n: bool = False,
    allow_exocyclic_multiple_bonds: bool = True,
) -> bool:
    """Check if atom can be an aromaticity candidate.
    
    Args:
        atom_idx: Atom index.
        mol: Parent molecule.
        edon: Electron donor type.
        ring_bonds: Set of ring bond indices.
        allow_third_row: If False, reject atoms with Z > 10.
        allow_triple_bonds: If False, reject atoms with triple bonds.
        allow_higher_exceptions: If True, allow Se (34) and Te (52).
        only_c_or_n: If True, only C and N can be aromatic.
        allow_exocyclic_multiple_bonds: If False, reject atoms with
            exocyclic double/triple bonds.
    
    Returns:
        True if atom can be aromatic candidate.
    """
    atom = mol.atoms[atom_idx]
    atomic_num = get_atomic_number(atom.symbol)
    
    if only_c_or_n and atomic_num not in (6, 7):
        return False
    
    if not allow_third_row and atomic_num > 10:
        return False
    
    # Limit to first two rows + Se and Te
    if atomic_num > 18:
        if not allow_higher_exceptions or atomic_num not in (34, 52):
            return False
    
    # Check donor type
    if edon not in (
        ElectronDonorType.VACANT,
        ElectronDonorType.ONE,
        ElectronDonorType.TWO,
        ElectronDonorType.ONE_OR_TWO,
        ElectronDonorType.ANY,
    ):
        return False
    
    # Check default valence
    dv = get_default_valence(atomic_num)
    if dv is not None and dv > 0:
        total_valence = _get_atom_valence(atom, mol)
        # For charged atoms, adjust the expected valence:
        # - Cations (e.g., N+) can have higher valence (N+ can have 4 bonds)
        # - Anions (e.g., O-) have lower valence
        # The formula: effective_dv = dv + charge (for N+: 3+1=4, for O-: 2-1=1)
        effective_dv = dv + atom.charge
        if total_valence > effective_dv:
            return False
    
    # Check for multiple unsaturations (e.g., C=C=N)
    valence = _get_atom_valence(atom, mol)
    degree = len(atom.bond_indices)
    n_unsaturations = valence - degree
    
    if n_unsaturations > 1:
        n_mult = 0
        for bond_idx in atom.bond_indices:
            bond = mol.bonds[bond_idx]
            if bond.order == 2:
                n_mult += 1
            elif bond.order == 3:
                if not allow_triple_bonds:
                    return False
                n_mult += 1
        if n_mult > 1:
            return False
    
    # Check exocyclic multiple bonds
    if not allow_exocyclic_multiple_bonds:
        for bond_idx in atom.bond_indices:
            bond = mol.bonds[bond_idx]
            if bond.order >= 2 and bond_idx not in ring_bonds:
                return False
    
    return True


def _get_min_max_elec(dtype: ElectronDonorType) -> tuple[int, int]:
    """Get min and max electrons for donor type.
    
    Args:
        dtype: Electron donor type.
    
    Returns:
        Tuple of (min_elec, max_elec).
    """
    if dtype == ElectronDonorType.ANY:
        return 1, 2
    elif dtype == ElectronDonorType.ONE_OR_TWO:
        return 1, 2
    elif dtype == ElectronDonorType.ONE:
        return 1, 1
    elif dtype == ElectronDonorType.TWO:
        return 2, 2
    else:  # VACANT or NONE
        return 0, 0


def apply_huckel(
    ring_atoms: list[int],
    edon: dict[int, ElectronDonorType],
    min_ring_size: int = 0,
) -> bool:
    """Apply Hückel rule to a ring.
    
    Hückel rule: (4n + 2) pi electrons for aromaticity.
    Special: at most 1 AnyElectronDonorType atom per ring.
    
    Args:
        ring_atoms: List of atom indices in the ring.
        edon: Dict mapping atom index to donor type.
        min_ring_size: Minimum ring size (0 = no limit).
    
    Returns:
        True if ring satisfies Hückel.
    """
    if min_ring_size and len(ring_atoms) < min_ring_size:
        return False
    
    rlw = 0  # Ring lower bound
    rup = 0  # Ring upper bound
    n_any = 0
    
    for idx in ring_atoms:
        dtype = edon.get(idx, ElectronDonorType.NONE)
        if dtype == ElectronDonorType.ANY:
            n_any += 1
            if n_any > 1:
                return False
        atlw, atup = _get_min_max_elec(dtype)
        rlw += atlw
        rup += atup
    
    # Check Hückel: (4n + 2) = 2, 6, 10, 14, ...
    if rup >= 6:
        for rie in range(rlw, rup + 1):
            if (rie - 2) % 4 == 0:
                return True
    elif rup == 2:
        return True
    
    return False


def _get_ring_bonds(mol: Molecule, rings: list[set[int]]) -> set[int]:
    """Get all bond indices that are in any ring.
    
    Args:
        mol: Parent molecule.
        rings: List of rings (as atom index sets).
    
    Returns:
        Set of bond indices that are ring bonds.
    """
    ring_bonds: set[int] = set()
    for ring in rings:
        for bond_idx, bond in enumerate(mol.bonds):
            if bond.atom1_idx in ring and bond.atom2_idx in ring:
                ring_bonds.add(bond_idx)
    return ring_bonds


def _make_ring_neighbor_map(
    rings: list[set[int]],
    mol: Molecule,
    max_size: int = 0,
) -> dict[int, list[int]]:
    """Make a map of ring neighbors (rings that share bonds).
    
    Args:
        rings: List of rings.
        mol: Parent molecule.
        max_size: Maximum ring size to consider (0 = no limit).
    
    Returns:
        Dict mapping ring index to list of neighbor ring indices.
    """
    n_rings = len(rings)
    neigh_map: dict[int, list[int]] = {i: [] for i in range(n_rings)}
    
    # Get bonds for each ring
    ring_bonds: list[set[tuple[int, int]]] = []
    for ring in rings:
        bonds = set()
        for bond in mol.bonds:
            a1, a2 = bond.atom1_idx, bond.atom2_idx
            if a1 in ring and a2 in ring:
                bonds.add((min(a1, a2), max(a1, a2)))
        ring_bonds.append(bonds)
    
    for i in range(n_rings):
        if max_size and len(rings[i]) > max_size:
            continue
        for j in range(i + 1, n_rings):
            if max_size and len(rings[j]) > max_size:
                continue
            # Check for shared bonds
            shared = ring_bonds[i] & ring_bonds[j]
            if shared:
                neigh_map[i].append(j)
                neigh_map[j].append(i)
    
    return neigh_map


def _pick_fused_rings(
    start: int,
    neigh_map: dict[int, list[int]],
    done: set[int],
) -> list[int]:
    """Pick all rings in a fused system starting from start.
    
    Args:
        start: Starting ring index.
        neigh_map: Ring neighbor map.
        done: Set of already processed ring indices.
    
    Returns:
        List of ring indices in the fused system.
    """
    result: list[int] = []
    stack = [start]
    
    while stack:
        curr = stack.pop()
        if curr in done:
            continue
        done.add(curr)
        result.append(curr)
        
        for neigh in neigh_map.get(curr, []):
            if neigh not in done:
                stack.append(neigh)
    
    return result


class AromaticityPerceiver:
    """Hückel-based aromaticity perception.
    
    This perceiver implements aromaticity detection using:
    
    1. Find all rings (SSSR)
    2. For each atom in a ring, compute electron donor type
    3. Check if atom is a candidate for aromaticity
    4. For candidate rings, check Hückel rule (4n+2 electrons)
    5. For fused systems, try combinations of rings
    
    Key features:
    - ElectronDonorType classification
    - Exocyclic bond electron stealing by more electronegative atoms
    - At most 1 AnyElectronDonorType atom per ring
    - Fused ring combination checking
    """
    
    # Maximum ring size for fused aromaticity
    MAX_FUSED_RING_SIZE = 24
    
    def __init__(self, max_ring_size: int | None = None) -> None:
        """Initialize perceiver.
        
        Args:
            max_ring_size: Maximum ring size to consider (default: no limit).
        """
        self._max_ring_size = max_ring_size
    
    def perceive(self, mol: Molecule) -> None:
        """Perceive and assign aromaticity.
        
        Args:
            mol: Molecule to analyze (modified in-place).
        """
        # Store parsed aromatic info for kekulization
        parsed_aromatic_atoms = {
            i for i, a in enumerate(mol.atoms) 
            if a.is_aromatic or (a.symbol and a.symbol[0].islower())
        }
        parsed_aromatic_bonds = {i for i, b in enumerate(mol.bonds) if b.is_aromatic}
        
        # Clear all aromaticity flags
        for atom in mol.atoms:
            atom.is_aromatic = False
        for bond in mol.bonds:
            bond.is_aromatic = False
        
        # Find rings
        rings = find_sssr(mol, self._max_ring_size)
        if not rings:
            if parsed_aromatic_atoms:
                self._kekulize_non_aromatic_region(mol, parsed_aromatic_atoms, parsed_aromatic_bonds)
            return
        
        # Get ring bonds
        ring_bonds = _get_ring_bonds(mol, rings)
        
        # Compute donor types for all atoms in rings
        all_ring_atoms: set[int] = set()
        for ring in rings:
            all_ring_atoms |= ring
        
        edon: dict[int, ElectronDonorType] = {}
        acands: dict[int, bool] = {}
        
        for atom_idx in all_ring_atoms:
            edon[atom_idx] = get_atom_donor_type(
                atom_idx, mol, ring_bonds, 
                exocyclic_bonds_steal_electrons=True,
                parsed_aromatic_bonds=parsed_aromatic_bonds,
            )
            acands[atom_idx] = is_atom_cand_for_arom(
                atom_idx, mol, edon[atom_idx], ring_bonds
            )
        
        # Find candidate rings (all atoms must be candidates)
        candidate_rings: list[set[int]] = []
        for ring in rings:
            all_candidates = True
            all_dummy = True
            
            for atom_idx in ring:
                atom = mol.atoms[atom_idx]
                if get_atomic_number(atom.symbol) != 0:
                    all_dummy = False
                if not acands.get(atom_idx, False):
                    all_candidates = False
            
            if all_candidates and not all_dummy:
                candidate_rings.append(ring)
        
        if not candidate_rings:
            if parsed_aromatic_atoms:
                self._kekulize_non_aromatic_region(mol, parsed_aromatic_atoms, parsed_aromatic_bonds)
            return
        
        # Build ring neighbor map
        neigh_map = _make_ring_neighbor_map(candidate_rings, mol, self.MAX_FUSED_RING_SIZE)
        
        # Process fused systems
        done_rings: set[int] = set()
        aromatic_ring_ids: set[int] = set()
        done_bonds: set[int] = set()
        
        curr = 0
        while curr < len(candidate_rings):
            if curr in done_rings:
                curr += 1
                continue
            
            # Get fused system
            fused = _pick_fused_rings(curr, neigh_map, done_rings)
            
            # Apply Hückel to fused system
            self._apply_huckel_to_fused(
                mol, candidate_rings, fused, edon,
                neigh_map, aromatic_ring_ids, done_bonds
            )
            
            # Find next ring
            for i in range(len(candidate_rings)):
                if i not in done_rings:
                    curr = i
                    break
            else:
                break
        
        # Mark aromatic atoms and bonds
        for ring_idx in aromatic_ring_ids:
            ring = candidate_rings[ring_idx]
            for atom_idx in ring:
                mol.atoms[atom_idx].is_aromatic = True
        
        for bond_idx in done_bonds:
            bond = mol.bonds[bond_idx]
            bond.is_aromatic = True
        
        # Kekulize non-aromatic atoms that were parsed as aromatic
        non_aromatic_parsed = {
            i for i in parsed_aromatic_atoms
            if not mol.atoms[i].is_aromatic
        }
        if non_aromatic_parsed:
            self._kekulize_non_aromatic_region(mol, non_aromatic_parsed, parsed_aromatic_bonds)
    
    def _apply_huckel_to_fused(
        self,
        mol: Molecule,
        rings: list[set[int]],
        fused_ids: list[int],
        edon: dict[int, ElectronDonorType],
        neigh_map: dict[int, list[int]],
        aromatic_ring_ids: set[int],
        done_bonds: set[int],
    ) -> None:
        """Apply Hückel rule to a fused ring system.
        
        Try increasing sizes of ring combinations until all bonds are covered
        or we've tried all combinations.
        
        Args:
            mol: Parent molecule.
            rings: List of all candidate rings.
            fused_ids: Indices of rings in this fused system.
            edon: Electron donor types.
            neigh_map: Ring neighbor map.
            aromatic_ring_ids: Set to add aromatic ring indices to.
            done_bonds: Set of bonds already marked aromatic.
        """
        n_fused = len(fused_ids)
        
        # Get all bonds in fused system
        all_fused_bonds: set[int] = set()
        for rid in fused_ids:
            ring = rings[rid]
            for bond_idx, bond in enumerate(mol.bonds):
                if bond.atom1_idx in ring and bond.atom2_idx in ring:
                    all_fused_bonds.add(bond_idx)
        
        # Try combinations of increasing size
        for size in range(1, n_fused + 1):
            # Limit for very large systems
            if size == 2 and n_fused > 300:
                break
            
            if len(done_bonds & all_fused_bonds) >= len(all_fused_bonds):
                break  # All bonds done
            
            for combo in combinations(range(n_fused), size):
                cur_ring_ids = [fused_ids[i] for i in combo]
                
                # Check if subset is fused
                if size > 1 and not self._check_fused(cur_ring_ids, neigh_map):
                    continue
                
                # Get union of atoms (but only count atoms in 1 or 2 rings)
                atom_ring_count: dict[int, int] = {}
                for rid in cur_ring_ids:
                    for atom_idx in rings[rid]:
                        atom_ring_count[atom_idx] = atom_ring_count.get(atom_idx, 0) + 1
                
                # Only include atoms in 1 or 2 rings (not bridgehead in >2)
                union_atoms = [
                    idx for idx, count in atom_ring_count.items()
                    if count == 1 or count == 2
                ]
                
                if apply_huckel(union_atoms, edon):
                    # Mark bonds as aromatic
                    self._mark_bonds_aromatic(
                        mol, rings, cur_ring_ids, done_bonds
                    )
                    aromatic_ring_ids.update(cur_ring_ids)
    
    def _check_fused(
        self,
        ring_ids: list[int],
        neigh_map: dict[int, list[int]],
    ) -> bool:
        """Check if a set of rings forms a connected fused system.
        
        Args:
            ring_ids: List of ring indices.
            neigh_map: Ring neighbor map.
        
        Returns:
            True if rings are connected.
        """
        if len(ring_ids) <= 1:
            return True
        
        # BFS from first ring
        visited: set[int] = set()
        ring_set = set(ring_ids)
        queue = [ring_ids[0]]
        
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            
            for neigh in neigh_map.get(curr, []):
                if neigh in ring_set and neigh not in visited:
                    queue.append(neigh)
        
        return len(visited) == len(ring_ids)
    
    def _mark_bonds_aromatic(
        self,
        mol: Molecule,
        rings: list[set[int]],
        ring_ids: list[int],
        done_bonds: set[int],
    ) -> None:
        """Mark bonds in rings as aromatic.
        
        Only bonds that appear in exactly one ring of the set are marked.
        
        Args:
            mol: Parent molecule.
            rings: List of all rings.
            ring_ids: Indices of rings to mark.
            done_bonds: Set to add marked bond indices to.
        """
        bond_count: dict[int, int] = {}
        
        for rid in ring_ids:
            ring = rings[rid]
            for bond_idx, bond in enumerate(mol.bonds):
                if bond.atom1_idx in ring and bond.atom2_idx in ring:
                    bond_count[bond_idx] = bond_count.get(bond_idx, 0) + 1
        
        # Mark bonds with count == 1
        for bond_idx, count in bond_count.items():
            if count == 1:
                done_bonds.add(bond_idx)
    
    def _kekulize_non_aromatic_region(
        self,
        mol: Molecule,
        non_aromatic_atoms: set[int],
        parsed_aromatic_bonds: set[int],
    ) -> None:
        """Kekulize atoms that were parsed as aromatic but are not.
        
        Args:
            mol: Parent molecule.
            non_aromatic_atoms: Atoms that are NOT aromatic.
            parsed_aromatic_bonds: Bonds that were aromatic in input.
        """
        from chiralipy.transform.kekulize import _atom_needs_double_bond, KekulizationError
        from itertools import combinations
        
        # Find bonds that need kekulization
        bonds_to_kekulize: set[int] = set()
        for bond_idx in parsed_aromatic_bonds:
            bond = mol.bonds[bond_idx]
            if bond.atom1_idx in non_aromatic_atoms and bond.atom2_idx in non_aromatic_atoms:
                bonds_to_kekulize.add(bond_idx)
        
        if not bonds_to_kekulize:
            return
        
        # Build adjacency
        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in non_aromatic_atoms}
        for bond_idx in bonds_to_kekulize:
            bond = mol.bonds[bond_idx]
            if bond.atom1_idx in non_aromatic_atoms and bond.atom2_idx in non_aromatic_atoms:
                adj[bond.atom1_idx].append((bond.atom2_idx, bond_idx))
                adj[bond.atom2_idx].append((bond.atom1_idx, bond_idx))
        
        # Determine which atoms need double bonds
        needs_double: dict[int, bool] = {}
        ambiguous: set[int] = set()
        for atom_idx in non_aromatic_atoms:
            atom = mol.atoms[atom_idx]
            needs, is_amb = _atom_needs_double_bond(atom, mol, bonds_to_kekulize)
            needs_double[atom_idx] = needs
            if is_amb:
                ambiguous.add(atom_idx)
        
        def try_kekulize(needs: dict[int, bool]) -> tuple[bool, set[int]]:
            atoms_needing = [i for i in non_aromatic_atoms if needs.get(i, False)]
            matching: dict[int, int] = {}
            double_bonds: set[int] = set()
            
            def try_match(atoms_left: list[int]) -> bool:
                if not atoms_left:
                    return True
                atom_idx = atoms_left[0]
                for nbr_idx, bond_idx in adj.get(atom_idx, []):
                    if nbr_idx in matching:
                        continue
                    if not needs.get(nbr_idx, False):
                        continue
                    matching[atom_idx] = nbr_idx
                    matching[nbr_idx] = atom_idx
                    double_bonds.add(bond_idx)
                    remaining = [a for a in atoms_left[1:] if a not in matching]
                    if try_match(remaining):
                        return True
                    del matching[atom_idx]
                    del matching[nbr_idx]
                    double_bonds.discard(bond_idx)
                return False
            
            atoms_needing.sort(key=lambda x: len(adj.get(x, [])))
            if not atoms_needing:
                return True, set()
            if try_match(atoms_needing):
                return True, double_bonds
            return False, set()
        
        success, double_bonds = try_kekulize(needs_double)
        
        if not success and ambiguous:
            amb_list = list(ambiguous)
            for n_flips in range(1, len(amb_list) + 1):
                for to_flip in combinations(amb_list, n_flips):
                    mod_needs = needs_double.copy()
                    for idx in to_flip:
                        mod_needs[idx] = not mod_needs[idx]
                    success, double_bonds = try_kekulize(mod_needs)
                    if success:
                        break
                if success:
                    break
        
        if not success:
            return
        
        # Apply
        for bond_idx in bonds_to_kekulize:
            bond = mol.bonds[bond_idx]
            bond.order = 2 if bond_idx in double_bonds else 1
            bond.is_aromatic = False
        
        for atom_idx in non_aromatic_atoms:
            atom = mol.atoms[atom_idx]
            atom.is_aromatic = False
            if atom.symbol and atom.symbol[0].islower():
                atom.symbol = atom.symbol[0].upper() + atom.symbol[1:]


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
