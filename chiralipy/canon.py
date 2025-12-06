"""
Canonical SMILES generation.

This module provides canonicalization algorithms for generating unique,
reproducible SMILES strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Final

from chiralipy.elements import BondOrder
from chiralipy.rings import _find_ring_atoms_and_bonds_fast, get_ring_info

if TYPE_CHECKING:
    from chiralipy.types import Molecule


# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Helper functions
# =============================================================================

def _count_swaps_to_interconvert(perm: list[int], sorted_perm: list[int]) -> int:
    """Count minimum swaps to convert perm to sorted_perm.
    
    Uses a cycle decomposition approach to count swaps.
    """
    if len(perm) != len(sorted_perm):
        return 0
    
    # Create a mapping from value to target position
    target_pos = {v: i for i, v in enumerate(sorted_perm)}
    
    # Work on a copy
    current = list(perm)
    n_swaps = 0
    
    for i in range(len(current)):
        # Find where current[i] should go
        target = target_pos[current[i]]
        while target != i:
            # Swap current[i] with current[target]
            current[i], current[target] = current[target], current[i]
            n_swaps += 1
            target = target_pos[current[i]]
    
    return n_swaps


# =============================================================================
# Data structures for canonicalization
# =============================================================================

@dataclass(slots=True)
class _BondHolder:
    """Internal bond representation for canonicalization."""
    bond_type: int
    bond_stereo: int
    nbr_sym_class: int
    nbr_idx: int
    bond_idx: int
    nbr_chiral_code: int = 0  # Chirality contribution from this neighbor
    
    @staticmethod
    def compare(x: "_BondHolder", y: "_BondHolder") -> int:
        """Compare two bondholders."""
        if x.bond_type != y.bond_type:
            return -1 if x.bond_type < y.bond_type else 1
        
        if x.bond_stereo != y.bond_stereo:
            return -1 if x.bond_stereo < y.bond_stereo else 1
        
        if x.nbr_sym_class != y.nbr_sym_class:
            return -1 if x.nbr_sym_class < y.nbr_sym_class else 1
        
        # Compare chirality contribution from neighbor
        if x.nbr_chiral_code != y.nbr_chiral_code:
            return -1 if x.nbr_chiral_code < y.nbr_chiral_code else 1
        
        return 0


@dataclass(slots=True)
class _CanonAtom:
    """Internal atom representation for canonicalization."""
    atom_idx: int
    degree: int
    atomic_num: int
    isotope: int
    formal_charge: int
    total_num_hs: int
    has_chirality: bool
    is_aromatic: bool
    is_in_ring: bool
    min_ring_size: int
    chirality_tag: int  # 0 = none, 1 = CW (@), 2 = CCW (@@)
    nbr_ids: list[int] = field(default_factory=list)
    bonds: list[_BondHolder] = field(default_factory=list)
    index: int = 0  # Current partition/symmetry class


# =============================================================================
# HanoiSort - merge sort with partition tracking
# =============================================================================

def _hanoi(
    base: list[int],
    nel: int,
    temp: list[int],
    count: list[int],
    changed: list[int],
    compar: Callable[[int, int], int],
    base_offset: int = 0,
    temp_offset: int = 0,
) -> bool:
    """Recursive hanoi merge-sort that updates count array.
    
    Returns True if result is in temp, False if in base.
    """
    if nel == 1:
        count[base[base_offset]] = 1
        return False
    
    if nel == 2:
        n1 = base[base_offset]
        n2 = base[base_offset + 1]
        
        stat = compar(n1, n2) if (changed[n1] or changed[n2]) else 0
        
        if stat == 0:
            count[n1] = 2
            count[n2] = 0
            return False
        elif stat < 0:
            count[n1] = 1
            count[n2] = 1
            return False
        else:
            count[n1] = 1
            count[n2] = 1
            base[base_offset] = n2
            base[base_offset + 1] = n1
            return False
    
    # Recursive case
    n1 = nel // 2
    n2 = nel - n1
    
    r1 = _hanoi(base, n1, temp, count, changed, compar, base_offset, temp_offset)
    r2 = _hanoi(base, n2, temp, count, changed, compar, base_offset + n1, temp_offset + n1)
    
    # Determine sources and destination
    s1_arr, s1_off = (temp, temp_offset) if r1 else (base, base_offset)
    s2_arr, s2_off = (temp, temp_offset + n1) if r2 else (base, base_offset + n1)
    
    if r1:
        result = False
        ptr_arr, ptr_off = base, base_offset
    else:
        result = True
        ptr_arr, ptr_off = temp, temp_offset
    
    # Merge
    i1, i2, ip = 0, 0, 0
    
    while i1 < n1 and i2 < n2:
        v1 = s1_arr[s1_off + i1]
        v2 = s2_arr[s2_off + i2]
        
        stat = compar(v1, v2) if (changed[v1] or changed[v2]) else 0
        len1 = count[v1]
        len2 = count[v2]
        
        if stat == 0:
            # Equal - merge partitions
            count[v1] = len1 + len2
            count[v2] = 0
            
            for k in range(len1):
                ptr_arr[ptr_off + ip] = s1_arr[s1_off + i1 + k]
                ip += 1
            i1 += len1
            
            if i1 >= n1:
                while i2 < n2:
                    ptr_arr[ptr_off + ip] = s2_arr[s2_off + i2]
                    ip += 1
                    i2 += 1
                return result
            
            for k in range(len2):
                ptr_arr[ptr_off + ip] = s2_arr[s2_off + i2 + k]
                ip += 1
            i2 += len2
            
            if i2 >= n2:
                while i1 < n1:
                    ptr_arr[ptr_off + ip] = s1_arr[s1_off + i1]
                    ip += 1
                    i1 += 1
                return result
                
        elif stat < 0:
            for k in range(len1):
                ptr_arr[ptr_off + ip] = s1_arr[s1_off + i1 + k]
                ip += 1
            i1 += len1
            
            if i1 >= n1:
                while i2 < n2:
                    ptr_arr[ptr_off + ip] = s2_arr[s2_off + i2]
                    ip += 1
                    i2 += 1
                return result
        else:
            for k in range(len2):
                ptr_arr[ptr_off + ip] = s2_arr[s2_off + i2 + k]
                ip += 1
            i2 += len2
            
            if i2 >= n2:
                while i1 < n1:
                    ptr_arr[ptr_off + ip] = s1_arr[s1_off + i1]
                    ip += 1
                    i1 += 1
                return result
    
    return result


def _hanoi_sort(
    order: list[int],
    start: int,
    length: int,
    count: list[int],
    changed: list[int],
    compar: Callable[[int, int], int],
) -> None:
    """Sort a segment using HanoiSort."""
    if length <= 1:
        if length == 1:
            count[order[start]] = 1
        return
    
    segment = order[start:start + length]
    temp = [0] * length
    
    result_in_temp = _hanoi(segment, length, temp, count, changed, compar)
    
    if result_in_temp:
        order[start:start + length] = temp
    else:
        order[start:start + length] = segment


# =============================================================================
# Atom comparison functor
# =============================================================================

class _AtomCompareFunctor:
    """Atom comparison functor for canonical ordering."""
    
    def __init__(
        self,
        atoms: list[_CanonAtom],
        use_isotopes: bool = True,
        use_chirality: bool = True,
    ) -> None:
        self.atoms = atoms
        self.use_isotopes = use_isotopes
        self.use_chirality = use_chirality
        self.use_nbrs = False
    
    def __call__(self, i: int, j: int) -> int:
        """Compare atoms i and j."""
        v = self._basecomp(i, j)
        if v != 0:
            return v
        
        if self.use_nbrs:
            self._update_neighbor_index(i)
            self._update_neighbor_index(j)
            
            ai = self.atoms[i]
            aj = self.atoms[j]
            
            for k in range(min(len(ai.bonds), len(aj.bonds))):
                cmp = _BondHolder.compare(ai.bonds[k], aj.bonds[k])
                if cmp != 0:
                    return cmp
            
            if len(ai.bonds) != len(aj.bonds):
                return -1 if len(ai.bonds) < len(aj.bonds) else 1
        
        return 0
    
    def _update_neighbor_index(self, atom_idx: int) -> None:
        """Update neighbor symmetry classes, chirality codes, and sort bonds."""
        atom = self.atoms[atom_idx]
        for bh in atom.bonds:
            nbr = self.atoms[bh.nbr_idx]
            bh.nbr_sym_class = nbr.index
            
            # Compute chirality code using chiral rank algorithm
            if self.use_chirality and nbr.chirality_tag != 0:
                bh.nbr_chiral_code = self._get_chiral_rank(bh.nbr_idx)
            else:
                bh.nbr_chiral_code = 0
        
        # Sort ascending by (bond_type, bond_stereo, nbr_sym_class, nbr_chiral_code)
        # Sort neighbors ascending to determine rank
        atom.bonds.sort(key=lambda bh: (bh.bond_type, bh.bond_stereo, bh.nbr_sym_class, bh.nbr_chiral_code))
    
    def _get_chiral_rank(self, atom_idx: int) -> int:
        """Compute chiral rank for an atom.
        
        Returns 0 for non-chiral atoms, 1 or 2 for chiral atoms based on
        the chirality tag and the permutation of neighbor ranks.
        
        IMPORTANT: Must use original neighbor order (nbr_ids), not sorted bonds.
        """
        atom = self.atoms[atom_idx]
        if atom.chirality_tag == 0:
            return 0
        
        # Get neighbor ranks in ORIGINAL bond iteration order (nbr_ids)
        # This is critical - uses original order
        perm = []
        for nbr_idx in atom.nbr_ids:
            rnk = self.atoms[nbr_idx].index
            # Make sure we don't have duplicate ranks
            if rnk in perm:
                return 0  # Can't determine chirality with duplicate ranks
            perm.append(rnk)
        
        if len(perm) != atom.degree:
            return 0
        
        # Sort the permutation
        sorted_perm = sorted(perm)
        
        # Count swaps to convert perm to sorted_perm
        n_swaps = _count_swaps_to_interconvert(perm, sorted_perm)
        
        # chirality_tag: 1 = CW (@@), 2 = CCW (@)
        # Return 2 for CW, 1 for CCW, then flip if odd swaps
        res = 2 if atom.chirality_tag == 1 else 1
        if n_swaps % 2:
            res = 1 if res == 2 else 2
        
        return res
    
    def _basecomp(self, i: int, j: int) -> int:
        """Base comparison without neighbor info."""
        ai = self.atoms[i]
        aj = self.atoms[j]
        
        # 1. Partition index
        if ai.index != aj.index:
            return -1 if ai.index < aj.index else 1
        
        # 2. Degree
        if ai.degree != aj.degree:
            return -1 if ai.degree < aj.degree else 1

        # 3. Ring membership (Not in ring < In ring)
        if ai.is_in_ring != aj.is_in_ring:
            return -1 if not ai.is_in_ring else 1

        # 4. Aromaticity (Aromatic < Not Aromatic)
        if ai.is_aromatic != aj.is_aromatic:
            return -1 if ai.is_aromatic else 1

        # 5. Atomic number
        if ai.atomic_num != aj.atomic_num:
            return -1 if ai.atomic_num < aj.atomic_num else 1

        # 6. Ring Size (Descending: 6 < 5)
        if ai.min_ring_size != aj.min_ring_size:
            return -1 if ai.min_ring_size > aj.min_ring_size else 1
        
        # 7. Isotope
        if self.use_isotopes and ai.isotope != aj.isotope:
            return -1 if ai.isotope < aj.isotope else 1
        
        # 8. Total Hs
        if ai.total_num_hs != aj.total_num_hs:
            return -1 if ai.total_num_hs < aj.total_num_hs else 1
        
        # 9. Formal charge - uses unsigned comparison for proper ordering
        ui = ai.formal_charge & 0xFFFFFFFF
        uj = aj.formal_charge & 0xFFFFFFFF
        if ui != uj:
            return -1 if ui < uj else 1
            
        # 9. Presence of chirality (has chirality or not)
        if self.use_chirality:
            ivi = 1 if ai.chirality_tag != 0 else 0
            ivj = 1 if aj.chirality_tag != 0 else 0
            if ivi != ivj:
                return -1 if ivi < ivj else 1
            
            # 8. If both are chiral, use chiral rank
            if ivi and ivj:
                chiral_rank_i = self._get_chiral_rank(i)
                chiral_rank_j = self._get_chiral_rank(j)
                if chiral_rank_i != chiral_rank_j:
                    return -1 if chiral_rank_i < chiral_rank_j else 1
        
        return 0


# =============================================================================
# Special Chirality Comparator for neighbor-based chirality comparison
# =============================================================================

class _SpecialChiralityAtomCompareFunctor:
    """Special atom comparison functor for chirality-based tie breaking.
    
    Compares atoms based on how their chiral neighbors would rank them.
    """
    
    def __init__(self, atoms: list[_CanonAtom]) -> None:
        self.atoms = atoms
    
    def __call__(self, i: int, j: int) -> int:
        """Compare atoms i and j based on chiral neighbor swaps."""
        # First update and compare neighbor indices
        self._update_neighbor_index(i)
        self._update_neighbor_index(j)
        
        ai = self.atoms[i]
        aj = self.atoms[j]
        
        for k in range(min(len(ai.bonds), len(aj.bonds))):
            cmp = _BondHolder.compare(ai.bonds[k], aj.bonds[k])
            if cmp != 0:
                return cmp
        
        # Then compare based on chiral neighbor swaps
        swaps_i = self._get_neighbor_num_swaps(i)
        swaps_j = self._get_neighbor_num_swaps(j)
        
        for k in range(min(len(swaps_i), len(swaps_j))):
            cmp = swaps_i[k][1] - swaps_j[k][1]
            if cmp != 0:
                return -1 if cmp < 0 else 1
        
        return 0
    
    def _update_neighbor_index(self, atom_idx: int) -> None:
        """Update neighbor symmetry classes and sort bonds."""
        atom = self.atoms[atom_idx]
        for bh in atom.bonds:
            nbr = self.atoms[bh.nbr_idx]
            bh.nbr_sym_class = nbr.index
        
        # Sort ascending by (bond_type, bond_stereo, nbr_sym_class)
        atom.bonds.sort(key=lambda bh: (bh.bond_type, bh.bond_stereo, bh.nbr_sym_class))
    
    def _get_neighbor_num_swaps(self, atom_idx: int) -> list[tuple[int, int]]:
        """Compute chiral swap counts for each neighbor.
        
        For each neighbor, if it's a chiral atom, compute the number of swaps
        needed to move atom_idx to the front of the neighbor's neighbor list.
        """
        result = []
        atom = self.atoms[atom_idx]
        
        for bh in atom.bonds:
            nbr_idx = bh.nbr_idx
            nbr = self.atoms[nbr_idx]
            
            if nbr.chirality_tag != 0:
                # Build reference order: neighbor's original neighbor IDs
                ref = list(nbr.nbr_ids)
                
                # Check for duplicate ranks among neighbors (except atom_idx)
                neighbors_seen = []
                too_many_similar = False
                for nid in ref:
                    if nid != atom_idx:
                        nbr_nbr_index = self.atoms[nid].index
                        if nbr_nbr_index in neighbors_seen:
                            too_many_similar = True
                            break
                        neighbors_seen.append(nbr_nbr_index)
                
                if too_many_similar:
                    result.append((bh.nbr_sym_class, 0))
                else:
                    # Build probe order: atom_idx first, then other neighbors in bond order
                    probe = [atom_idx]
                    for other_bh in nbr.bonds:
                        if other_bh.nbr_idx != atom_idx:
                            probe.append(other_bh.nbr_idx)
                    
                    n_swaps = _count_swaps_to_interconvert(ref, probe)
                    
                    # Determine result based on chirality tag and swap parity
                    # chirality_tag: 1 = CW (@@), 2 = CCW (@)
                    if nbr.chirality_tag == 1:  # CW
                        if n_swaps % 2:
                            result.append((bh.nbr_sym_class, 2))
                        else:
                            result.append((bh.nbr_sym_class, 1))
                    else:  # CCW
                        if n_swaps % 2:
                            result.append((bh.nbr_sym_class, 1))
                        else:
                            result.append((bh.nbr_sym_class, 2))
            else:
                result.append((bh.nbr_sym_class, 0))
        
        result.sort()
        return result


# =============================================================================
# Core canonicalization algorithm
# =============================================================================

def _create_single_partition(
    n_atoms: int,
    order: list[int],
    count: list[int],
    atoms: list[_CanonAtom],
) -> None:
    """Initialize single partition."""
    for i in range(n_atoms):
        atoms[i].index = 0
        order[i] = i
        count[i] = 0
    count[0] = n_atoms


def _activate_partitions(
    n_atoms: int,
    order: list[int],
    count: list[int],
    next_arr: list[int],
    changed: list[int],
) -> int:
    """Activate partitions needing refinement."""
    for i in range(n_atoms):
        next_arr[i] = -2
    
    activeset = -1
    i = 0
    while i < n_atoms:
        j = order[i]
        if count[j] > 1:
            next_arr[j] = activeset
            activeset = j
            i += count[j]
        else:
            i += 1
    
    for i in range(n_atoms):
        changed[order[i]] = 1
    
    return activeset


def _refine_partitions(
    atoms: list[_CanonAtom],
    ftor: Callable[[int, int], int],
    order: list[int],
    count: list[int],
    activeset: int,
    next_arr: list[int],
    changed: list[int],
    touched: list[int],
) -> int:
    """Refine partitions using comparison functor."""
    n_atoms = len(atoms)
    
    while activeset != -1:
        partition = activeset
        activeset = next_arr[partition]
        next_arr[partition] = -2
        
        length = count[partition]
        offset = atoms[partition].index
        
        if length <= 1:
            continue
        
        _hanoi_sort(order, offset, length, count, changed, ftor)
        
        for k in range(length):
            changed[order[offset + k]] = 0
        
        first_idx = order[offset]
        symclass = offset
        
        i = count[first_idx]
        
        while i < length:
            idx = order[offset + i]
            cnt = count[idx]
            
            if cnt > 0:
                symclass = offset + i
            
            atoms[idx].index = symclass
            
            for nbr_idx in atoms[idx].nbr_ids:
                changed[nbr_idx] = 1
            
            i += 1
        
        for i in range(count[first_idx], length):
            idx = order[offset + i]
            for nbr_idx in atoms[idx].nbr_ids:
                touched[atoms[nbr_idx].index] = 1
        
        for ii in range(n_atoms):
            if touched[ii] == 1:
                touched[ii] = 0
                npart = order[ii]
                if count[npart] > 1 and next_arr[npart] == -2:
                    next_arr[npart] = activeset
                    activeset = npart
    
    return activeset


def _break_ties(
    atoms: list[_CanonAtom],
    ftor: _AtomCompareFunctor,
    order: list[int],
    count: list[int],
    activeset: int,
    next_arr: list[int],
    changed: list[int],
    touched: list[int],
) -> None:
    """Break remaining ties by isolating atoms."""
    n_atoms = len(atoms)
    i = 0
    
    while i < n_atoms:
        partition = order[i]
        old_part = atoms[partition].index
        
        while count[partition] > 1:
            length = count[partition]
            offset = atoms[partition].index + length - 1
            
            index = order[offset]
            atoms[index].index = offset
            count[partition] = length - 1
            count[index] = 1
            
            if len(atoms[index].nbr_ids) < 1:
                continue
            
            for nbr_idx in atoms[index].nbr_ids:
                touched[atoms[nbr_idx].index] = 1
                changed[nbr_idx] = 1
            
            for ii in range(n_atoms):
                if touched[ii] == 1:
                    npart = order[ii]
                    if count[npart] > 1 and next_arr[npart] == -2:
                        next_arr[npart] = activeset
                        activeset = npart
                    touched[ii] = 0
            
            activeset = _refine_partitions(
                atoms, ftor, order, count, activeset,
                next_arr, changed, touched
            )
        
        if atoms[partition].index != old_part:
            i -= 1
        
        i += 1


def _rank_mol_atoms(
    atoms: list[_CanonAtom],
    break_ties_flag: bool = True,
    include_chirality: bool = True,
    include_isotopes: bool = True,
) -> list[int]:
    """Main ranking function."""
    n_atoms = len(atoms)
    if n_atoms == 0:
        return []
    
    order = list(range(n_atoms))
    count = [0] * n_atoms
    next_arr = [-2] * n_atoms
    changed = [1] * n_atoms
    touched = [0] * n_atoms
    
    ftor = _AtomCompareFunctor(
        atoms,
        use_isotopes=include_isotopes,
        use_chirality=include_chirality,
    )
    
    _create_single_partition(n_atoms, order, count, atoms)
    
    ftor.use_nbrs = True
    
    activeset = _activate_partitions(n_atoms, order, count, next_arr, changed)
    activeset = _refine_partitions(
        atoms, ftor, order, count, activeset,
        next_arr, changed, touched
    )
    
    # Check for ties after initial refinement
    ties = any(c == 0 for c in count)
    
    # Apply special chirality functor if there are ties and chirality is enabled
    if include_chirality and ties:
        scftor = _SpecialChiralityAtomCompareFunctor(atoms)
        activeset = _activate_partitions(n_atoms, order, count, next_arr, changed)
        activeset = _refine_partitions(
            atoms, scftor, order, count, activeset,
            next_arr, changed, touched
        )
    
    if break_ties_flag:
        _break_ties(
            atoms, ftor, order, count, activeset,
            next_arr, changed, touched
        )
    
    result = [0] * n_atoms
    for i in range(n_atoms):
        result[atoms[order[i]].atom_idx] = i
    
    return result


# =============================================================================
# Default valence lookup
# =============================================================================

_DEFAULT_VALENCES: Final[dict[int, int]] = {
    1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1,
    15: 3, 16: 2, 17: 1, 35: 1, 53: 1,
}


def _get_default_valence(atomic_num: int) -> int:
    """Get default valence for an atomic number."""
    return _DEFAULT_VALENCES.get(atomic_num, 4)


# =============================================================================
# Public API
# =============================================================================

class Canonicalizer:
    """Compute canonical atom ordering for a molecule.
    
    This class implements canonical ranking using
    partition refinement with the HanoiSort algorithm.
    
    Example:
        >>> from chiralipy import parse
        >>> mol = parse("C(C)CC")
        >>> canonicalizer = Canonicalizer(mol)
        >>> ranks = canonicalizer.compute_ranks()
        >>> # Atom at index with lowest rank should come first
    """
    
    def __init__(self, mol: Molecule) -> None:
        """Initialize canonicalizer.
        
        Args:
            mol: Molecule to canonicalize.
        """
        self._mol = mol
        self._atoms: list[_CanonAtom] | None = None
    
    def compute_ranks(
        self,
        break_ties: bool = True,
        include_chirality: bool = True,
        include_isotopes: bool = True,
    ) -> list[int]:
        """Compute canonical ranks for all atoms.
        
        Args:
            break_ties: Whether to break remaining ties (default True).
            include_chirality: Consider chirality in ranking.
            include_isotopes: Consider isotopes in ranking.
        
        Returns:
            List of ranks indexed by atom index.
            Lower rank = earlier in canonical ordering.
        """
        self._build_canon_atoms()
        
        return _rank_mol_atoms(
            self._atoms,
            break_ties_flag=break_ties,
            include_chirality=include_chirality,
            include_isotopes=include_isotopes,
        )
    
    def _build_canon_atoms(self) -> None:
        """Build internal atom representations."""
        mol = self._mol
        n = len(mol.atoms)
        
        ring_atoms, _ = _find_ring_atoms_and_bonds_fast(mol)
        _, ring_sizes_map = get_ring_info(mol, _ring_atoms=ring_atoms)
        
        self._atoms = []
        
        for atom in mol.atoms:
            nbr_ids = list(atom.neighbors(mol))
            
            # Compute total Hs
            total_hs = atom.explicit_hydrogens
            if atom.atomic_number in _DEFAULT_VALENCES:
                default_val = _get_default_valence(atom.atomic_number)
                bond_order_sum = 0.0
                for bond in atom.get_bonds(mol):
                    if bond.is_aromatic:
                        bond_order_sum += 1.5
                    else:
                        bond_order_sum += bond.order
                implicit_hs = max(
                    0,
                    default_val - int(round(bond_order_sum)) - abs(atom.charge) - atom.explicit_hydrogens
                )
                total_hs = atom.explicit_hydrogens + implicit_hs
            
            # Determine chirality tag:
            # 0 = none, 1 = CW (@@), 2 = CCW (@)
            chirality_tag = 0
            if atom.chirality == "@":
                chirality_tag = 2  # @ = CCW
            elif atom.chirality == "@@":
                chirality_tag = 1  # @@ = CW
            
            min_ring = 0
            if atom.idx in ring_sizes_map and ring_sizes_map[atom.idx]:
                min_ring = min(ring_sizes_map[atom.idx])
            
            canon_atom = _CanonAtom(
                atom_idx=atom.idx,
                degree=len(atom.bond_indices),
                atomic_num=atom.atomic_number,
                isotope=atom.isotope or 0,
                formal_charge=atom.charge,
                total_num_hs=total_hs,
                has_chirality=(atom.chirality is not None),
                is_aromatic=atom.is_aromatic,
                is_in_ring=(atom.idx in ring_atoms),
                min_ring_size=min_ring,
                chirality_tag=chirality_tag,
                nbr_ids=nbr_ids,
            )
            self._atoms.append(canon_atom)
        
        # Build bond lists
        for canon_atom in self._atoms:
            atom = mol.atoms[canon_atom.atom_idx]
            for bond in atom.get_bonds(mol):
                nbr = bond.other_atom(canon_atom.atom_idx)
                
                bond_type = BondOrder.AROMATIC if bond.is_aromatic else bond.order
                
                bond_stereo = 0
                if bond.stereo == "/":
                    bond_stereo = 1
                elif bond.stereo == "\\":
                    bond_stereo = 2
                
                bh = _BondHolder(
                    bond_type=bond_type,
                    bond_stereo=bond_stereo,
                    nbr_sym_class=0,
                    nbr_idx=nbr,
                    bond_idx=bond.idx,
                )
                canon_atom.bonds.append(bh)


def canonical_ranks(mol: Molecule) -> list[int]:
    """Compute canonical ranks for a molecule.
    
    This is a convenience function that creates a Canonicalizer
    and computes ranks with default settings.
    
    Args:
        mol: Molecule to rank.
    
    Returns:
        List of ranks indexed by atom index.
    
    Example:
        >>> mol = parse("C(C)CC")
        >>> ranks = canonical_ranks(mol)
        >>> min_rank_atom = ranks.index(min(ranks))
    """
    return Canonicalizer(mol).compute_ranks()
