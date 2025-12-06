"""
BRICS decomposition algorithm.

Implementation of the BRICS (Breaking of Retrosynthetically Interesting
Chemical Substructures) algorithm based on:

    Degen et al. "On the Art of Compiling and Using 'Drug-Like' 
    Chemical Fragment Spaces" ChemMedChem 2008, 3, 1503-1507.
    DOI: 10.1002/cmdc.200800178

This module provides functions for fragmenting molecules at strategic
bonds commonly found in medicinal chemistry, and for recombining
fragments to generate new molecules.

The BRICS algorithm identifies 16 different chemical environments (L1-L16)
and defines cleavage rules for bonds between specific environment pairs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterator

from chiralipy.parser import parse
from chiralipy.writer import to_smiles, get_output_chirality, count_swaps_to_interconvert
from chiralipy.types import Atom, Bond, Molecule
from chiralipy.rings import _find_ring_atoms_and_bonds_fast


ENVIRONS: dict[str, str] = {
    # L1: Acyl carbon attached to heteroatom
    'L1': '[C;D3]([#0,#6,#7,#8])(=O)',
    # L3: Ether/ester oxygen - single bond, not in ring
    'L3': '[O;D2]-;!@[#0,#6,#1]',
    # L4: Aliphatic carbon, not degree 1, not double bonded
    'L4': '[C;!D1;!$(C=*)]-;!@[#6]',
    # L5: Amine nitrogen - complex exclusions for amides etc
    'L5': '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]',
    # L6: Carbonyl carbon not in ring
    'L6': '[C;D3;!R](=O)-;!@[#0,#6,#7,#8]',
    # L7a: Olefin carbon (for C=C cleavage)
    'L7a': '[C;D2,D3]-[#6]',
    # L7b: Olefin carbon (paired with L7a)
    'L7b': '[C;D2,D3]-[#6]',
    # L8: Aliphatic carbon, not in ring, not degree 1
    'L8': '[C;!R;!D1;!$(C!-*)]',
    # L9: Aromatic nitrogen in ring (pyridine-like)
    'L9': '[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]',
    # L10: Lactam nitrogen in ring
    'L10': '[N;R;$(N(@C(=O))@[C,N,O,S])]',
    # L11: Thioether sulfur
    'L11': '[S;D2](-;!@[#0,#6])',
    # L12: Sulfonyl sulfur
    'L12': '[S;D4]([#6,#0])(=O)(=O)',
    # L13: Ring carbon attached to heteroatom via ring bonds
    'L13': '[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]',
    # L14: Aromatic carbon adjacent to heteroatom
    'L14': '[c;$(c(:[c,n,o,s]):[n,o,s])]',
    # L15: Ring carbon attached to carbons
    'L15': '[C;$(C(-;@C)-;@C)]',
    # L16: Aromatic carbon in benzene-like ring
    'L16': '[c;$(c(:c):c)]',
}

ENVIRONS['L14b'] = ENVIRONS['L14']
ENVIRONS['L16b'] = ENVIRONS['L16']

REACTION_DEFS: list[list[tuple[str, str, str]]] = [
    [('1', '3', '-'), ('1', '5', '-'), ('1', '10', '-')],
    [('3', '4', '-'), ('3', '13', '-'), ('3', '14', '-'), ('3', '15', '-'), ('3', '16', '-')],
    [('4', '5', '-'), ('4', '11', '-')],
    [('5', '12', '-'), ('5', '14', '-'), ('5', '16', '-'), ('5', '13', '-'), ('5', '15', '-')],
    [('6', '13', '-'), ('6', '14', '-'), ('6', '15', '-'), ('6', '16', '-')],
    [('7a', '7b', '=')],
    [('8', '9', '-'), ('8', '10', '-'), ('8', '13', '-'), ('8', '14', '-'), ('8', '15', '-'), ('8', '16', '-')],
    [('9', '13', '-'), ('9', '14', '-'), ('9', '15', '-'), ('9', '16', '-')],
    [('10', '13', '-'), ('10', '14', '-'), ('10', '15', '-'), ('10', '16', '-')],
    [('11', '13', '-'), ('11', '14', '-'), ('11', '15', '-'), ('11', '16', '-')],
    [('13', '14', '-'), ('13', '15', '-'), ('13', '16', '-')],
    [('14', '14', '-'), ('14', '15', '-'), ('14', '16', '-')],
    [('15', '16', '-')],
    [('16', '16', '-')],
]

BRICS_RULES: list[tuple[str, str, int]] = []
for rule_group in REACTION_DEFS:
    for label1, label2, bond_char in rule_group:
        bond_order = 2 if bond_char == '=' else 1
        l1 = f'L{label1}' if not label1.startswith('L') else label1
        l2 = f'L{label2}' if not label2.startswith('L') else label2
        BRICS_RULES.append((l1, l2, bond_order))

# Pre-group BRICS rules by expected bond order for faster lookup
BRICS_RULES_BY_ORDER: dict[int, list[tuple[str, str]]] = {1: [], 2: []}
for label1, label2, order in BRICS_RULES:
    BRICS_RULES_BY_ORDER[order].append((label1, label2))

ATOM_ENVIRONS = ENVIRONS

_COMPILED_PATTERNS: dict[str, Molecule | None] = {}
_PATTERN_ADJS: dict[str, dict[int, list[tuple[int, int]]]] = {}
_PATTERNS_BY_SYMBOL: dict[str, list[str]] | None = None
_PATTERNS_WILDCARD: list[str] | None = None


def _build_pattern_adj(pattern: Molecule) -> dict[int, list[tuple[int, int]]]:
    """Build pattern adjacency list."""
    adj: dict[int, list[tuple[int, int]]] = {
        i: [] for i in range(pattern.num_atoms)
    }
    for bond in pattern.bonds:
        adj[bond.atom1_idx].append((bond.atom2_idx, bond.idx))
        adj[bond.atom2_idx].append((bond.atom1_idx, bond.idx))
    return adj


def _get_pattern(label: str) -> Molecule | None:
    """Get compiled SMARTS pattern for a label."""
    if label not in _COMPILED_PATTERNS:
        smarts = ENVIRONS.get(label, '')
        if smarts:
            try:
                # Use perceive_aromaticity=False for SMARTS patterns
                # to preserve explicit aromaticity markers
                _COMPILED_PATTERNS[label] = parse(smarts, perceive_aromaticity=False)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to parse BRICS pattern {label}: {smarts} - {e}")
                _COMPILED_PATTERNS[label] = None
        else:
            _COMPILED_PATTERNS[label] = None
    return _COMPILED_PATTERNS[label]


def _ensure_pattern_groups():
    """Initialize pattern groups for fast lookup."""
    global _PATTERNS_BY_SYMBOL, _PATTERNS_WILDCARD
    if _PATTERNS_BY_SYMBOL is not None:
        return
        
    _PATTERNS_BY_SYMBOL = {}
    _PATTERNS_WILDCARD = []
    
    for label in ENVIRONS:
        pattern = _get_pattern(label)
        if pattern is None or pattern.num_atoms == 0:
            continue
            
        # Precompute adjacency
        if label not in _PATTERN_ADJS:
            _PATTERN_ADJS[label] = _build_pattern_adj(pattern)

        first_atom = pattern.atoms[0]
        if first_atom.is_wildcard or first_atom.symbol == '*':
            _PATTERNS_WILDCARD.append(label)
        elif first_atom.atom_list:
             _PATTERNS_WILDCARD.append(label)
        else:
            symbol = first_atom.symbol.upper()
            if symbol not in _PATTERNS_BY_SYMBOL:
                _PATTERNS_BY_SYMBOL[symbol] = []
            _PATTERNS_BY_SYMBOL[symbol].append(label)


def find_brics_bonds(
    mol: Molecule,
    bond_orders: set[int] | None = None,
) -> Iterator[tuple[tuple[int, int], tuple[str, str]]]:
    """Find bonds in a molecule that BRICS would cleave.
    
    Args:
        mol: Molecule to analyze.
        bond_orders: Set of bond orders to consider for cleavage.
            If None, all bond orders are considered (default BRICS behavior).
            Use {1} for single bonds only, {2} for double bonds only.
            BRICS primarily cleaves single bonds; double bond cleavage is rare
            (only L7a-L7b olefin C=C bonds).
    
    Yields:
        Tuples of ((atom1_idx, atom2_idx), (label1, label2)) for each cleavable bond.
    
    Example:
        >>> mol = parse("CCCOCC")
        >>> bonds = list(find_brics_bonds(mol))
        >>> # Returns bonds that can be cleaved according to BRICS rules
        >>> bonds_single = list(find_brics_bonds(mol, bond_orders={1}))
        >>> # Returns only single bonds that can be cleaved
    """
    from chiralipy.match import RingInfo, match_at_root

    # Optimized RingInfo for BRICS (only needs membership, not counts/sizes)
    # This avoids the expensive get_ring_info() call which enumerates all rings
    ring_atoms, ring_bonds = _find_ring_atoms_and_bonds_fast(mol)
    ring_count = {i: 1 if i in ring_atoms else 0 for i in range(mol.num_atoms)}
    ring_sizes = {i: set() for i in range(mol.num_atoms)}
    ring_info = RingInfo(ring_count, ring_sizes, ring_bonds)

    _ensure_pattern_groups()
    
    env_matches: dict[str, set[int]] = {label: set() for label in ENVIRONS}
    
    # Iterate over atoms and check relevant patterns
    for atom in mol.atoms:
        symbol = atom.symbol.upper()
        candidates = _PATTERNS_BY_SYMBOL.get(symbol)
        
        if candidates:
            if _PATTERNS_WILDCARD:
                candidates = candidates + _PATTERNS_WILDCARD
        elif _PATTERNS_WILDCARD:
            candidates = _PATTERNS_WILDCARD
        else:
            continue
        
        for label in candidates:
            pattern = _get_pattern(label)
            # Pass precomputed adjacency to avoid rebuilding it in match_at_root
            if pattern and match_at_root(mol, atom.idx, pattern, ring_info, pattern_adj=_PATTERN_ADJS.get(label)):
                env_matches[label].add(atom.idx)
    
    bonds_found: set[tuple[int, int]] = set()
    ring_bonds = ring_info.ring_bonds
    
    # Pre-compute label numeric parts to avoid repeated string operations
    label_nums: dict[str, str] = {label: ''.join(c for c in label if c.isdigit()) for label in ENVIRONS}
    
    for bond in mol.bonds:
        atom1_idx = bond.atom1_idx
        atom2_idx = bond.atom2_idx
        bond_key = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
        
        if bond_key in ring_bonds:
            continue
        
        if bond_key in bonds_found:
            continue
        
        # Filter by bond order if specified
        if bond_orders is not None and bond.order not in bond_orders:
            continue
        
        # Get rules for this bond order only (avoids checking order for every rule)
        rules_for_order = BRICS_RULES_BY_ORDER.get(bond.order)
        if not rules_for_order:
            continue
        
        for label1, label2 in rules_for_order:
            # Direct access to pre-initialized sets (no .get() allocation)
            set1 = env_matches[label1]
            set2 = env_matches[label2]
            
            if atom1_idx in set1 and atom2_idx in set2:
                bonds_found.add(bond_key)
                yield ((atom1_idx, atom2_idx), (label_nums[label1], label_nums[label2]))
                break
            elif atom2_idx in set1 and atom1_idx in set2:
                bonds_found.add(bond_key)
                yield ((atom2_idx, atom1_idx), (label_nums[label1], label_nums[label2]))
                break


def break_brics_bonds(
    mol: Molecule,
    bonds: list[tuple[tuple[int, int], tuple[str, str]]] | None = None,
) -> Molecule:
    """Break BRICS bonds in a molecule and add dummy atom labels.
    
    Args:
        mol: Molecule to fragment.
        bonds: Specific bonds to break. If None, finds all BRICS bonds.
    
    Returns:
        New molecule with broken bonds and dummy atoms ([N*] labels).
    
    Example:
        >>> mol = parse("CCCOCC")
        >>> fragmented = break_brics_bonds(mol)
        >>> # Returns molecule with broken bonds marked with [3*], [4*] etc.
    """
    if bonds is None:
        bonds = list(find_brics_bonds(mol))
    
    if not bonds:
        return deepcopy(mol)
    
    # Pre-compute the intended output chirality for each chiral atom.
    # This is the chirality symbol that should appear in the output SMILES.
    # We compute this BEFORE modifying the molecule structure, because the
    # canonical traversal order depends on the full molecular structure.
    # After fragmenting, the traversal order changes, so we store this info
    # to ensure fragments output the correct chirality.
    #
    # Use get_output_chirality which runs the full writer algorithm to
    # correctly determine output chirality including ring handling.
    intended_output_chirality = get_output_chirality(mol)
    
    new_atoms: list[Atom] = []
    new_bonds: list[Bond] = []
    
    for atom in mol.atoms:
        # Copy atom data and add intended output chirality if computed
        atom_data = dict(atom.data) if atom.data else {}
        if atom.idx in intended_output_chirality:
            atom_data['_intended_output_chirality'] = intended_output_chirality[atom.idx]
        
        new_atom = Atom(
            idx=atom.idx,
            symbol=atom.symbol,
            charge=atom.charge,
            explicit_hydrogens=atom.explicit_hydrogens,
            is_aromatic=atom.is_aromatic,
            isotope=atom.isotope,
            chirality=atom.chirality,
            atom_class=atom.atom_class,
            data=atom_data if atom_data else None,
            _was_first_in_component=atom._was_first_in_component,
        )
        # Copy _smiles_neighbor_bonds for chirality preservation
        if atom._smiles_neighbor_bonds is not None:
            new_atom._smiles_neighbor_bonds = list(atom._smiles_neighbor_bonds)
        new_atoms.append(new_atom)
    
    bonds_to_break: dict[tuple[int, int], tuple[str, str]] = {}
    for (atom1, atom2), (label1, label2) in bonds:
        key = (min(atom1, atom2), max(atom1, atom2))
        bonds_to_break[key] = (label1, label2) if atom1 < atom2 else (label2, label1)
    
    next_atom_idx = len(new_atoms)
    dummy_bonds: list[tuple[int, int, str, int, int]] = []  # parent_idx, dummy_idx, label, order, original_bond_idx
    
    # Track mapping from old bond idx -> new bond idx
    old_bond_to_new: dict[int, int] = {}
    
    for bond in mol.bonds:
        key = (min(bond.atom1_idx, bond.atom2_idx), max(bond.atom1_idx, bond.atom2_idx))
        
        if key in bonds_to_break:
            labels = bonds_to_break[key]
            
            if bond.atom1_idx < bond.atom2_idx:
                label1, label2 = labels
            else:
                label2, label1 = labels
            
            dummy1 = Atom(
                idx=next_atom_idx,
                symbol='*',
                charge=0,
                explicit_hydrogens=0,
                is_aromatic=False,
                isotope=int(label1) if label1.isdigit() else None,
                atom_class=bond.atom1_idx,  # Store original atom index
            )
            new_atoms.append(dummy1)
            # Store original bond idx to map _smiles_neighbor_bonds
            dummy_bonds.append((bond.atom1_idx, next_atom_idx, label1, bond.order, bond.idx))
            next_atom_idx += 1
            
            dummy2 = Atom(
                idx=next_atom_idx,
                symbol='*',
                charge=0,
                explicit_hydrogens=0,
                is_aromatic=False,
                isotope=int(label2) if label2.isdigit() else None,
                atom_class=bond.atom2_idx,  # Store original atom index
            )
            new_atoms.append(dummy2)
            dummy_bonds.append((bond.atom2_idx, next_atom_idx, label2, bond.order, bond.idx))
            next_atom_idx += 1
        else:
            new_bond_idx = len(new_bonds)
            new_bond = Bond(
                idx=new_bond_idx,
                atom1_idx=bond.atom1_idx,
                atom2_idx=bond.atom2_idx,
                order=bond.order,
                is_aromatic=bond.is_aromatic,
                stereo=bond.stereo,
            )
            new_bonds.append(new_bond)
            old_bond_to_new[bond.idx] = new_bond_idx
    
    # Track broken bond -> new dummy bond for each atom
    # broken_bond_replacement[atom_idx][old_bond_idx] = new_bond_idx
    broken_bond_replacement: dict[int, dict[int, int]] = {}
    
    for parent_idx, dummy_idx, label, order, orig_bond_idx in dummy_bonds:
        new_bond_idx = len(new_bonds)
        new_bond = Bond(
            idx=new_bond_idx,
            atom1_idx=parent_idx,
            atom2_idx=dummy_idx,
            order=1,  # Always use single bond for dummy atoms (BRICS convention)
            is_aromatic=False,
        )
        new_bonds.append(new_bond)
        
        if parent_idx not in broken_bond_replacement:
            broken_bond_replacement[parent_idx] = {}
        broken_bond_replacement[parent_idx][orig_bond_idx] = new_bond_idx
    
    new_mol = Molecule()
    
    for atom in new_atoms:
        atom.bond_indices = []
        new_mol.atoms.append(atom)
    
    for bond in new_bonds:
        new_mol.bonds.append(bond)
        new_mol.atoms[bond.atom1_idx].bond_indices.append(bond.idx)
        new_mol.atoms[bond.atom2_idx].bond_indices.append(bond.idx)
    
    # Update _smiles_neighbor_bonds and potentially invert chirality
    # 
    # Approach for chirality during fragmentation:
    # When a fragment is created, the dummy atom (replacing the broken bond target)
    # appears at position 0 in the atom list, so its bond to the chiral center
    # comes FIRST in the neighbor list. This changes the neighbor order relative
    # to the original molecule, and we need to invert chirality if the permutation
    # has odd parity.
    #
    # Algorithm:
    # 1. For each broken bond at a chiral atom, determine the position of the 
    #    broken bond in the original neighbor order
    # 2. In the fragment, the dummy bond will be at position 0 (first)
    # 3. Calculate the permutation parity and invert chirality if odd
    for atom_idx, atom in enumerate(new_mol.atoms):
        if atom_idx >= len(mol.atoms):
            continue  # Skip dummy atoms
        
        orig_atom = mol.atoms[atom_idx]
        if orig_atom._smiles_neighbor_bonds is not None:
            # First pass: collect non-broken bonds in their original order
            new_neighbor_bonds = []
            dummy_bonds_to_prepend = []  # Changed: prepend instead of append
            
            for old_bond_idx in orig_atom._smiles_neighbor_bonds:
                # Check if this bond was broken and replaced
                if atom_idx in broken_bond_replacement and old_bond_idx in broken_bond_replacement[atom_idx]:
                    # Collect dummy bonds to prepend at the beginning
                    dummy_bonds_to_prepend.append(broken_bond_replacement[atom_idx][old_bond_idx])
                elif old_bond_idx in old_bond_to_new:
                    new_neighbor_bonds.append(old_bond_to_new[old_bond_idx])
                # Else the bond was removed (shouldn't happen for stereochem atoms)
            
            # Prepend dummy bonds at the beginning (matching fragment neighbor order)
            new_neighbor_bonds = dummy_bonds_to_prepend + new_neighbor_bonds
            
            if new_neighbor_bonds:
                atom._smiles_neighbor_bonds = new_neighbor_bonds
        
        # Handle chirality inversion
        # Only process atoms with tetrahedral chirality
        if orig_atom.chirality in ('@', '@@') and atom_idx in broken_bond_replacement:
            # In fragments, the dummy atom is at position 0, so its bond
            # comes FIRST in the chiral atom's neighbor list. The original neighbor
            # that was replaced is now represented by the dummy at the front.
            #
            # Original: [N, C(broken), C, C] at positions [0, 1, 2, 3]
            # Fragment: [*, N, C, C] where * replaces C(broken) but is at position 0
            #
            # The permutation moves the element from its original position to position 0,
            # shifting other elements. We need to count the swaps.
            
            if orig_atom._smiles_neighbor_bonds is not None:
                original_bond_order = list(orig_atom._smiles_neighbor_bonds)
            else:
                original_bond_order = list(orig_atom.bond_indices)
            
            broken_bond_idxs = set(broken_bond_replacement[atom_idx].keys())
            
            # Find the position(s) of broken bonds in original order
            broken_positions = []
            non_broken_positions = []
            for pos, bond_idx in enumerate(original_bond_order):
                if bond_idx in broken_bond_idxs:
                    broken_positions.append(pos)
                else:
                    non_broken_positions.append(pos)
            
            # New order: [broken positions (now at front)] + [non-broken positions]
            # This represents [*, ...remaining...] in the fragment
            new_position_order = broken_positions + non_broken_positions
            
            # Count swaps from new_position_order to [0, 1, 2, 3, ...]
            if len(new_position_order) >= 3:
                original_position_order = list(range(len(original_bond_order)))
                try:
                    n_swaps = count_swaps_to_interconvert(new_position_order, original_position_order)
                    if n_swaps % 2 == 1:
                        # Invert chirality: @ <-> @@
                        if atom.chirality == '@':
                            atom.chirality = '@@'
                        elif atom.chirality == '@@':
                            atom.chirality = '@'
                except ValueError:
                    pass  # Lists don't contain same elements - shouldn't happen
    
    return new_mol


def _get_fragments(mol: Molecule) -> list[Molecule]:
    """Split a molecule into disconnected fragments.
    
    Args:
        mol: Molecule to split.
    
    Returns:
        List of fragment molecules.
    """
    num_atoms = mol.num_atoms
    if num_atoms == 0:
        return []
    
    # Use list[bool] instead of set[int] for O(1) access without hashing
    visited: list[bool] = [False] * num_atoms
    components: list[set[int]] = []
    atoms = mol.atoms  # Local reference for faster access
    bonds = mol.bonds
    
    def dfs(start: int) -> set[int]:
        component: set[int] = set()
        stack = [start]
        while stack:
            atom_idx = stack.pop()
            if visited[atom_idx]:
                continue
            visited[atom_idx] = True
            component.add(atom_idx)
            
            atom = atoms[atom_idx]
            for bond_idx in atom.bond_indices:
                bond = bonds[bond_idx]
                neighbor = bond.other_atom(atom_idx)
                if not visited[neighbor]:
                    stack.append(neighbor)
        return component
    
    for i in range(num_atoms):
        if not visited[i]:
            component = dfs(i)
            components.append(component)
    
    if len(components) <= 1:
        return [deepcopy(mol)]
    
    fragments: list[Molecule] = []
    
    for component in components:
        old_to_new: dict[int, int] = {}
        for new_idx, old_idx in enumerate(sorted(component)):
            old_to_new[old_idx] = new_idx
        
        frag = Molecule()
        
        # First pass: create atoms
        for old_idx in sorted(component):
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
                data=dict(old_atom.data) if old_atom.data else None,
                _was_first_in_component=old_atom._was_first_in_component,
            )
            new_atom.bond_indices = []
            frag.atoms.append(new_atom)
        
        # Second pass: create bonds and track old->new bond mapping
        old_bond_to_new: dict[int, int] = {}
        added_bonds: set[int] = set()
        for old_idx in component:
            old_atom = mol.atoms[old_idx]
            for bond_idx in old_atom.bond_indices:
                if bond_idx in added_bonds:
                    continue
                
                bond = mol.bonds[bond_idx]
                if bond.atom1_idx in component and bond.atom2_idx in component:
                    new_bond_idx = len(frag.bonds)
                    new_bond = Bond(
                        idx=new_bond_idx,
                        atom1_idx=old_to_new[bond.atom1_idx],
                        atom2_idx=old_to_new[bond.atom2_idx],
                        order=bond.order,
                        is_aromatic=bond.is_aromatic,
                        stereo=bond.stereo,
                    )
                    frag.bonds.append(new_bond)
                    frag.atoms[new_bond.atom1_idx].bond_indices.append(new_bond.idx)
                    frag.atoms[new_bond.atom2_idx].bond_indices.append(new_bond.idx)
                    old_bond_to_new[bond_idx] = new_bond_idx
                    added_bonds.add(bond_idx)
        
        # Third pass: Remap _smiles_neighbor_bonds
        for old_idx in component:
            old_atom = mol.atoms[old_idx]
            new_atom = frag.atoms[old_to_new[old_idx]]
            
            if old_atom._smiles_neighbor_bonds:
                new_neighbors = []
                for bond_idx in old_atom._smiles_neighbor_bonds:
                    if bond_idx in old_bond_to_new:
                        new_neighbors.append(old_bond_to_new[bond_idx])
                new_atom._smiles_neighbor_bonds = new_neighbors
        
        # Fourth pass: Reorder bond_indices for chiral atoms
        # Place bonds to dummy atoms FIRST in the neighbor list. Since we use
        # bond_indices as the reference for chirality inversion calculations (when
        # _smiles_neighbor_bonds is None), we must match this order.
        for new_atom in frag.atoms:
            if new_atom.chirality in ('@', '@@'):
                # Separate bonds to dummy atoms from bonds to real atoms
                dummy_bonds = []
                real_bonds = []
                for bond_idx in new_atom.bond_indices:
                    bond = frag.bonds[bond_idx]
                    neighbor_idx = bond.other_atom(new_atom.idx)
                    neighbor = frag.atoms[neighbor_idx]
                    if neighbor.symbol == '*':
                        dummy_bonds.append(bond_idx)
                    else:
                        real_bonds.append(bond_idx)
                # Put dummy bonds first, then real bonds
                new_atom.bond_indices = dummy_bonds + real_bonds
        
        fragments.append(frag)
    
    return fragments


def _get_stem_atom_indices(mol: Molecule) -> set[int]:
    """Find indices of real atoms that are bonded to dummy atoms.
    
    Args:
        mol: Molecule with dummy atoms still present.
    
    Returns:
        Set of atom indices (in the current molecule) that are stems.
    """
    stem_indices: set[int] = set()
    for atom in mol.atoms:
        if atom.symbol == '*':
            for bond_idx in atom.bond_indices:
                bond = mol.bonds[bond_idx]
                neighbor_idx = bond.other_atom(atom.idx)
                neighbor = mol.atoms[neighbor_idx]
                if neighbor.symbol != '*':
                    stem_indices.add(neighbor_idx)
    return stem_indices


def _strip_dummy_atoms_and_mark_stems(mol: Molecule) -> Molecule:
    r"""Remove dummy atoms (*) from a molecule and mark stem atoms in data dict.
    
    The bonds to dummy atoms are removed, leaving the attachment point atoms
    with their original connectivity (minus the dummy). Stem atoms are marked
    with data['is_stem']=True so they can be identified after canonicalization.
    
    Bond stereo markers (/ and \\) that are no longer valid after dummy removal
    are also cleared - this happens when the double bond they referred to has
    been cleaved.
    
    Args:
        mol: Molecule to strip dummy atoms from.
    
    Returns:
        New molecule without dummy atoms, with stem atoms marked via data['is_stem'].
    """
    # Find stem atoms (real atoms bonded to dummies)
    stem_old_indices = _get_stem_atom_indices(mol)
    
    # Find indices of real (non-dummy) atoms
    real_atom_indices: list[int] = []
    for atom in mol.atoms:
        if atom.symbol != '*':
            real_atom_indices.append(atom.idx)
    
    if len(real_atom_indices) == mol.num_atoms:
        # No dummy atoms, return a copy
        return deepcopy(mol)
    
    # Create mapping from old indices to new indices
    old_to_new: dict[int, int] = {}
    for new_idx, old_idx in enumerate(real_atom_indices):
        old_to_new[old_idx] = new_idx
    
    real_atoms_set = set(real_atom_indices)
    
    # Find atoms that were bonded to dummy atoms (cleaved bond endpoints)
    # These atoms may have invalid stereo on their remaining bonds
    # Also count how many bonds each atom loses to dummies (for H addition)
    atoms_with_cleaved_bonds: set[int] = set()
    bonds_lost_to_dummies: dict[int, int] = {}  # atom_idx -> count of bonds lost
    for bond in mol.bonds:
        a1, a2 = bond.atom1_idx, bond.atom2_idx
        a1_is_dummy = mol.atoms[a1].symbol == '*'
        a2_is_dummy = mol.atoms[a2].symbol == '*'
        if a1_is_dummy and not a2_is_dummy:
            atoms_with_cleaved_bonds.add(a2)
            bonds_lost_to_dummies[a2] = bonds_lost_to_dummies.get(a2, 0) + bond.order
        elif a2_is_dummy and not a1_is_dummy:
            atoms_with_cleaved_bonds.add(a1)
            bonds_lost_to_dummies[a1] = bonds_lost_to_dummies.get(a1, 0) + bond.order
    
    # Calculate remaining bond order for each real atom (bonds to other real atoms)
    remaining_bond_order: dict[int, int] = {i: 0 for i in real_atom_indices}
    for bond in mol.bonds:
        a1, a2 = bond.atom1_idx, bond.atom2_idx
        if a1 in real_atoms_set and a2 in real_atoms_set:
            order = 1.5 if bond.is_aromatic else bond.order
            remaining_bond_order[a1] += order
            remaining_bond_order[a2] += order
    
    # Build new molecule
    new_mol = Molecule()
    
    for old_idx in real_atom_indices:
        old_atom = mol.atoms[old_idx]
        is_stem = old_idx in stem_old_indices
        # Copy existing data and add is_stem flag
        new_data = dict(old_atom.data) if old_atom.data else {}
        if is_stem:
            new_data['is_stem'] = True
        
        # Clear chirality if this atom was bonded to a dummy
        # (losing a substituent invalidates tetrahedral chirality)
        chirality = old_atom.chirality
        explicit_hydrogens = old_atom.explicit_hydrogens
        if old_idx in atoms_with_cleaved_bonds:
            chirality = None
            
            from chiralipy.elements import get_default_valence, get_atomic_number
            atomic_num = get_atomic_number(old_atom.symbol)
            default_val = get_default_valence(atomic_num)
            remaining_bonds = int(round(remaining_bond_order.get(old_idx, 0)))
            
            if default_val is not None and remaining_bonds > default_val:
                # Hypervalent atom: need explicit H to show the actual valence
                # The atom had more bonds than its default valence, so we need to
                # explicitly show H to indicate its true valence
                lost_bonds = bonds_lost_to_dummies.get(old_idx, 0)
                explicit_hydrogens += lost_bonds
            else:
                # Normal atom: clear explicit H that was only used for chirality
                # Implicit H will be calculated correctly by the writer
                if explicit_hydrogens == 1 and chirality is None:
                    explicit_hydrogens = 0
        
        new_atom = Atom(
            idx=old_to_new[old_idx],
            symbol=old_atom.symbol,
            charge=old_atom.charge,
            explicit_hydrogens=explicit_hydrogens,
            is_aromatic=old_atom.is_aromatic,
            isotope=old_atom.isotope,
            chirality=chirality,
            atom_class=old_atom.atom_class,
            data=new_data if new_data else None,
        )
        new_atom.bond_indices = []
        new_mol.atoms.append(new_atom)
    
    # Add bonds between real atoms only
    added_bonds: set[tuple[int, int]] = set()
    for bond in mol.bonds:
        if bond.atom1_idx in real_atoms_set and bond.atom2_idx in real_atoms_set:
            bond_key = (min(bond.atom1_idx, bond.atom2_idx), max(bond.atom1_idx, bond.atom2_idx))
            if bond_key not in added_bonds:
                # Clear stereo if either endpoint was bonded to a dummy
                # (indicating the double bond was cleaved)
                stereo = bond.stereo
                if stereo and (bond.atom1_idx in atoms_with_cleaved_bonds or 
                               bond.atom2_idx in atoms_with_cleaved_bonds):
                    stereo = None
                
                new_bond = Bond(
                    idx=len(new_mol.bonds),
                    atom1_idx=old_to_new[bond.atom1_idx],
                    atom2_idx=old_to_new[bond.atom2_idx],
                    order=bond.order,
                    is_aromatic=bond.is_aromatic,
                    stereo=stereo,
                )
                new_mol.bonds.append(new_bond)
                new_mol.atoms[new_bond.atom1_idx].bond_indices.append(new_bond.idx)
                new_mol.atoms[new_bond.atom2_idx].bond_indices.append(new_bond.idx)
                added_bonds.add(bond_key)
    
    return new_mol


def _to_smiles_with_stems(
    mol: Molecule,
    include_stereo: bool = True,
) -> tuple[str, set[int]]:
    """Convert molecule to canonical SMILES and extract stem indices.
    
    Stem atoms must be marked with data['is_stem']=True (via _strip_dummy_atoms_and_mark_stems).
    Returns the canonical SMILES and the set of atom indices in the canonical SMILES that are stems.
    
    Args:
        mol: Molecule with stem atoms marked via data['is_stem'].
        include_stereo: If True, include bond stereochemistry (/, \\) in SMILES.
    
    Returns:
        Tuple of (canonical_smiles, set of stem indices in the canonical SMILES).
    """
    from chiralipy.canon import canonical_ranks
    
    # Get canonical ranks
    ranks = canonical_ranks(mol)
    
    # Generate canonical SMILES
    smiles = to_smiles(mol, ranks, include_stereo=include_stereo)
    
    # Parse the canonical SMILES to get atoms in canonical order
    parsed = parse(smiles)
    
    # The parsed molecule has atoms in SMILES traversal order.
    # We need to match atoms from original mol to parsed mol to transfer stem info.
    # Since both are canonical representations of the same molecule,
    # we can use the canonical ranks to create a mapping.
    
    # For each atom in parsed mol, find the corresponding atom in original mol
    # by matching their canonical positions
    
    # Create rank -> original_idx mapping
    rank_to_orig: dict[int, int] = {ranks[i]: i for i in range(len(mol.atoms))}
    
    # Get ranks for parsed molecule
    parsed_ranks = canonical_ranks(parsed)
    
    # Find stems in parsed molecule by matching ranks
    stem_indices: set[int] = set()
    for parsed_idx in range(len(parsed.atoms)):
        parsed_rank = parsed_ranks[parsed_idx]
        if parsed_rank in rank_to_orig:
            orig_idx = rank_to_orig[parsed_rank]
            orig_atom = mol.atoms[orig_idx]
            if orig_atom.data and orig_atom.data.get('is_stem'):
                stem_indices.add(parsed_idx)
    
    return smiles, stem_indices


def brics_decompose(
    mol: Molecule | str,
    min_fragment_size: int = 1,
    keep_non_leaf_nodes: bool = False,
    single_pass: bool = False,
    return_mols: bool = False,
    return_stems: bool = False,
    bond_orders: set[int] | None = None,
    include_atom_class: bool = False,
    include_stereo: bool = True,
) -> set[str] | list[Molecule] | dict[str, set[int]] | list[tuple[Molecule, set[int]]]:
    """Perform BRICS decomposition on a molecule.
    
    Identifies and breaks all BRICS bonds in the molecule.
    
    Args:
        mol: Molecule to decompose, or SMILES string.
        min_fragment_size: Minimum number of heavy atoms in a fragment.
        keep_non_leaf_nodes: If True, include the original molecule in output.
        single_pass: If True, only break bonds once. If False (default),
            recursively decompose fragments until no more BRICS bonds remain.
        return_mols: If True, return Molecule objects instead of SMILES.
        return_stems: If True, return fragment-local atom indices at cleavage points.
            Dummy atoms are removed from the fragments, leaving only real atoms.
            The indices refer to atom positions within each fragment's SMILES.
            When combined with return_mols=False, returns dict mapping SMILES to atom index sets.
            When combined with return_mols=True, returns list of (Molecule, atom_indices) tuples.
        bond_orders: Set of bond orders to consider for cleavage.
            If None, all bond orders are considered (default BRICS behavior).
            Use {1} for single bonds only, {2} for double bonds only.
            BRICS primarily cleaves single bonds; double bond cleavage is rare
            (only L7a-L7b olefin C=C bonds).
        include_atom_class: If True, include atom class annotations (:N) in SMILES.
            Default is False for cleaner output.
        include_stereo: If True (default), include bond stereochemistry (/, \\) in SMILES.
            Set to False to exclude stereochemistry markers.
    
    Returns:
        - set[str]: SMILES strings (default)
        - list[Molecule]: If return_mols=True
        - dict[str, set[int]]: If return_stems=True (SMILES -> fragment-local atom indices)
        - list[tuple[Molecule, set[int]]]: If return_mols=True and return_stems=True
    
    Example:
        >>> mol = parse("CCOc1ccc(CC)cc1")
        >>> frags = brics_decompose(mol)
        >>> sorted(frags)
        ['[16*]c1ccc([16*])cc1', '[3*]O[3*]', '[4*]CC', '[8*]CC']
        >>> frags_with_stems = brics_decompose(mol, return_stems=True)
        >>> # Returns atom indices where cleavages occurred
        >>> frags_single = brics_decompose(mol, bond_orders={1})
        >>> # Only cleave single bonds
    """
    # Handle string input
    if isinstance(mol, str):
        mol = parse(mol)
    
    mol_smi = to_smiles(mol, include_atom_class=include_atom_class, include_stereo=include_stereo)
    
    # Find all cleavable bonds
    bonds = list(find_brics_bonds(mol, bond_orders=bond_orders))
    
    if not bonds:
        if return_mols and return_stems:
            return [(mol, set())]
        if return_mols:
            return [mol]
        if return_stems:
            return {mol_smi: set()}
        return {mol_smi}
    
    # Break all bonds simultaneously
    fragmented = break_brics_bonds(mol, bonds)
    fragments = _get_fragments(fragmented)
    
    # Filter fragments by size
    valid_fragments = []
    for frag in fragments:
        heavy_count = sum(1 for a in frag.atoms if a.symbol != '*')
        if heavy_count >= min_fragment_size:
            valid_fragments.append(frag)
    
    if single_pass:
        # Single pass: return fragments from one round of decomposition
        if keep_non_leaf_nodes:
            all_frags = [mol] + valid_fragments
        else:
            all_frags = valid_fragments
        
        if return_mols and return_stems:
            result = []
            for f in all_frags:
                stripped = _strip_dummy_atoms_and_mark_stems(f)
                stems = {a.idx for a in stripped.atoms if a.data and a.data.get('is_stem')}
                result.append((stripped, stems))
            return result
        if return_mols:
            return all_frags
        if return_stems:
            result_dict: dict[str, set[int]] = {}
            for f in all_frags:
                stripped = _strip_dummy_atoms_and_mark_stems(f)
                smiles, stems = _to_smiles_with_stems(stripped, include_stereo=include_stereo)
                result_dict[smiles] = stems
            return result_dict
        return {to_smiles(f, include_atom_class=include_atom_class, include_stereo=include_stereo) for f in all_frags}
    
    # Recursive decomposition: continue until no more bonds can be broken
    final_fragments: list[Molecule] = []
    intermediate_fragments: list[Molecule] = [mol] if keep_non_leaf_nodes else []
    
    to_process = valid_fragments
    while to_process:
        next_round: list[Molecule] = []
        for frag in to_process:
            frag_bonds = list(find_brics_bonds(frag, bond_orders=bond_orders))
            if not frag_bonds:
                # No more bonds to break, this is a leaf fragment
                final_fragments.append(frag)
            else:
                # More bonds to break
                if keep_non_leaf_nodes:
                    intermediate_fragments.append(frag)
                sub_fragmented = break_brics_bonds(frag, frag_bonds)
                sub_fragments = _get_fragments(sub_fragmented)
                for sub_frag in sub_fragments:
                    heavy_count = sum(1 for a in sub_frag.atoms if a.symbol != '*')
                    if heavy_count >= min_fragment_size:
                        next_round.append(sub_frag)
        to_process = next_round
    
    all_fragments = intermediate_fragments + final_fragments
    
    if return_mols and return_stems:
        result = []
        for f in all_fragments:
            stripped = _strip_dummy_atoms_and_mark_stems(f)
            stems = {a.idx for a in stripped.atoms if a.data and a.data.get('is_stem')}
            result.append((stripped, stems))
        return result
    if return_mols:
        return all_fragments
    if return_stems:
        result_dict = {}
        for f in all_fragments:
            stripped = _strip_dummy_atoms_and_mark_stems(f)
            smiles, stems = _to_smiles_with_stems(stripped, include_stereo=include_stereo)
            result_dict[smiles] = stems
        return result_dict
    return {to_smiles(f, include_atom_class=include_atom_class, include_stereo=include_stereo) for f in all_fragments}
