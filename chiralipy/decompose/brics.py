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
from chiralipy.writer import to_smiles
from chiralipy.types import Atom, Bond, Molecule


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
_PATTERNS_BY_SYMBOL: dict[str, list[str]] | None = None
_PATTERNS_WILDCARD: list[str] | None = None


def _get_pattern(label: str) -> Molecule | None:
    """Get compiled SMARTS pattern for a label."""
    if label not in _COMPILED_PATTERNS:
        smarts = ENVIRONS.get(label, '')
        if smarts:
            try:
                _COMPILED_PATTERNS[label] = parse(smarts)
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


def find_brics_bonds(mol: Molecule) -> Iterator[tuple[tuple[int, int], tuple[str, str]]]:
    """Find bonds in a molecule that BRICS would cleave.
    
    Args:
        mol: Molecule to analyze.
    
    Yields:
        Tuples of ((atom1_idx, atom2_idx), (label1, label2)) for each cleavable bond.
    
    Example:
        >>> mol = parse("CCCOCC")
        >>> bonds = list(find_brics_bonds(mol))
        >>> # Returns bonds that can be cleaved according to BRICS rules
    """
    from chiralipy.match import RingInfo, match_at_root

    ring_info = RingInfo.from_molecule(mol)
    _ensure_pattern_groups()
    
    env_matches: dict[str, set[int]] = {label: set() for label in ENVIRONS}
    
    # Iterate over atoms and check relevant patterns
    for atom in mol.atoms:
        symbol = atom.symbol.upper()
        candidates = _PATTERNS_BY_SYMBOL.get(symbol, [])
        if _PATTERNS_WILDCARD:
            candidates = candidates + _PATTERNS_WILDCARD
        
        for label in candidates:
            pattern = _get_pattern(label)
            if pattern and match_at_root(mol, atom.idx, pattern, ring_info):
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
    
    new_atoms: list[Atom] = []
    new_bonds: list[Bond] = []
    
    for atom in mol.atoms:
        new_atom = Atom(
            idx=atom.idx,
            symbol=atom.symbol,
            charge=atom.charge,
            explicit_hydrogens=atom.explicit_hydrogens,
            is_aromatic=atom.is_aromatic,
            isotope=atom.isotope,
            chirality=atom.chirality,
            atom_class=atom.atom_class,
        )
        new_atoms.append(new_atom)
    
    bonds_to_break: dict[tuple[int, int], tuple[str, str]] = {}
    for (atom1, atom2), (label1, label2) in bonds:
        key = (min(atom1, atom2), max(atom1, atom2))
        bonds_to_break[key] = (label1, label2) if atom1 < atom2 else (label2, label1)
    
    next_atom_idx = len(new_atoms)
    dummy_bonds: list[tuple[int, int, str, int]] = []
    
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
            dummy_bonds.append((bond.atom1_idx, next_atom_idx, label1, bond.order))
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
            dummy_bonds.append((bond.atom2_idx, next_atom_idx, label2, bond.order))
            next_atom_idx += 1
        else:
            new_bond = Bond(
                idx=len(new_bonds),
                atom1_idx=bond.atom1_idx,
                atom2_idx=bond.atom2_idx,
                order=bond.order,
                is_aromatic=bond.is_aromatic,
                stereo=bond.stereo,
            )
            new_bonds.append(new_bond)
    
    for parent_idx, dummy_idx, label, order in dummy_bonds:
        new_bond = Bond(
            idx=len(new_bonds),
            atom1_idx=parent_idx,
            atom2_idx=dummy_idx,
            order=order,
            is_aromatic=False,
        )
        new_bonds.append(new_bond)
    
    new_mol = Molecule()
    
    for atom in new_atoms:
        atom.bond_indices = []
        new_mol.atoms.append(atom)
    
    for bond in new_bonds:
        new_mol.bonds.append(bond)
        new_mol.atoms[bond.atom1_idx].bond_indices.append(bond.idx)
        new_mol.atoms[bond.atom2_idx].bond_indices.append(bond.idx)
    
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
            )
            new_atom.bond_indices = []
            frag.atoms.append(new_atom)
        
        added_bonds: set[int] = set()
        for old_idx in component:
            old_atom = mol.atoms[old_idx]
            for bond_idx in old_atom.bond_indices:
                if bond_idx in added_bonds:
                    continue
                
                bond = mol.bonds[bond_idx]
                if bond.atom1_idx in component and bond.atom2_idx in component:
                    new_bond = Bond(
                        idx=len(frag.bonds),
                        atom1_idx=old_to_new[bond.atom1_idx],
                        atom2_idx=old_to_new[bond.atom2_idx],
                        order=bond.order,
                        is_aromatic=bond.is_aromatic,
                        stereo=bond.stereo,
                    )
                    frag.bonds.append(new_bond)
                    frag.atoms[new_bond.atom1_idx].bond_indices.append(new_bond.idx)
                    frag.atoms[new_bond.atom2_idx].bond_indices.append(new_bond.idx)
                    added_bonds.add(bond_idx)
        
        fragments.append(frag)
    
    return fragments


def _get_stems(mol: Molecule) -> set[int]:
    """Extract original atom indices at cleavage points from a molecule.
    
    The atom indices are stored in the atom_class field of dummy atoms
    during bond breaking.
    
    Args:
        mol: Molecule to extract stems from.
    
    Returns:
        Set of original atom indices where cleavages occurred.
    """
    stems: set[int] = set()
    for atom in mol.atoms:
        if atom.symbol == '*' and atom.atom_class is not None:
            stems.add(atom.atom_class)
    return stems


def _strip_dummy_atoms(mol: Molecule) -> Molecule:
    """Remove dummy atoms (*) from a molecule, keeping only real atoms.
    
    The bonds to dummy atoms are removed, leaving the attachment point atoms
    with their original connectivity (minus the dummy).
    
    Args:
        mol: Molecule to strip dummy atoms from.
    
    Returns:
        New molecule without dummy atoms.
    """
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
    
    # Build new molecule
    new_mol = Molecule()
    
    for old_idx in real_atom_indices:
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
    
    # Add bonds between real atoms only
    added_bonds: set[tuple[int, int]] = set()
    for bond in mol.bonds:
        if bond.atom1_idx in real_atoms_set and bond.atom2_idx in real_atoms_set:
            bond_key = (min(bond.atom1_idx, bond.atom2_idx), max(bond.atom1_idx, bond.atom2_idx))
            if bond_key not in added_bonds:
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
                added_bonds.add(bond_key)
    
    return new_mol


def brics_decompose(
    mol: Molecule,
    min_fragment_size: int = 1,
    keep_non_leaf_nodes: bool = False,
    single_pass: bool = False,
    return_mols: bool = False,
    return_stems: bool = False,
) -> set[str] | list[Molecule] | dict[str, set[int]] | list[tuple[Molecule, set[int]]]:
    """Perform BRICS decomposition on a molecule.
    
    Identifies and breaks all BRICS bonds in the molecule.
    
    Args:
        mol: Molecule to decompose.
        min_fragment_size: Minimum number of heavy atoms in a fragment.
        keep_non_leaf_nodes: If True, include the original molecule in output.
        single_pass: If True, only break bonds once. If False (default),
            recursively decompose fragments until no more BRICS bonds remain.
        return_mols: If True, return Molecule objects instead of SMILES.
        return_stems: If True, return original atom indices at cleavage points.
            Dummy atoms are removed from the fragments, leaving only real atoms.
            When combined with return_mols=False, returns dict mapping SMILES to atom index sets.
            When combined with return_mols=True, returns list of (Molecule, atom_indices) tuples.
    
    Returns:
        - set[str]: SMILES strings (default)
        - list[Molecule]: If return_mols=True
        - dict[str, set[int]]: If return_stems=True (SMILES -> original atom indices)
        - list[tuple[Molecule, set[int]]]: If return_mols=True and return_stems=True
    
    Example:
        >>> mol = parse("CCOc1ccc(CC)cc1")
        >>> frags = brics_decompose(mol)
        >>> sorted(frags)
        ['[16*]c1ccc([16*])cc1', '[3*]O[3*]', '[4*]CC', '[8*]CC']
        >>> frags_with_stems = brics_decompose(mol, return_stems=True)
        >>> # Returns atom indices where cleavages occurred
    """
    mol_smi = to_smiles(mol)
    
    # Find all cleavable bonds
    bonds = list(find_brics_bonds(mol))
    
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
            return [(_strip_dummy_atoms(f), _get_stems(f)) for f in all_frags]
        if return_mols:
            return all_frags
        if return_stems:
            return {to_smiles(_strip_dummy_atoms(f)): _get_stems(f) for f in all_frags}
        return {to_smiles(f) for f in all_frags}
    
    # Recursive decomposition: continue until no more bonds can be broken
    final_fragments: list[Molecule] = []
    intermediate_fragments: list[Molecule] = [mol] if keep_non_leaf_nodes else []
    
    to_process = valid_fragments
    while to_process:
        next_round: list[Molecule] = []
        for frag in to_process:
            frag_bonds = list(find_brics_bonds(frag))
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
        return [(_strip_dummy_atoms(f), _get_stems(f)) for f in all_fragments]
    if return_mols:
        return all_fragments
    if return_stems:
        return {to_smiles(_strip_dummy_atoms(f)): _get_stems(f) for f in all_fragments}
    return {to_smiles(f) for f in all_fragments}
