"""
BRICS decomposition algorithm.

Implementation of the BRICS (Breaking of Retrosynthetically Interesting
Chemical Substructures) algorithm from:
Degen et al. ChemMedChem 2008, 3, 1503-7.

This module provides functions for fragmenting molecules at strategic
bonds commonly found in medicinal chemistry, and for recombining
fragments to generate new molecules.

SMARTS patterns and reaction definitions are from RDKit's BRICS implementation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Iterator

from .match import substructure_search
from .parser import parse
from .writer import to_smiles
from .types import Atom, Bond, Molecule
from .rings import find_sssr

if TYPE_CHECKING:
    pass


# RDKit BRICS environment definitions (exact SMARTS patterns)
# These define atom environments that participate in BRICS bond cleavage
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

# Aliases used in RDKit (same patterns, different labels)
ENVIRONS['L14b'] = ENVIRONS['L14']
ENVIRONS['L16b'] = ENVIRONS['L16']

# RDKit BRICS reaction definitions
# Each tuple is (label1, label2, bond_type) where bond_type is '-' for single, '=' for double
# These define which atom environment pairs can have their connecting bond cleaved
REACTION_DEFS: list[list[tuple[str, str, str]]] = [
    # L1 cleavages: amides, esters, lactams
    [('1', '3', '-'), ('1', '5', '-'), ('1', '10', '-')],
    # L3 cleavages: ethers
    [('3', '4', '-'), ('3', '13', '-'), ('3', '14', '-'), ('3', '15', '-'), ('3', '16', '-')],
    # L4 cleavages: aliphatic C-N, C-S
    [('4', '5', '-'), ('4', '11', '-')],
    # L5 cleavages: amine to sulfonamide/aromatic
    [('5', '12', '-'), ('5', '14', '-'), ('5', '16', '-'), ('5', '13', '-'), ('5', '15', '-')],
    # L6 cleavages: carbonyl to ring/aromatic
    [('6', '13', '-'), ('6', '14', '-'), ('6', '15', '-'), ('6', '16', '-')],
    # L7 cleavages: olefin C=C
    [('7a', '7b', '=')],
    # L8 cleavages: aliphatic to aromatic/ring
    [('8', '9', '-'), ('8', '10', '-'), ('8', '13', '-'), ('8', '14', '-'), ('8', '15', '-'), ('8', '16', '-')],
    # L9 cleavages: aromatic N to ring
    [('9', '13', '-'), ('9', '14', '-'), ('9', '15', '-'), ('9', '16', '-')],
    # L10 cleavages: lactam to ring
    [('10', '13', '-'), ('10', '14', '-'), ('10', '15', '-'), ('10', '16', '-')],
    # L11 cleavages: thioether to ring/aromatic
    [('11', '13', '-'), ('11', '14', '-'), ('11', '15', '-'), ('11', '16', '-')],
    # L13 cleavages: ring hetero to aromatic
    [('13', '14', '-'), ('13', '15', '-'), ('13', '16', '-')],
    # L14 cleavages: aromatic to aromatic/ring
    [('14', '14', '-'), ('14', '15', '-'), ('14', '16', '-')],
    # L15 cleavages: ring to aromatic
    [('15', '16', '-')],
    # L16 cleavages: aromatic-aromatic (biphenyl etc)
    [('16', '16', '-')],
]

# Flatten reaction defs into rules list with proper label format
BRICS_RULES: list[tuple[str, str, int]] = []
for rule_group in REACTION_DEFS:
    for label1, label2, bond_char in rule_group:
        # Convert bond character to order
        bond_order = 2 if bond_char == '=' else 1
        # Convert numeric labels to L-prefixed format
        l1 = f'L{label1}' if not label1.startswith('L') else label1
        l2 = f'L{label2}' if not label2.startswith('L') else label2
        BRICS_RULES.append((l1, l2, bond_order))

# Legacy alias for backward compatibility
ATOM_ENVIRONS = ENVIRONS

# Compiled patterns cache
_COMPILED_PATTERNS: dict[str, Molecule | None] = {}


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


def _find_ring_bonds(mol: Molecule) -> set[tuple[int, int]]:
    """Find all bonds that are part of a ring."""
    ring_bonds: set[tuple[int, int]] = set()
    rings = find_sssr(mol)
    
    for ring in rings:
        # Convert to list if it's a set
        ring_list = list(ring) if isinstance(ring, (set, frozenset)) else ring
        ring_set = set(ring_list)
        for i, atom_idx in enumerate(ring_list):
            next_idx = ring_list[(i + 1) % len(ring_list)]
            key = (min(atom_idx, next_idx), max(atom_idx, next_idx))
            ring_bonds.add(key)
    
    return ring_bonds


def _matches_environment(mol: Molecule, atom_idx: int, label: str) -> bool:
    """Check if an atom matches a BRICS environment."""
    pattern = _get_pattern(label)
    if pattern is None:
        return False
    
    matches = substructure_search(mol, pattern, uniquify=True)
    return any(atom_idx in match for match in matches)


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
    # Find ring bonds (we don't cleave ring bonds)
    ring_bonds = _find_ring_bonds(mol)
    
    # Find atoms matching each environment
    env_matches: dict[str, set[int]] = {}
    for label in ENVIRONS.keys():
        pattern = _get_pattern(label)
        if pattern is not None:
            matches = substructure_search(mol, pattern, uniquify=True)
            env_matches[label] = {m[0] for m in matches if m}
        else:
            env_matches[label] = set()
    
    bonds_found: set[tuple[int, int]] = set()
    
    # Check each bond against BRICS rules
    for bond in mol.bonds:
        atom1_idx = bond.atom1_idx
        atom2_idx = bond.atom2_idx
        bond_key = (min(atom1_idx, atom2_idx), max(atom1_idx, atom2_idx))
        
        # Skip ring bonds
        if bond_key in ring_bonds:
            continue
        
        # Skip already found bonds
        if bond_key in bonds_found:
            continue
        
        # Check each BRICS rule
        for label1, label2, expected_order in BRICS_RULES:
            # Check bond order
            if bond.order != expected_order:
                continue
            
            # Check if atoms match the environments (either direction)
            if atom1_idx in env_matches.get(label1, set()) and \
               atom2_idx in env_matches.get(label2, set()):
                bonds_found.add(bond_key)
                # Extract numeric label
                num1 = ''.join(c for c in label1 if c.isdigit())
                num2 = ''.join(c for c in label2 if c.isdigit())
                yield ((atom1_idx, atom2_idx), (num1, num2))
                break
            elif atom2_idx in env_matches.get(label1, set()) and \
                 atom1_idx in env_matches.get(label2, set()):
                bonds_found.add(bond_key)
                num1 = ''.join(c for c in label1 if c.isdigit())
                num2 = ''.join(c for c in label2 if c.isdigit())
                yield ((atom2_idx, atom1_idx), (num1, num2))
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
    
    # Create new molecule
    new_atoms: list[Atom] = []
    new_bonds: list[Bond] = []
    
    # Copy existing atoms
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
    
    # Find bonds to break
    bonds_to_break: dict[tuple[int, int], tuple[str, str]] = {}
    for (atom1, atom2), (label1, label2) in bonds:
        key = (min(atom1, atom2), max(atom1, atom2))
        bonds_to_break[key] = (label1, label2) if atom1 < atom2 else (label2, label1)
    
    # Copy bonds, skipping broken ones
    next_atom_idx = len(new_atoms)
    dummy_bonds: list[tuple[int, int, str, int]] = []  # (parent_idx, dummy_idx, label, bond_order)
    
    for bond in mol.bonds:
        key = (min(bond.atom1_idx, bond.atom2_idx), max(bond.atom1_idx, bond.atom2_idx))
        
        if key in bonds_to_break:
            # This bond will be broken
            labels = bonds_to_break[key]
            
            # Get correct label for each atom
            if bond.atom1_idx < bond.atom2_idx:
                label1, label2 = labels
            else:
                label2, label1 = labels
            
            # Create dummy atoms for each side
            # Dummy for atom1 side
            dummy1 = Atom(
                idx=next_atom_idx,
                symbol='*',
                charge=0,
                explicit_hydrogens=0,
                is_aromatic=False,
                isotope=int(label1) if label1.isdigit() else None,
            )
            new_atoms.append(dummy1)
            dummy_bonds.append((bond.atom1_idx, next_atom_idx, label1, bond.order))
            next_atom_idx += 1
            
            # Dummy for atom2 side
            dummy2 = Atom(
                idx=next_atom_idx,
                symbol='*',
                charge=0,
                explicit_hydrogens=0,
                is_aromatic=False,
                isotope=int(label2) if label2.isdigit() else None,
            )
            new_atoms.append(dummy2)
            dummy_bonds.append((bond.atom2_idx, next_atom_idx, label2, bond.order))
            next_atom_idx += 1
        else:
            # Keep this bond
            new_bond = Bond(
                idx=len(new_bonds),
                atom1_idx=bond.atom1_idx,
                atom2_idx=bond.atom2_idx,
                order=bond.order,
                is_aromatic=bond.is_aromatic,
                stereo=bond.stereo,
            )
            new_bonds.append(new_bond)
    
    # Add bonds to dummy atoms
    for parent_idx, dummy_idx, label, order in dummy_bonds:
        new_bond = Bond(
            idx=len(new_bonds),
            atom1_idx=parent_idx,
            atom2_idx=dummy_idx,
            order=order,
            is_aromatic=False,
        )
        new_bonds.append(new_bond)
    
    # Build molecule
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
    if mol.num_atoms == 0:
        return []
    
    # Find connected components using DFS
    visited: set[int] = set()
    components: list[set[int]] = []
    
    def dfs(start: int) -> set[int]:
        component: set[int] = set()
        stack = [start]
        while stack:
            atom_idx = stack.pop()
            if atom_idx in visited:
                continue
            visited.add(atom_idx)
            component.add(atom_idx)
            
            atom = mol.atoms[atom_idx]
            for bond_idx in atom.bond_indices:
                bond = mol.bonds[bond_idx]
                neighbor = bond.other_atom(atom_idx)
                if neighbor not in visited:
                    stack.append(neighbor)
        return component
    
    for i in range(mol.num_atoms):
        if i not in visited:
            component = dfs(i)
            components.append(component)
    
    if len(components) <= 1:
        return [deepcopy(mol)]
    
    # Create fragment molecules
    fragments: list[Molecule] = []
    
    for component in components:
        # Map old indices to new
        old_to_new: dict[int, int] = {}
        for new_idx, old_idx in enumerate(sorted(component)):
            old_to_new[old_idx] = new_idx
        
        frag = Molecule()
        
        # Add atoms
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
        
        # Add bonds
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


def brics_decompose(
    mol: Molecule,
    min_fragment_size: int = 1,
    keep_non_leaf_nodes: bool = False,
    single_pass: bool = False,
    return_mols: bool = False,
) -> set[str] | list[Molecule]:
    """Perform BRICS decomposition on a molecule.
    
    Recursively breaks BRICS bonds to generate fragments.
    
    Args:
        mol: Molecule to decompose.
        min_fragment_size: Minimum number of heavy atoms in a fragment.
        keep_non_leaf_nodes: If True, include intermediate fragments.
        single_pass: If True, only do one round of decomposition.
        return_mols: If True, return Molecule objects instead of SMILES.
    
    Returns:
        Set of SMILES strings (or list of Molecules if return_mols=True).
    
    Example:
        >>> mol = parse("CCOc1ccc(CC)cc1")
        >>> frags = brics_decompose(mol)
        >>> sorted(frags)
        ['[16*]c1ccc([16*])cc1', '[3*]O[3*]', '[4*]CC', '[8*]CC']
    """
    from .aromaticity import perceive_aromaticity
    
    # Get initial SMILES
    mol_smi = to_smiles(mol)
    
    all_nodes: set[str] = set()
    all_nodes.add(mol_smi)
    
    found_mols: dict[str, Molecule] = {mol_smi: mol}
    active_pool: dict[str, Molecule] = {mol_smi: mol}
    
    while active_pool:
        new_pool: dict[str, Molecule] = {}
        
        for smi, current_mol in list(active_pool.items()):
            # Find BRICS bonds
            bonds = list(find_brics_bonds(current_mol))
            
            if not bonds:
                # No bonds to break, this is a leaf
                new_pool[smi] = current_mol
                continue
            
            matched = False
            
            # Try breaking each bond individually
            for bond_info in bonds:
                # Break this single bond
                fragmented = break_brics_bonds(current_mol, [bond_info])
                fragments = _get_fragments(fragmented)
                
                # Check fragment sizes
                valid_frags = True
                for frag in fragments:
                    # Count heavy atoms (non-dummy)
                    heavy_count = sum(1 for a in frag.atoms if a.symbol != '*')
                    if heavy_count < min_fragment_size:
                        valid_frags = False
                        break
                
                if not valid_frags:
                    continue
                
                matched = True
                
                for frag in fragments:
                    frag_smi = to_smiles(frag)
                    
                    if frag_smi not in all_nodes:
                        all_nodes.add(frag_smi)
                        found_mols[frag_smi] = frag
                        
                        if not single_pass:
                            new_pool[frag_smi] = frag
            
            if single_pass or keep_non_leaf_nodes or not matched:
                new_pool[smi] = current_mol
        
        if single_pass:
            break
        
        active_pool = new_pool
        
        # Check if we made progress
        if set(active_pool.keys()) == set(new_pool.keys()):
            break
    
    # Return results
    if not (single_pass or keep_non_leaf_nodes):
        result_smis = set(active_pool.keys())
    else:
        result_smis = all_nodes
    
    if return_mols:
        return [found_mols[smi] for smi in result_smis if smi in found_mols]
    else:
        return result_smis


# Compatibility aliases
BRICSDecompose = brics_decompose
FindBRICSBonds = find_brics_bonds
BreakBRICSBonds = break_brics_bonds
