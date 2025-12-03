"""
Hydrogen manipulation functions.

This module provides functions for adding and removing explicit hydrogens
from molecular structures.
"""

from __future__ import annotations

from copy import deepcopy

from chiralipy.types import Atom, Bond, Molecule


def add_explicit_hydrogens(mol: Molecule) -> Molecule:
    """Add explicit hydrogen atoms to a molecule.
    
    Converts implicit hydrogens to explicit hydrogen atoms with bonds.
    
    Args:
        mol: Input molecule.
    
    Returns:
        New molecule with explicit hydrogens added.
    
    Example:
        >>> mol = parse("CCO")
        >>> mol_h = add_explicit_hydrogens(mol)
        >>> mol_h.num_atoms
        9
    """
    # Create a new molecule
    new_atoms: list[Atom] = []
    new_bonds: list[Bond] = []
    
    # Copy existing atoms
    for atom in mol.atoms:
        new_atom = Atom(
            idx=atom.idx,
            symbol=atom.symbol,
            charge=atom.charge,
            explicit_hydrogens=0,  # Will be converted to explicit H atoms
            is_aromatic=atom.is_aromatic,
            isotope=atom.isotope,
            chirality=atom.chirality,
            atom_class=atom.atom_class,
        )
        new_atoms.append(new_atom)
    
    # Copy existing bonds
    for bond in mol.bonds:
        new_bond = Bond(
            idx=bond.idx,
            atom1_idx=bond.atom1_idx,
            atom2_idx=bond.atom2_idx,
            order=bond.order,
            is_aromatic=bond.is_aromatic,
            stereo=bond.stereo,
        )
        new_bonds.append(new_bond)
    
    # Add explicit hydrogens
    h_idx = len(new_atoms)
    bond_idx = len(new_bonds)
    
    for atom in mol.atoms:
        # Calculate total hydrogens for this atom
        total_h = atom.total_hydrogens(mol)
        
        for _ in range(total_h):
            # Create hydrogen atom
            h_atom = Atom(
                idx=h_idx,
                symbol='H',
                charge=0,
                explicit_hydrogens=0,
                is_aromatic=False,
            )
            new_atoms.append(h_atom)
            
            # Create bond to parent atom
            h_bond = Bond(
                idx=bond_idx,
                atom1_idx=atom.idx,
                atom2_idx=h_idx,
                order=1,
                is_aromatic=False,
            )
            new_bonds.append(h_bond)
            
            h_idx += 1
            bond_idx += 1
    
    # Build the new molecule
    new_mol = Molecule()
    
    # Add atoms
    for atom in new_atoms:
        atom.bond_indices = []
        new_mol.atoms.append(atom)
    
    # Add bonds and update atom bond_indices
    for bond in new_bonds:
        bond.idx = len(new_mol.bonds)
        new_mol.bonds.append(bond)
        new_mol.atoms[bond.atom1_idx].bond_indices.append(bond.idx)
        new_mol.atoms[bond.atom2_idx].bond_indices.append(bond.idx)
    
    return new_mol


def remove_explicit_hydrogens(mol: Molecule, keep_isotopes: bool = False) -> Molecule:
    """Remove explicit hydrogen atoms from a molecule.
    
    Converts explicit hydrogen atoms to implicit hydrogens on their
    parent atoms.
    
    Args:
        mol: Input molecule.
        keep_isotopes: If True, keep hydrogens with isotope labels (e.g., deuterium).
    
    Returns:
        New molecule with explicit hydrogens removed.
    
    Example:
        >>> mol = parse("[H]OC([H])([H])C([H])([H])[H]")
        >>> mol_noh = remove_explicit_hydrogens(mol)
        >>> mol_noh.num_atoms
        3
    """
    # Find which atoms are hydrogens to remove
    h_to_remove: set[int] = set()
    h_counts: dict[int, int] = {}  # parent_idx -> count of removed H
    
    for atom in mol.atoms:
        if atom.symbol.upper() == 'H':
            # Check if we should keep this hydrogen
            if keep_isotopes and atom.isotope is not None:
                continue
            
            # Check if hydrogen is bonded to exactly one non-H atom
            if len(atom.bond_indices) == 1:
                bond = mol.bonds[atom.bond_indices[0]]
                parent_idx = bond.other_atom(atom.idx)
                parent = mol.atoms[parent_idx]
                
                # Don't remove H from other H atoms
                if parent.symbol.upper() != 'H':
                    h_to_remove.add(atom.idx)
                    h_counts[parent_idx] = h_counts.get(parent_idx, 0) + 1
    
    if not h_to_remove:
        # No hydrogens to remove, return copy
        return deepcopy(mol)
    
    # Build new molecule without removed hydrogens
    # Create mapping from old indices to new indices
    old_to_new: dict[int, int] = {}
    new_idx = 0
    for atom in mol.atoms:
        if atom.idx not in h_to_remove:
            old_to_new[atom.idx] = new_idx
            new_idx += 1
    
    # Create new atoms
    new_atoms: list[Atom] = []
    for atom in mol.atoms:
        if atom.idx in h_to_remove:
            continue
        
        # Calculate new explicit_hydrogens
        added_h = h_counts.get(atom.idx, 0)
        
        new_atom = Atom(
            idx=old_to_new[atom.idx],
            symbol=atom.symbol,
            charge=atom.charge,
            explicit_hydrogens=atom.explicit_hydrogens + added_h,
            is_aromatic=atom.is_aromatic,
            isotope=atom.isotope,
            chirality=atom.chirality,
            atom_class=atom.atom_class,
        )
        new_atoms.append(new_atom)
    
    # Create new bonds (excluding bonds to removed H)
    new_bonds: list[Bond] = []
    for bond in mol.bonds:
        if bond.atom1_idx in h_to_remove or bond.atom2_idx in h_to_remove:
            continue
        
        new_bond = Bond(
            idx=len(new_bonds),
            atom1_idx=old_to_new[bond.atom1_idx],
            atom2_idx=old_to_new[bond.atom2_idx],
            order=bond.order,
            is_aromatic=bond.is_aromatic,
            stereo=bond.stereo,
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
