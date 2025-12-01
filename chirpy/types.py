"""
Core molecular data types.

This module defines the fundamental data structures for representing molecules:
Atom, Bond, and Molecule classes with modern Python type annotations and
dataclass features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from .elements import (
    BondOrder,
    get_atomic_number,
    get_default_valence,
)

if TYPE_CHECKING:
    from typing import Self


@dataclass(slots=True)
class Bond:
    """Represents a chemical bond between two atoms.
    
    Attributes:
        idx: Unique index of this bond in the molecule.
        atom1_idx: Index of the first atom.
        atom2_idx: Index of the second atom.
        order: Bond order (1=single, 2=double, 3=triple, 4=aromatic, 5=quadruple, 6=dative, 0=any).
        is_aromatic: Whether this bond is aromatic.
        stereo: Stereochemistry marker ('/' or '\\' for E/Z).
        is_dative: Whether this is a dative/coordinate bond.
        dative_direction: Direction of dative bond (1 = atom1->atom2, -1 = atom2->atom1).
        is_any: Whether this is a SMARTS "any" bond (~).
        
        # SMARTS bond query fields:
        is_ring_bond: Whether bond must be in a ring (@) - None means no constraint.
        is_not_ring_bond: Whether bond must NOT be in a ring (!@).
    """
    
    idx: int
    atom1_idx: int
    atom2_idx: int
    order: int = BondOrder.SINGLE
    is_aromatic: bool = False
    stereo: str | None = None
    is_dative: bool = False
    dative_direction: int = 0  # 1 means atom1->atom2, -1 means atom2->atom1
    is_any: bool = False  # SMARTS any bond (~)
    
    # SMARTS bond query fields
    is_ring_bond: bool | None = None  # True = must be ring bond (@), False = no constraint
    is_not_ring_bond: bool = False    # True = must NOT be ring bond (!@)
    
    def other_atom(self, atom_idx: int) -> int:
        """Get the index of the atom on the other end of this bond.
        
        Args:
            atom_idx: Index of one atom in the bond.
        
        Returns:
            Index of the other atom.
        
        Raises:
            ValueError: If atom_idx is not part of this bond.
        """
        if atom_idx == self.atom1_idx:
            return self.atom2_idx
        if atom_idx == self.atom2_idx:
            return self.atom1_idx
        raise ValueError(f"Atom {atom_idx} not in bond {self.idx}")
    
    @property
    def bond_order(self) -> BondOrder:
        """Get bond order as enum."""
        return BondOrder(self.order)
    
    def __contains__(self, atom_idx: int) -> bool:
        """Check if atom is part of this bond."""
        return atom_idx in (self.atom1_idx, self.atom2_idx)


@dataclass(slots=True)
class Atom:
    """Represents an atom in a molecule.
    
    Attributes:
        idx: Unique index of this atom in the molecule.
        symbol: Element symbol (e.g., "C", "N", "Cl"). For SMARTS, can be "*" (any).
        charge: Formal charge.
        explicit_hydrogens: Explicit hydrogen count from bracket notation.
        is_aromatic: Whether this atom is aromatic.
        isotope: Mass number (isotope), or None for natural abundance.
        chirality: Stereochemistry marker ('@' or '@@').
        atom_class: Atom class number from SMILES (for reaction mapping).
        bond_indices: Indices of bonds connected to this atom.
        _was_first_in_component: Internal flag for chirality handling.
        
        # SMARTS query fields:
        is_wildcard: True if this is a wildcard atom (*).
        atom_list: List of allowed symbols (for [C,N,O] queries).
        atom_list_negated: True if atom_list is negated ([!C]).
        ring_count: Required ring membership count (R0, R1, R2...).
        ring_size: Required ring size (r5, r6...).
        degree_query: Required degree (D1, D2...).
        valence_query: Required valence (v1, v2...).
        connectivity_query: Required connectivity (X1, X2...).
        is_recursive: True if contains recursive SMARTS.
        recursive_smarts: The recursive SMARTS pattern.
    """
    
    idx: int
    symbol: str
    charge: int = 0
    explicit_hydrogens: int = 0
    is_aromatic: bool = False
    isotope: int | None = None
    chirality: str | None = None
    atom_class: int | None = None
    bond_indices: list[int] = field(default_factory=list)
    _was_first_in_component: bool = False
    
    # SMARTS query fields
    is_wildcard: bool = False
    atom_list: list[str] | None = None
    atom_list_negated: bool = False
    atomic_number_list: list[int] | None = None  # For [#0,#6,#7] patterns
    ring_count: int | None = None
    ring_size: int | None = None
    degree_query: int | None = None
    valence_query: int | None = None
    connectivity_query: int | None = None
    is_recursive: bool = False
    recursive_smarts: str | None = None
    charge_query: int | None = None  # For [+0] exact charge query
    
    @property
    def atomic_number(self) -> int:
        """Get the atomic number for this element."""
        return get_atomic_number(self.symbol)
    
    @property
    def default_valence(self) -> int | None:
        """Get the default valence for this element."""
        return get_default_valence(self.atomic_number)
    
    def degree(self, mol: "Molecule") -> int:
        """Get the number of bonds to this atom.
        
        Args:
            mol: Parent molecule.
        
        Returns:
            Number of bonds (degree).
        """
        return len(self.bond_indices)
    
    def neighbors(self, mol: "Molecule") -> Iterator[int]:
        """Iterate over indices of neighboring atoms.
        
        Args:
            mol: Parent molecule.
        
        Yields:
            Indices of atoms bonded to this atom.
        """
        for bond_idx in self.bond_indices:
            bond = mol.bonds[bond_idx]
            yield bond.other_atom(self.idx)
    
    def get_bonds(self, mol: "Molecule") -> Iterator[Bond]:
        """Iterate over bonds connected to this atom.
        
        Args:
            mol: Parent molecule.
        
        Yields:
            Bond objects connected to this atom.
        """
        for bond_idx in self.bond_indices:
            yield mol.bonds[bond_idx]
    
    def total_hydrogens(self, mol: "Molecule") -> int:
        """Calculate total hydrogen count (explicit + implicit).
        
        Args:
            mol: Parent molecule.
        
        Returns:
            Total number of hydrogens attached to this atom.
        """
        default_val = self.default_valence
        if default_val is None:
            return self.explicit_hydrogens
        
        # Calculate bond order sum
        bond_order_sum = 0.0
        for bond in self.get_bonds(mol):
            if bond.is_aromatic:
                bond_order_sum += 1.5
            else:
                bond_order_sum += bond.order
        
        # For charged atoms:
        # - Positive charge means atom can accept more bonds (e.g., NH4+ has 4 bonds)
        # - Negative charge means atom has fewer bonds (e.g., O- has 1 bond, CH3- has 3)
        # Implicit H = default_valence - bond_order + charge - explicit_H
        implicit = max(
            0,
            default_val - int(round(bond_order_sum)) + self.charge - self.explicit_hydrogens
        )
        return self.explicit_hydrogens + implicit


@dataclass
class Molecule:
    """Represents a molecular structure.
    
    A molecule consists of atoms connected by bonds. This class provides
    methods for building and querying molecular structures.
    
    Attributes:
        atoms: List of atoms in the molecule.
        bonds: List of bonds in the molecule.
        name: Optional molecule name/identifier.
    
    Example:
        >>> mol = Molecule()
        >>> c1 = mol.add_atom("C")
        >>> c2 = mol.add_atom("C")
        >>> mol.add_bond(c1, c2)
        >>> len(mol)
        2
    """
    
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    name: str | None = None
    
    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self.atoms)
    
    def __iter__(self) -> Iterator[Atom]:
        """Iterate over atoms."""
        return iter(self.atoms)
    
    def __getitem__(self, idx: int) -> Atom:
        """Get atom by index."""
        return self.atoms[idx]
    
    def add_atom(
        self,
        symbol: str,
        *,
        charge: int = 0,
        explicit_hydrogens: int = 0,
        is_aromatic: bool = False,
        isotope: int | None = None,
        chirality: str | None = None,
        atom_class: int | None = None,
        # SMARTS query fields
        is_wildcard: bool = False,
        atom_list: list[str] | None = None,
        atom_list_negated: bool = False,
        atomic_number_list: list[int] | None = None,
        ring_count: int | None = None,
        ring_size: int | None = None,
        degree_query: int | None = None,
        valence_query: int | None = None,
        connectivity_query: int | None = None,
        is_recursive: bool = False,
        recursive_smarts: str | None = None,
        charge_query: int | None = None,
    ) -> int:
        """Add an atom to the molecule.
        
        Args:
            symbol: Element symbol.
            charge: Formal charge.
            explicit_hydrogens: Explicit hydrogen count.
            is_aromatic: Whether atom is aromatic.
            isotope: Mass number.
            chirality: Stereochemistry marker.
            atom_class: Atom class for reaction mapping.
            is_wildcard: SMARTS wildcard atom.
            atom_list: SMARTS atom list [C,N,O].
            atom_list_negated: Whether atom list is negated [!C].
            atomic_number_list: SMARTS atomic number list [#0,#6,#7].
            ring_count: SMARTS ring membership query.
            ring_size: SMARTS ring size query.
            degree_query: SMARTS degree query.
            valence_query: SMARTS valence query.
            connectivity_query: SMARTS connectivity query.
            is_recursive: Whether this is a recursive SMARTS.
            recursive_smarts: The recursive SMARTS pattern.
            charge_query: SMARTS exact charge query [+0].
        
        Returns:
            Index of the newly added atom.
        """
        idx = len(self.atoms)
        atom = Atom(
            idx=idx,
            symbol=symbol,
            charge=charge,
            explicit_hydrogens=explicit_hydrogens,
            is_aromatic=is_aromatic,
            isotope=isotope,
            chirality=chirality,
            atom_class=atom_class,
            is_wildcard=is_wildcard,
            atom_list=atom_list,
            atom_list_negated=atom_list_negated,
            atomic_number_list=atomic_number_list,
            ring_count=ring_count,
            ring_size=ring_size,
            degree_query=degree_query,
            valence_query=valence_query,
            connectivity_query=connectivity_query,
            is_recursive=is_recursive,
            recursive_smarts=recursive_smarts,
            charge_query=charge_query,
        )
        self.atoms.append(atom)
        return idx
    
    def add_bond(
        self,
        atom1_idx: int,
        atom2_idx: int,
        *,
        order: int = BondOrder.SINGLE,
        is_aromatic: bool = False,
        stereo: str | None = None,
        is_dative: bool = False,
        dative_direction: int = 0,
        is_any: bool = False,
        is_ring_bond: bool | None = None,
        is_not_ring_bond: bool = False,
    ) -> int:
        """Add a bond between two atoms.
        
        Args:
            atom1_idx: Index of the first atom.
            atom2_idx: Index of the second atom.
            order: Bond order.
            is_aromatic: Whether bond is aromatic.
            stereo: Stereochemistry marker.
            is_dative: Whether this is a dative/coordinate bond.
            dative_direction: Direction (1 = atom1->atom2, -1 = atom2->atom1).
            is_any: Whether this is a SMARTS any bond.
            is_ring_bond: SMARTS ring bond constraint (@).
            is_not_ring_bond: SMARTS not ring bond constraint (!@).
        
        Returns:
            Index of the newly added bond.
        
        Raises:
            IndexError: If atom indices are out of bounds.
        """
        if atom1_idx >= len(self.atoms) or atom2_idx >= len(self.atoms):
            raise IndexError(f"Atom index out of bounds: {atom1_idx}, {atom2_idx}")
        
        idx = len(self.bonds)
        bond = Bond(
            idx=idx,
            atom1_idx=atom1_idx,
            atom2_idx=atom2_idx,
            order=order,
            is_aromatic=is_aromatic,
            stereo=stereo,
            is_dative=is_dative,
            dative_direction=dative_direction,
            is_any=is_any,
            is_ring_bond=is_ring_bond,
            is_not_ring_bond=is_not_ring_bond,
        )
        self.bonds.append(bond)
        self.atoms[atom1_idx].bond_indices.append(idx)
        self.atoms[atom2_idx].bond_indices.append(idx)
        return idx
    
    def get_bond_between(self, atom1_idx: int, atom2_idx: int) -> Bond | None:
        """Find the bond between two atoms.
        
        Args:
            atom1_idx: Index of the first atom.
            atom2_idx: Index of the second atom.
        
        Returns:
            Bond object if found, None otherwise.
        """
        for bond_idx in self.atoms[atom1_idx].bond_indices:
            bond = self.bonds[bond_idx]
            if atom1_idx in bond and atom2_idx in bond:
                return bond
        return None
    
    def connected_components(self) -> list[list[int]]:
        """Find connected components in the molecule.
        
        Returns:
            List of components, each being a sorted list of atom indices.
        """
        visited: set[int] = set()
        components: list[list[int]] = []
        
        for start in range(len(self.atoms)):
            if start in visited:
                continue
            
            # BFS to find component
            component: list[int] = []
            stack = [start]
            visited.add(start)
            
            while stack:
                atom_idx = stack.pop()
                component.append(atom_idx)
                
                for neighbor in self.atoms[atom_idx].neighbors(self):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            components.append(sorted(component))
        
        return components
    
    def copy(self) -> "Self":
        """Create a deep copy of the molecule.
        
        Returns:
            New Molecule instance with copied data.
        """
        mol = Molecule(name=self.name)
        
        # Copy atoms
        for atom in self.atoms:
            mol.atoms.append(Atom(
                idx=atom.idx,
                symbol=atom.symbol,
                charge=atom.charge,
                explicit_hydrogens=atom.explicit_hydrogens,
                is_aromatic=atom.is_aromatic,
                isotope=atom.isotope,
                chirality=atom.chirality,
                atom_class=atom.atom_class,
                bond_indices=list(atom.bond_indices),
                _was_first_in_component=atom._was_first_in_component,
                is_wildcard=atom.is_wildcard,
                atom_list=list(atom.atom_list) if atom.atom_list else None,
                atom_list_negated=atom.atom_list_negated,
                ring_count=atom.ring_count,
                ring_size=atom.ring_size,
                degree_query=atom.degree_query,
                valence_query=atom.valence_query,
                connectivity_query=atom.connectivity_query,
                is_recursive=atom.is_recursive,
                recursive_smarts=atom.recursive_smarts,
            ))
        
        # Copy bonds
        for bond in self.bonds:
            mol.bonds.append(Bond(
                idx=bond.idx,
                atom1_idx=bond.atom1_idx,
                atom2_idx=bond.atom2_idx,
                order=bond.order,
                is_aromatic=bond.is_aromatic,
                stereo=bond.stereo,
                is_dative=bond.is_dative,
                dative_direction=bond.dative_direction,
                is_any=bond.is_any,
            ))
        
        return mol
    
    @property
    def num_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)
    
    @property
    def num_bonds(self) -> int:
        """Number of bonds in the molecule."""
        return len(self.bonds)
    
    @property 
    def is_connected(self) -> bool:
        """Check if molecule is a single connected component."""
        return len(self.connected_components()) <= 1
