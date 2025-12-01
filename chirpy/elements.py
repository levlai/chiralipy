"""
Chemical elements and constants.

This module provides element data, periodic table information, and constants
used throughout the chemistry library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Final, FrozenSet


class BondOrder(IntEnum):
    """Bond order enumeration."""
    
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    QUADRUPLE = 5
    DATIVE = 6  # Coordinate/dative bond
    ANY = 0     # SMARTS wildcard bond
    
    def __str__(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Element:
    """Immutable element data.
    
    Attributes:
        atomic_number: Atomic number (proton count).
        symbol: Element symbol (e.g., "C", "Cl").
        name: Full element name.
        default_valence: Common valence for organic chemistry.
    """
    
    atomic_number: int
    symbol: str
    name: str
    default_valence: int | None = None
    
    # Class-level registry
    _by_symbol: ClassVar[dict[str, "Element"]] = {}
    _by_number: ClassVar[dict[int, "Element"]] = {}
    
    def __post_init__(self) -> None:
        # Register in class-level dictionaries
        Element._by_symbol[self.symbol] = self
        Element._by_symbol[self.symbol.lower()] = self  # Aromatic lowercase
        Element._by_number[self.atomic_number] = self
    
    @classmethod
    def from_symbol(cls, symbol: str) -> "Element | None":
        """Look up element by symbol (case-insensitive for single letters)."""
        # Try exact match first
        if symbol in cls._by_symbol:
            return cls._by_symbol[symbol]
        # Try capitalized version (e.g., "cl" -> "Cl")
        capitalized = symbol.capitalize()
        return cls._by_symbol.get(capitalized)
    
    @classmethod
    def from_atomic_number(cls, num: int) -> "Element | None":
        """Look up element by atomic number."""
        return cls._by_number.get(num)


# Initialize periodic table elements
# Common organic chemistry elements with default valences
_ELEMENTS_DATA: Final[list[tuple[int, str, str, int | None]]] = [
    # (atomic_number, symbol, name, default_valence)
    (1, "H", "Hydrogen", 1),
    (2, "He", "Helium", None),
    (3, "Li", "Lithium", 1),
    (4, "Be", "Beryllium", 2),
    (5, "B", "Boron", 3),
    (6, "C", "Carbon", 4),
    (7, "N", "Nitrogen", 3),
    (8, "O", "Oxygen", 2),
    (9, "F", "Fluorine", 1),
    (10, "Ne", "Neon", None),
    (11, "Na", "Sodium", 1),
    (12, "Mg", "Magnesium", 2),
    (13, "Al", "Aluminum", 3),
    (14, "Si", "Silicon", 4),
    (15, "P", "Phosphorus", 3),
    (16, "S", "Sulfur", 2),
    (17, "Cl", "Chlorine", 1),
    (18, "Ar", "Argon", None),
    (19, "K", "Potassium", 1),
    (20, "Ca", "Calcium", 2),
    (21, "Sc", "Scandium", None),
    (22, "Ti", "Titanium", None),
    (23, "V", "Vanadium", None),
    (24, "Cr", "Chromium", None),
    (25, "Mn", "Manganese", None),
    (26, "Fe", "Iron", None),
    (27, "Co", "Cobalt", None),
    (28, "Ni", "Nickel", None),
    (29, "Cu", "Copper", None),
    (30, "Zn", "Zinc", 2),
    (31, "Ga", "Gallium", 3),
    (32, "Ge", "Germanium", 4),
    (33, "As", "Arsenic", 3),
    (34, "Se", "Selenium", 2),
    (35, "Br", "Bromine", 1),
    (36, "Kr", "Krypton", None),
    (37, "Rb", "Rubidium", 1),
    (38, "Sr", "Strontium", 2),
    (39, "Y", "Yttrium", None),
    (40, "Zr", "Zirconium", None),
    (41, "Nb", "Niobium", None),
    (42, "Mo", "Molybdenum", None),
    (43, "Tc", "Technetium", None),
    (44, "Ru", "Ruthenium", None),
    (45, "Rh", "Rhodium", None),
    (46, "Pd", "Palladium", None),
    (47, "Ag", "Silver", 1),
    (48, "Cd", "Cadmium", 2),
    (49, "In", "Indium", 3),
    (50, "Sn", "Tin", 4),
    (51, "Sb", "Antimony", 3),
    (52, "Te", "Tellurium", 2),
    (53, "I", "Iodine", 1),
    (54, "Xe", "Xenon", None),
    (55, "Cs", "Cesium", 1),
    (56, "Ba", "Barium", 2),
    (57, "La", "Lanthanum", None),
    (58, "Ce", "Cerium", None),
    (59, "Pr", "Praseodymium", None),
    (60, "Nd", "Neodymium", None),
    (61, "Pm", "Promethium", None),
    (62, "Sm", "Samarium", None),
    (63, "Eu", "Europium", None),
    (64, "Gd", "Gadolinium", None),
    (65, "Tb", "Terbium", None),
    (66, "Dy", "Dysprosium", None),
    (67, "Ho", "Holmium", None),
    (68, "Er", "Erbium", None),
    (69, "Tm", "Thulium", None),
    (70, "Yb", "Ytterbium", None),
    (71, "Lu", "Lutetium", None),
    (72, "Hf", "Hafnium", None),
    (73, "Ta", "Tantalum", None),
    (74, "W", "Tungsten", None),
    (75, "Re", "Rhenium", None),
    (76, "Os", "Osmium", None),
    (77, "Ir", "Iridium", None),
    (78, "Pt", "Platinum", None),
    (79, "Au", "Gold", 1),
    (80, "Hg", "Mercury", 2),
    (81, "Tl", "Thallium", 3),
    (82, "Pb", "Lead", 4),
    (83, "Bi", "Bismuth", 3),
    (84, "Po", "Polonium", 2),
    (85, "At", "Astatine", 1),
    (86, "Rn", "Radon", None),
    (87, "Fr", "Francium", 1),
    (88, "Ra", "Radium", 2),
    (89, "Ac", "Actinium", None),
    (90, "Th", "Thorium", None),
    (91, "Pa", "Protactinium", None),
    (92, "U", "Uranium", None),
    (93, "Np", "Neptunium", None),
    (94, "Pu", "Plutonium", None),
    (95, "Am", "Americium", None),
    (96, "Cm", "Curium", None),
    (97, "Bk", "Berkelium", None),
    (98, "Cf", "Californium", None),
    (99, "Es", "Einsteinium", None),
    (100, "Fm", "Fermium", None),
    (101, "Md", "Mendelevium", None),
    (102, "No", "Nobelium", None),
    (103, "Lr", "Lawrencium", None),
    (104, "Rf", "Rutherfordium", None),
    (105, "Db", "Dubnium", None),
    (106, "Sg", "Seaborgium", None),
    (107, "Bh", "Bohrium", None),
    (108, "Hs", "Hassium", None),
    (109, "Mt", "Meitnerium", None),
    (110, "Ds", "Darmstadtium", None),
    (111, "Rg", "Roentgenium", None),
    (112, "Cn", "Copernicium", None),
    (113, "Nh", "Nihonium", None),
    (114, "Fl", "Flerovium", None),
    (115, "Mc", "Moscovium", None),
    (116, "Lv", "Livermorium", None),
    (117, "Ts", "Tennessine", None),
    (118, "Og", "Oganesson", None),
]

# Initialize elements
ELEMENTS: Final[tuple[Element, ...]] = tuple(
    Element(num, sym, name, valence)
    for num, sym, name, valence in _ELEMENTS_DATA
)

# Daylight "organic subset" - atoms that can appear without brackets
# when they have standard valence and no charge
ORGANIC_SUBSET: Final[FrozenSet[str]] = frozenset({
    "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I",
})

# Aromatic element symbols allowed in lowercase SMILES form
AROMATIC_SUBSET: Final[FrozenSet[str]] = frozenset({
    "b", "c", "n", "o", "p", "s", "as", "se",
})

# Two-letter elements in organic subset (need special handling in parser)
TWO_LETTER_ORGANIC: Final[FrozenSet[str]] = frozenset({"Cl", "Br"})

# Default valences for implicit hydrogen calculation
DEFAULT_VALENCES: Final[dict[int, int]] = {
    1: 1,   # H
    5: 3,   # B
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    9: 1,   # F
    15: 3,  # P
    16: 2,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}


def get_atomic_number(symbol: str) -> int:
    """Get atomic number for an element symbol.
    
    Args:
        symbol: Element symbol (e.g., "C", "cl", "Cl").
    
    Returns:
        Atomic number, or 0 if not found.
    """
    elem = Element.from_symbol(symbol)
    return elem.atomic_number if elem else 0


def get_default_valence(atomic_num: int) -> int | None:
    """Get default valence for an element.
    
    Args:
        atomic_num: Atomic number.
    
    Returns:
        Default valence, or None if not applicable.
    """
    return DEFAULT_VALENCES.get(atomic_num)


def is_organic_symbol(symbol: str) -> bool:
    """Check if symbol is in the organic subset."""
    return symbol in ORGANIC_SUBSET or symbol.lower() in AROMATIC_SUBSET


def is_aromatic_symbol(symbol: str) -> bool:
    """Check if symbol represents an aromatic atom."""
    return symbol in AROMATIC_SUBSET


# ============================================================================
# Aromaticity Properties
# ============================================================================
# 
# Principled approach to aromaticity based on electronic structure:
# 1. Outer electrons (valence shell) - determines bonding capacity
# 2. Hybridization state - sp2 required for aromatic systems
# 3. Formal charge - modifies electron count
# 4. Orbital availability - p orbital perpendicular to ring plane
#
# The number of π electrons an atom contributes depends on:
# - How many outer electrons it has (periodic table group)
# - How many are used in σ bonds (hybridization)
# - Whether lone pairs are available for π system
# - Formal charge adjustments

# Outer (valence) electrons for each element
# This is the fundamental property that determines aromaticity behavior
OUTER_ELECTRONS: Final[dict[int, int]] = {
    # Group 13: 3 outer electrons
    5: 3,    # B
    13: 3,   # Al
    31: 3,   # Ga
    49: 3,   # In
    81: 3,   # Tl
    # Group 14: 4 outer electrons
    6: 4,    # C
    14: 4,   # Si
    32: 4,   # Ge
    50: 4,   # Sn
    82: 4,   # Pb
    # Group 15: 5 outer electrons
    7: 5,    # N
    15: 5,   # P
    33: 5,   # As
    51: 5,   # Sb
    83: 5,   # Bi
    # Group 16: 6 outer electrons
    8: 6,    # O
    16: 6,   # S
    34: 6,   # Se
    52: 6,   # Te
    84: 6,   # Po
    # Group 17: 7 outer electrons (halogens - not aromatic)
    9: 7,    # F
    17: 7,   # Cl
    35: 7,   # Br
    53: 7,   # I
    85: 7,   # At
}

# Periodic table group for common organic elements
# Group determines outer electrons and thus π electron contribution
ELEMENT_GROUPS: Final[dict[int, int]] = {
    # Group 13 (B, Al, Ga, In, Tl): 3 outer electrons
    5: 13,   # B
    13: 13,  # Al
    31: 13,  # Ga
    49: 13,  # In
    81: 13,  # Tl
    # Group 14 (C, Si, Ge, Sn, Pb): 4 outer electrons
    6: 14,   # C
    14: 14,  # Si
    32: 14,  # Ge
    50: 14,  # Sn
    82: 14,  # Pb
    # Group 15 (N, P, As, Sb, Bi): 5 outer electrons
    7: 15,   # N
    15: 15,  # P
    33: 15,  # As
    51: 15,  # Sb
    83: 15,  # Bi
    # Group 16 (O, S, Se, Te, Po): 6 outer electrons
    8: 16,   # O
    16: 16,  # S
    34: 16,  # Se
    52: 16,  # Te
    84: 16,  # Po
    # Group 17 (Halogens): 7 outer electrons - generally not aromatic
    9: 17,   # F
    17: 17,  # Cl
    35: 17,  # Br
    53: 17,  # I
    85: 17,  # At
}

# Elements that can participate in aromatic rings (can be sp2)
# Requirements for aromaticity:
# 1. Ability to adopt sp2 hybridization (planar geometry)
# 2. Have a p orbital perpendicular to ring plane available for π system
# 3. Either contribute electrons to π system or have empty p orbital
AROMATIC_CAPABLE_ELEMENTS: Final[frozenset[int]] = frozenset({
    5,   # B  - 3 outer e⁻, sp2 with empty p orbital (0 π electrons)
    6,   # C  - 4 outer e⁻, sp2 uses 3 for σ, 1 for π (1 π electron)
    7,   # N  - 5 outer e⁻, pyridine-like (1 π) or pyrrole-like (2 π)
    8,   # O  - 6 outer e⁻, furan-like uses lone pair (2 π electrons)
    14,  # Si - 4 outer e⁻, silabenzene (rare but possible)
    15,  # P  - 5 outer e⁻, phosphole
    16,  # S  - 6 outer e⁻, thiophene (2 π from lone pair)
    33,  # As - 5 outer e⁻, arsole
    34,  # Se - 6 outer e⁻, selenophene
    52,  # Te - 6 outer e⁻, tellurophene
})

# Maximum degree (σ bonds) for sp2 hybridization by group
# sp2 has 3 hybrid orbitals for σ bonds, but some groups have lone pairs
# that occupy orbitals, limiting σ bond capacity
MAX_SP2_DEGREE: Final[dict[int, int]] = {
    13: 3,  # Group 13 (B): 3 outer e⁻, all 3 in sp2 σ bonds, empty p
    14: 3,  # Group 14 (C): 4 outer e⁻, 3 in sp2 σ bonds, 1 in p orbital
    15: 3,  # Group 15 (N): 5 outer e⁻, up to 3 σ bonds, lone pair + p
    16: 2,  # Group 16 (O, S): 6 outer e⁻, 2 σ bonds, 2 lone pairs
}


def get_outer_electrons(atomic_num: int) -> int:
    """Get number of outer (valence) electrons for an element.
    
    This is the fundamental property determining bonding behavior.
    
    Args:
        atomic_num: Atomic number.
    
    Returns:
        Number of outer electrons, or 0 if unknown.
    """
    return OUTER_ELECTRONS.get(atomic_num, 0)


def get_element_group(atomic_num: int) -> int | None:
    """Get periodic table group for an element.
    
    Args:
        atomic_num: Atomic number.
    
    Returns:
        Group number (13-17 for main group elements), or None.
    """
    return ELEMENT_GROUPS.get(atomic_num)


def can_be_sp2(atomic_num: int, total_bonds: int, charge: int = 0) -> bool:
    """Check if an atom can adopt sp2 hybridization.
    
    sp2 hybridization requires:
    - Planar geometry with 3 sp2 orbitals for σ bonds
    - One unhybridized p orbital perpendicular to plane
    - Appropriate electron count for bonding pattern
    
    Args:
        atomic_num: Atomic number.
        total_bonds: Total number of σ bonds.
        charge: Formal charge.
    
    Returns:
        True if sp2 hybridization is possible.
    """
    if atomic_num not in AROMATIC_CAPABLE_ELEMENTS:
        return False
    
    group = ELEMENT_GROUPS.get(atomic_num)
    if group is None:
        return False
    
    max_degree = MAX_SP2_DEGREE.get(group, 3)
    
    # Charge affects electron count and thus max bonds
    # Positive charge: fewer electrons, can accommodate more bonds in some cases
    # Negative charge: more electrons, may limit bonds
    effective_max = max_degree
    if group == 14 and charge == 1:  # Carbocation
        effective_max = 3  # Still 3 bonds but empty p orbital
    elif group == 15 and charge == 1:  # Quaternary N+
        effective_max = 4  # Can have 4 bonds
    
    return total_bonds <= effective_max


def can_be_aromatic(atomic_num: int) -> bool:
    """Check if an element can participate in aromatic rings.
    
    An element can be aromatic if it can:
    1. Adopt sp2 hybridization
    2. Provide a p orbital for the π system
    3. Contribute 0, 1, or 2 electrons to the π system
    
    Args:
        atomic_num: Atomic number.
    
    Returns:
        True if element can be sp2 hybridized in aromatic ring.
    """
    return atomic_num in AROMATIC_CAPABLE_ELEMENTS


def get_max_sp2_degree(atomic_num: int) -> int:
    """Get maximum number of bonds for sp2 hybridization.
    
    Args:
        atomic_num: Atomic number.
    
    Returns:
        Maximum degree (number of bonds) for sp2.
    """
    group = ELEMENT_GROUPS.get(atomic_num)
    if group is None:
        return 0
    return MAX_SP2_DEGREE.get(group, 3)


def get_pi_contribution(
    atomic_num: int,
    has_pi_bond: bool,
    has_exo_double: bool,
    has_h: bool,
    total_connections: int,
    charge: int,
) -> tuple[int | None, int | None]:
    """Determine π electron contribution for aromaticity.
    
    Calculates how many electrons an atom contributes to the aromatic π system
    based on its electronic structure:
    
    The calculation considers:
    1. Outer electrons: Base electron count from periodic table group
    2. Hybridization: sp2 uses 3 electrons for σ bonds
    3. Formal charge: Modifies available electron count
    4. Bonding pattern: Determines electron distribution
    
    For sp2 atoms in aromatic rings:
    - Group 13 (3 outer e⁻): 3 in σ bonds → 0 π electrons (empty p orbital)
    - Group 14 (4 outer e⁻): 3 in σ bonds → 1 π electron
    - Group 15 (5 outer e⁻): Depends on bonding:
        * Pyridine-like (3 σ bonds): 1 π electron + 1 lone pair in plane
        * Pyrrole-like (2 σ bonds + H): 2 π electrons from lone pair
    - Group 16 (6 outer e⁻): 2 σ bonds → 2 π electrons (one lone pair)
    
    Args:
        atomic_num: Atomic number (determines outer electrons).
        has_pi_bond: Has double or aromatic bond in ring.
        has_exo_double: Has exocyclic double bond (e.g., C=O).
        has_h: Has explicit hydrogen attached.
        total_connections: Total number of σ bonds (hybridization indicator).
        charge: Formal charge (modifies electron count).
    
    Returns:
        Tuple of (fixed_contribution, optional_contribution).
        - fixed: Primary π electron count
        - optional: Alternative count if ambiguous (for assignment optimization)
        - If fixed is None, atom cannot be aromatic.
    """
    outer_e = OUTER_ELECTRONS.get(atomic_num, 0)
    group = ELEMENT_GROUPS.get(atomic_num)
    
    if group is None or outer_e == 0:
        # Unknown element - assume can contribute 1 if has pi bond
        return (1, None) if has_pi_bond else (None, None)
    
    # Adjust outer electrons for formal charge
    effective_outer_e = outer_e - charge
    
    # Group 13 (3 outer e⁻): All electrons in σ bonds, empty p orbital
    # Contributes 0 electrons but provides orbital for conjugation
    if group == 13:
        return (0, None)
    
    # Group 14 (4 outer e⁻): sp2 uses 3 for σ, leaves 1 for π
    if group == 14:
        if charge == 1:
            # Carbocation: 3 outer e⁻, all in σ bonds, empty p orbital
            return (0, None)
        elif charge == -1:
            # Carbanion: 5 outer e⁻, 3 in σ bonds, 2 in p orbital
            return (2, None)
        elif has_pi_bond:
            # Neutral sp2: 4 outer e⁻, 3 in σ bonds, 1 in p orbital
            return (1, None)
        else:
            # No pi bonds - likely sp3, cannot be aromatic
            return (None, None)
    
    # Group 15 (5 outer e⁻): Complex - depends on bonding pattern
    # Can be pyridine-like (1 π) or pyrrole-like (2 π)
    # 
    # Pyridine-like: =N- with 3 σ bonds
    #   5 outer e⁻ - 3 σ bonds = 2 remaining
    #   1 electron in p orbital (π), 1 lone pair in sp2 orbital (in plane)
    #   Contributes: 1 π electron
    #
    # Pyrrole-like: -NH- with 2 σ bonds + 1 H
    #   5 outer e⁻ - 3 σ bonds (including N-H) = 2 remaining
    #   Both electrons in p orbital as lone pair
    #   Contributes: 2 π electrons
    if group == 15:
        if charge == 1:
            # N⁺: 4 outer e⁻
            # If has pi bond: 3 in σ, 1 in π → 1 electron
            # Else: 2 in σ, 2 in lone pair → 2 electrons
            if has_pi_bond:
                return (1, None)
            else:
                return (2, None)
        elif charge == -1:
            # N⁻: 6 outer e⁻, has extra lone pair → 2 electrons
            return (2, None)
        elif has_pi_bond:
            # Neutral with pi bond
            if has_h:
                # N-H bond: pyrrole-like, lone pair in p orbital
                return (2, None)
            elif total_connections == 2:
                # Only 2 σ bonds in ring: pyridine-like, 1 π electron
                return (1, None)
            else:
                # Tertiary N (3 bonds) with pi bonds - ambiguous hybridization
                return (1, 2)
        else:
            # No explicit pi bonds
            if has_h:
                return (2, None)  # Pyrrole-like
            elif total_connections >= 3:
                # Tertiary - ambiguous, could be sp2 or sp3
                return (1, 2)
            else:
                return (2, None)  # Default pyrrole-like
    
    # Group 16 (6 outer e⁻): Furan/thiophene-like
    # sp2: 2 σ bonds use 2 electrons
    # Remaining 4 electrons: 2 lone pairs
    # One lone pair in p orbital (π system), one in sp2 orbital (in plane)
    # Contributes: 2 π electrons
    if group == 16:
        if charge == 1:
            # O⁺/S⁺: 5 outer e⁻, 2 in σ, can have 1 in π
            return (1, None)
        elif charge == -1:
            # O⁻/S⁻: 7 outer e⁻, extra electrons → 2 π
            return (2, None)
        elif has_pi_bond or has_exo_double:
            # Has explicit double bond - 1 electron in π bond
            return (1, None)
        else:
            # Furan-like: lone pair contributes 2 electrons
            return (2, None)
    
    # Group 17 (7 outer e⁻): Halogens
    # Generally cannot be sp2 in rings - too many lone pairs
    return (None, None)
