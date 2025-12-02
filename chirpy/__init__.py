"""Chirpy - Pure Python SMILES/SMARTS chemistry library.

A zero-dependency library for parsing, manipulating, and canonicalizing SMILES
(Simplified Molecular Input Line Entry System) strings and SMARTS
(SMILES Arbitrary Target Specification) patterns.

Example usage:
    >>> from chirpy import Molecule, parse, canonical_smiles
    >>> mol = parse("CCO")
    >>> canonical_smiles(mol)
    'CCO'
    >>> mol.atoms[0].symbol
    'C'
    
    # SMARTS support
    >>> pattern = parse("[NX3H2]")  # Primary amine
    >>> pattern.atoms[0].connectivity_query
    3
    
    # Substructure matching
    >>> mol = parse("CCO")
    >>> pattern = parse("[OH]")
    >>> from chirpy import substructure_search
    >>> substructure_search(mol, pattern)
    [(2,)]

Chirpy provides deterministic canonical SMILES output while remaining
pure Python for easy deployment and modification.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Vladimir Lekić"
__all__ = [
    # Core types
    "Atom",
    "Bond",
    "Molecule",
    # Parsing
    "parse",
    "SmilesParser",
    # Writing
    "canonical_smiles",
    "to_smiles",
    "SmilesWriter",
    # Canonicalization
    "canonical_ranks",
    "Canonicalizer",
    # Aromaticity
    "perceive_aromaticity",
    "AromaticityPerceiver",
    # Ring detection
    "find_sssr",
    "find_ring_systems",
    "get_ring_info",
    # Substructure matching
    "substructure_search",
    "has_substructure",
    "count_matches",
    "RingInfo",
    # Kekulization
    "kekulize",
    # Hydrogen manipulation
    "add_explicit_hydrogens",
    "remove_explicit_hydrogens",
    # BRICS decomposition
    "find_brics_bonds",
    "break_brics_bonds",
    "brics_decompose",
    # Exceptions
    "ChemError",
    "ParseError",
    "ValenceError",
    "KekulizationError",
    # Elements
    "Element",
    "BondOrder",
    "ORGANIC_SUBSET",
    "AROMATIC_SUBSET",
]

# Lazy imports to avoid circular dependencies and improve startup time
def __getattr__(name: str):
    """Lazy import module members."""
    if name in ("Atom", "Bond", "Molecule"):
        from .types import Atom, Bond, Molecule
        return locals()[name]
    
    if name in ("parse", "SmilesParser"):
        from .parser import parse, SmilesParser
        return locals()[name]
    
    if name in ("canonical_smiles", "to_smiles", "SmilesWriter"):
        from .writer import canonical_smiles, to_smiles, SmilesWriter
        return locals()[name]
    
    if name in ("canonical_ranks", "Canonicalizer"):
        from .canon import canonical_ranks, Canonicalizer
        return locals()[name]
    
    if name in ("perceive_aromaticity", "AromaticityPerceiver"):
        from .aromaticity import perceive_aromaticity, AromaticityPerceiver
        return locals()[name]
    
    if name in ("substructure_search", "has_substructure", "count_matches", "RingInfo"):
        from .match import substructure_search, has_substructure, count_matches, RingInfo
        return locals()[name]
    
    if name in ("find_sssr", "find_ring_systems", "get_ring_info"):
        from .rings import find_sssr, find_ring_systems, get_ring_info
        return locals()[name]
    
    if name == "kekulize":
        from .kekulization import kekulize
        return kekulize
    
    if name == "KekulizationError":
        from .kekulization import KekulizationError
        return KekulizationError
    
    if name in ("add_explicit_hydrogens", "remove_explicit_hydrogens"):
        from .hydrogen import add_explicit_hydrogens, remove_explicit_hydrogens
        return locals()[name]
    
    if name in ("find_brics_bonds", "break_brics_bonds", "brics_decompose"):
        from .brics import find_brics_bonds, break_brics_bonds, brics_decompose
        return locals()[name]
    
    if name in ("ChemError", "ParseError", "ValenceError"):
        from .exceptions import ChemError, ParseError, ValenceError
        return locals()[name]
    
    if name in ("Element", "BondOrder", "ORGANIC_SUBSET", "AROMATIC_SUBSET"):
        from .elements import Element, BondOrder, ORGANIC_SUBSET, AROMATIC_SUBSET
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
