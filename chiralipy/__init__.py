"""
Chiralipy - Pure Python SMILES/SMARTS library.

A zero-dependency library for parsing, manipulating, and canonicalizing
molecular structures using SMILES notation.

    >>> from chiralipy import parse, canonical_smiles
    >>> mol = parse("CCO")
    >>> canonical_smiles(mol)
    'CCO'

Submodules:
    chiralipy.decompose - BRICS and Murcko decomposition
    chiralipy.rings     - Ring detection (SSSR)
    chiralipy.match     - Substructure search (SMARTS)
    chiralipy.transform - Kekulization, aromaticity, hydrogens
"""

__version__ = "0.1.0"
__author__ = "Vladimir Lekić"

# Core types
from chiralipy.types import Atom, Bond, Molecule

# Parsing and writing
from chiralipy.parser import parse, SmilesParser
from chiralipy.writer import canonical_smiles, to_smiles, SmilesWriter
from chiralipy.canon import canonical_ranks, Canonicalizer

# Exceptions
from chiralipy.exceptions import ChemError, ParseError, ValenceError

# Element data
from chiralipy.elements import Element, BondOrder, ORGANIC_SUBSET, AROMATIC_SUBSET

# Submodules
from chiralipy import decompose, rings, match, transform

__all__ = [
    # Types
    "Atom", "Bond", "Molecule",
    # Parsing
    "parse", "SmilesParser",
    # Writing
    "canonical_smiles", "to_smiles", "SmilesWriter",
    # Canonicalization
    "canonical_ranks", "Canonicalizer",
    # Exceptions
    "ChemError", "ParseError", "ValenceError",
    # Elements
    "Element", "BondOrder", "ORGANIC_SUBSET", "AROMATIC_SUBSET",
    # Submodules
    "decompose", "rings", "match", "transform",
]
