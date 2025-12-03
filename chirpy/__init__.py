"""
Chirpy - Pure Python SMILES/SMARTS library.

A zero-dependency library for parsing, manipulating, and canonicalizing
molecular structures using SMILES notation.

    >>> from chirpy import parse, canonical_smiles
    >>> mol = parse("CCO")
    >>> canonical_smiles(mol)
    'CCO'

Submodules:
    chirpy.decompose - BRICS and Murcko decomposition
    chirpy.rings     - Ring detection (SSSR)
    chirpy.match     - Substructure search (SMARTS)
    chirpy.transform - Kekulization, aromaticity, hydrogens
"""

__version__ = "0.2.0"
__author__ = "Vladimir Lekić"

# Core types
from chirpy.types import Atom, Bond, Molecule

# Parsing and writing
from chirpy.parser import parse, SmilesParser
from chirpy.writer import canonical_smiles, to_smiles, SmilesWriter
from chirpy.canon import canonical_ranks, Canonicalizer

# Exceptions
from chirpy.exceptions import ChemError, ParseError, ValenceError

# Element data
from chirpy.elements import Element, BondOrder, ORGANIC_SUBSET, AROMATIC_SUBSET

# Submodules
from chirpy import decompose, rings, match, transform

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
