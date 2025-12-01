"""
Custom exceptions for the chem library.

This module defines a hierarchy of exceptions for handling chemistry-related
errors in a structured way.
"""

from __future__ import annotations


class ChemError(Exception):
    """Base exception for all chemistry-related errors."""
    
    pass


class ParseError(ChemError):
    """Error during SMILES parsing.
    
    Attributes:
        position: Character position in the SMILES string where error occurred.
        smiles: The original SMILES string being parsed.
        message: Description of what went wrong.
    """
    
    def __init__(
        self,
        message: str,
        smiles: str | None = None,
        position: int | None = None,
    ) -> None:
        self.message = message
        self.smiles = smiles
        self.position = position
        
        # Build detailed error message
        parts = [message]
        if smiles is not None and position is not None:
            parts.append(f"\n  {smiles}")
            parts.append(f"\n  {' ' * position}^")
        elif smiles is not None:
            parts.append(f" in: {smiles}")
        
        super().__init__("".join(parts))


class ValenceError(ChemError):
    """Error related to invalid valence or bonding.
    
    Attributes:
        atom_symbol: The element symbol of the problematic atom.
        expected_valence: The expected/allowed valence.
        actual_valence: The actual valence found.
    """
    
    def __init__(
        self,
        message: str,
        atom_symbol: str | None = None,
        expected_valence: int | None = None,
        actual_valence: int | None = None,
    ) -> None:
        self.message = message
        self.atom_symbol = atom_symbol
        self.expected_valence = expected_valence
        self.actual_valence = actual_valence
        super().__init__(message)


class RingError(ChemError):
    """Error related to ring handling in SMILES.
    
    Attributes:
        ring_index: The problematic ring closure index.
    """
    
    def __init__(self, message: str, ring_index: int | None = None) -> None:
        self.message = message
        self.ring_index = ring_index
        super().__init__(message)


class AromaticityError(ChemError):
    """Error during aromaticity perception or kekulization."""
    
    pass


class CanonicalizeError(ChemError):
    """Error during canonicalization."""
    
    pass
