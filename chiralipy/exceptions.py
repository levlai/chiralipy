"""Custom exceptions for chirpy."""

from __future__ import annotations


class ChemError(Exception):
    """Base exception for chemistry-related errors."""
    pass


class ParseError(ChemError):
    """Error during SMILES parsing."""
    
    def __init__(self, message: str, smiles: str | None = None, position: int | None = None):
        self.message = message
        self.smiles = smiles
        self.position = position
        
        if smiles is not None and position is not None:
            super().__init__(f"{message}\n  {smiles}\n  {' ' * position}^")
        elif smiles is not None:
            super().__init__(f"{message} in: {smiles}")
        else:
            super().__init__(message)


class ValenceError(ChemError):
    """Invalid valence or bonding."""
    pass


class RingError(ChemError):
    """Invalid ring closure."""
    
    def __init__(self, message: str, ring_index: int | None = None):
        self.ring_index = ring_index
        super().__init__(message)


class AromaticityError(ChemError):
    """Error during aromaticity perception or kekulization."""
    pass


class CanonicalizeError(ChemError):
    """Error during canonicalization."""
    pass
