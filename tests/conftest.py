"""Test configuration and fixtures for chirpy tests."""

import pytest
from typing import Iterator

# RDKit is used as reference for canonical SMILES
from rdkit import Chem


def rdkit_canonical(smiles: str) -> str:
    """Get RDKit canonical SMILES for comparison.
    
    Args:
        smiles: Input SMILES string.
        
    Returns:
        RDKit's canonical SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def rdkit_canonical_isomeric(smiles: str) -> str:
    """Get RDKit canonical SMILES with stereochemistry for comparison.
    
    Args:
        smiles: Input SMILES string.
        
    Returns:
        RDKit's canonical isomeric SMILES (includes stereochemistry).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@pytest.fixture
def simple_smiles() -> list[str]:
    """Basic valid SMILES strings for smoke testing."""
    return [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "C=C",
        "C#C",
        "C=O",
        "C#N",
    ]


@pytest.fixture
def aromatic_smiles() -> list[str]:
    """Aromatic SMILES strings."""
    return [
        "c1ccccc1",
        "c1cnccc1",
        "c1ccncc1",
        "n1ccccc1",
        "c1ccc2ccccc2c1",
        "c1cc2ccccc2cc1",
    ]


@pytest.fixture
def ring_smiles() -> list[str]:
    """SMILES with ring closures."""
    return [
        "C1CC1",
        "C1CCC1",
        "C1CCCC1",
        "C1CCCCC1",
        "C1CC2CCCCC2C1",
        "C12CC1CC2",
    ]


@pytest.fixture
def charged_smiles() -> list[str]:
    """SMILES with charged atoms."""
    return [
        "[O-]",
        "[NH4+]",
        "[Na+]",
        "[Cl-]",
        "[O-]C=O",
        "[NH4+].[Cl-]",
        "[Na+].[Cl-]",
        "CC([O-])=O",
        "[N+](=O)[O-]",
    ]


@pytest.fixture
def chiral_smiles() -> list[str]:
    """SMILES with tetrahedral chirality."""
    return [
        "C[C@H](O)F",
        "C[C@@H](O)F",
        "F[C@H](Cl)Br",
        "F[C@@H](Cl)Br",
        "[C@H](Br)(Cl)F",
        "[C@@H](Br)(Cl)F",
        "C[C@H]1CCCCC1",
        "C[C@@H]1CCCCC1",
    ]


@pytest.fixture
def isotope_smiles() -> list[str]:
    """SMILES with isotopes."""
    return [
        "[2H]",
        "[13C]",
        "[14C]",
        "[2H]C([2H])([2H])[2H]",
        "C[13C](C)(C)C",
        "[35Cl]",
    ]


@pytest.fixture
def stereo_bond_smiles() -> list[str]:
    """SMILES with E/Z stereochemistry."""
    return [
        "F/C=C/F",
        r"F/C=C\F",
        "C/C=C/C",
        r"C/C=C\C",
        r"Cl/C=C/Cl",
        r"Cl/C=C\Cl",
    ]


@pytest.fixture
def bracket_atom_smiles() -> list[str]:
    """SMILES with bracket atoms."""
    return [
        "[CH4]",
        "[CH3]",
        "[CH2]",
        "[CH]",
        "[NH3]",
        "[NH2]",
        "[NH]",
        "[OH]",
        "[SH]",
        "[Cu]",
        "[Fe]",
        "[Pt]",
    ]


@pytest.fixture
def multi_component_smiles() -> list[str]:
    """SMILES with multiple disconnected components."""
    return [
        "[Na+].[Cl-]",
        "O.O",
        "CO.OC",
        "[Na+].[Cl-].[NH4+].[Cl-]",
        "c1ccccc1.c1ccccc1",
    ]


@pytest.fixture
def high_ring_closure_smiles() -> list[str]:
    """SMILES with high ring closure numbers."""
    return [
        "C%10CC%10",
        "C%11CC%11",
        "C%99CC%99",
    ]


@pytest.fixture
def complex_smiles() -> list[str]:
    """Complex real-world molecules."""
    return [
        # Aspirin
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        # Caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # Ibuprofen
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        # Acetaminophen
        "CC(=O)Nc1ccc(O)cc1",
        # Ethanol
        "CCO",
        # Benzene
        "c1ccccc1",
        # Naphthalene
        "c1ccc2ccccc2c1",
        # Anthracene
        "c1ccc2cc3ccccc3cc2c1",
        # Pyrene
        "c1cc2ccc3cccc4ccc(c1)c2c34",
        # Biphenyl
        "c1ccc(-c2ccccc2)cc1",
    ]
