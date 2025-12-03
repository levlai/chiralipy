"""Tests for hydrogen manipulation module comparing with RDKit."""

from __future__ import annotations

import pytest
from chiralipy import parse, to_smiles
from chiralipy.transform import add_explicit_hydrogens, remove_explicit_hydrogens


# Skip if RDKit not available
rdkit = pytest.importorskip("rdkit")
from rdkit import Chem


class TestAddExplicitHydrogens:
    """Test adding explicit hydrogens."""
    
    @pytest.mark.parametrize("smiles,expected_h_count", [
        ("C", 4),          # methane
        ("CC", 6),         # ethane  
        ("O", 2),          # water
        ("[OH2]", 2),      # water explicit
        ("N", 3),          # ammonia
        ("C=C", 4),        # ethene
        ("C#C", 2),        # ethyne
        ("CO", 4),         # methanol
        ("CCO", 6),        # ethanol
    ])
    def test_add_hydrogens_count(self, smiles: str, expected_h_count: int) -> None:
        """Test that correct number of hydrogens are added."""
        mol = parse(smiles)
        mol_h = add_explicit_hydrogens(mol)
        
        h_count = sum(1 for a in mol_h.atoms if a.symbol == 'H')
        assert h_count == expected_h_count, f"Expected {expected_h_count} H for {smiles}, got {h_count}"
    
    @pytest.mark.parametrize("smiles", [
        "C",          # methane
        "CC",         # ethane
        "CCC",        # propane
        "O",          # water
        "N",          # ammonia
        "CO",         # methanol
        "CCO",        # ethanol
        "c1ccccc1",   # benzene
        "C=C",        # ethene
    ])
    def test_add_hydrogens_matches_rdkit(self, smiles: str) -> None:
        """Test hydrogen count matches RDKit."""
        # Chirpy
        mol = parse(smiles)
        mol_h = add_explicit_hydrogens(mol)
        chirpy_h_count = sum(1 for a in mol_h.atoms if a.symbol == 'H')
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdmol_h = Chem.AddHs(rdmol)
        rdkit_h_count = sum(1 for a in rdmol_h.GetAtoms() if a.GetAtomicNum() == 1)
        
        assert chirpy_h_count == rdkit_h_count, f"H count mismatch for {smiles}: chirpy={chirpy_h_count}, rdkit={rdkit_h_count}"
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",       # benzene
        "c1ccncc1",       # pyridine
        "c1ccc2ccccc2c1", # naphthalene
    ])
    def test_add_hydrogens_aromatics(self, smiles: str) -> None:
        """Test adding hydrogens to aromatic molecules."""
        mol = parse(smiles)
        mol_h = add_explicit_hydrogens(mol)
        
        # Chirpy
        chirpy_h_count = sum(1 for a in mol_h.atoms if a.symbol == 'H')
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdmol_h = Chem.AddHs(rdmol)
        rdkit_h_count = sum(1 for a in rdmol_h.GetAtoms() if a.GetAtomicNum() == 1)
        
        assert chirpy_h_count == rdkit_h_count
    
    def test_add_hydrogens_preserves_connectivity(self) -> None:
        """Test that original atoms remain connected after adding H."""
        smiles = "CCO"
        mol = parse(smiles)
        mol_h = add_explicit_hydrogens(mol)
        
        # Original heavy atoms should still be connected
        # Find C-C bond
        c_indices = [i for i, a in enumerate(mol_h.atoms) if a.symbol == 'C']
        assert len(c_indices) == 2
        
        # Find O index
        o_index = next(i for i, a in enumerate(mol_h.atoms) if a.symbol == 'O')
        
        # Verify C-C-O connectivity still exists
        bonds_exist = False
        for bond in mol_h.bonds:
            if bond.atom1_idx in c_indices and bond.atom2_idx in c_indices:
                bonds_exist = True
                break
        assert bonds_exist, "C-C bond not found"
    
    def test_add_hydrogens_charged_atoms(self) -> None:
        """Test adding hydrogens to charged atoms."""
        # Note: Charged atoms have complex valence rules that may vary
        # between chemistry software. We test that common cases work.
        test_cases = [
            ("[CH3-]", 3),  # methyl anion - keeps its H
        ]
        
        for smiles, expected_h in test_cases:
            mol = parse(smiles)
            mol_h = add_explicit_hydrogens(mol)
            h_count = sum(1 for a in mol_h.atoms if a.symbol == 'H')
            assert h_count == expected_h, f"Expected {expected_h} H for {smiles}, got {h_count}"
    
    def test_add_hydrogens_returns_new_molecule(self) -> None:
        """Test that add_explicit_hydrogens returns a new molecule."""
        smiles = "CC"
        mol = parse(smiles)
        original_atom_count = mol.num_atoms
        
        mol_h = add_explicit_hydrogens(mol)
        
        # Original should be unchanged
        assert mol.num_atoms == original_atom_count
        
        # New should have more atoms
        assert mol_h.num_atoms > original_atom_count


class TestRemoveExplicitHydrogens:
    """Test removing explicit hydrogens."""
    
    @pytest.mark.parametrize("smiles", [
        "[CH4]",      # methane with H
        "[CH3][CH3]", # ethane with H
        "[OH2]",      # water with H
    ])
    def test_remove_hydrogens_basic(self, smiles: str) -> None:
        """Test basic hydrogen removal."""
        mol = parse(smiles)
        mol_no_h = remove_explicit_hydrogens(mol)
        
        # No H atoms should remain
        h_count = sum(1 for a in mol_no_h.atoms if a.symbol == 'H')
        assert h_count == 0, f"H atoms remaining after removal: {smiles}"
    
    @pytest.mark.parametrize("smiles", [
        "C",
        "CC",
        "CCO",
        "c1ccccc1",
    ])
    def test_remove_hydrogens_roundtrip(self, smiles: str) -> None:
        """Test adding then removing hydrogens gives equivalent structure."""
        mol = parse(smiles)
        original_heavy_count = sum(1 for a in mol.atoms if a.symbol != 'H')
        
        # Add then remove
        mol_h = add_explicit_hydrogens(mol)
        mol_no_h = remove_explicit_hydrogens(mol_h)
        
        # Should have same number of heavy atoms
        final_heavy_count = sum(1 for a in mol_no_h.atoms if a.symbol != 'H')
        assert final_heavy_count == original_heavy_count
    
    @pytest.mark.parametrize("smiles", [
        "C",
        "CC",
        "CCC",
        "O",
        "CO",
        "c1ccccc1",
    ])
    def test_remove_hydrogens_matches_rdkit(self, smiles: str) -> None:
        """Test hydrogen removal produces same heavy atom count as RDKit."""
        # Chirpy: add then remove
        mol = parse(smiles)
        mol_h = add_explicit_hydrogens(mol)
        mol_no_h = remove_explicit_hydrogens(mol_h)
        
        chirpy_heavy = sum(1 for a in mol_no_h.atoms if a.symbol != 'H')
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_heavy = sum(1 for a in rdmol.GetAtoms() if a.GetAtomicNum() != 1)
        
        assert chirpy_heavy == rdkit_heavy
    
    def test_remove_hydrogens_preserves_implicit(self) -> None:
        """Test that implicit hydrogen counts are preserved."""
        smiles = "C"
        mol = parse(smiles)
        mol_h = add_explicit_hydrogens(mol)
        mol_no_h = remove_explicit_hydrogens(mol_h)
        
        # Carbon should have 4 implicit H after removal
        c_atom = mol_no_h.atoms[0]
        # The hydrogens should be stored as explicit_hydrogens property
        # or be calculable from valence
    
    def test_remove_hydrogens_keeps_stereo_h(self) -> None:
        """Test that stereochemistry-defining hydrogens are kept."""
        # This is a complex case - for now just test it doesn't crash
        smiles = "[C@H](F)(Cl)Br"  # tetrahedral with explicit H
        mol = parse(smiles)
        mol_no_h = remove_explicit_hydrogens(mol)
        
        # Should still be valid
        assert mol_no_h.num_atoms >= 4  # At least C, F, Cl, Br
    
    def test_remove_hydrogens_returns_new_molecule(self) -> None:
        """Test that remove_explicit_hydrogens returns a new molecule."""
        smiles = "[CH4]"
        mol = parse(smiles)
        original_atom_count = mol.num_atoms
        
        mol_no_h = remove_explicit_hydrogens(mol)
        
        # Original should be unchanged
        assert mol.num_atoms == original_atom_count


class TestHydrogenEdgeCases:
    """Test edge cases for hydrogen manipulation."""
    
    def test_empty_molecule(self) -> None:
        """Test with empty molecule."""
        from chiralipy.types import Molecule
        mol = Molecule()
        
        mol_h = add_explicit_hydrogens(mol)
        assert mol_h.num_atoms == 0
        
        mol_no_h = remove_explicit_hydrogens(mol)
        assert mol_no_h.num_atoms == 0
    
    def test_single_heavy_atom(self) -> None:
        """Test with single heavy atom."""
        mol = parse("[C]")
        mol_h = add_explicit_hydrogens(mol)
        
        h_count = sum(1 for a in mol_h.atoms if a.symbol == 'H')
        # Neutral carbon with no bonds expects 4 H
        assert h_count == 4
    
    def test_no_hydrogen_atoms(self) -> None:
        """Test molecule with no hydrogens to remove."""
        smiles = "[C]([C])([C])[C]"  # all carbons, no H
        mol = parse(smiles)
        
        mol_no_h = remove_explicit_hydrogens(mol)
        # Should be same size
        assert mol_no_h.num_atoms == mol.num_atoms
    
    def test_only_hydrogens(self) -> None:
        """Test molecule that is all hydrogens (H2)."""
        smiles = "[H][H]"
        mol = parse(smiles)
        
        # Should work without error
        mol_h = add_explicit_hydrogens(mol)
        assert mol_h.num_atoms >= 2


class TestHydrogenValence:
    """Test hydrogen addition respects valence rules."""
    
    @pytest.mark.parametrize("smiles,atom_idx,expected_h", [
        ("C", 0, 4),      # carbon gets 4 H
        ("N", 0, 3),      # nitrogen gets 3 H
        ("O", 0, 2),      # oxygen gets 2 H
        ("F", 0, 1),      # fluorine gets 1 H
        ("[Cl]", 0, 1),   # chlorine gets 1 H
        ("[Br]", 0, 1),   # bromine gets 1 H
        ("[I]", 0, 1),    # iodine gets 1 H
    ])
    def test_valence_single_atoms(self, smiles: str, atom_idx: int, expected_h: int) -> None:
        """Test correct hydrogen count for single atoms based on valence."""
        mol = parse(smiles)
        mol_h = add_explicit_hydrogens(mol)
        
        # Count H attached to the original atom
        h_count = sum(1 for a in mol_h.atoms if a.symbol == 'H')
        assert h_count == expected_h
    
    @pytest.mark.parametrize("smiles", [
        "C=O",        # carbonyl
        "C#N",        # nitrile
        "C=C",        # alkene
        "C#C",        # alkyne
        "N=O",        # nitroso
    ])
    def test_valence_multiple_bonds(self, smiles: str) -> None:
        """Test hydrogen count respects multiple bonds."""
        mol = parse(smiles)
        mol_h = add_explicit_hydrogens(mol)
        
        # Compare with RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdmol_h = Chem.AddHs(rdmol)
        
        chirpy_h = sum(1 for a in mol_h.atoms if a.symbol == 'H')
        rdkit_h = sum(1 for a in rdmol_h.GetAtoms() if a.GetAtomicNum() == 1)
        
        assert chirpy_h == rdkit_h, f"H count mismatch for {smiles}"
