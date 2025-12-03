"""Tests for kekulize module comparing with RDKit."""

from __future__ import annotations

import pytest
from chiralipy import parse, to_smiles
from chiralipy.transform import kekulize


# Skip if RDKit not available
rdkit = pytest.importorskip("rdkit")
from rdkit import Chem


class TestKekulize:
    """Test kekulization functionality."""
    
    @pytest.mark.parametrize("smiles", [
        # Simple aromatics
        "c1ccccc1",       # benzene
        "c1ccncc1",       # pyridine
        "c1ccc2ccccc2c1", # naphthalene
        "c1ccoc1",        # furan
        "c1cc[nH]c1",     # pyrrole
        "c1ccsc1",        # thiophene
        "c1cnc2ccccc2n1", # quinazoline
        "c1ccc2[nH]ccc2c1", # indole
        "c1ccc2c(c1)cccc2", # naphthalene variant
        # Fused aromatics
        "c1ccc2c(c1)ccc1ccccc12", # phenanthrene
        # Heterocycles
        "c1cncnc1",       # pyrimidine
        "c1ccnnc1",       # pyridazine
        "c1ccccn1",       # pyridine variant
        "c1ncc2ccccc2n1", # quinoxaline
    ])
    def test_kekulize_simple_aromatics(self, smiles: str) -> None:
        """Test kekulization of simple aromatic molecules."""
        mol = parse(smiles)
        kek_mol = kekulize(mol)
        
        # Verify no aromatic bonds remain
        for bond in kek_mol.bonds:
            assert not bond.is_aromatic, f"Aromatic bond found after kekulization: {smiles}"
        
        # Verify no aromatic atoms remain
        for atom in kek_mol.atoms:
            assert not atom.is_aromatic, f"Aromatic atom found after kekulization: {smiles}"
        
        # Verify we have alternating single/double bonds
        double_bonds = sum(1 for b in kek_mol.bonds if b.order == 2)
        assert double_bonds > 0, f"No double bonds after kekulization: {smiles}"
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",       # benzene
        "c1ccncc1",       # pyridine
        "c1ccoc1",        # furan
        "c1cc[nH]c1",     # pyrrole
        "c1ccsc1",        # thiophene
        "CCc1ccccc1",     # ethylbenzene
        "c1ccc(O)cc1",    # phenol
    ])
    def test_kekulize_matches_rdkit(self, smiles: str) -> None:
        """Test that kekulization produces valid structures like RDKit."""
        # Chirpy
        mol = parse(smiles)
        kek_mol = kekulize(mol)
        chirpy_smi = to_smiles(kek_mol)
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(rdmol)
        rdkit_smi = Chem.MolToSmiles(rdmol, kekuleSmiles=True)
        
        # Both should be parseable and equivalent molecules
        # (exact SMILES may differ, but structure should be valid)
        rdmol_from_chirpy = Chem.MolFromSmiles(chirpy_smi)
        assert rdmol_from_chirpy is not None, f"RDKit couldn't parse chirpy kekulized: {chirpy_smi}"
        
        # Verify same number of double bonds
        chirpy_doubles = sum(1 for b in kek_mol.bonds if b.order == 2)
        rdkit_doubles = sum(1 for b in rdmol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        assert chirpy_doubles == rdkit_doubles, f"Different double bond count for {smiles}"
    
    def test_kekulize_non_aromatic_unchanged(self) -> None:
        """Test that non-aromatic molecules pass through unchanged."""
        smiles = "CCCCC"
        mol = parse(smiles)
        kek_mol = kekulize(mol)
        
        # Should be same structure
        assert kek_mol.num_atoms == mol.num_atoms
        assert kek_mol.num_bonds == mol.num_bonds
        
        for i, bond in enumerate(kek_mol.bonds):
            assert bond.order == mol.bonds[i].order
    
    def test_kekulize_preserves_connectivity(self) -> None:
        """Test that kekulization preserves atom connectivity."""
        smiles = "c1ccccc1C"  # toluene
        mol = parse(smiles)
        kek_mol = kekulize(mol)
        
        # Same number of atoms and bonds
        assert kek_mol.num_atoms == mol.num_atoms
        assert kek_mol.num_bonds == mol.num_bonds
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",       # benzene (6 carbons, 3 double bonds)
        "c1ccc2ccccc2c1", # naphthalene (10 carbons, 5 double bonds)
    ])
    def test_kekulize_correct_double_bond_count(self, smiles: str) -> None:
        """Test correct number of double bonds after kekulization."""
        mol = parse(smiles)
        kek_mol = kekulize(mol)
        
        # Count aromatic carbons (should have n/2 double bonds for n-membered ring)
        rdmol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(rdmol)
        expected_doubles = sum(1 for b in rdmol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        
        actual_doubles = sum(1 for b in kek_mol.bonds if b.order == 2)
        assert actual_doubles == expected_doubles
    
    def test_kekulize_mixed_aromatic_aliphatic(self) -> None:
        """Test kekulization of molecule with both aromatic and aliphatic parts."""
        smiles = "CCCc1ccccc1CCC"  # propylbenzene
        mol = parse(smiles)
        kek_mol = kekulize(mol)
        
        # Ring should have alternating bonds, aliphatic should be single
        ring_atoms = {i for i, a in enumerate(mol.atoms) if a.is_aromatic}
        
        for bond in kek_mol.bonds:
            is_ring_bond = bond.atom1_idx in ring_atoms and bond.atom2_idx in ring_atoms
            if not is_ring_bond:
                assert bond.order == 1, "Non-ring bond should be single"
    
    def test_kekulize_returns_new_molecule(self) -> None:
        """Test that kekulize returns a new molecule, not modifying original."""
        smiles = "c1ccccc1"
        mol = parse(smiles)
        
        # Store original state
        original_aromatic_bonds = sum(1 for b in mol.bonds if b.is_aromatic)
        
        kek_mol = kekulize(mol)
        
        # Original should be unchanged
        assert sum(1 for b in mol.bonds if b.is_aromatic) == original_aromatic_bonds
        
        # Kekulized should be different
        assert sum(1 for b in kek_mol.bonds if b.is_aromatic) == 0
    
    def test_kekulize_with_substituents(self) -> None:
        """Test kekulization of substituted aromatics."""
        test_cases = [
            "c1ccc(C)cc1",     # toluene
            "c1ccc(O)cc1",     # phenol
            "c1ccc(N)cc1",     # aniline
            "c1ccc(Cl)cc1",    # chlorobenzene
            "c1ccc(F)cc1",     # fluorobenzene
            "c1ccc(C(=O)O)cc1",# benzoic acid
        ]
        
        for smiles in test_cases:
            mol = parse(smiles)
            kek_mol = kekulize(mol)
            
            # Should produce valid kekulized structure
            kek_smi = to_smiles(kek_mol)
            rdmol = Chem.MolFromSmiles(kek_smi)
            assert rdmol is not None, f"Invalid kekulized structure for {smiles}: {kek_smi}"


class TestKekulizeEdgeCases:
    """Test edge cases for kekulization."""
    
    def test_kekulize_empty_molecule(self) -> None:
        """Test kekulization of empty molecule."""
        from chiralipy.types import Molecule
        mol = Molecule()
        kek_mol = kekulize(mol)
        assert kek_mol.num_atoms == 0
        assert kek_mol.num_bonds == 0
    
    def test_kekulize_single_atom(self) -> None:
        """Test kekulization of single atom."""
        mol = parse("[OH2]")
        kek_mol = kekulize(mol)
        assert kek_mol.num_atoms == 1
    
    def test_kekulize_non_aromatic_ring(self) -> None:
        """Test kekulization of non-aromatic ring."""
        smiles = "C1CCCCC1"  # cyclohexane
        mol = parse(smiles)
        kek_mol = kekulize(mol)
        
        # Should have all single bonds
        for bond in kek_mol.bonds:
            assert bond.order == 1


class TestKekulizeFromSmiles:
    """Test convenience function for kekulizing SMILES strings."""
    
    def test_kekulize_smiles_string(self) -> None:
        """Test kekulizing from SMILES string."""
        smiles = "c1ccccc1"
        mol = parse(smiles)
        kek_mol = kekulize(mol)
        kek_smi = to_smiles(kek_mol)
        
        # Result should be valid and not contain lowercase
        assert 'c' not in kek_smi
        
        # Should be parseable by RDKit
        rdmol = Chem.MolFromSmiles(kek_smi)
        assert rdmol is not None
