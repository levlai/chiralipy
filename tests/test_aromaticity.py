"""Tests for aromaticity perception.

This module tests the chirpy aromaticity perception algorithm.
"""

import pytest
from chiralipy import parse
from chiralipy.transform import perceive_aromaticity, AromaticityPerceiver


class TestAromaticPerception:
    """Test aromaticity perception."""

    def test_benzene_aromatic(self):
        """Benzene should be detected as aromatic."""
        mol = parse("c1ccccc1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)

    def test_kekule_benzene(self):
        """Kekulé benzene should be perceived as aromatic."""
        mol = parse("C1=CC=CC=C1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)

    def test_cyclohexane_not_aromatic(self):
        """Cyclohexane should not be aromatic."""
        mol = parse("C1CCCCC1")
        perceive_aromaticity(mol)
        assert not any(a.is_aromatic for a in mol.atoms)

    def test_pyridine_aromatic(self):
        """Pyridine should be aromatic."""
        mol = parse("c1ccncc1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)

    def test_furan_aromatic(self):
        """Furan should be aromatic."""
        mol = parse("c1ccoc1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)

    def test_thiophene_aromatic(self):
        """Thiophene should be aromatic."""
        mol = parse("c1ccsc1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)

    def test_pyrrole_aromatic(self):
        """Pyrrole should be aromatic."""
        mol = parse("c1cc[nH]c1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)

    def test_naphthalene_aromatic(self):
        """Naphthalene should be aromatic."""
        mol = parse("c1ccc2ccccc2c1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)

    def test_cyclopentadiene_not_aromatic(self):
        """Cyclopentadiene is not aromatic (no lone pair contributing)."""
        mol = parse("C1=CC=CC1")
        perceive_aromaticity(mol)
        # The sp3 carbon breaks aromaticity
        aromatic_count = sum(1 for a in mol.atoms if a.is_aromatic)
        assert aromatic_count < len(mol.atoms)


class TestMixedAromaticity:
    """Test molecules with both aromatic and non-aromatic parts."""

    def test_toluene(self):
        """Toluene: aromatic ring with methyl substituent."""
        mol = parse("Cc1ccccc1")
        perceive_aromaticity(mol)
        # Methyl carbon should not be aromatic
        methyl_c = [a for a in mol.atoms if a.symbol == "C" and not a.is_aromatic]
        assert len(methyl_c) >= 1

    def test_phenol(self):
        """Phenol: aromatic ring with OH."""
        mol = parse("Oc1ccccc1")
        perceive_aromaticity(mol)
        # Oxygen is not aromatic, ring carbons are
        aromatic_atoms = [a for a in mol.atoms if a.is_aromatic]
        assert len(aromatic_atoms) == 6

    def test_biphenyl(self):
        """Biphenyl: two aromatic rings."""
        mol = parse("c1ccc(-c2ccccc2)cc1")
        perceive_aromaticity(mol)
        aromatic_atoms = [a for a in mol.atoms if a.is_aromatic]
        assert len(aromatic_atoms) == 12


class TestFusedRingAromaticity:
    """Test aromaticity in fused ring systems."""
    
    def test_naphthalene_bridgehead_aromatic(self):
        """In naphthalene, the bridgehead bond IS aromatic.
        
        Naphthalene is a fully aromatic system - all 11 bonds are aromatic.
        """
        mol = parse("c1ccc2ccccc2c1")
        perceive_aromaticity(mol)
        
        # All atoms should be aromatic
        assert all(a.is_aromatic for a in mol.atoms)
        
        # All bonds should be aromatic (including bridgehead)
        assert all(b.is_aromatic for b in mol.bonds)
    
    def test_hypoxanthine_bridgehead_not_aromatic(self):
        """In purine-like systems, the bridgehead bond is NOT aromatic.
        
        This matches RDKit's aromaticity model where bonds shared between
        multiple rings (bridgehead bonds) are marked as non-aromatic SINGLE bonds.
        
        The canonical SMILES should include explicit '-' for the bridgehead:
        RDKit: Oc1[nH]cnc2ncnc1-2
        """
        mol = parse("c1nc2ncnc-2c(O)[nH]1")
        
        # All atoms in the fused system should be aromatic
        # (except O which is outside the ring)
        aromatic_atoms = [a for a in mol.atoms if a.is_aromatic]
        assert len(aromatic_atoms) >= 8  # 8 atoms in the fused ring system
        
        # The bridgehead bond (between the two rings) should NOT be aromatic
        # Find the bond connecting the two ring systems
        bridgehead_bonds = []
        for bond in mol.bonds:
            a1 = mol.atoms[bond.atom1_idx]
            a2 = mol.atoms[bond.atom2_idx]
            # Bridgehead is C-C bond between two aromatic carbons
            if (a1.symbol.upper() == 'C' and a2.symbol.upper() == 'C' and 
                a1.is_aromatic and a2.is_aromatic and not bond.is_aromatic):
                bridgehead_bonds.append(bond)
        
        # Should have exactly one non-aromatic bridgehead bond
        assert len(bridgehead_bonds) == 1, (
            f"Expected 1 non-aromatic bridgehead bond, found {len(bridgehead_bonds)}"
        )
    
    def test_hypoxanthine_canonical_smiles_matches_rdkit(self):
        """The canonical SMILES for hypoxanthine should match RDKit.
        
        This tests that the writer outputs explicit '-' for bridgehead bonds.
        """
        rdkit = pytest.importorskip("rdkit")
        from rdkit import Chem
        from chiralipy import to_smiles
        
        # Parse the molecule
        smiles = "c1nc2ncnc-2c(O)[nH]1"
        mol = parse(smiles)
        
        # Get canonical SMILES from both
        chirpy_canonical = to_smiles(mol)
        
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_canonical = Chem.MolToSmiles(rdkit_mol)
        
        assert chirpy_canonical == rdkit_canonical, (
            f"Canonical SMILES mismatch:\n"
            f"  chiralipy: {chirpy_canonical}\n"
            f"  RDKit:     {rdkit_canonical}"
        )

    def test_quinone_indole_fused_not_aromatic(self):
        """Quinone fused with indole - quinone ring is NOT aromatic.
        
        The molecule Cc1=c-c(=O)-c2c([nH]c3ccc(Cl)cc23)-c-1=O has:
        - An indole-like aromatic system (fused pyrrole + benzene)
        - A quinone-like ring with C=O groups that is NOT aromatic
        
        The carbonyl carbons break the aromaticity of the quinone ring.
        RDKit canonical: CC1=CC(=O)c2c([nH]c3ccc(Cl)cc23)C1=O
        """
        rdkit = pytest.importorskip("rdkit")
        from rdkit import Chem
        from chiralipy import to_smiles
        
        smiles = "Cc1=c-c(=O)-c2c([nH]c3ccc(Cl)cc23)-c-1=O"
        mol = parse(smiles)
        
        # Get canonical SMILES from both
        chirpy_canonical = to_smiles(mol)
        
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_canonical = Chem.MolToSmiles(rdkit_mol)
        
        assert chirpy_canonical == rdkit_canonical, (
            f"Canonical SMILES mismatch:\n"
            f"  chiralipy: {chirpy_canonical}\n"
            f"  RDKit:     {rdkit_canonical}"
        )

    def test_triazole_fused_to_saturated_ring(self):
        """Triazole ring fused to a saturated ring should still be aromatic.
        
        The molecule C1Cn2c(nnc2C(F)(F)F)C1 has:
        - A triazole ring (5-membered with 3 N) which IS aromatic
        - A saturated 5-membered ring (piperidine-like) fused to it
        
        The triazole should be aromatic even though it's fused to a saturated ring.
        This is important because the sp3 carbons in the saturated ring don't
        break the aromaticity of the triazole - they're not part of that ring.
        
        RDKit canonical: FC(F)(F)c1nnc2n1CCC2
        """
        rdkit = pytest.importorskip("rdkit")
        from rdkit import Chem
        from chiralipy import to_smiles
        
        smiles = "C1Cn2c(nnc2C(F)(F)F)C1"
        mol = parse(smiles)
        
        # Get canonical SMILES from both
        chirpy_canonical = to_smiles(mol)
        
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_canonical = Chem.MolToSmiles(rdkit_mol)
        
        assert chirpy_canonical == rdkit_canonical, (
            f"Canonical SMILES mismatch:\n"
            f"  chiralipy: {chirpy_canonical}\n"
            f"  RDKit:     {rdkit_canonical}"
        )

    def test_difluorophenyl_triazole_piperazine(self):
        """Full molecule with difluorophenyl, triazole, and piperazine.
        
        The molecule O=C(c1cccc(F)c1F)N1CCn2c(nnc2C(F)(F)F)C1 has:
        - A difluorobenzene (aromatic)
        - A triazole ring (aromatic) fused to a saturated ring
        - A piperazine-like ring (non-aromatic)
        
        RDKit canonical: O=C(c1cccc(F)c1F)N1CCn2c(nnc2C(F)(F)F)C1
        """
        rdkit = pytest.importorskip("rdkit")
        from rdkit import Chem
        from chiralipy import to_smiles
        
        smiles = "O=C(c1cccc(F)c1F)N1CCn2c(nnc2C(F)(F)F)C1"
        mol = parse(smiles)
        
        # Get canonical SMILES from both
        chirpy_canonical = to_smiles(mol)
        
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_canonical = Chem.MolToSmiles(rdkit_mol)
        
        assert chirpy_canonical == rdkit_canonical, (
            f"Canonical SMILES mismatch:\n"
            f"  chiralipy: {chirpy_canonical}\n"
            f"  RDKit:     {rdkit_canonical}"
        )

    def test_thiadiazole_lactam_exocyclic_double_bond(self):
        """Thiadiazole-lactam: ring with exocyclic double bond is NOT aromatic.
        
        The molecule O=C1Cn2nc(-c3ccccc3)sc2=N1 has:
        - A thiadiazole-like ring that would normally be aromatic
        - But it has an exocyclic C=N double bond from the ring carbon
        
        RDKit rule: atoms with exocyclic double bonds cannot be aromatic candidates
        (allowExocyclicMultipleBonds = false in RDKit's aromaticity perception).
        
        So the thiadiazole ring is NOT aromatic, while the phenyl is.
        
        RDKit canonical: O=C1CN2N=C(c3ccccc3)SC2=N1
        """
        rdkit = pytest.importorskip("rdkit")
        from rdkit import Chem
        from chiralipy import to_smiles
        
        smiles = "O=C1Cn2nc(-c3ccccc3)sc2=N1"
        mol = parse(smiles)
        
        # Get canonical SMILES from both
        chirpy_canonical = to_smiles(mol)
        
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_canonical = Chem.MolToSmiles(rdkit_mol)
        
        assert chirpy_canonical == rdkit_canonical, (
            f"Canonical SMILES mismatch:\n"
            f"  chiralipy: {chirpy_canonical}\n"
            f"  RDKit:     {rdkit_canonical}"
        )


class TestAromaticityPreserver:
    """Test AromaticityPerceiver class."""

    def test_perceiver_class(self):
        """AromaticityPerceiver should work correctly."""
        mol = parse("c1ccccc1")
        perceiver = AromaticityPerceiver()  # No args, uses default max_ring_size=6
        perceiver.perceive(mol)
        assert all(a.is_aromatic for a in mol.atoms)

    def test_perceiver_multiple_calls(self):
        """Multiple perception calls should be idempotent."""
        mol = parse("c1ccccc1")
        perceive_aromaticity(mol)
        state1 = [a.is_aromatic for a in mol.atoms]
        perceive_aromaticity(mol)
        state2 = [a.is_aromatic for a in mol.atoms]
        assert state1 == state2
