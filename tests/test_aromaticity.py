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
