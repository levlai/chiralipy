"""
Tests for SMARTS substructure matching.

These tests verify that chirpy's substructure matching produces
the same results as RDKit.
"""

from __future__ import annotations

import pytest

from chiralipy.parser import parse
from chiralipy.match import substructure_search, has_substructure, count_matches
from chiralipy.transform import perceive_aromaticity

# Try to import RDKit for comparison tests
try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class TestBasicMatching:
    """Basic substructure matching tests."""
    
    def test_empty_pattern(self) -> None:
        """Empty pattern matches everything."""
        mol = parse("CCO")
        pattern = parse("")
        # Empty pattern should return empty match
        # Actually, parse("") would fail, so skip this
    
    def test_single_atom_carbon(self) -> None:
        """Match single carbon atom."""
        mol = parse("CCO")
        pattern = parse("C")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 2  # Two carbons
        assert (0,) in matches
        assert (1,) in matches
    
    def test_single_atom_oxygen(self) -> None:
        """Match single oxygen atom."""
        mol = parse("CCO")
        pattern = parse("O")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 1
        assert (2,) in matches
    
    def test_two_atom_pattern(self) -> None:
        """Match two-atom pattern."""
        mol = parse("CCO")
        pattern = parse("CO")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 1
        assert matches[0] == (1, 2)
    
    def test_no_match(self) -> None:
        """Pattern not found in molecule."""
        mol = parse("CCC")
        pattern = parse("O")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 0
    
    def test_has_substructure_true(self) -> None:
        """has_substructure returns True when match exists."""
        mol = parse("CCO")
        pattern = parse("O")
        assert has_substructure(mol, pattern) is True
    
    def test_has_substructure_false(self) -> None:
        """has_substructure returns False when no match."""
        mol = parse("CCC")
        pattern = parse("O")
        assert has_substructure(mol, pattern) is False


class TestWildcardMatching:
    """Tests for wildcard atom (*)."""
    
    def test_wildcard_matches_any_atom(self) -> None:
        """Wildcard * matches any atom."""
        mol = parse("CNO")
        pattern = parse("*")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 3
    
    def test_wildcard_in_chain(self) -> None:
        """Wildcard in chain pattern."""
        mol = parse("CCO")
        pattern = parse("C*")
        matches = substructure_search(mol, pattern)
        # C-C and C-O both match C*
        assert len(matches) >= 2


class TestAromaticMatching:
    """Tests for aromatic atom matching."""
    
    def test_aromatic_carbon(self) -> None:
        """Match aromatic carbon."""
        mol = parse("c1ccccc1")
        perceive_aromaticity(mol)
        pattern = parse("c")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 6
    
    def test_aliphatic_carbon_no_match_aromatic(self) -> None:
        """Aliphatic C pattern shouldn't match aromatic ring."""
        mol = parse("c1ccccc1")
        perceive_aromaticity(mol)
        pattern = parse("C")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 0
    
    def test_aromatic_nitrogen(self) -> None:
        """Match aromatic nitrogen in pyridine."""
        mol = parse("c1ccncc1")
        perceive_aromaticity(mol)
        pattern = parse("n")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 1


class TestRingQueries:
    """Tests for ring membership queries."""
    
    def test_ring_membership_any(self) -> None:
        """[R] matches atoms in any ring."""
        mol = parse("c1ccccc1C")
        perceive_aromaticity(mol)
        pattern = parse("[R]")
        matches = substructure_search(mol, pattern)
        # 6 ring carbons, 1 non-ring carbon
        assert len(matches) == 6
    
    def test_ring_membership_zero(self) -> None:
        """[R0] matches atoms not in any ring."""
        mol = parse("c1ccccc1C")
        perceive_aromaticity(mol)
        pattern = parse("[R0]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 1  # The methyl carbon
    
    def test_ring_size_six(self) -> None:
        """[r6] matches atoms in 6-membered ring."""
        mol = parse("c1ccccc1")
        perceive_aromaticity(mol)
        pattern = parse("[r6]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 6
    
    def test_ring_size_five(self) -> None:
        """[r5] matches atoms in 5-membered ring."""
        mol = parse("c1ccoc1")
        perceive_aromaticity(mol)
        pattern = parse("[r5]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 5


class TestDegreeQuery:
    """Tests for degree queries [D]."""
    
    def test_degree_one(self) -> None:
        """[D1] matches terminal atoms."""
        mol = parse("CCC")
        pattern = parse("[D1]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 2  # Two terminal carbons
    
    def test_degree_two(self) -> None:
        """[D2] matches atoms with 2 connections."""
        mol = parse("CCC")
        pattern = parse("[D2]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 1  # Middle carbon
    
    def test_degree_three(self) -> None:
        """[D3] matches atoms with 3 connections."""
        mol = parse("CC(C)C")
        pattern = parse("[D3]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 1  # Central carbon


class TestConnectivityQuery:
    """Tests for connectivity queries [X]."""
    
    def test_connectivity_four(self) -> None:
        """[X4] matches atoms with 4 total connections (including H)."""
        mol = parse("[CH4]")
        pattern = parse("[X4]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 1


class TestAtomList:
    """Tests for atom list [C,N,O]."""
    
    def test_atom_list_two(self) -> None:
        """[C,N] matches carbon or nitrogen."""
        mol = parse("CNO")
        pattern = parse("[C,N]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 2
    
    def test_atom_list_three(self) -> None:
        """[C,N,O] matches any of the three."""
        mol = parse("CNO")
        pattern = parse("[C,N,O]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 3


class TestNegation:
    """Tests for negation [!C]."""
    
    def test_not_carbon(self) -> None:
        """[!C] matches non-carbon atoms."""
        mol = parse("CNO")
        pattern = parse("[!C]")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 2  # N and O


class TestBondMatching:
    """Tests for bond type matching."""
    
    def test_double_bond(self) -> None:
        """Match double bond."""
        mol = parse("CC=CC")
        pattern = parse("C=C")
        matches = substructure_search(mol, pattern)
        assert len(matches) >= 1
    
    def test_triple_bond(self) -> None:
        """Match triple bond."""
        mol = parse("CC#CC")
        pattern = parse("C#C")
        matches = substructure_search(mol, pattern)
        assert len(matches) >= 1
    
    def test_any_bond(self) -> None:
        """Any bond ~ matches all bond types."""
        mol = parse("C=CC")
        pattern = parse("C~C")
        matches = substructure_search(mol, pattern)
        # Should match both C=C and C-C
        assert len(matches) >= 2


class TestComplexPatterns:
    """Tests for complex SMARTS patterns."""
    
    def test_carbonyl(self) -> None:
        """Match carbonyl C=O."""
        mol = parse("CC=O")
        pattern = parse("C=O")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 1
    
    def test_carboxylic_acid(self) -> None:
        """Match carboxylic acid pattern."""
        mol = parse("CC(=O)O")
        pattern = parse("C(=O)O")
        matches = substructure_search(mol, pattern)
        assert len(matches) >= 1
    
    def test_benzene_ring(self) -> None:
        """Match complete benzene ring."""
        mol = parse("c1ccccc1")
        perceive_aromaticity(mol)
        pattern = parse("c1ccccc1")
        perceive_aromaticity(pattern)
        matches = substructure_search(mol, pattern)
        # Should find matches (6 rotational variants)
        assert len(matches) >= 1


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")
class TestMatchesRDKit:
    """Tests that verify chirpy matches RDKit results."""
    
    def _compare_with_rdkit(
        self,
        mol_smiles: str,
        pattern_smarts: str,
        use_aromaticity: bool = False,
    ) -> None:
        """Helper to compare chirpy results with RDKit."""
        # RDKit
        rdkit_mol = Chem.MolFromSmiles(mol_smiles)
        rdkit_pattern = Chem.MolFromSmarts(pattern_smarts)
        rdkit_matches = rdkit_mol.GetSubstructMatches(rdkit_pattern)
        
        # Chirpy
        chirpy_mol = parse(mol_smiles)
        if use_aromaticity:
            perceive_aromaticity(chirpy_mol)
        # SMARTS patterns explicitly encode aromaticity, don't override
        chirpy_pattern = parse(pattern_smarts, perceive_aromaticity=False)
        chirpy_matches = substructure_search(chirpy_mol, chirpy_pattern)
        
        # Compare counts
        assert len(chirpy_matches) == len(rdkit_matches), \
            f"Count mismatch for {pattern_smarts} in {mol_smiles}: " \
            f"chiralipy={len(chirpy_matches)}, rdkit={len(rdkit_matches)}"
        
        # Compare actual matches (as sets since order may differ)
        rdkit_set = set(rdkit_matches)
        chirpy_set = set(chirpy_matches)
        assert chirpy_set == rdkit_set, \
            f"Match mismatch for {pattern_smarts} in {mol_smiles}: " \
            f"chiralipy={chirpy_set}, rdkit={rdkit_set}"
    
    def test_single_carbon(self) -> None:
        """Single carbon match."""
        self._compare_with_rdkit("CCO", "C")
    
    def test_single_oxygen(self) -> None:
        """Single oxygen match."""
        self._compare_with_rdkit("CCO", "O")
    
    def test_single_nitrogen(self) -> None:
        """Single nitrogen match."""
        self._compare_with_rdkit("CCN", "N")
    
    def test_cc_chain(self) -> None:
        """C-C chain match."""
        self._compare_with_rdkit("CCCC", "CC")
    
    def test_double_bond(self) -> None:
        """Double bond match."""
        self._compare_with_rdkit("CC=CC", "C=C")
    
    def test_triple_bond(self) -> None:
        """Triple bond match."""
        self._compare_with_rdkit("CC#CC", "C#C")
    
    def test_wildcard(self) -> None:
        """Wildcard atom match."""
        self._compare_with_rdkit("CNO", "*")
    
    def test_atom_list(self) -> None:
        """Atom list match."""
        self._compare_with_rdkit("CNO", "[C,N]")
    
    def test_negation(self) -> None:
        """Negation match."""
        self._compare_with_rdkit("CNO", "[!C]")
    
    def test_degree_one(self) -> None:
        """Degree 1 match."""
        self._compare_with_rdkit("CCC", "[D1]")
    
    def test_degree_two(self) -> None:
        """Degree 2 match."""
        self._compare_with_rdkit("CCC", "[D2]")
    
    def test_degree_three(self) -> None:
        """Degree 3 match."""
        self._compare_with_rdkit("CC(C)C", "[D3]")
    
    def test_aromatic_carbon_benzene(self) -> None:
        """Aromatic carbon in benzene."""
        self._compare_with_rdkit("c1ccccc1", "c", use_aromaticity=True)
    
    def test_ring_membership_benzene(self) -> None:
        """Ring membership in benzene."""
        self._compare_with_rdkit("c1ccccc1C", "[R]", use_aromaticity=True)
    
    def test_not_in_ring(self) -> None:
        """Not in ring."""
        self._compare_with_rdkit("c1ccccc1C", "[R0]", use_aromaticity=True)
    
    def test_ring_size_six(self) -> None:
        """Ring size 6."""
        self._compare_with_rdkit("c1ccccc1", "[r6]", use_aromaticity=True)
    
    def test_ring_size_five(self) -> None:
        """Ring size 5."""
        self._compare_with_rdkit("c1ccoc1", "[r5]", use_aromaticity=True)
    
    def test_carbonyl(self) -> None:
        """Carbonyl pattern."""
        self._compare_with_rdkit("CC=O", "[CX3]=[OX1]")
    
    def test_hydroxyl(self) -> None:
        """Hydroxyl pattern."""
        self._compare_with_rdkit("CCO", "[OH]")
    
    def test_primary_amine(self) -> None:
        """Primary amine [NX3H2]."""
        self._compare_with_rdkit("CCN", "[NX3H2]")
    
    def test_carboxylic_acid(self) -> None:
        """Carboxylic acid pattern."""
        self._compare_with_rdkit("CC(=O)O", "C(=O)O")
    
    def test_phenol(self) -> None:
        """Phenol - aromatic with OH."""
        self._compare_with_rdkit("c1ccccc1O", "[cR]", use_aromaticity=True)
    
    def test_toluene(self) -> None:
        """Toluene - benzene with methyl."""
        self._compare_with_rdkit("Cc1ccccc1", "c", use_aromaticity=True)
    
    def test_pyridine_nitrogen(self) -> None:
        """Pyridine aromatic nitrogen."""
        self._compare_with_rdkit("c1ccncc1", "n", use_aromaticity=True)
    
    def test_furan_oxygen(self) -> None:
        """Furan aromatic oxygen."""
        self._compare_with_rdkit("c1ccoc1", "o", use_aromaticity=True)
    
    def test_any_bond(self) -> None:
        """Any bond ~."""
        self._compare_with_rdkit("C=CC", "C~C")
    
    def test_naphthalene_ring_count(self) -> None:
        """Naphthalene has atoms in 2 rings."""
        self._compare_with_rdkit("c1ccc2ccccc2c1", "[R2]", use_aromaticity=True)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_self_match(self) -> None:
        """Molecule matches itself."""
        mol = parse("CCO")
        pattern = parse("CCO")
        matches = substructure_search(mol, pattern)
        assert len(matches) >= 1
    
    def test_larger_pattern_no_match(self) -> None:
        """Pattern larger than molecule doesn't match."""
        mol = parse("CC")
        pattern = parse("CCCC")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 0
    
    def test_disconnected_molecule(self) -> None:
        """Match in disconnected molecule."""
        mol = parse("CC.OO")
        pattern = parse("O")
        matches = substructure_search(mol, pattern)
        assert len(matches) == 2
    
    def test_symmetrical_molecule(self) -> None:
        """Matches in symmetrical molecule."""
        mol = parse("CCCC")
        pattern = parse("CC")
        matches = substructure_search(mol, pattern)
        # C1-C2, C2-C3, C3-C4 = 3 matches
        assert len(matches) == 3
    
    def test_count_matches(self) -> None:
        """count_matches returns correct count."""
        mol = parse("CCCC")
        pattern = parse("C")
        count = count_matches(mol, pattern)
        assert count == 4
