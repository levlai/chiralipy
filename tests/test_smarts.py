"""
Tests for SMARTS support in chirpy.

This module tests parsing and writing of SMARTS patterns including:
- Wildcard atoms (*)
- Atom lists [C,N,O]
- Negation [!C]
- Ring membership (R, R0, R1)
- Ring size (r5, r6)
- Degree (D, D2)
- Valence (v, v4)
- Connectivity (X, X2)
- Recursive SMARTS ($(...))
- Dative bonds (->, <-)
- Any bonds (~)
- Quadruple bonds ($)
"""

from __future__ import annotations

import pytest

from chirpy.parser import parse
from chirpy.writer import to_smiles
from chirpy.elements import BondOrder


class TestWildcardAtom:
    """Tests for wildcard atom (*)."""
    
    def test_bare_wildcard(self) -> None:
        """Test bare * atom."""
        mol = parse("*")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
        assert mol.atoms[0].is_wildcard is True
    
    def test_bracket_wildcard(self) -> None:
        """Test [*] atom."""
        mol = parse("[*]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
        assert mol.atoms[0].is_wildcard is True
    
    def test_wildcard_in_chain(self) -> None:
        """Test wildcard in a chain."""
        mol = parse("C*N")
        assert len(mol.atoms) == 3
        assert mol.atoms[1].is_wildcard is True
    
    def test_wildcard_roundtrip(self) -> None:
        """Test round-trip of wildcard."""
        mol = parse("*")
        smiles = to_smiles(mol)
        assert smiles == "*"


class TestAtomLists:
    """Tests for atom lists [C,N,O]."""
    
    def test_two_element_list(self) -> None:
        """Test [C,N] atom list."""
        mol = parse("[C,N]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].atom_list == ["C", "N"]
    
    def test_three_element_list(self) -> None:
        """Test [C,N,O] atom list."""
        mol = parse("[C,N,O]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].atom_list == ["C", "N", "O"]
    
    def test_atom_list_roundtrip(self) -> None:
        """Test round-trip of atom list."""
        mol = parse("[C,N]")
        smiles = to_smiles(mol)
        assert smiles == "[C,N]"


class TestNegation:
    """Tests for atom negation [!C]."""
    
    def test_negated_carbon(self) -> None:
        """Test [!C] negation."""
        mol = parse("[!C]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].atom_list_negated is True
    
    def test_negated_roundtrip(self) -> None:
        """Test round-trip of negation."""
        mol = parse("[!C]")
        smiles = to_smiles(mol)
        assert smiles == "[!C]"


class TestRingQueries:
    """Tests for ring membership and size queries."""
    
    def test_ring_membership_any(self) -> None:
        """Test [R] - any ring."""
        mol = parse("[R]")
        assert mol.atoms[0].ring_count == -1  # -1 = any ring
    
    def test_ring_membership_zero(self) -> None:
        """Test [R0] - not in ring."""
        mol = parse("[R0]")
        assert mol.atoms[0].ring_count == 0
    
    def test_ring_membership_two(self) -> None:
        """Test [R2] - in two rings."""
        mol = parse("[R2]")
        assert mol.atoms[0].ring_count == 2
    
    def test_ring_size_five(self) -> None:
        """Test [r5] - in 5-membered ring."""
        mol = parse("[r5]")
        assert mol.atoms[0].ring_size == 5
    
    def test_ring_size_six(self) -> None:
        """Test [r6] - in 6-membered ring."""
        mol = parse("[r6]")
        assert mol.atoms[0].ring_size == 6
    
    def test_element_with_ring_query(self) -> None:
        """Test [CR] - carbon in any ring."""
        mol = parse("[CR]")
        assert mol.atoms[0].symbol == "C"
        assert mol.atoms[0].ring_count == -1
    
    def test_element_not_in_ring(self) -> None:
        """Test [NR0] - nitrogen not in ring."""
        mol = parse("[NR0]")
        assert mol.atoms[0].symbol == "N"
        assert mol.atoms[0].ring_count == 0


class TestDegreeQuery:
    """Tests for degree queries [D]."""
    
    def test_degree_any(self) -> None:
        """Test [D] - any degree."""
        mol = parse("[D]")
        assert mol.atoms[0].degree_query == -1
    
    def test_degree_two(self) -> None:
        """Test [D2] - degree 2."""
        mol = parse("[D2]")
        assert mol.atoms[0].degree_query == 2
    
    def test_degree_three(self) -> None:
        """Test [D3] - degree 3."""
        mol = parse("[D3]")
        assert mol.atoms[0].degree_query == 3


class TestValenceQuery:
    """Tests for valence queries [v]."""
    
    def test_valence_any(self) -> None:
        """Test [v] - any valence."""
        mol = parse("[v]")
        assert mol.atoms[0].valence_query == -1
    
    def test_valence_four(self) -> None:
        """Test [v4] - valence 4."""
        mol = parse("[v4]")
        assert mol.atoms[0].valence_query == 4


class TestConnectivityQuery:
    """Tests for connectivity queries [X]."""
    
    def test_connectivity_any(self) -> None:
        """Test [X] - any connectivity."""
        mol = parse("[X]")
        assert mol.atoms[0].connectivity_query == -1
    
    def test_connectivity_two(self) -> None:
        """Test [X2] - connectivity 2."""
        mol = parse("[X2]")
        assert mol.atoms[0].connectivity_query == 2
    
    def test_connectivity_four(self) -> None:
        """Test [X4] - connectivity 4."""
        mol = parse("[X4]")
        assert mol.atoms[0].connectivity_query == 4


class TestDativeBonds:
    """Tests for dative/coordinate bonds (-> and <-)."""
    
    def test_forward_dative(self) -> None:
        """Test N->B dative bond."""
        mol = parse("N->B")
        assert len(mol.bonds) == 1
        assert mol.bonds[0].is_dative is True
        assert mol.bonds[0].dative_direction == 1
        assert mol.bonds[0].order == BondOrder.DATIVE
    
    def test_reverse_dative(self) -> None:
        """Test B<-N dative bond."""
        mol = parse("B<-N")
        assert len(mol.bonds) == 1
        assert mol.bonds[0].is_dative is True
        assert mol.bonds[0].dative_direction == -1
    
    def test_dative_in_complex(self) -> None:
        """Test dative bond in metal complex."""
        mol = parse("[Cu](<-N)(<-N)(<-N)<-N")
        assert len(mol.atoms) == 5
        dative_bonds = [b for b in mol.bonds if b.is_dative]
        assert len(dative_bonds) == 4
    
    def test_dative_roundtrip(self) -> None:
        """Test round-trip of dative bond."""
        mol = parse("N->B")
        smiles = to_smiles(mol)
        # Direction may be canonical
        assert "->" in smiles or "<-" in smiles


class TestAnyBond:
    """Tests for SMARTS any bond (~)."""
    
    def test_any_bond(self) -> None:
        """Test C~N any bond."""
        mol = parse("C~N")
        assert len(mol.bonds) == 1
        assert mol.bonds[0].is_any is True
        assert mol.bonds[0].order == BondOrder.ANY
    
    def test_any_bond_roundtrip(self) -> None:
        """Test round-trip of any bond."""
        mol = parse("C~N")
        smiles = to_smiles(mol)
        assert "~" in smiles


class TestQuadrupleBond:
    """Tests for quadruple bonds ($)."""
    
    def test_quadruple_bond(self) -> None:
        """Test C$C quadruple bond."""
        mol = parse("C$C")
        assert len(mol.bonds) == 1
        assert mol.bonds[0].order == BondOrder.QUADRUPLE
    
    def test_quadruple_bond_roundtrip(self) -> None:
        """Test round-trip of quadruple bond."""
        mol = parse("C$C")
        smiles = to_smiles(mol)
        assert smiles == "C$C"


class TestRecursiveSmarts:
    """Tests for recursive SMARTS $(...).
    
    Note: This tests parsing of recursive SMARTS notation.
    Actual recursive matching is a separate concern.
    """
    
    def test_simple_recursive(self) -> None:
        """Test simple recursive SMARTS."""
        mol = parse("[$(C)]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].is_recursive is True
        assert mol.atoms[0].recursive_smarts == "C"
    
    def test_recursive_with_chain(self) -> None:
        """Test recursive SMARTS with chain."""
        mol = parse("[$(CC)]")
        assert mol.atoms[0].is_recursive is True
        assert mol.atoms[0].recursive_smarts == "CC"
    
    def test_recursive_with_ring(self) -> None:
        """Test recursive SMARTS with ring."""
        mol = parse("[$(C1CCCCC1)]")
        assert mol.atoms[0].is_recursive is True
        assert "C1CCCCC1" in mol.atoms[0].recursive_smarts


class TestCombinedQueries:
    """Tests for combined SMARTS queries."""
    
    def test_element_with_ring_and_degree(self) -> None:
        """Test [CRD2] - carbon in ring with degree 2."""
        mol = parse("[CRD2]")
        assert mol.atoms[0].symbol == "C"
        assert mol.atoms[0].ring_count == -1
        assert mol.atoms[0].degree_query == 2
    
    def test_aromatic_nitrogen_in_ring(self) -> None:
        """Test [nR] - aromatic nitrogen in ring."""
        mol = parse("[nR]")
        assert mol.atoms[0].symbol == "n"
        assert mol.atoms[0].is_aromatic is True
        assert mol.atoms[0].ring_count == -1
    
    def test_complex_smarts_chain(self) -> None:
        """Test complex SMARTS chain."""
        mol = parse("[CR]~[NR]")
        assert len(mol.atoms) == 2
        assert mol.atoms[0].ring_count == -1
        assert mol.atoms[1].ring_count == -1
        assert mol.bonds[0].is_any is True


class TestSmartsPatterns:
    """Tests for common SMARTS patterns."""
    
    def test_primary_amine(self) -> None:
        """Test [NX3H2] - primary amine."""
        mol = parse("[NX3H2]")
        assert mol.atoms[0].symbol == "N"
        assert mol.atoms[0].connectivity_query == 3
        assert mol.atoms[0].explicit_hydrogens == 2
    
    def test_secondary_amine(self) -> None:
        """Test [NX3H1] - secondary amine."""
        mol = parse("[NX3H1]")
        assert mol.atoms[0].symbol == "N"
        assert mol.atoms[0].connectivity_query == 3
        assert mol.atoms[0].explicit_hydrogens == 1
    
    def test_carbonyl_carbon(self) -> None:
        """Test [CX3]=[OX1] - carbonyl."""
        mol = parse("[CX3]=[OX1]")
        assert len(mol.atoms) == 2
        assert mol.atoms[0].connectivity_query == 3
        assert mol.atoms[1].connectivity_query == 1
        assert mol.bonds[0].order == 2


class TestBackwardCompatibility:
    """Tests to ensure SMARTS features don't break SMILES parsing."""
    
    def test_regular_smiles_still_work(self) -> None:
        """Test that regular SMILES still parse correctly."""
        test_cases = [
            "C",
            "CC",
            "CCO",
            "c1ccccc1",
            "[Na+]",
            "[O-]",
            "C[C@H](O)F",
            "F/C=C/F",
        ]
        for smiles in test_cases:
            mol = parse(smiles)
            assert len(mol.atoms) > 0, f"Failed to parse {smiles}"
    
    def test_stereo_bonds_still_work(self) -> None:
        """Test that stereo bonds aren't confused with dative."""
        mol = parse("F/C=C/F")
        assert len(mol.bonds) == 3
        # None should be dative
        assert all(not b.is_dative for b in mol.bonds)
