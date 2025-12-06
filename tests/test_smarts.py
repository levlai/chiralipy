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

from chiralipy.parser import parse
from chiralipy.writer import to_smiles
from chiralipy.elements import BondOrder


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
        # Use perceive_aromaticity=False for SMARTS patterns
        mol = parse("[nR]", perceive_aromaticity=False)
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


class TestChargeQueries:
    """Tests for bare charge SMARTS queries."""
    
    def test_positive_charge_only(self) -> None:
        """Test [+] - any positive atom."""
        mol = parse("[+]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
        assert mol.atoms[0].charge == 1
    
    def test_negative_charge_only(self) -> None:
        """Test [-] - any negative atom."""
        mol = parse("[-]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
        assert mol.atoms[0].charge == -1
    
    def test_positive_charge_numeric(self) -> None:
        """Test [+2] - charge +2."""
        mol = parse("[+2]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
        assert mol.atoms[0].charge == 2
    
    def test_negative_charge_numeric(self) -> None:
        """Test [-3] - charge -3."""
        mol = parse("[-3]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
        assert mol.atoms[0].charge == -3
    
    def test_double_plus(self) -> None:
        """Test [++] - charge +2."""
        mol = parse("[++]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].charge == 2
    
    def test_double_minus(self) -> None:
        """Test [--] - charge -2."""
        mol = parse("[--]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].charge == -2


class TestHybridizationQueries:
    """Tests for hybridization SMARTS queries (^)."""
    
    def test_sp_hybridization(self) -> None:
        """Test [^1] - sp hybridized."""
        mol = parse("[^1]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
    
    def test_sp2_hybridization(self) -> None:
        """Test [^2] - sp2 hybridized."""
        mol = parse("[^2]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
    
    def test_sp3_hybridization(self) -> None:
        """Test [^3] - sp3 hybridized."""
        mol = parse("[^3]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
    
    def test_carbon_sp2(self) -> None:
        """Test [C^2] - sp2 carbon."""
        mol = parse("[C^2]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "C"
    
    def test_carbon_sp3(self) -> None:
        """Test [C^3] - sp3 carbon."""
        mol = parse("[C^3]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "C"
    
    def test_atomic_number_with_hybridization(self) -> None:
        """Test [#6^2] - sp2 carbon by atomic number."""
        mol = parse("[#6^2]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "C"


class TestBondExpressions:
    """Tests for SMARTS bond expressions (-;!@, -;@, etc.)."""
    
    def test_not_ring_bond(self) -> None:
        """Test -;!@ - single bond not in ring."""
        mol = parse("C-;!@C")
        assert len(mol.bonds) == 1
        assert mol.bonds[0].is_not_ring_bond is True
    
    def test_ring_bond(self) -> None:
        """Test -;@ - single bond in ring."""
        mol = parse("C-;@C")
        assert len(mol.bonds) == 1
        assert mol.bonds[0].is_ring_bond is True
    
    def test_standalone_ring_bond(self) -> None:
        """Test @ as standalone ring bond marker."""
        mol = parse("C@C")
        assert len(mol.bonds) == 1
        assert mol.bonds[0].is_ring_bond is True
    
    def test_double_not_ring_bond(self) -> None:
        """Test =;!@ - double bond not in ring."""
        mol = parse("C=;!@C")
        assert len(mol.bonds) == 1
        assert mol.bonds[0].order == 2
        assert mol.bonds[0].is_not_ring_bond is True


class TestAtomicNumberLists:
    """Tests for atomic number list SMARTS ([#0,#6,#7])."""
    
    def test_single_atomic_number(self) -> None:
        """Test [#6] - carbon by atomic number."""
        mol = parse("[#6]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "C"
    
    def test_dummy_atom(self) -> None:
        """Test [#0] - dummy atom / attachment point."""
        mol = parse("[#0]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "*"
    
    def test_atomic_number_list(self) -> None:
        """Test [#0,#6,#7] - list of atomic numbers."""
        mol = parse("[#0,#6,#7]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].atomic_number_list == [0, 6, 7]
    
    def test_mixed_atomic_and_symbol(self) -> None:
        """Test [#6,N,O] - mixed atomic number and symbols."""
        mol = parse("[#6,N,O]")
        assert len(mol.atoms) == 1


class TestBRICSPatterns:
    """Tests for RDKit BRICS decomposition patterns."""
    
    @pytest.mark.parametrize("smarts,label", [
        ("[C;!R;!D1]-;!@[#6]", "L8_variant"),
        ("[C;D3]([#0,#6,#7,#8])(=O)", "L1"),
        ("[N;R;$(N(@C(=O))@[C,N,O,S])]", "L10"),
        ("[S;D2](-;!@[#0,#6])", "L11"),
        ("[S;D4]([#6,#0])(=O)(=O)", "L12"),
        ("[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]", "L13"),
        ("[c;$(c(:[c,n,o,s]):[n,o,s])]", "L14"),
        ("[C;$(C(-;@C)-;@C)]", "L15"),
        ("[c;$(c(:c):c)]", "L16"),
        ("[O;D2]-;!@[#0,#6,#1]", "L3"),
        ("[C;!D1;!$(C=*)]-;!@[#6]", "L4"),
        ("[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]", "L5"),
        ("[C;D3;!R](=O)-;!@[#0,#6,#7,#8]", "L6"),
        ("[C;D2,D3]-[#6]", "L7a"),
        ("[C;!R;!D1;!$(C!-*)]", "L8"),
        ("[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]", "L9"),
    ])
    def test_brics_pattern(self, smarts: str, label: str) -> None:
        """Test that BRICS patterns parse correctly."""
        mol = parse(smarts)
        assert len(mol.atoms) >= 1, f"Failed to parse {label}: {smarts}"


class TestComplexRecursiveSmarts:
    """Tests for complex recursive SMARTS patterns."""
    
    def test_recursive_with_atom_list(self) -> None:
        """Test recursive SMARTS containing atom lists."""
        mol = parse("[N;$(C[N,O])]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].is_recursive is True
    
    def test_recursive_with_ring_bond(self) -> None:
        """Test recursive SMARTS with ring bond (@)."""
        mol = parse("[C;$(C@C)]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].is_recursive is True
    
    def test_multiple_recursive(self) -> None:
        """Test multiple recursive SMARTS in one pattern."""
        mol = parse("[N;!$(N=*);!$(N-C)]")
        assert len(mol.atoms) == 1
    
    def test_nested_brackets_in_recursive(self) -> None:
        """Test recursive SMARTS with nested bracket atoms."""
        mol = parse("[N;$(N(@C(=O))@[C,N,O,S])]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].is_recursive is True


class TestDegreeQueryList:
    """Tests for OR'd degree query parsing [D2,D3].
    
    This tests the fix for BRICS L7a/L7b pattern [C;D2,D3]-[#6]
    where the comma-separated degrees should be parsed as OR'd constraints.
    """
    
    def test_degree_query_list(self) -> None:
        """Test [C;D2,D3] parses with degree_query_list."""
        mol = parse("[C;D2,D3]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].degree_query_list == [2, 3]
    
    def test_degree_query_list_in_brics_l7a(self) -> None:
        """Test full L7a pattern [C;D2,D3]-[#6] parses correctly."""
        mol = parse("[C;D2,D3]-[#6]")
        assert len(mol.atoms) == 2
        assert mol.atoms[0].degree_query_list == [2, 3]
        assert mol.atoms[1].atomic_number_list == [6]
    
    def test_single_degree_not_in_list(self) -> None:
        """Test single [D2] uses degree_query, not degree_query_list."""
        mol = parse("[D2]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].degree_query == 2
        assert mol.atoms[0].degree_query_list is None or mol.atoms[0].degree_query_list == []


class TestNegatedBondParsing:
    """Tests for negated bond parsing (!-).
    
    This tests the fix for BRICS L8 pattern [C;!R;!D1;!$(C!-*)]
    where !- means "negated single bond" (must not have any single bonds).
    """
    
    def test_negated_single_bond(self) -> None:
        """Test C!-C parses with negated bond."""
        mol = parse("C!-C")
        assert len(mol.atoms) == 2
        assert len(mol.bonds) == 1
        assert mol.bonds[0].is_negated is True
        assert mol.bonds[0].order == 1
    
    def test_negated_double_bond(self) -> None:
        """Test C!=C parses with negated double bond."""
        mol = parse("C!=C")
        assert len(mol.atoms) == 2
        assert len(mol.bonds) == 1
        assert mol.bonds[0].is_negated is True
        assert mol.bonds[0].order == 2
    
    def test_negated_bond_in_recursive_smarts(self) -> None:
        """Test !-* inside recursive SMARTS."""
        mol = parse("[C;!$(C!-*)]")
        assert len(mol.atoms) == 1
        # The negated recursive SMARTS should parse without error
        assert mol.atoms[0].negated_recursive_smarts is not None
    
    def test_brics_l8_pattern(self) -> None:
        """Test full L8 pattern parses correctly."""
        mol = parse("[C;!R;!D1;!$(C!-*)]")
        assert len(mol.atoms) == 1
        # Atom should have the negated recursive SMARTS for the !-* part
        assert mol.atoms[0].negated_recursive_smarts is not None


class TestNegatedAtomicNumberList:
    """Tests for negated atomic number list parsing [!#6;!#16;!#0;!#1].
    
    This tests the fix for BRICS L5 pattern 
    [N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]
    where [!#6;!#16;!#0;!#1] means "not carbon, not sulfur, not dummy, not hydrogen".
    """
    
    def test_single_negated_atomic_number(self) -> None:
        """Test [!#6] - not carbon."""
        mol = parse("[!#6]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].negated_atomic_number_list == [6]
        assert mol.atoms[0].symbol == "*"  # Wildcard since it's "not carbon"
    
    def test_multiple_negated_atomic_numbers(self) -> None:
        """Test [!#6;!#16;!#0;!#1] - not C, S, *, H."""
        mol = parse("[!#6;!#16;!#0;!#1]")
        assert len(mol.atoms) == 1
        # All negated atomic numbers should be in the list
        assert set(mol.atoms[0].negated_atomic_number_list) == {6, 16, 0, 1}
    
    def test_negated_atomic_number_in_recursive(self) -> None:
        """Test negated atomic numbers inside recursive SMARTS."""
        mol = parse("[N;$(N-[!#6;!#1])]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].is_recursive is True
    
    def test_brics_l5_pattern(self) -> None:
        """Test full L5 pattern parses correctly."""
        mol = parse("[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]")
        assert len(mol.atoms) == 1
        # Should have multiple negated recursive SMARTS
        assert mol.atoms[0].negated_recursive_smarts is not None
        assert len(mol.atoms[0].negated_recursive_smarts) >= 3
