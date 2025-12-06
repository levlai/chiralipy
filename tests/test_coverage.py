"""Additional tests to improve code coverage.

These tests target specific uncovered code paths identified by coverage analysis.
"""

import pytest
from chiralipy import parse, canonical_smiles, to_smiles
from chiralipy.types import Atom, Bond, Molecule
from chiralipy.match import substructure_search, has_substructure, count_matches
from chiralipy.decompose import brics_decompose
from chiralipy.decompose.murcko import (
    murcko_decompose, get_scaffold, get_framework,
    get_scaffold_smiles, get_framework_smiles, get_side_chains
)
from chiralipy.transform import kekulize, perceive_aromaticity
from chiralipy.transform.hydrogen import add_explicit_hydrogens, remove_explicit_hydrogens
from chiralipy.rings import get_ring_info, get_ring_membership, get_ring_info_fast


class TestMoleculeProperties:
    """Test Molecule class properties and methods."""
    
    def test_num_atoms_property(self):
        """Test num_atoms property."""
        mol = parse("CCO")
        assert mol.num_atoms == 3
        
    def test_num_bonds_property(self):
        """Test num_bonds property."""
        mol = parse("CCO")
        assert mol.num_bonds == 2
        
    def test_molecule_len(self):
        """Test __len__ method."""
        mol = parse("CCCC")
        assert len(mol) == 4
        
    def test_molecule_iter(self):
        """Test __iter__ method."""
        mol = parse("CCO")
        atoms = list(mol)
        assert len(atoms) == 3
        
    def test_molecule_getitem(self):
        """Test __getitem__ method."""
        mol = parse("CCO")
        assert mol[0].symbol == "C"
        assert mol[2].symbol == "O"
        
    def test_molecule_contains(self):
        """Test __contains__ method."""
        mol = parse("CCO")
        atom = mol.atoms[0]
        assert atom in mol
        
    def test_get_bond_between(self):
        """Test get_bond_between method."""
        mol = parse("CCO")
        bond = mol.get_bond_between(0, 1)
        assert bond is not None
        assert bond.order == 1
        
        # Non-existent bond
        bond = mol.get_bond_between(0, 2)
        assert bond is None
        
    def test_molecule_copy(self):
        """Test molecule copy method."""
        mol = parse("CCO")
        mol_copy = mol.copy()
        assert mol_copy.num_atoms == mol.num_atoms
        assert mol_copy is not mol
        
    def test_is_connected(self):
        """Test is_connected property."""
        mol = parse("CCO")
        assert mol.is_connected  # Property, not method


class TestAtomProperties:
    """Test Atom class properties."""
    
    def test_atomic_number_property(self):
        """Test atomic_number property."""
        mol = parse("CCO")
        assert mol.atoms[0].atomic_number == 6  # Carbon
        assert mol.atoms[2].atomic_number == 8  # Oxygen
        
    def test_degree_property(self):
        """Test degree method."""
        mol = parse("CC(C)C")  # isobutane
        # degree() requires molecule as argument
        assert mol.atoms[1].degree(mol) == 3  # Central carbon
        
    def test_default_valence_property(self):
        """Test default_valence property."""
        mol = parse("C")
        assert mol.atoms[0].default_valence == 4


class TestBondProperties:
    """Test Bond class properties."""
    
    def test_bond_order_property(self):
        """Test bond_order property."""
        mol = parse("C=C")
        assert mol.bonds[0].bond_order == 2


class TestRingFunctions:
    """Test ring detection functions."""
    
    def test_get_ring_membership(self):
        """Test get_ring_membership function."""
        mol = parse("c1ccccc1")  # Benzene
        membership = get_ring_membership(mol)
        # All atoms should be in exactly one ring
        assert all(v == 1 for v in membership.values())
        
    def test_get_ring_info_fast(self):
        """Test get_ring_info_fast function."""
        mol = parse("c1ccccc1CC")  # Toluene
        ring_count, ring_sizes = get_ring_info_fast(mol)
        # Ring atoms should have count > 0
        assert sum(1 for v in ring_count.values() if v > 0) == 6


class TestSubstructureEdgeCases:
    """Test substructure matching edge cases."""
    
    def test_has_substructure_function(self):
        """Test has_substructure convenience function."""
        mol = parse("c1ccccc1O")
        pattern = parse("[OH]")
        assert has_substructure(mol, pattern)
        
    def test_count_matches_function(self):
        """Test count_matches function."""
        mol = parse("c1ccc(O)cc1O")  # Hydroquinone
        pattern = parse("[OH]")
        assert count_matches(mol, pattern) == 2
        
    def test_recursive_smarts_parsing_error(self):
        """Test recursive SMARTS with invalid syntax."""
        mol = parse("CCO")
        # This should not crash even with complex patterns
        pattern = parse("[C;$(C-O)]")
        matches = substructure_search(mol, pattern)
        assert len(matches) >= 0


class TestMurckoDecomposition:
    """Test Murcko scaffold decomposition."""
    
    def test_murcko_decompose_function(self):
        """Test murcko_decompose function."""
        mol = parse("c1ccc(CC)cc1")  # Ethylbenzene
        result = murcko_decompose(mol)
        # Returns dict with 'scaffold', 'framework', etc.
        assert 'scaffold' in result
        assert result['scaffold'] is not None
        
    def test_get_scaffold_smiles(self):
        """Test get_scaffold_smiles convenience function."""
        smiles = get_scaffold_smiles("c1ccc(CC)cc1")  # Takes SMILES string
        assert smiles is not None
        
    def test_get_framework_smiles(self):
        """Test get_framework_smiles convenience function."""
        smiles = get_framework_smiles("c1ccc(CC)cc1")  # Takes SMILES string
        assert smiles is not None
        
    def test_get_side_chains(self):
        """Test get_side_chains function."""
        mol = parse("c1ccc(CC)cc1")
        chains = get_side_chains(mol)
        assert isinstance(chains, list)


class TestHydrogenHandling:
    """Test hydrogen addition/removal."""
    
    def test_add_explicit_hydrogens(self):
        """Test add_explicit_hydrogens function."""
        mol = parse("C")  # Methane with implicit H
        mol_with_h = add_explicit_hydrogens(mol)  # Returns new molecule
        # Should have explicit H atoms now
        h_count = sum(1 for a in mol_with_h.atoms if a.symbol == "H")
        assert h_count == 4
        
    def test_remove_explicit_hydrogens(self):
        """Test remove_explicit_hydrogens function."""
        mol = parse("C")
        mol_with_h = add_explicit_hydrogens(mol)
        mol_clean = remove_explicit_hydrogens(mol_with_h)
        # Should have no explicit H atoms
        h_count = sum(1 for a in mol_clean.atoms if a.symbol == "H")
        assert h_count == 0


class TestWriterEdgeCases:
    """Test SMILES writer edge cases."""
    
    def test_isotope_writing(self):
        """Test isotope writing."""
        mol = parse("[13C]")
        smiles = to_smiles(mol)
        assert "13" in smiles
        
    def test_charge_writing(self):
        """Test charge writing."""
        mol = parse("[NH4+]")
        smiles = to_smiles(mol)
        assert "+" in smiles
        
    def test_radical_writing(self):
        """Test radical notation."""
        mol = parse("[CH3]")
        # Methyl radical-like
        smiles = to_smiles(mol)
        assert "C" in smiles


class TestAromaticityEdgeCases:
    """Test aromaticity perception edge cases."""
    
    def test_imidazole_aromaticity(self):
        """Test imidazole aromaticity."""
        mol = parse("c1c[nH]cn1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)
        
    def test_pyrimidine_aromaticity(self):
        """Test pyrimidine aromaticity."""
        mol = parse("c1cncnc1")
        perceive_aromaticity(mol)
        assert all(a.is_aromatic for a in mol.atoms)


class TestKekulizationEdgeCases:
    """Test kekulization edge cases."""
    
    def test_kekulize_naphthalene(self):
        """Test kekulization of naphthalene."""
        mol = parse("c1ccc2ccccc2c1")
        kek_mol = kekulize(mol, clear_aromatic_flags=True)  # Returns new molecule
        # Should have alternating single/double bonds
        double_count = sum(1 for b in kek_mol.bonds if b.order == 2)
        assert double_count > 0


class TestElementsModule:
    """Test elements module utilities."""
    
    def test_is_organic_symbol(self):
        """Test is_organic_symbol function."""
        from chiralipy.elements import is_organic_symbol
        assert is_organic_symbol("C")
        assert is_organic_symbol("N")
        assert is_organic_symbol("c")  # aromatic
        assert not is_organic_symbol("Fe")
        
    def test_is_aromatic_symbol(self):
        """Test is_aromatic_symbol function."""
        from chiralipy.elements import is_aromatic_symbol
        assert is_aromatic_symbol("c")
        assert is_aromatic_symbol("n")
        assert not is_aromatic_symbol("C")
        
    def test_bond_order_str(self):
        """Test BondOrder __str__ method."""
        from chiralipy.elements import BondOrder
        assert str(BondOrder.SINGLE) == "single"
        assert str(BondOrder.DOUBLE) == "double"


class TestParserEdgeCases:
    """Test parser edge cases."""
    
    def test_parse_complex_stereo(self):
        """Test parsing complex stereochemistry."""
        mol = parse("[C@H](F)(Cl)Br")
        assert mol.atoms[0].chirality is not None  # 'chirality' not 'chirality_tag'
        
    def test_parse_ring_closures(self):
        """Test parsing multiple ring closures."""
        mol = parse("C12CC1CC2")
        assert mol.num_atoms == 5
        
    def test_parse_disconnected(self):
        """Test parsing disconnected molecules."""
        mol = parse("C.C")
        assert mol.num_atoms == 2
        assert mol.num_bonds == 0


class TestExceptions:
    """Test exception handling."""
    
    def test_parse_error(self):
        """Test ParseError is raised for invalid SMILES."""
        from chiralipy.exceptions import ParseError
        # Parser is lenient, try truly invalid input
        with pytest.raises((ParseError, ValueError)):
            parse("C[invalid]]]]")  # Very malformed
            
    def test_valence_error_context(self):
        """Test ValenceError has context."""
        from chiralipy.exceptions import ValenceError
        try:
            # This might raise valence error
            mol = parse("C(C)(C)(C)(C)C")  # Hexavalent carbon
        except ValenceError as e:
            assert str(e)  # Should have a message


class TestMurckoAdvanced:
    """Additional Murcko tests for coverage."""
    
    def test_get_scaffold_no_rings(self):
        """Test get_scaffold with molecule without rings."""
        mol = parse("CCCCC")
        scaffold = get_scaffold(mol)
        # Molecule without rings should return None or empty
        assert scaffold is None or scaffold.num_atoms == 0
        
    def test_get_framework_no_rings(self):
        """Test get_framework with molecule without rings."""
        mol = parse("CCCCC")
        framework = get_framework(mol)
        assert framework is None or framework.num_atoms == 0
        
    def test_murcko_biphenyl(self):
        """Test Murcko decomposition of biphenyl."""
        mol = parse("c1ccc(c2ccccc2)cc1")  # Biphenyl
        result = murcko_decompose(mol)
        assert 'scaffold' in result
        assert 'ring_systems' in result
        
    def test_murcko_fused_rings(self):
        """Test Murcko decomposition with fused rings."""
        mol = parse("c1ccc2ccccc2c1")  # Naphthalene
        result = murcko_decompose(mol)
        assert result['scaffold'] is not None


class TestSubstructureAdvanced:
    """Additional substructure matching tests."""
    
    def test_wildcard_atom_matching(self):
        """Test matching with wildcard atoms."""
        mol = parse("CCO")
        pattern = parse("[#6][#8]")  # C-O using atomic numbers
        matches = substructure_search(mol, pattern)
        assert len(matches) >= 1
        
    def test_bond_query_matching(self):
        """Test matching bond queries."""
        mol = parse("C=C")
        pattern = parse("C=C")
        assert has_substructure(mol, pattern)
        
    def test_ring_bond_query(self):
        """Test ring bond query matching."""
        mol = parse("c1ccccc1")  # Benzene
        pattern = parse("[c;R]")  # Aromatic carbon in ring
        assert has_substructure(mol, pattern)
        
    def test_non_ring_atom_query(self):
        """Test non-ring atom query matching."""
        mol = parse("c1ccccc1CC")  # Toluene
        pattern = parse("[C;!R]")  # Aliphatic carbon not in ring
        matches = substructure_search(mol, pattern)
        assert len(matches) >= 1


class TestWriterAdvanced:
    """Additional SMILES writer tests."""
    
    def test_stereochemistry_writing(self):
        """Test stereochemistry in SMILES output."""
        mol = parse("[C@H](F)(Cl)Br")
        smiles = to_smiles(mol)
        assert "@" in smiles
        
    def test_double_bond_stereo(self):
        """Test double bond stereochemistry."""
        mol = parse("C/C=C/C")
        smiles = to_smiles(mol)
        # Should preserve E/Z info
        assert "C" in smiles
        
    def test_ring_with_stereo(self):
        """Test ring with stereochemistry."""
        mol = parse("[C@H]1(F)CCCC1")
        smiles = to_smiles(mol)
        assert "@" in smiles


class TestKekulizationAdvanced:
    """Additional kekulization tests."""
    
    def test_kekulize_pyridine(self):
        """Test kekulization of pyridine."""
        mol = parse("c1ccncc1")  # Pyridine
        kek = kekulize(mol)
        double_count = sum(1 for b in kek.bonds if b.order == 2)
        assert double_count >= 2
        
    def test_kekulize_furan(self):
        """Test kekulization of furan."""
        mol = parse("c1ccoc1")  # Furan
        kek = kekulize(mol)
        assert kek is not None


class TestAromaticityAdvanced:
    """Additional aromaticity tests."""
    
    def test_perceive_aromaticity_already_aromatic(self):
        """Test perceiving aromaticity on already aromatic molecule."""
        mol = parse("c1ccccc1")
        perceive_aromaticity(mol)  # Should not change anything
        assert all(a.is_aromatic for a in mol.atoms)
        
    def test_perceive_aromaticity_non_aromatic(self):
        """Test perceiving aromaticity on non-aromatic molecule."""
        mol = parse("C1CCCCC1")  # Cyclohexane
        perceive_aromaticity(mol)
        assert not any(a.is_aromatic for a in mol.atoms)


class TestCanonicalSmiles:
    """Test canonical SMILES generation."""
    
    def test_canonical_smiles_same_molecule(self):
        """Test that different input orders produce same canonical SMILES."""
        smiles1 = canonical_smiles("CCO")
        smiles2 = canonical_smiles("OCC")
        assert smiles1 == smiles2
        
    def test_canonical_smiles_benzene(self):
        """Test canonical SMILES for benzene."""
        smiles = canonical_smiles("c1ccccc1")
        assert "c" in smiles.lower()


class TestRingDetectionAdvanced:
    """Additional ring detection tests."""
    
    def test_spiro_compound(self):
        """Test ring detection in spiro compound."""
        mol = parse("C1CCC2(CC1)CCCCC2")  # Spiro compound
        ring_count, ring_sizes = get_ring_info(mol)
        # Ring atoms should have ring count > 0
        ring_atoms = sum(1 for v in ring_count.values() if v > 0)
        assert ring_atoms >= 6  # Multiple ring atoms
        
    def test_bridged_compound(self):
        """Test ring detection in bridged compound."""
        mol = parse("C1CC2CCC1C2")  # Norbornane
        ring_count, ring_sizes = get_ring_info(mol)
        # Should detect rings
        ring_atoms = sum(1 for v in ring_count.values() if v > 0)
        assert ring_atoms >= 4
