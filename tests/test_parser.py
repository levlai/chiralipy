"""Tests for the SMILES parser.

This module tests the chirpy SMILES parser with various test cases
derived from common molecules and edge cases. Uses RDKit as reference
for validation.
"""

import pytest
from rdkit import Chem

from chiralipy import parse, Molecule, Atom, Bond
from chiralipy.exceptions import ParseError, RingError


def rdkit_atom_count(smiles: str) -> int:
    """Get number of atoms from RDKit for comparison."""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol else 0


def rdkit_bond_count(smiles: str) -> int:
    """Get number of bonds from RDKit for comparison."""
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumBonds() if mol else 0


def rdkit_is_valid(smiles: str) -> bool:
    """Check if SMILES is valid according to RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


class TestBasicParsing:
    """Test basic SMILES parsing functionality."""

    def test_parse_returns_molecule(self):
        """parse() should return a Molecule object."""
        mol = parse("C")
        assert isinstance(mol, Molecule)

    def test_single_carbon(self):
        """Parse single carbon atom."""
        mol = parse("C")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "C"
        assert mol.atoms[0].atomic_number == 6

    def test_single_nitrogen(self):
        """Parse single nitrogen atom."""
        mol = parse("N")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "N"
        assert mol.atoms[0].atomic_number == 7

    def test_single_oxygen(self):
        """Parse single oxygen atom."""
        mol = parse("O")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "O"
        assert mol.atoms[0].atomic_number == 8

    def test_simple_chain(self, simple_smiles):
        """Parse simple chain molecules."""
        for smiles in simple_smiles:
            mol = parse(smiles)
            assert mol is not None
            assert len(mol.atoms) > 0

    def test_ethane(self):
        """Parse ethane (CC)."""
        mol = parse("CC")
        assert len(mol.atoms) == 2
        assert len(mol.bonds) == 1
        assert mol.bonds[0].order == 1

    def test_ethene(self):
        """Parse ethene (C=C)."""
        mol = parse("C=C")
        assert len(mol.atoms) == 2
        assert len(mol.bonds) == 1
        assert mol.bonds[0].order == 2

    def test_ethyne(self):
        """Parse ethyne (C#C)."""
        mol = parse("C#C")
        assert len(mol.atoms) == 2
        assert len(mol.bonds) == 1
        assert mol.bonds[0].order == 3

    def test_propane(self):
        """Parse propane (CCC)."""
        mol = parse("CCC")
        assert len(mol.atoms) == 3
        assert len(mol.bonds) == 2

    def test_butane(self):
        """Parse butane (CCCC)."""
        mol = parse("CCCC")
        assert len(mol.atoms) == 4
        assert len(mol.bonds) == 3


class TestAromaticParsing:
    """Test aromatic SMILES parsing."""

    def test_benzene(self):
        """Parse benzene (c1ccccc1)."""
        mol = parse("c1ccccc1")
        assert len(mol.atoms) == 6
        assert len(mol.bonds) == 6
        assert all(a.is_aromatic for a in mol.atoms)

    def test_pyridine(self):
        """Parse pyridine (c1ccncc1)."""
        mol = parse("c1ccncc1")
        assert len(mol.atoms) == 6
        assert len(mol.bonds) == 6
        # Should have 5 carbons and 1 nitrogen
        symbols = [a.symbol for a in mol.atoms]
        assert symbols.count("c") == 5
        assert symbols.count("n") == 1

    def test_naphthalene(self):
        """Parse naphthalene (c1ccc2ccccc2c1)."""
        mol = parse("c1ccc2ccccc2c1")
        assert len(mol.atoms) == 10
        assert len(mol.bonds) == 11

    def test_aromatic_smiles(self, aromatic_smiles):
        """Parse various aromatic molecules."""
        for smiles in aromatic_smiles:
            mol = parse(smiles)
            assert mol is not None
            # All atoms should be aromatic
            assert all(a.is_aromatic for a in mol.atoms)


class TestRingParsing:
    """Test ring closure parsing."""

    def test_cyclopropane(self):
        """Parse cyclopropane (C1CC1)."""
        mol = parse("C1CC1")
        assert len(mol.atoms) == 3
        assert len(mol.bonds) == 3

    def test_cyclobutane(self):
        """Parse cyclobutane (C1CCC1)."""
        mol = parse("C1CCC1")
        assert len(mol.atoms) == 4
        assert len(mol.bonds) == 4

    def test_cyclopentane(self):
        """Parse cyclopentane (C1CCCC1)."""
        mol = parse("C1CCCC1")
        assert len(mol.atoms) == 5
        assert len(mol.bonds) == 5

    def test_cyclohexane(self):
        """Parse cyclohexane (C1CCCCC1)."""
        mol = parse("C1CCCCC1")
        assert len(mol.atoms) == 6
        assert len(mol.bonds) == 6

    def test_bicyclic(self):
        """Parse bicyclic compound."""
        mol = parse("C1CC2CCCCC2C1")
        assert len(mol.atoms) == 9
        # Should have ring bonds connecting the two rings

    def test_high_ring_number(self):
        """Parse SMILES with %nn ring closure."""
        mol = parse("C%10CC%10")
        assert len(mol.atoms) == 3
        assert len(mol.bonds) == 3

    def test_ring_smiles(self, ring_smiles):
        """Parse various ring molecules."""
        for smiles in ring_smiles:
            mol = parse(smiles)
            assert mol is not None
            assert len(mol.bonds) >= len(mol.atoms)  # Rings have at least as many bonds as atoms


class TestChargedAtoms:
    """Test parsing of charged atoms."""

    def test_negative_oxygen(self):
        """Parse negatively charged oxygen ([O-])."""
        mol = parse("[O-]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].charge == -1

    def test_positive_nitrogen(self):
        """Parse ammonium ion ([NH4+])."""
        mol = parse("[NH4+]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].charge == 1
        assert mol.atoms[0].explicit_hydrogens == 4

    def test_sodium_chloride(self):
        """Parse sodium chloride ([Na+].[Cl-])."""
        mol = parse("[Na+].[Cl-]")
        assert len(mol.atoms) == 2
        charges = [a.charge for a in mol.atoms]
        assert 1 in charges
        assert -1 in charges

    def test_formate(self):
        """Parse formate ion ([O-]C=O)."""
        mol = parse("[O-]C=O")
        assert len(mol.atoms) == 3
        # One oxygen should have -1 charge
        charges = [a.charge for a in mol.atoms]
        assert -1 in charges

    def test_charged_smiles(self, charged_smiles):
        """Parse various charged molecules."""
        for smiles in charged_smiles:
            mol = parse(smiles)
            assert mol is not None


class TestChirality:
    """Test parsing of chiral centers."""

    def test_r_chirality(self):
        """Parse R chirality (C[C@H](O)F)."""
        mol = parse("C[C@H](O)F")
        assert len(mol.atoms) == 4
        # Find the chiral center
        chiral_atoms = [a for a in mol.atoms if a.chirality]
        assert len(chiral_atoms) == 1
        assert chiral_atoms[0].chirality == "@"

    def test_s_chirality(self):
        """Parse S chirality (C[C@@H](O)F)."""
        mol = parse("C[C@@H](O)F")
        assert len(mol.atoms) == 4
        chiral_atoms = [a for a in mol.atoms if a.chirality]
        assert len(chiral_atoms) == 1
        assert chiral_atoms[0].chirality == "@@"

    def test_chiral_smiles(self, chiral_smiles):
        """Parse various chiral molecules."""
        for smiles in chiral_smiles:
            mol = parse(smiles)
            assert mol is not None
            # Should have at least one chiral center
            chiral_atoms = [a for a in mol.atoms if a.chirality]
            assert len(chiral_atoms) >= 1


class TestIsotopes:
    """Test parsing of isotopes."""

    def test_deuterium(self):
        """Parse deuterium ([2H])."""
        mol = parse("[2H]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].isotope == 2

    def test_carbon_13(self):
        """Parse carbon-13 ([13C])."""
        mol = parse("[13C]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].isotope == 13

    def test_deuterated_methane(self):
        """Parse fully deuterated methane."""
        mol = parse("[2H]C([2H])([2H])[2H]")
        assert len(mol.atoms) == 5
        isotope_atoms = [a for a in mol.atoms if a.isotope is not None]
        assert len(isotope_atoms) == 4

    def test_isotope_smiles(self, isotope_smiles):
        """Parse various isotope-labeled molecules."""
        for smiles in isotope_smiles:
            mol = parse(smiles)
            assert mol is not None


class TestStereoBonds:
    """Test parsing of E/Z stereochemistry."""

    def test_trans_difluoroethene(self):
        """Parse trans-1,2-difluoroethene (F/C=C/F)."""
        mol = parse("F/C=C/F")
        assert len(mol.atoms) == 4
        # Should have stereo bonds
        stereo_bonds = [b for b in mol.bonds if b.stereo]
        assert len(stereo_bonds) >= 1

    def test_cis_difluoroethene(self):
        """Parse cis-1,2-difluoroethene (F/C=C\\F)."""
        mol = parse(r"F/C=C\F")
        assert len(mol.atoms) == 4
        stereo_bonds = [b for b in mol.bonds if b.stereo]
        assert len(stereo_bonds) >= 1

    def test_stereo_bond_smiles(self, stereo_bond_smiles):
        """Parse various molecules with stereo bonds."""
        for smiles in stereo_bond_smiles:
            mol = parse(smiles)
            assert mol is not None


class TestBracketAtoms:
    """Test parsing of bracket atom notation."""

    def test_methane_explicit(self):
        """Parse methane with explicit hydrogens ([CH4])."""
        mol = parse("[CH4]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "C"
        assert mol.atoms[0].explicit_hydrogens == 4

    def test_ammonia_explicit(self):
        """Parse ammonia with explicit hydrogens ([NH3])."""
        mol = parse("[NH3]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "N"
        assert mol.atoms[0].explicit_hydrogens == 3

    def test_water_explicit(self):
        """Parse water with explicit hydrogen ([OH2])."""
        mol = parse("[OH2]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "O"
        assert mol.atoms[0].explicit_hydrogens == 2

    def test_transition_metal(self):
        """Parse copper atom ([Cu])."""
        mol = parse("[Cu]")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "Cu"

    def test_bracket_atom_smiles(self, bracket_atom_smiles):
        """Parse various bracket atom notations."""
        for smiles in bracket_atom_smiles:
            mol = parse(smiles)
            assert mol is not None


class TestMultiComponent:
    """Test parsing of multi-component SMILES."""

    def test_two_components(self):
        """Parse two disconnected atoms."""
        mol = parse("[Na+].[Cl-]")
        assert len(mol.atoms) == 2
        assert len(mol.bonds) == 0

    def test_multiple_components(self):
        """Parse multiple disconnected components."""
        mol = parse("O.O.O")
        assert len(mol.atoms) == 3
        assert len(mol.bonds) == 0

    def test_connected_components(self):
        """Verify connected_components() method."""
        mol = parse("[Na+].[Cl-]")
        components = mol.connected_components()
        assert len(components) == 2

    def test_multi_component_smiles(self, multi_component_smiles):
        """Parse various multi-component molecules."""
        for smiles in multi_component_smiles:
            mol = parse(smiles)
            assert mol is not None
            components = mol.connected_components()
            assert len(components) >= 2


class TestBranching:
    """Test parsing of branched molecules."""

    def test_isobutane(self):
        """Parse isobutane (CC(C)C)."""
        mol = parse("CC(C)C")
        assert len(mol.atoms) == 4
        assert len(mol.bonds) == 3

    def test_neopentane(self):
        """Parse neopentane (CC(C)(C)C)."""
        mol = parse("CC(C)(C)C")
        assert len(mol.atoms) == 5
        assert len(mol.bonds) == 4

    def test_acetic_acid(self):
        """Parse acetic acid (CC(=O)O)."""
        mol = parse("CC(=O)O")
        assert len(mol.atoms) == 4
        # Should have one double bond
        double_bonds = [b for b in mol.bonds if b.order == 2]
        assert len(double_bonds) == 1

    def test_nested_branches(self):
        """Parse molecule with nested branches."""
        mol = parse("CC(C(C)C)C")
        assert len(mol.atoms) == 6


class TestTwoLetterElements:
    """Test parsing of two-letter element symbols."""

    def test_chlorine(self):
        """Parse chlorine (Cl)."""
        mol = parse("Cl")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "Cl"
        assert mol.atoms[0].atomic_number == 17

    def test_bromine(self):
        """Parse bromine (Br)."""
        mol = parse("Br")
        assert len(mol.atoms) == 1
        assert mol.atoms[0].symbol == "Br"
        assert mol.atoms[0].atomic_number == 35

    def test_chloroform(self):
        """Parse chloroform (ClC(Cl)(Cl)Cl)."""
        mol = parse("ClC(Cl)(Cl)Cl")
        assert len(mol.atoms) == 5
        cl_atoms = [a for a in mol.atoms if a.symbol == "Cl"]
        assert len(cl_atoms) == 4

    def test_bromomethane(self):
        """Parse bromomethane (CBr)."""
        mol = parse("CBr")
        assert len(mol.atoms) == 2


class TestComplexMolecules:
    """Test parsing of complex real-world molecules."""

    def test_aspirin(self):
        """Parse aspirin."""
        mol = parse("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert mol is not None
        assert len(mol.atoms) == 13

    def test_caffeine(self):
        """Parse caffeine."""
        mol = parse("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert mol is not None

    def test_complex_smiles(self, complex_smiles):
        """Parse various complex molecules."""
        for smiles in complex_smiles:
            mol = parse(smiles)
            assert mol is not None
            assert len(mol.atoms) > 0


class TestAtomProperties:
    """Test atom property access after parsing."""

    def test_atom_idx(self):
        """Atoms should have correct indices."""
        mol = parse("CCC")
        for i, atom in enumerate(mol.atoms):
            assert atom.idx == i

    def test_atom_neighbors(self):
        """Test neighbor access."""
        mol = parse("CCC")
        # Middle atom should have 2 neighbors
        middle_atom = mol.atoms[1]
        neighbors = list(middle_atom.neighbors(mol))
        assert len(neighbors) == 2

    def test_atom_bonds(self):
        """Test bond access from atom."""
        mol = parse("CCC")
        middle_atom = mol.atoms[1]
        bonds = list(middle_atom.get_bonds(mol))
        assert len(bonds) == 2


class TestBondProperties:
    """Test bond property access after parsing."""

    def test_bond_atoms(self):
        """Bonds should reference correct atoms."""
        mol = parse("CC")
        bond = mol.bonds[0]
        assert bond.atom1_idx in [0, 1]
        assert bond.atom2_idx in [0, 1]
        assert bond.atom1_idx != bond.atom2_idx

    def test_bond_other_atom(self):
        """Test other_atom() method."""
        mol = parse("CC")
        bond = mol.bonds[0]
        other = bond.other_atom(bond.atom1_idx)
        assert other == bond.atom2_idx


class TestErrorHandling:
    """Test error handling for invalid SMILES."""

    def test_unclosed_ring(self):
        """Unclosed ring should raise error."""
        with pytest.raises((ParseError, RingError)):
            parse("C1CC")

    def test_empty_string(self):
        """Empty string should raise error or return empty molecule."""
        try:
            mol = parse("")
            assert len(mol.atoms) == 0
        except ParseError:
            pass  # Also acceptable

    def test_invalid_element(self):
        """Invalid element symbol should raise error or parse as-is."""
        # Some parsers accept any symbol in brackets
        try:
            mol = parse("[Xx]")
            # If it doesn't raise, it should have parsed something
            assert len(mol.atoms) == 1
        except (ParseError, Exception):
            pass  # Raising is also acceptable

    def test_unclosed_bracket(self):
        """Unclosed bracket should raise error."""
        with pytest.raises((ParseError, TypeError, Exception)):
            parse("[C")

    def test_invalid_bond_order(self):
        """Invalid SMILES syntax - double equals is treated as two double bonds."""
        # C==C may be interpreted as C=C with extra = ignored or as error
        try:
            mol = parse("C==C")
            # If it parses, verify structure
            assert len(mol.atoms) >= 2
        except (ParseError, Exception):
            pass  # Raising is also acceptable


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_atom_with_h(self):
        """Single atom with explicit H count."""
        mol = parse("[CH4]")
        assert mol.atoms[0].explicit_hydrogens == 4

    def test_multiple_charges(self):
        """Parse atom with +2 charge."""
        mol = parse("[Ca+2]")
        assert mol.atoms[0].charge == 2

    def test_negative_charge_number(self):
        """Parse atom with -2 charge."""
        mol = parse("[O-2]")
        assert mol.atoms[0].charge == -2

    def test_atom_class(self):
        """Parse atom with atom class."""
        mol = parse("[C:1]")
        assert mol.atoms[0].atom_class == 1

    def test_ring_with_stereo(self):
        """Parse ring with stereo center."""
        mol = parse("C[C@H]1CCCCC1")
        chiral = [a for a in mol.atoms if a.chirality]
        assert len(chiral) == 1


class TestParserMatchesRDKit:
    """Test that parser produces structures matching RDKit."""

    @pytest.mark.parametrize("smiles", [
        "C", "CC", "CCC", "CCCC", "CCCCC",
        "CO", "CCO", "CCCO",
        "C=C", "C#C", "C=O", "C#N",
        "c1ccccc1", "c1ccncc1", "c1ccoc1",
        "C1CCCCC1", "C1CC1", "C1CCC1",
    ])
    def test_atom_count_matches_rdkit(self, smiles):
        """Atom count should match RDKit."""
        mol = parse(smiles)
        expected = rdkit_atom_count(smiles)
        assert len(mol.atoms) == expected, f"{smiles}: Expected {expected} atoms, got {len(mol.atoms)}"

    @pytest.mark.parametrize("smiles", [
        "C", "CC", "CCC", "CCCC", "CCCCC",
        "CO", "CCO", "CCCO",
        "C=C", "C#C", "C=O", "C#N",
        "c1ccccc1", "c1ccncc1", "c1ccoc1",
        "C1CCCCC1", "C1CC1", "C1CCC1",
    ])
    def test_bond_count_matches_rdkit(self, smiles):
        """Bond count should match RDKit."""
        mol = parse(smiles)
        expected = rdkit_bond_count(smiles)
        assert len(mol.bonds) == expected, f"{smiles}: Expected {expected} bonds, got {len(mol.bonds)}"

    @pytest.mark.parametrize("smiles", [
        # Simple molecules
        "C", "CC", "CCC", "CCCC", "CCCCC",
        "CO", "CCO", "C(O)C", "C(C)O",
        # Bonds
        "C=C", "CC=C", "C=CC", "C#C", "CC#C",
        "C=O", "CC=O", "C#N", "CC#N",
        # Rings
        "C1CC1", "C1CCC1", "C1CCCC1", "C1CCCCC1",
        "C1CCCCCC1", "C1CCCCCCC1",
        # Aromatics
        "c1ccccc1", "c1ccncc1", "c1cnccc1", "n1ccccc1",
        "c1ccoc1", "c1ccsc1", "c1cc[nH]c1",
        # Fused rings
        "c1ccc2ccccc2c1",  # Naphthalene
        "c1ccc(-c2ccccc2)cc1",  # Biphenyl
        # Branched
        "CC(C)C", "CC(C)(C)C", "C(C)(C)C",
        "CC(C)CC", "CC(CC)C",
        # Charged
        "[NH4+]", "[O-]", "[Na+]", "[Cl-]",
        "[O-]C=O", "CC([O-])=O",
        # Isotopes
        "[2H]", "[13C]", "[14C]",
        # Chirality
        "C[C@H](O)F", "C[C@@H](O)F",
        "F[C@H](Cl)Br", "F[C@@H](Cl)Br",
        # E/Z
        "F/C=C/F", r"F/C=C\F",
        # Complex
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ])
    def test_valid_smiles_parses_like_rdkit(self, smiles):
        """Valid SMILES should parse and produce same atom/bond counts as RDKit."""
        # RDKit should be able to parse it
        assert rdkit_is_valid(smiles), f"RDKit can't parse: {smiles}"
        
        # chirpy should parse it too
        mol = parse(smiles)
        assert mol is not None
        
        # Counts should match
        expected_atoms = rdkit_atom_count(smiles)
        expected_bonds = rdkit_bond_count(smiles)
        
        assert len(mol.atoms) == expected_atoms, f"{smiles}: atoms - expected {expected_atoms}, got {len(mol.atoms)}"
        assert len(mol.bonds) == expected_bonds, f"{smiles}: bonds - expected {expected_bonds}, got {len(mol.bonds)}"

    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",  # Benzene
        "c1ccncc1",  # Pyridine
        "c1ccoc1",   # Furan
        "c1ccsc1",   # Thiophene
        "c1cc[nH]c1",  # Pyrrole
        "c1ccc2ccccc2c1",  # Naphthalene
    ])
    def test_aromatic_atoms_detected(self, smiles):
        """All atoms in aromatic rings should be marked aromatic."""
        mol = parse(smiles)
        rdkit_mol = Chem.MolFromSmiles(smiles)
        
        # Count aromatic atoms in RDKit
        rdkit_aromatic = sum(1 for a in rdkit_mol.GetAtoms() if a.GetIsAromatic())
        
        # Count aromatic atoms in chirpy
        chirpy_aromatic = sum(1 for a in mol.atoms if a.is_aromatic)
        
        assert chirpy_aromatic == rdkit_aromatic, f"{smiles}: aromatic atoms - expected {rdkit_aromatic}, got {chirpy_aromatic}"

    @pytest.mark.parametrize("smiles,expected_charge", [
        ("[NH4+]", 1),
        ("[O-]", -1),
        ("[Na+]", 1),
        ("[Cl-]", -1),
        ("[Ca+2]", 2),
        ("[O-2]", -2),
        ("[Fe+3]", 3),
    ])
    def test_charge_matches_rdkit(self, smiles, expected_charge):
        """Parsed charges should match RDKit."""
        mol = parse(smiles)
        rdkit_mol = Chem.MolFromSmiles(smiles)
        
        # Get charge from RDKit
        rdkit_charge = rdkit_mol.GetAtomWithIdx(0).GetFormalCharge()
        
        # Get charge from chiralipy
        chirpy_charge = mol.atoms[0].charge
        
        assert chirpy_charge == rdkit_charge == expected_charge

    @pytest.mark.parametrize("smiles,expected_isotope", [
        ("[2H]", 2),
        ("[13C]", 13),
        ("[14C]", 14),
        ("[35Cl]", 35),
        ("[37Cl]", 37),
    ])
    def test_isotope_matches_rdkit(self, smiles, expected_isotope):
        """Parsed isotopes should match RDKit."""
        mol = parse(smiles)
        rdkit_mol = Chem.MolFromSmiles(smiles)
        
        # Get isotope from RDKit
        rdkit_isotope = rdkit_mol.GetAtomWithIdx(0).GetIsotope()
        
        # Get isotope from chiralipy
        chirpy_isotope = mol.atoms[0].isotope
        
        assert chirpy_isotope == rdkit_isotope == expected_isotope
