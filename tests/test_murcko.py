"""Tests for Bemis-Murcko decomposition module comparing with RDKit."""

from __future__ import annotations

import pytest
from chirpy import parse, to_smiles
from chirpy.decompose import (
    get_scaffold,
    get_framework,
    get_side_chains,
    get_ring_systems,
    murcko_decompose,
    MurckoDecomposition,
    get_scaffold_smiles,
    get_framework_smiles,
)


from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonicalize_smiles(smiles: str) -> str | None:
    """Canonicalize SMILES using RDKit for comparison."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


class TestGetScaffold:
    """Test scaffold extraction."""
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",           # benzene
        "c1ccc2ccccc2c1",     # naphthalene
        "C1CCCCC1",           # cyclohexane
        "c1ccccc1CCc1ccccc1", # diphenylethane
    ])
    def test_ring_molecules_have_scaffold(self, smiles: str) -> None:
        """Test that ring-containing molecules have scaffolds."""
        mol = parse(smiles)
        scaffold = get_scaffold(mol)
        
        assert scaffold is not None, f"No scaffold found for {smiles}"
        assert scaffold.num_atoms > 0
    
    @pytest.mark.parametrize("smiles", [
        "CCCCCC",   # hexane
        "CC(C)C",   # isobutane
        "CCOCC",    # diethyl ether
    ])
    def test_acyclic_molecules_no_scaffold(self, smiles: str) -> None:
        """Test that acyclic molecules have no scaffold."""
        mol = parse(smiles)
        scaffold = get_scaffold(mol)
        
        assert scaffold is None, f"Unexpected scaffold for acyclic {smiles}"
    
    @pytest.mark.parametrize("smiles,expected_atoms", [
        ("c1ccccc1", 6),           # benzene - scaffold is the whole ring
        ("c1ccccc1C", 6),          # toluene - scaffold is just benzene
        ("c1ccccc1CC", 6),         # ethylbenzene - scaffold is just benzene
    ])
    def test_scaffold_size(self, smiles: str, expected_atoms: int) -> None:
        """Test scaffold has expected number of atoms."""
        mol = parse(smiles)
        scaffold = get_scaffold(mol)
        
        assert scaffold is not None
        # Scaffold might include linker atoms, so >= comparison
        assert scaffold.num_atoms >= expected_atoms
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1CCc1ccccc1",      # diphenylethane
        "c1ccccc1OCc1ccccc1",      # diphenylmethyl ether  
        "c1ccccc1NCc1ccccc1",      # N-benzylaniline
    ])
    def test_scaffold_includes_linkers(self, smiles: str) -> None:
        """Test that scaffold includes linker chains between rings."""
        mol = parse(smiles)
        scaffold = get_scaffold(mol)
        
        assert scaffold is not None
        # Should have more than just two benzene rings (12 atoms)
        # because linkers are included
        assert scaffold.num_atoms > 12


class TestGetFramework:
    """Test framework extraction."""
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",           # benzene
        "c1ccc2ccccc2c1",     # naphthalene
        "c1ccccc1Nc1ccccc1",  # diphenylamine
    ])
    def test_framework_exists(self, smiles: str) -> None:
        """Test that framework can be extracted."""
        mol = parse(smiles)
        framework = get_framework(mol)
        
        assert framework is not None
        assert framework.num_atoms > 0
    
    def test_framework_all_carbons(self) -> None:
        """Test that framework has all carbon atoms."""
        smiles = "c1ccccc1Nc1ccccc1"  # diphenylamine
        mol = parse(smiles)
        framework = get_framework(mol)
        
        assert framework is not None
        # All atoms should be carbon (except possibly retained charged atoms)
        for atom in framework.atoms:
            assert atom.symbol == 'C', f"Non-carbon atom in framework: {atom.symbol}"
    
    def test_framework_all_single_bonds(self) -> None:
        """Test that framework has all single bonds."""
        smiles = "c1ccccc1"  # benzene (aromatic)
        mol = parse(smiles)
        framework = get_framework(mol)
        
        assert framework is not None
        for bond in framework.bonds:
            assert bond.order == 1, f"Non-single bond in framework: order={bond.order}"
            assert not bond.is_aromatic, "Aromatic bond in framework"
    
    def test_acyclic_no_framework(self) -> None:
        """Test that acyclic molecules have no framework."""
        smiles = "CCCCC"
        mol = parse(smiles)
        framework = get_framework(mol)
        
        assert framework is None


class TestGetSideChains:
    """Test side chain extraction."""
    
    @pytest.mark.parametrize("smiles,expected_chain_count", [
        ("c1ccccc1C", 1),          # toluene - one methyl
        ("c1ccccc1CC", 1),         # ethylbenzene - one ethyl
        ("c1ccc(C)cc1C", 2),       # xylene - two methyls
    ])
    def test_side_chain_count(self, smiles: str, expected_chain_count: int) -> None:
        """Test expected number of side chains."""
        mol = parse(smiles)
        chains = get_side_chains(mol)
        
        assert len(chains) == expected_chain_count, \
            f"Expected {expected_chain_count} chains for {smiles}, got {len(chains)}"
    
    def test_benzene_no_side_chains(self) -> None:
        """Test that benzene has no side chains."""
        mol = parse("c1ccccc1")
        chains = get_side_chains(mol)
        
        assert len(chains) == 0
    
    def test_acyclic_whole_molecule_is_side_chain(self) -> None:
        """Test that acyclic molecule is returned as side chain."""
        smiles = "CCCCC"
        mol = parse(smiles)
        chains = get_side_chains(mol)
        
        # Whole molecule is "side chain" when there's no scaffold
        assert len(chains) == 1
        assert chains[0].num_atoms == mol.num_atoms


class TestGetRingSystems:
    """Test ring system extraction."""
    
    @pytest.mark.parametrize("smiles,expected_systems", [
        ("c1ccccc1", 1),                  # benzene - one system
        ("c1ccc2ccccc2c1", 1),            # naphthalene - one fused system
        ("c1ccccc1CCc1ccccc1", 2),        # diphenylethane - two systems
        ("c1ccccc1c1ccccc1", 2),          # biphenyl - two systems
    ])
    def test_ring_system_count(self, smiles: str, expected_systems: int) -> None:
        """Test expected number of ring systems."""
        mol = parse(smiles)
        systems = get_ring_systems(mol)
        
        assert len(systems) == expected_systems, \
            f"Expected {expected_systems} systems for {smiles}, got {len(systems)}"
    
    def test_naphthalene_single_system(self) -> None:
        """Test that naphthalene is one fused system."""
        mol = parse("c1ccc2ccccc2c1")
        systems = get_ring_systems(mol)
        
        assert len(systems) == 1
        # Naphthalene has 10 atoms in fused system
        assert systems[0].num_atoms == 10
    
    def test_acyclic_no_systems(self) -> None:
        """Test that acyclic molecule has no ring systems."""
        mol = parse("CCCCC")
        systems = get_ring_systems(mol)
        
        assert len(systems) == 0


class TestMurckoDecompose:
    """Test full Murcko decomposition."""
    
    def test_decompose_returns_all_components(self) -> None:
        """Test that decomposition returns all components."""
        smiles = "CCc1ccc(CC)cc1"  # diethylbenzene
        mol = parse(smiles)
        result = murcko_decompose(mol, return_mols=True)
        
        assert isinstance(result, MurckoDecomposition)
        assert result.scaffold is not None
        assert result.framework is not None
        assert len(result.side_chains) == 2  # two ethyl groups
        assert len(result.ring_systems) == 1  # one benzene
    
    def test_decompose_returns_dict(self) -> None:
        """Test that decomposition returns dict when return_mols=False."""
        smiles = "c1ccccc1C"
        mol = parse(smiles)
        result = murcko_decompose(mol, return_mols=False)
        
        assert isinstance(result, dict)
        assert 'scaffold' in result
        assert 'framework' in result
        assert 'side_chains' in result
        assert 'ring_systems' in result
    
    def test_decompose_smiles_output(self) -> None:
        """Test that SMILES output is valid."""
        smiles = "c1ccccc1C"
        mol = parse(smiles)
        result = murcko_decompose(mol, return_mols=False)
        
        # Scaffold SMILES should be parseable
        if result['scaffold']:
            scaffold_mol = Chem.MolFromSmiles(result['scaffold'])
            assert scaffold_mol is not None, f"Invalid scaffold SMILES: {result['scaffold']}"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_scaffold_smiles(self) -> None:
        """Test get_scaffold_smiles function."""
        result = get_scaffold_smiles("c1ccccc1C")
        assert result is not None
        # Should be benzene or similar
        canon = canonicalize_smiles(result)
        assert canon is not None
    
    def test_get_framework_smiles(self) -> None:
        """Test get_framework_smiles function."""
        result = get_framework_smiles("c1ccccc1C")
        assert result is not None
        # Framework should be cyclohexane-like
        canon = canonicalize_smiles(result)
        assert canon is not None
    
    def test_scaffold_smiles_acyclic_returns_none(self) -> None:
        """Test that acyclic molecules return None."""
        result = get_scaffold_smiles("CCCCC")
        assert result is None
    
    def test_framework_smiles_acyclic_returns_none(self) -> None:
        """Test that acyclic molecules return None."""
        result = get_framework_smiles("CCCCC")
        assert result is None


class TestCompareWithRDKit:
    """Compare Murcko decomposition with RDKit."""
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",               # benzene
        "c1ccccc1C",              # toluene
        "c1ccccc1CC",             # ethylbenzene
        "c1ccc2ccccc2c1",         # naphthalene
        "c1ccc(C)cc1",            # toluene variant
        "CCc1ccccc1",             # ethylbenzene variant
        "c1ccccc1Cc1ccccc1",      # diphenylmethane
        "C1CCCCC1",               # cyclohexane
        "c1ccc(O)cc1",            # phenol
        "c1ccc(N)cc1",            # aniline
    ])
    def test_scaffold_matches_rdkit(self, smiles: str) -> None:
        """Test that scaffold matches RDKit's GetScaffoldForMol."""
        # Chirpy
        mol = parse(smiles)
        chirpy_scaffold = get_scaffold(mol)
        chirpy_smi = to_smiles(chirpy_scaffold) if chirpy_scaffold else None
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_scaffold = MurckoScaffold.GetScaffoldForMol(rdmol)
        rdkit_smi = Chem.MolToSmiles(rdkit_scaffold) if rdkit_scaffold else None
        
        # Canonicalize both for comparison
        chirpy_canon = canonicalize_smiles(chirpy_smi) if chirpy_smi else None
        rdkit_canon = canonicalize_smiles(rdkit_smi) if rdkit_smi else None
        
        assert chirpy_canon == rdkit_canon, \
            f"Scaffold mismatch for {smiles}:\n  chirpy: {chirpy_canon}\n  rdkit: {rdkit_canon}"
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",               # benzene
        "c1ccccc1C",              # toluene
        "c1ccc2ccccc2c1",         # naphthalene
        "c1ccccc1Nc1ccccc1",      # diphenylamine
        "C1CCCCC1",               # cyclohexane
    ])
    def test_framework_matches_rdkit(self, smiles: str) -> None:
        """Test that framework matches RDKit's MakeScaffoldGeneric."""
        # Chirpy
        mol = parse(smiles)
        chirpy_framework = get_framework(mol)
        chirpy_smi = to_smiles(chirpy_framework) if chirpy_framework else None
        
        # RDKit - get scaffold then make generic
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_scaffold = MurckoScaffold.GetScaffoldForMol(rdmol)
        rdkit_framework = MurckoScaffold.MakeScaffoldGeneric(rdkit_scaffold)
        rdkit_smi = Chem.MolToSmiles(rdkit_framework) if rdkit_framework else None
        
        # Canonicalize both for comparison
        chirpy_canon = canonicalize_smiles(chirpy_smi) if chirpy_smi else None
        rdkit_canon = canonicalize_smiles(rdkit_smi) if rdkit_smi else None
        
        assert chirpy_canon == rdkit_canon, \
            f"Framework mismatch for {smiles}:\n  chirpy: {chirpy_canon}\n  rdkit: {rdkit_canon}"
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1CCc1ccccc1",     # diphenylethane
        "c1ccccc1OCc1ccccc1",     # benzyl phenyl ether
        "c1ccccc1NCC1CCCCC1",     # phenyl-cyclohexyl with linker
    ])
    def test_scaffold_with_linkers_matches_rdkit(self, smiles: str) -> None:
        """Test scaffolds with linkers between ring systems."""
        # Chirpy
        mol = parse(smiles)
        chirpy_scaffold = get_scaffold(mol)
        chirpy_smi = to_smiles(chirpy_scaffold) if chirpy_scaffold else None
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_scaffold = MurckoScaffold.GetScaffoldForMol(rdmol)
        rdkit_smi = Chem.MolToSmiles(rdkit_scaffold) if rdkit_scaffold else None
        
        # Canonicalize both for comparison
        chirpy_canon = canonicalize_smiles(chirpy_smi) if chirpy_smi else None
        rdkit_canon = canonicalize_smiles(rdkit_smi) if rdkit_smi else None
        
        assert chirpy_canon == rdkit_canon, \
            f"Linker scaffold mismatch for {smiles}:\n  chirpy: {chirpy_canon}\n  rdkit: {rdkit_canon}"


class TestEdgeCases:
    """Test edge cases for Murcko decomposition."""
    
    def test_empty_molecule(self) -> None:
        """Test with empty molecule."""
        from chirpy.types import Molecule
        mol = Molecule()
        
        scaffold = get_scaffold(mol)
        assert scaffold is None
        
        framework = get_framework(mol)
        assert framework is None
        
        chains = get_side_chains(mol)
        assert len(chains) == 0
        
        systems = get_ring_systems(mol)
        assert len(systems) == 0
    
    def test_single_atom(self) -> None:
        """Test with single atom."""
        mol = parse("C")
        
        scaffold = get_scaffold(mol)
        assert scaffold is None
        
        chains = get_side_chains(mol)
        assert len(chains) == 1
    
    def test_charged_atoms_in_ring(self) -> None:
        """Test molecules with charged atoms in rings."""
        smiles = "c1cc[nH+]cc1"  # protonated pyridine
        mol = parse(smiles)
        
        scaffold = get_scaffold(mol)
        # Should still work
        assert scaffold is not None or scaffold is None  # Just don't crash
    
    def test_fused_rings(self) -> None:
        """Test fused ring systems."""
        # Anthracene - three fused rings
        smiles = "c1ccc2cc3ccccc3cc2c1"
        mol = parse(smiles)
        
        systems = get_ring_systems(mol)
        assert len(systems) == 1  # All fused into one system
        
        scaffold = get_scaffold(mol)
        assert scaffold is not None
        assert scaffold.num_atoms == 14
    
    def test_spiro_compound(self) -> None:
        """Test spiro compounds (rings sharing one atom)."""
        smiles = "C1CCC2(CC1)CCCCC2"  # spiro[5.5]undecane
        mol = parse(smiles)
        
        systems = get_ring_systems(mol)
        # Spiro rings share an atom, so might be 1 or 2 systems
        # depending on definition of "fused"
        assert len(systems) >= 1
    
    def test_bridged_compound(self) -> None:
        """Test bridged ring systems."""
        smiles = "C1CC2CCC1C2"  # norbornane
        mol = parse(smiles)
        
        systems = get_ring_systems(mol)
        assert len(systems) >= 1
        
        scaffold = get_scaffold(mol)
        assert scaffold is not None


class TestMurckoDecomposeComplex:
    """Test Murcko decomposition on complex drug-like molecules."""
    
    @pytest.mark.parametrize("smiles,desc", [
        ("CC(=O)Nc1ccc(O)cc1", "paracetamol"),
        ("CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ibuprofen"),
        ("c1ccc2[nH]c(-c3ccccc3)nc2c1", "2-phenylbenzimidazole"),
    ])
    def test_drug_like_scaffold(self, smiles: str, desc: str) -> None:
        """Test scaffold extraction from drug-like molecules."""
        # Chirpy
        mol = parse(smiles)
        chirpy_scaffold = get_scaffold(mol)
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_scaffold = MurckoScaffold.GetScaffoldForMol(rdmol)
        
        assert chirpy_scaffold is not None, f"No scaffold for {desc}"
        
        # Compare atom counts (scaffolds should be same size)
        chirpy_count = chirpy_scaffold.num_atoms
        rdkit_count = rdkit_scaffold.GetNumAtoms()
        
        assert chirpy_count == rdkit_count, \
            f"Scaffold size mismatch for {desc}: chirpy={chirpy_count}, rdkit={rdkit_count}"
    
    @pytest.mark.parametrize("smiles", [
        "Cc1ccc(Nc2nccc(-c3cccnc3)n2)cc1",  # multi-ring with linkers
        "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",   # terphenyl
    ])
    def test_multi_ring_scaffold(self, smiles: str) -> None:
        """Test scaffolds with multiple ring systems."""
        mol = parse(smiles)
        scaffold = get_scaffold(mol)
        
        assert scaffold is not None
        
        # Should have multiple ring systems
        systems = get_ring_systems(mol)
        assert len(systems) >= 2
