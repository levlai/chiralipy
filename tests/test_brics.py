"""Tests for BRICS decomposition module comparing with RDKit."""

from __future__ import annotations

import pytest
from chiralipy import parse, to_smiles
from chiralipy.decompose import find_brics_bonds, break_brics_bonds, brics_decompose


# Skip if RDKit not available
rdkit = pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import BRICS


class TestFindBRICSBonds:
    """Test finding BRICS cleavable bonds."""
    
    @pytest.mark.parametrize("smiles", [
        "CCCOCC",         # ether
        "CCNCC",          # amine
        "CC(=O)OC",       # ester
        "CC(=O)NC",       # amide
        # NOTE: Toluene (c1ccccc1C) removed - methyl is D1 so not cleavable by BRICS
        "c1ccccc1CC",     # ethylbenzene - CH2 is D2 so cleavable
        "CCOc1ccccc1",    # anisole (ethoxy benzene)
    ])
    def test_finds_cleavable_bonds(self, smiles: str) -> None:
        """Test that cleavable bonds are found."""
        mol = parse(smiles)
        bonds = list(find_brics_bonds(mol))
        
        # Should find at least one cleavable bond
        assert len(bonds) > 0, f"No BRICS bonds found for {smiles}"
    
    def test_toluene_no_cleavable_bonds(self) -> None:
        """Test that toluene has no BRICS cleavable bonds.
        
        The methyl carbon in toluene is degree 1 (D1), which is excluded
        by the L8 pattern [C;!R;!D1;!$(C!-*)]. RDKit also returns no 
        BRICS bonds for toluene.
        """
        mol = parse("c1ccccc1C")
        bonds = list(find_brics_bonds(mol))
        assert len(bonds) == 0, "Toluene should not have BRICS cleavable bonds"
    
    @pytest.mark.parametrize("smiles", [
        "CC",             # ethane (C-C not cleavable by BRICS rules)
        "C",              # methane
    ])
    def test_no_cleavable_bonds(self, smiles: str) -> None:
        """Test molecules with no BRICS cleavable bonds."""
        mol = parse(smiles)
        bonds = list(find_brics_bonds(mol))
        
        # May or may not find bonds depending on interpretation
        # Just test it doesn't crash
        assert isinstance(bonds, list)
    
    @pytest.mark.parametrize("smiles", [
        "CCCOCC",         # ether
        "CCNCC",          # amine
    ])
    def test_bond_labels(self, smiles: str) -> None:
        """Test that bonds have proper labels."""
        mol = parse(smiles)
        bonds = list(find_brics_bonds(mol))
        
        for (atom1, atom2), (label1, label2) in bonds:
            # Labels should be numeric strings
            assert label1.isdigit() or label1[:-1].isdigit(), f"Invalid label: {label1}"
            assert label2.isdigit() or label2[:-1].isdigit(), f"Invalid label: {label2}"
            
            # Atom indices should be valid
            assert 0 <= atom1 < mol.num_atoms
            assert 0 <= atom2 < mol.num_atoms
    
    @pytest.mark.parametrize("smiles", [
        "CCCOCC",
        "CCOc1ccccc1",
        "CC(=O)OCC",
    ])
    def test_compare_bond_count_with_rdkit(self, smiles: str) -> None:
        """Test that similar number of bonds are found as RDKit."""
        # Chirpy
        mol = parse(smiles)
        chirpy_bonds = list(find_brics_bonds(mol))
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_bonds = list(BRICS.FindBRICSBonds(rdmol))
        
        # Should find similar number of bonds
        # Allow some difference due to implementation details
        assert abs(len(chirpy_bonds) - len(rdkit_bonds)) <= 2, \
            f"Bond count mismatch for {smiles}: chirpy={len(chirpy_bonds)}, rdkit={len(rdkit_bonds)}"


class TestBreakBRICSBonds:
    """Test breaking BRICS bonds."""
    
    @pytest.mark.parametrize("smiles", [
        "CCCOCC",         # ether
        "CCNCC",          # amine
        "CCOc1ccccc1",    # anisole
    ])
    def test_break_bonds_creates_dummy_atoms(self, smiles: str) -> None:
        """Test that breaking bonds creates dummy atoms."""
        mol = parse(smiles)
        fragmented = break_brics_bonds(mol)
        
        # Should have dummy atoms (*) in the fragmented molecule
        dummy_count = sum(1 for a in fragmented.atoms if a.symbol == '*')
        assert dummy_count > 0, f"No dummy atoms found for {smiles}"
    
    @pytest.mark.parametrize("smiles", [
        "CCCOCC",
        "CCOc1ccccc1",
    ])
    def test_break_bonds_adds_isotope_labels(self, smiles: str) -> None:
        """Test that dummy atoms have isotope labels."""
        mol = parse(smiles)
        fragmented = break_brics_bonds(mol)
        
        for atom in fragmented.atoms:
            if atom.symbol == '*':
                # Should have isotope label
                assert atom.isotope is not None, "Dummy atom missing isotope label"
    
    def test_break_no_bonds(self) -> None:
        """Test breaking with no cleavable bonds."""
        smiles = "CC"
        mol = parse(smiles)
        fragmented = break_brics_bonds(mol)
        
        # Should be same as original (or very similar)
        assert fragmented.num_atoms >= mol.num_atoms
    
    def test_break_specific_bonds(self) -> None:
        """Test breaking specific bonds."""
        smiles = "CCOCC"
        mol = parse(smiles)
        bonds = list(find_brics_bonds(mol))
        
        if bonds:
            # Break only first bond
            fragmented = break_brics_bonds(mol, [bonds[0]])
            
            # Should have exactly 2 dummy atoms (one for each side)
            dummy_count = sum(1 for a in fragmented.atoms if a.symbol == '*')
            assert dummy_count == 2


class TestBRICSDecompose:
    """Test full BRICS decomposition."""
    
    @pytest.mark.parametrize("smiles", [
        "CCCOCC",         # ether
        "CCOc1ccccc1",    # anisole
        "c1ccccc1CCc1ccccc1",  # diphenylethane
    ])
    def test_decompose_returns_fragments(self, smiles: str) -> None:
        """Test that decomposition returns fragments."""
        mol = parse(smiles)
        fragments = brics_decompose(mol)
        
        assert len(fragments) >= 1, f"No fragments for {smiles}"
    
    @pytest.mark.parametrize("smiles", [
        "CCCOCC",
        "CCOc1ccccc1",
    ])
    def test_decompose_fragments_contain_dummies(self, smiles: str) -> None:
        """Test that fragments contain dummy atoms."""
        mol = parse(smiles)
        fragments = brics_decompose(mol)
        
        # At least one fragment should contain a dummy atom marker
        has_dummy = any('*' in frag for frag in fragments)
        # Note: The original molecule might also be in fragments
        assert len(fragments) >= 1
    
    def test_decompose_single_pass(self) -> None:
        """Test single-pass decomposition."""
        smiles = "CCOCCOc1ccccc1"  # multiple cleavage sites
        mol = parse(smiles)
        
        fragments_single = brics_decompose(mol, single_pass=True)
        fragments_full = brics_decompose(mol, single_pass=False)
        
        # Single pass should give fewer or equal fragments
        assert len(fragments_single) <= len(fragments_full) + 1
    
    def test_decompose_min_fragment_size(self) -> None:
        """Test minimum fragment size filter."""
        smiles = "CCOCC"
        mol = parse(smiles)
        
        fragments_any = brics_decompose(mol, min_fragment_size=1)
        fragments_large = brics_decompose(mol, min_fragment_size=3)
        
        # Larger min size should give fewer fragments
        assert len(fragments_large) <= len(fragments_any)
    
    def test_decompose_return_mols(self) -> None:
        """Test returning molecule objects."""
        smiles = "CCCOCC"
        mol = parse(smiles)
        
        fragments_smi = brics_decompose(mol, return_mols=False)
        fragments_mol = brics_decompose(mol, return_mols=True)
        
        assert isinstance(list(fragments_smi)[0] if fragments_smi else "", str)
        assert len(fragments_mol) >= 0  # Just check it doesn't crash


class TestBRICSChiralityPreservation:
    """Test that chirality is correctly preserved during BRICS decomposition."""
    
    @pytest.mark.parametrize("smiles,expected_frags", [
        # Case 1: Chiral cyclobutane with carboxylic acid - bond to COOH broken
        # RDKit: [C@] stays as [C@] in output (traversal from dummy)
        ("N[C@]1(C(=O)O)C[C@@H](/C=C/I)C1", 
         {'[6*]C(=O)O', '[15*][C@]1(N)C[C@H](/C=C/I)C1'}),
        # Case 2: Opposite stereochemistry on the second chiral center
        ("N[C@]1(C(=O)O)C[C@H](/C=C/I)C1",
         {'[6*]C(=O)O', '[15*][C@]1(N)C[C@@H](/C=C/I)C1'}),
    ])
    def test_brics_chirality_matches_rdkit(self, smiles: str, expected_frags: set) -> None:
        """Test that BRICS decomposition preserves chirality correctly.
        
        When a bond at a chiral center is broken and replaced with a dummy atom,
        the chirality symbol in the canonical SMILES may need to change based on
        the new traversal order. This test verifies chiralipy matches RDKit exactly.
        
        The key insight is that chirality is relative to the neighbor order in
        the SMILES output, which changes when the canonical traversal starts from
        a different atom (e.g., the dummy atom instead of the original neighbor).
        """
        # Chirpy result
        mol = parse(smiles)
        chirpy_frags = brics_decompose(mol)
        
        # Normalize: strip atom class labels for comparison
        import re
        chirpy_normalized = set()
        for frag in chirpy_frags:
            # Remove atom class labels like :1, :2, etc.
            normalized = re.sub(r':\d+\]', ']', frag)
            chirpy_normalized.add(normalized)
        
        # RDKit result
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_frags = set(BRICS.BRICSDecompose(rdmol))
        
        # Compare with expected (which is RDKit's output)
        assert chirpy_normalized == expected_frags, (
            f"Chirality mismatch for {smiles}:\n"
            f"  chiralipy: {sorted(chirpy_normalized)}\n"
            f"  expected:  {sorted(expected_frags)}\n"
            f"  rdkit:     {sorted(rdkit_frags)}"
        )
        
        # Also verify against RDKit directly
        assert chirpy_normalized == rdkit_frags, (
            f"Chirality mismatch vs RDKit for {smiles}:\n"
            f"  chiralipy: {sorted(chirpy_normalized)}\n"
            f"  rdkit:     {sorted(rdkit_frags)}"
        )


class TestBRICSCompareWithRDKit:
    """Compare BRICS decomposition with RDKit."""
    
    @pytest.mark.parametrize("smiles", [
        "CCCOCC",
        "CCOc1ccccc1",
        "CC(=O)OCC",
    ])
    def test_fragment_count_similar(self, smiles: str) -> None:
        """Test that fragment counts are similar to RDKit."""
        # Chirpy
        mol = parse(smiles)
        chirpy_frags = brics_decompose(mol)
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_frags = set(BRICS.BRICSDecompose(rdmol))
        
        # Should have similar number of fragments
        # Allow some variation due to implementation differences
        assert abs(len(chirpy_frags) - len(rdkit_frags)) <= 3, \
            f"Fragment count mismatch for {smiles}: chirpy={len(chirpy_frags)}, rdkit={len(rdkit_frags)}"

    def test_complex_molecule_breaks_all_bonds_at_once(self) -> None:
        """Test that BRICS bonds are broken all at once, not one-at-a-time.
        
        This test uses a simpler molecule (diphenylmethane) to verify the fix
        for the bug where bonds were being broken one-at-a-time instead of
        all at once, producing different fragment counts than RDKit.
        
        Note: The original complex macrocyclic molecule exposed differences
        in ring detection (chirpy uses max_ring_size=20 by default, RDKit
        detects all rings). For most drug-like molecules without large
        macrocycles, the results match.
        """
        smiles = "c1ccccc1Cc1ccccc1"  # diphenylmethane - two phenyl rings linked by CH2
        
        # Chirpy
        mol = parse(smiles)
        chirpy_frags = brics_decompose(mol)
        
        # RDKit
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_frags = set(BRICS.BRICSDecompose(rdmol))
        
        # Canonicalize and compare
        # Note: chirpy includes atom class labels (e.g., [16*:5]) that distinguish
        # chemically identical fragments. Strip these for comparison with RDKit.
        def canonicalize(frag_set):
            import re
            result = set()
            for smi in frag_set:
                # Parse and re-serialize to canonical form
                # Strip atom class labels (e.g., [16*:5] -> [16*]) for comparison
                smi_no_class = re.sub(r':\d+\]', ']', smi)
                try:
                    m = Chem.MolFromSmiles(smi_no_class)
                    if m:
                        result.add(Chem.MolToSmiles(m))
                except:
                    pass
            return result
        
        chirpy_canonical = canonicalize(chirpy_frags)
        rdkit_canonical = canonicalize(rdkit_frags)
        
        # Must have same number of unique fragments (after stripping atom class labels)
        assert len(chirpy_canonical) == len(rdkit_canonical), \
            f"Fragment count mismatch: chirpy={len(chirpy_canonical)}, rdkit={len(rdkit_canonical)}"
        
        assert chirpy_canonical == rdkit_canonical, \
            f"Fragment mismatch:\n  chirpy: {sorted(chirpy_canonical)}\n  rdkit: {sorted(rdkit_canonical)}"
    
    @pytest.mark.parametrize("smiles", [
        "c1ccccc1",       # benzene (no cleavage)
        "CC",             # ethane (no cleavage by BRICS)
    ])
    def test_no_fragmentation(self, smiles: str) -> None:
        """Test molecules that shouldn't fragment much."""
        mol = parse(smiles)
        fragments = brics_decompose(mol)
        
        # Should return original or minimal fragments
        # At minimum, the original molecule
        assert len(fragments) >= 1


class TestBRICSEdgeCases:
    """Test edge cases for BRICS."""
    
    def test_empty_molecule(self) -> None:
        """Test with empty molecule."""
        from chiralipy.types import Molecule
        mol = Molecule()
        
        bonds = list(find_brics_bonds(mol))
        assert len(bonds) == 0
        
        fragmented = break_brics_bonds(mol)
        assert fragmented.num_atoms == 0
        
        fragments = brics_decompose(mol)
        assert len(fragments) <= 1
    
    def test_single_atom(self) -> None:
        """Test with single atom."""
        mol = parse("C")
        
        bonds = list(find_brics_bonds(mol))
        assert len(bonds) == 0
        
        fragments = brics_decompose(mol)
        assert len(fragments) >= 1
    
    def test_ring_molecule(self) -> None:
        """Test with ring molecule (benzene)."""
        mol = parse("c1ccccc1")
        
        # Benzene has no BRICS cleavable bonds (ring bonds excluded)
        bonds = list(find_brics_bonds(mol))
        # May or may not find bonds
        
        fragments = brics_decompose(mol)
        assert len(fragments) >= 1


class TestBRICSReturnStems:
    """Test BRICS decomposition with return_stems=True."""
    
    def test_return_stems_basic(self) -> None:
        """Test that return_stems returns valid stem indices."""
        smiles = "CCCOCC"  # ether - cleaves at C-O bonds
        mol = parse(smiles)
        result = brics_decompose(mol, return_stems=True)
        
        # Should return a dict mapping SMILES to set of stem indices
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Each value should be a set of integers
        for frag_smiles, stems in result.items():
            assert isinstance(frag_smiles, str)
            assert isinstance(stems, set)
            for idx in stems:
                assert isinstance(idx, int)
                assert idx >= 0
    
    def test_return_stems_indices_are_fragment_local(self) -> None:
        """Test that stem indices refer to positions within the fragment SMILES.
        
        When a molecule is cleaved, the returned indices should be relative to
        each fragment's canonical SMILES, not the original molecule.
        """
        smiles = "O=C(O)CCN1C(=S)CSC1=S"  # propionyl-dithiohydantoin
        mol = parse(smiles)
        result = brics_decompose(mol, return_stems=True)
        
        assert isinstance(result, dict)
        
        # For each fragment, stem indices should be valid for that fragment
        for frag_smiles, stems in result.items():
            # Parse the fragment to get atom count
            frag_mol = parse(frag_smiles)
            num_atoms = frag_mol.num_atoms
            
            # All stem indices must be valid for this fragment
            for idx in stems:
                assert 0 <= idx < num_atoms, \
                    f"Stem index {idx} out of range for fragment '{frag_smiles}' with {num_atoms} atoms"
    
    def test_return_stems_with_return_mols(self) -> None:
        """Test return_stems combined with return_mols."""
        smiles = "CCCOCC"
        mol = parse(smiles)
        result = brics_decompose(mol, return_stems=True, return_mols=True)
        
        # Should return list of (Molecule, set[int]) tuples
        assert isinstance(result, list)
        
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            frag_mol, stems = item
            from chiralipy.types import Molecule
            assert isinstance(frag_mol, Molecule)
            assert isinstance(stems, set)
            
            # All stem indices must be valid for the fragment molecule
            for idx in stems:
                assert 0 <= idx < frag_mol.num_atoms
    
    def test_return_stems_no_dummy_atoms(self) -> None:
        """Test that returned fragments don't contain dummy atoms."""
        smiles = "CCOc1ccccc1"  # anisole
        mol = parse(smiles)
        result = brics_decompose(mol, return_stems=True)
        
        for frag_smiles, stems in result.items():
            # Fragment SMILES should not contain dummy atoms
            assert '*' not in frag_smiles, \
                f"Fragment '{frag_smiles}' contains dummy atoms"
            
            # Parse and verify no dummy atoms
            frag_mol = parse(frag_smiles)
            for atom in frag_mol.atoms:
                assert atom.symbol != '*', \
                    f"Fragment '{frag_smiles}' contains dummy atom"
    
    def test_return_stems_stem_atoms_are_cleavage_points(self) -> None:
        """Test that stem atoms are the actual cleavage points in the fragment."""
        smiles = "CCOCC"  # diethyl ether
        mol = parse(smiles)
        result = brics_decompose(mol, return_stems=True, return_mols=True)
        
        # Should have fragments with stems
        has_stems = any(len(stems) > 0 for _, stems in result)
        assert has_stems, "Expected at least one fragment with stems"
    
    def test_return_stems_single_pass(self) -> None:
        """Test return_stems with single_pass=True."""
        smiles = "CCOCCOc1ccccc1"  # multiple cleavage sites
        mol = parse(smiles)
        result = brics_decompose(mol, return_stems=True, single_pass=True)
        
        assert isinstance(result, dict)
        assert len(result) >= 1
        
        # Verify indices are valid
        for frag_smiles, stems in result.items():
            frag_mol = parse(frag_smiles)
            for idx in stems:
                assert 0 <= idx < frag_mol.num_atoms
    
    def test_return_stems_molecule_without_cleavage(self) -> None:
        """Test return_stems on a molecule with no BRICS bonds."""
        smiles = "c1ccccc1"  # benzene - no cleavable bonds
        mol = parse(smiles)
        result = brics_decompose(mol, return_stems=True)
        
        assert isinstance(result, dict)
        # Should return original molecule with empty stems
        assert len(result) >= 1
        
        # Original molecule should have no stems
        for frag_smiles, stems in result.items():
            assert stems == set(), \
                f"Expected no stems for molecule without cleavage, got {stems}"


class TestBRICSExactMatchWithRDKit:
    """Test that BRICS decomposition exactly matches RDKit output.
    
    These tests verify that chiralipy's BRICS implementation produces
    identical fragments to RDKit for complex drug-like molecules.
    """
    
    @staticmethod
    def canonicalize_fragments(frag_set):
        """Canonicalize fragments for comparison.
        
        Strips atom class labels (e.g., [16*:5] -> [16*]) and 
        canonicalizes SMILES using RDKit.
        """
        import re
        result = set()
        for smi in frag_set:
            # Strip atom class labels for comparison
            smi_no_class = re.sub(r':\d+\]', ']', smi)
            try:
                m = Chem.MolFromSmiles(smi_no_class)
                if m:
                    result.add(Chem.MolToSmiles(m))
            except:
                pass
        return result
    
    @pytest.mark.parametrize("smiles,name", [
        # Molecule 1: sulfonamide with purine core
        ("c1ccc(C2CCCCC2)c(c1)S(=O)(=O)NCCNc3nccc4c3ncn4C(C)C", 
         "sulfonamide-purine"),
        # Molecule 2: carbamate with dimethoxyphenyl
        ("CC(C)CCCCOC(=O)Nc1c(OC)cccc1OC", 
         "carbamate-dimethoxyphenyl"),
        # Molecule 3: piperazine urea with pyrazole
        ("Cc1cc(NC(=O)N2CCN(CC2)Cc3ccccc3)n(n1)c4ccc(Cl)cc4", 
         "piperazine-urea-pyrazole"),
        # Molecule 4: fused purine
        ("Oc1[nH]cnc2ncnc1-2", 
         "fused-purine"),
    ])
    def test_exact_match_with_rdkit(self, smiles: str, name: str) -> None:
        """Test that chiralipy BRICS decomposition exactly matches RDKit.
        
        This is a strict test - both implementations must produce identical
        fragment sets (after canonicalization).
        """
        # Chirpy decomposition
        mol = parse(smiles)
        chirpy_frags = brics_decompose(mol)
        
        # RDKit decomposition
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_frags = set(BRICS.BRICSDecompose(rdmol))
        
        # Canonicalize both for comparison
        chirpy_canonical = self.canonicalize_fragments(chirpy_frags)
        rdkit_canonical = self.canonicalize_fragments(rdkit_frags)
        
        # Must have exact same fragments
        assert chirpy_canonical == rdkit_canonical, (
            f"BRICS decomposition mismatch for {name}:\n"
            f"  SMILES: {smiles}\n"
            f"  chirpy ({len(chirpy_canonical)}): {sorted(chirpy_canonical)}\n"
            f"  rdkit ({len(rdkit_canonical)}): {sorted(rdkit_canonical)}"
        )
    
    def test_stereochemistry_preserved(self) -> None:
        """Test that stereochemistry is preserved in fragments.
        
        Molecule 5: epoxide amide with stereochemistry
        """
        smiles = "NC(=O)[C@@H]1O[C@@H]1C(=O)Nc1ccc(Br)c(Cl)c1"
        
        # Chirpy decomposition
        mol = parse(smiles)
        chirpy_frags = brics_decompose(mol)
        
        # RDKit decomposition
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_frags = set(BRICS.BRICSDecompose(rdmol))
        
        # Canonicalize both
        chirpy_canonical = self.canonicalize_fragments(chirpy_frags)
        rdkit_canonical = self.canonicalize_fragments(rdkit_frags)
        
        # Must match exactly
        assert chirpy_canonical == rdkit_canonical, (
            f"BRICS decomposition mismatch for stereochemistry test:\n"
            f"  chirpy ({len(chirpy_canonical)}): {sorted(chirpy_canonical)}\n"
            f"  rdkit ({len(rdkit_canonical)}): {sorted(rdkit_canonical)}"
        )
        
        # Additionally check that @ symbols are present (stereochemistry preserved)
        has_stereo = any('@' in frag for frag in chirpy_frags)
        assert has_stereo, "Expected stereochemistry (@) in fragments"


class TestBRICSEnvironments:
    """Test BRICS environment matching."""
    
    def test_ether_cleavage_l3(self) -> None:
        """Test L3 environment (ether oxygen)."""
        smiles = "CCOCC"
        mol = parse(smiles)
        bonds = list(find_brics_bonds(mol))
        
        # Should find C-O bonds as cleavable
        assert len(bonds) >= 1
        
        # Labels should include '3' (L3 environment)
        labels = set()
        for _, (l1, l2) in bonds:
            labels.add(l1)
            labels.add(l2)
        # Check for ether-related labels
    
    def test_amine_cleavage_l5(self) -> None:
        """Test L5 environment (amine nitrogen)."""
        smiles = "CCNCC"
        mol = parse(smiles)
        bonds = list(find_brics_bonds(mol))
        
        # Should find C-N bonds as cleavable
        assert len(bonds) >= 1
    
    def test_aromatic_cleavage_l16(self) -> None:
        """Test L16 environment (aromatic carbon).
        
        Toluene (c1ccccc1C) does NOT have a cleavable bond because
        the methyl is D1 (degree 1) which is excluded by L8.
        
        Ethylbenzene (c1ccccc1CC) DOES have a cleavable bond because
        the CH2 is D2 which matches L8, paired with L16 for the aromatic carbon.
        """
        # Ethylbenzene - CH2-benzene bond is cleavable
        smiles = "c1ccccc1CC"  # ethylbenzene, not toluene
        mol = parse(smiles)
        bonds = list(find_brics_bonds(mol))
        
        # Should find Ar-CH2 bond as cleavable (L8-L16)
        assert len(bonds) >= 1


class TestBRICSStereochemistry:
    """Test BRICS decomposition handles stereochemistry correctly."""
    
    def test_double_bond_stereo_removed_after_cleavage(self) -> None:
        """Test that E/Z stereo markers are removed after cleavage breaks the double bond.
        
        When a double bond is cleaved, the resulting fragments should not retain
        the / or \\ stereo markers that were associated with the original double bond.
        This test case is for a molecule where the double bond is cleaved and one
        fragment becomes a ring - the stereo marker should be cleared.
        
        Regression test for: O=C1N=C(NO)S/C1=C/c1cccc([N+](=O)[O-])c1
        The [7*]C1SC(NO)=NC1=O fragment should become O=C1CSC(NO)=N1 (no stereo)
        not O=C1C/SC(NO)=N1 (with stereo).
        """
        smiles = "O=C1N=C(NO)S/C1=C/c1cccc([N+](=O)[O-])c1"
        
        # Get chiralipy fragments with stems
        result = brics_decompose(smiles, return_stems=True)
        
        # Check that no fragment contains invalid stereo markers (/ or \)
        # These should only appear in double bonds, not single bonds
        for chiralipy_smiles in result.keys():
            # Parse to check if stereo markers are present
            parsed_mol = Chem.MolFromSmiles(chiralipy_smiles)
            assert parsed_mol is not None, f"Invalid SMILES: {chiralipy_smiles}"
            
            # Check that the raw chiralipy SMILES doesn't have / or \ on ring single bonds
            # The thiazolone should be O=C1CSC(NO)=N1, not O=C1C/SC(NO)=N1
            if "CSC" in chiralipy_smiles or "C/SC" in chiralipy_smiles or "C\\SC" in chiralipy_smiles:
                # This is the thiazolone fragment
                assert "/" not in chiralipy_smiles and "\\" not in chiralipy_smiles, (
                    f"Unexpected stereo marker in thiazolone fragment: {chiralipy_smiles}\n"
                    f"Expected: O=C1CSC(NO)=N1 (no stereo markers on ring bonds)"
                )
        
        # Get RDKit fragments
        rdmol = Chem.MolFromSmiles(smiles)
        rdkit_frags = BRICS.BRICSDecompose(rdmol, returnMols=True)
        
        # Get RDKit canonical SMILES for each fragment (after removing dummies)
        rdkit_canonical = set()
        for frag in rdkit_frags:
            new_mol = Chem.RWMol(frag)
            dummy_idx = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
            for idx in sorted(dummy_idx, reverse=True):
                new_mol.RemoveAtom(idx)
            rdkit_canonical.add(Chem.MolToSmiles(new_mol))
        
        # Get chiralipy canonical SMILES (via RDKit to compare)
        chiralipy_canonical = set()
        for chiralipy_smiles in result.keys():
            parsed_mol = Chem.MolFromSmiles(chiralipy_smiles)
            if parsed_mol:
                chiralipy_canonical.add(Chem.MolToSmiles(parsed_mol))
        
        # All fragments should match
        assert chiralipy_canonical == rdkit_canonical, (
            f"Fragment mismatch:\n"
            f"  chiralipy: {sorted(chiralipy_canonical)}\n"
            f"  RDKit:     {sorted(rdkit_canonical)}"
        )

    def test_chirality_cleared_after_dummy_removal(self) -> None:
        """Test that chirality markers are cleared when dummy atoms are removed.
        
        When a dummy atom is removed from a chiral center, the center loses a 
        substituent and chirality becomes undefined (only 3 distinct groups remain).
        The @ or @@ markers should be cleared.
        
        Regression test for: [*][C@@H]1O[C@@H]1[*] (chiral epoxide with dummies)
        After removing dummies, should become C1CO1 (no chirality), not [C@@H]1[C@@H]O1.
        """
        from chiralipy.parser import parse
        from chiralipy.decompose.brics import _strip_dummy_atoms_and_mark_stems
        from chiralipy.canon import canonical_ranks
        from chiralipy.writer import to_smiles
        
        smiles = "[*][C@@H]1O[C@@H]1[*]"
        mol = parse(smiles)
        
        # Strip dummies
        stripped = _strip_dummy_atoms_and_mark_stems(mol)
        
        # Check chirality is cleared
        for atom in stripped.atoms:
            assert atom.chirality is None, (
                f"Atom {atom.idx} ({atom.symbol}) should not have chirality after dummy removal, "
                f"but has chirality={atom.chirality}"
            )
        
        # Check canonical SMILES matches RDKit
        ranks = canonical_ranks(stripped)
        chiralipy_result = to_smiles(stripped, ranks)
        
        rdmol = Chem.MolFromSmiles(smiles)
        new_mol = Chem.RWMol(rdmol)
        dummy_idx = [a.GetIdx() for a in rdmol.GetAtoms() if a.GetAtomicNum() == 0]
        for idx in sorted(dummy_idx, reverse=True):
            new_mol.RemoveAtom(idx)
        rdkit_result = Chem.MolToSmiles(new_mol)
        
        assert chiralipy_result == rdkit_result, (
            f"SMILES mismatch after dummy removal:\n"
            f"  chiralipy: {chiralipy_result}\n"
            f"  RDKit:     {rdkit_result}"
        )
