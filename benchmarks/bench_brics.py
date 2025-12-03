#!/usr/bin/env python3
"""
Benchmark script comparing BRICS decomposition speed between RDKit and chirpy.

Usage:
    python benchmarks/bench_brics.py [--extended]

Run from the chirpy directory to use the local version:
    cd external_libs/chirpy && python benchmarks/bench_brics.py

Options:
    --extended    Run extended benchmark with multiple molecules and detailed metrics
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import Optional

# Ensure local chirpy is used (not installed version)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test molecules with varying complexity
TEST_MOLECULES = {
    "small_ether": "CCOCC",
    "medium_drug": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "drug_like": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C",  # Imatinib-like
    "large_complex": "CCn1c2ccc3cc2c2cc(ccc21)C(=O)c1ccc(cc1)Cn1c[n+](c2ccccc21)Cc1ccc(cc1)C(=O)c1ccc2c(c1)c1cc(ccc1n2CC)C(=O)c1ccc(cc1)C[n+]1cn(c2ccccc21)Cc1ccc(cc1)C3=O",
}

# Default molecule for quick benchmark
LARGE_MOLECULE = TEST_MOLECULES["large_complex"]

ITERATIONS = 1000
EXTENDED_ITERATIONS = 500


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    smiles: str
    time_seconds: float
    iterations: int
    num_fragments: int
    num_atoms: int
    num_bonds: int
    
    @property
    def time_per_call_ms(self) -> float:
        return (self.time_seconds / self.iterations) * 1000
    
    @property
    def time_per_atom_us(self) -> float:
        """Microseconds per atom per call."""
        return (self.time_seconds / self.iterations / self.num_atoms) * 1_000_000
    
    @property
    def time_per_bond_us(self) -> float:
        """Microseconds per bond per call."""
        return (self.time_seconds / self.iterations / self.num_bonds) * 1_000_000


def get_mol_stats_rdkit(smiles: str) -> tuple[int, int]:
    """Get atom and bond count using RDKit."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0, 0
    return mol.GetNumAtoms(), mol.GetNumBonds()


def get_mol_stats_chirpy(smiles: str) -> tuple[int, int]:
    """Get atom and bond count using chirpy."""
    from chiralipy import parse
    mol = parse(smiles)
    return mol.num_atoms, len(mol.bonds)


def benchmark_rdkit(smiles: str, iterations: int) -> BenchmarkResult:
    """Benchmark RDKit BRICS decomposition."""
    from rdkit import Chem
    from rdkit.Chem import BRICS
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    
    # Warmup
    _ = BRICS.BRICSDecompose(mol)
    
    start = time.perf_counter()
    for _ in range(iterations):
        fragments = BRICS.BRICSDecompose(mol)
    end = time.perf_counter()
    
    return BenchmarkResult(
        smiles=smiles,
        time_seconds=end - start,
        iterations=iterations,
        num_fragments=len(fragments),
        num_atoms=num_atoms,
        num_bonds=num_bonds,
    )


def benchmark_chirpy(smiles: str, iterations: int) -> BenchmarkResult:
    """Benchmark chirpy BRICS decomposition."""
    from chiralipy import parse
    from chiralipy.decompose import brics_decompose
    
    mol = parse(smiles)
    num_atoms = mol.num_atoms
    num_bonds = len(mol.bonds)
    
    # Warmup
    _ = brics_decompose(mol)
    
    start = time.perf_counter()
    for _ in range(iterations):
        fragments = brics_decompose(mol)
    end = time.perf_counter()
    
    return BenchmarkResult(
        smiles=smiles,
        time_seconds=end - start,
        iterations=iterations,
        num_fragments=len(fragments),
        num_atoms=num_atoms,
        num_bonds=num_bonds,
    )


def run_single_benchmark():
    """Run basic single-molecule benchmark."""
    print("=" * 70)
    print("BRICS Decomposition Benchmark: RDKit vs chirpy")
    print("=" * 70)
    print(f"\nTest molecule ({len(LARGE_MOLECULE)} chars):")
    print(f"  {LARGE_MOLECULE[:60]}...")
    print(f"\nIterations: {ITERATIONS}")
    print("-" * 70)
    
    rdkit_result: Optional[BenchmarkResult] = None
    chirpy_result: Optional[BenchmarkResult] = None
    
    # Run RDKit benchmark
    print("\nRunning RDKit benchmark...", end=" ", flush=True)
    try:
        rdkit_result = benchmark_rdkit(LARGE_MOLECULE, ITERATIONS)
        print("done")
        print(f"  Time: {rdkit_result.time_seconds:.3f}s ({rdkit_result.time_per_call_ms:.3f}ms per call)")
        print(f"  Fragments: {rdkit_result.num_fragments}")
    except ImportError:
        print("SKIPPED (rdkit not installed)")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Run chirpy benchmark
    print("\nRunning chirpy benchmark...", end=" ", flush=True)
    try:
        chirpy_result = benchmark_chirpy(LARGE_MOLECULE, ITERATIONS)
        print("done")
        print(f"  Time: {chirpy_result.time_seconds:.3f}s ({chirpy_result.time_per_call_ms:.3f}ms per call)")
        print(f"  Fragments: {chirpy_result.num_fragments}")
    except ImportError:
        print("SKIPPED (chirpy not installed)")
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if rdkit_result and chirpy_result:
        ratio = chirpy_result.time_seconds / rdkit_result.time_seconds
        if ratio < 1:
            print(f"chiralipy is {1/ratio:.2f}x FASTER than RDKit")
        else:
            print(f"chiralipy is {ratio:.2f}x SLOWER than RDKit")
    else:
        print("Could not compare (one or both libraries failed)")


def run_extended_benchmark():
    """Run extended benchmark with multiple molecules and detailed metrics."""
    print("=" * 90)
    print("EXTENDED BRICS Decomposition Benchmark: RDKit vs chirpy")
    print("=" * 90)
    print(f"\nIterations per molecule: {EXTENDED_ITERATIONS}")
    print("-" * 90)
    
    # Collect results
    results: dict[str, dict[str, Optional[BenchmarkResult]]] = {}
    
    for name, smiles in TEST_MOLECULES.items():
        print(f"\n[{name}] ({len(smiles)} chars)")
        print(f"  SMILES: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
        
        results[name] = {"rdkit": None, "chiralipy": None}
        
        # RDKit
        try:
            result = benchmark_rdkit(smiles, EXTENDED_ITERATIONS)
            results[name]["rdkit"] = result
            print(f"  RDKit:  {result.time_per_call_ms:.4f} ms/call | {result.num_fragments} frags")
        except ImportError:
            print("  RDKit:  SKIPPED (not installed)")
        except Exception as e:
            print(f"  RDKit:  ERROR ({e})")
        
        # chirpy
        try:
            result = benchmark_chirpy(smiles, EXTENDED_ITERATIONS)
            results[name]["chiralipy"] = result
            print(f"  chirpy: {result.time_per_call_ms:.4f} ms/call | {result.num_fragments} frags")
        except ImportError:
            print("  chirpy: SKIPPED (not installed)")
        except Exception as e:
            print(f"  chirpy: ERROR ({e})")
    
    # Detailed comparison table
    print("\n" + "=" * 90)
    print("Detailed Comparison")
    print("=" * 90)
    
    header = f"{'Molecule':<18} {'Atoms':>6} {'Bonds':>6} {'RDKit ms':>10} {'chirpy ms':>10} {'Ratio':>8} {'µs/atom':>10} {'µs/bond':>10}"
    print(header)
    print("-" * 90)
    
    for name in TEST_MOLECULES:
        rdkit_res = results[name]["rdkit"]
        chirpy_res = results[name]["chiralipy"]
        
        if rdkit_res and chirpy_res:
            ratio = chirpy_res.time_seconds / rdkit_res.time_seconds
            ratio_str = f"{ratio:.2f}x"
            
            print(f"{name:<18} "
                  f"{chirpy_res.num_atoms:>6} "
                  f"{chirpy_res.num_bonds:>6} "
                  f"{rdkit_res.time_per_call_ms:>10.4f} "
                  f"{chirpy_res.time_per_call_ms:>10.4f} "
                  f"{ratio_str:>8} "
                  f"{chirpy_res.time_per_atom_us:>10.2f} "
                  f"{chirpy_res.time_per_bond_us:>10.2f}")
        elif chirpy_res:
            print(f"{name:<18} "
                  f"{chirpy_res.num_atoms:>6} "
                  f"{chirpy_res.num_bonds:>6} "
                  f"{'N/A':>10} "
                  f"{chirpy_res.time_per_call_ms:>10.4f} "
                  f"{'N/A':>8} "
                  f"{chirpy_res.time_per_atom_us:>10.2f} "
                  f"{chirpy_res.time_per_bond_us:>10.2f}")
        elif rdkit_res:
            print(f"{name:<18} "
                  f"{rdkit_res.num_atoms:>6} "
                  f"{rdkit_res.num_bonds:>6} "
                  f"{rdkit_res.time_per_call_ms:>10.4f} "
                  f"{'N/A':>10} "
                  f"{'N/A':>8} "
                  f"{'N/A':>10} "
                  f"{'N/A':>10}")
        else:
            print(f"{name:<18} {'N/A':>6} {'N/A':>6} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>10} {'N/A':>10}")
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("Summary Statistics")
    print("=" * 90)
    
    ratios = []
    for name in TEST_MOLECULES:
        rdkit_res = results[name]["rdkit"]
        chirpy_res = results[name]["chiralipy"]
        if rdkit_res and chirpy_res:
            ratios.append(chirpy_res.time_seconds / rdkit_res.time_seconds)
    
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        
        print(f"Average slowdown: {avg_ratio:.2f}x")
        print(f"Best case:        {min_ratio:.2f}x")
        print(f"Worst case:       {max_ratio:.2f}x")
        
        if avg_ratio < 1:
            print(f"\nchirpy is on average {1/avg_ratio:.2f}x FASTER than RDKit")
        else:
            print(f"\nchirpy is on average {avg_ratio:.2f}x SLOWER than RDKit")
    else:
        print("Could not compute summary (missing data)")
    
    # Per-atom cost analysis for chirpy
    print("\n" + "-" * 90)
    print("chiralipy Per-Atom/Per-Bond Cost Analysis")
    print("-" * 90)
    
    chirpy_results = [r["chiralipy"] for r in results.values() if r["chiralipy"]]
    if chirpy_results:
        avg_per_atom = sum(r.time_per_atom_us for r in chirpy_results) / len(chirpy_results)
        avg_per_bond = sum(r.time_per_bond_us for r in chirpy_results) / len(chirpy_results)
        
        print(f"Average time per atom: {avg_per_atom:.2f} µs")
        print(f"Average time per bond: {avg_per_bond:.2f} µs")
        print("\nThis helps identify if performance scales linearly with molecule size.")


def main():
    if "--extended" in sys.argv or "-e" in sys.argv:
        run_extended_benchmark()
    else:
        run_single_benchmark()
        print("\n" + "-" * 70)
        print("TIP: Run with --extended for detailed multi-molecule analysis")


if __name__ == "__main__":
    main()
