#!/usr/bin/env python3
"""
Benchmark script comparing BRICS decomposition speed between RDKit and chirpy.

Usage:
    python benchmarks/bench_brics.py

Run from the chirpy directory to use the local version:
    cd external_libs/chirpy && python benchmarks/bench_brics.py
"""

import sys
import os
import time

# Ensure local chirpy is used (not installed version)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Large test molecule
LARGE_MOLECULE = "CCn1c2ccc3cc2c2cc(ccc21)C(=O)c1ccc(cc1)Cn1c[n+](c2ccccc21)Cc1ccc(cc1)C(=O)c1ccc2c(c1)c1cc(ccc1n2CC)C(=O)c1ccc(cc1)C[n+]1cn(c2ccccc21)Cc1ccc(cc1)C3=O"

ITERATIONS = 1000


def benchmark_rdkit(smiles: str, iterations: int) -> tuple[float, int]:
    """Benchmark RDKit BRICS decomposition."""
    from rdkit import Chem
    from rdkit.Chem import BRICS
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    
    # Warmup
    _ = BRICS.BRICSDecompose(mol)
    
    start = time.perf_counter()
    for _ in range(iterations):
        fragments = BRICS.BRICSDecompose(mol)
    end = time.perf_counter()
    
    return end - start, len(fragments)


def benchmark_chirpy(smiles: str, iterations: int) -> tuple[float, int]:
    """Benchmark chirpy BRICS decomposition."""
    from chirpy import parse
    from chirpy.brics import brics_decompose
    
    mol = parse(smiles)
    
    # Warmup
    _ = brics_decompose(mol)
    
    start = time.perf_counter()
    for _ in range(iterations):
        fragments = brics_decompose(mol)
    end = time.perf_counter()
    
    return end - start, len(fragments)


def main():
    print("=" * 60)
    print("BRICS Decomposition Benchmark: RDKit vs chirpy")
    print("=" * 60)
    print(f"\nTest molecule ({len(LARGE_MOLECULE)} chars):")
    print(f"  {LARGE_MOLECULE[:60]}...")
    print(f"\nIterations: {ITERATIONS}")
    print("-" * 60)
    
    # Run RDKit benchmark
    print("\nRunning RDKit benchmark...", end=" ", flush=True)
    try:
        rdkit_time, rdkit_frags = benchmark_rdkit(LARGE_MOLECULE, ITERATIONS)
        print("done")
        print(f"  Time: {rdkit_time:.3f}s ({rdkit_time/ITERATIONS*1000:.3f}ms per call)")
        print(f"  Fragments: {rdkit_frags}")
    except ImportError:
        print("SKIPPED (rdkit not installed)")
        rdkit_time = None
    except Exception as e:
        print(f"ERROR: {e}")
        rdkit_time = None
    
    # Run chirpy benchmark
    print("\nRunning chirpy benchmark...", end=" ", flush=True)
    try:
        chirpy_time, chirpy_frags = benchmark_chirpy(LARGE_MOLECULE, ITERATIONS)
        print("done")
        print(f"  Time: {chirpy_time:.3f}s ({chirpy_time/ITERATIONS*1000:.3f}ms per call)")
        print(f"  Fragments: {chirpy_frags}")
    except ImportError:
        print("SKIPPED (chirpy not installed)")
        chirpy_time = None
    except Exception as e:
        print(f"ERROR: {e}")
        chirpy_time = None
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if rdkit_time and chirpy_time:
        ratio = chirpy_time / rdkit_time
        if ratio < 1:
            print(f"chirpy is {1/ratio:.2f}x FASTER than RDKit")
        else:
            print(f"chirpy is {ratio:.2f}x SLOWER than RDKit")
    else:
        print("Could not compare (one or both libraries failed)")


if __name__ == "__main__":
    main()
