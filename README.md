# Chirpy 🐦

A pure Python library for SMILES/SMARTS parsing, canonicalization, and molecular manipulation.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from chirpy import parse, canonical_smiles

# Parse and canonicalize
mol = parse("C(C)CC")
print(canonical_smiles(mol))  # CCCC

# Substructure matching
from chirpy import substructure_search
mol = parse("c1ccccc1CCO")
pattern = parse("[OH]")
matches = substructure_search(mol, pattern)  # [(7,)]

# BRICS decomposition
from chirpy import brics_decompose
mol = parse("CCOc1ccc(CC)cc1")
fragments = brics_decompose(mol)
print(sorted(fragments))
# ['[16*]c1ccc([16*])cc1', '[3*]O[4*]', '[4*]CC', '[8*]CC']
```

## Features

- **SMILES/SMARTS parsing** with full stereochemistry support
- **Canonical SMILES** generation
- **Substructure matching** (RDKit-compatible)
- **BRICS decomposition** for retrosynthetic fragmentation
- **Aromaticity perception** based on Hückel's 4n+2 rule
- **Ring detection** (SSSR algorithm)
- **Zero dependencies** — pure Python

## Core API

```python
from chirpy import (
    parse,              # SMILES/SMARTS string → Molecule
    canonical_smiles,   # Molecule → canonical SMILES
    to_smiles,          # Molecule → SMILES (custom ordering)
    substructure_search,
    has_substructure,
    brics_decompose,
    find_brics_bonds,
)
```

## Benchmark: BRICS Decomposition

Comparison against RDKit (C++ implementation):

```
Molecule            Atoms  Bonds   RDKit ms  chirpy ms    Ratio
----------------------------------------------------------------
small_ether             5      4     0.37       0.39      1.05x
medium_drug            15     15     0.87       1.21      1.39x
drug_like              37     41     3.48       3.24      0.93x  ← faster
large_complex          84     98     5.04       8.71      1.73x

Average: 1.28x slower than RDKit
```

Run benchmark: `python benchmarks/bench_brics.py --extended`

## License

MIT
