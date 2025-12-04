# chiralipy

Chiralipy is a pure Python library for SMILES/SMARTS parsing, canonicalization, and molecular manipulation.

## Installation from source

```bash
pip install -e .
```

## Installation with pip
pip install chiralipy


## Quick Start

```python
from chiralipy import parse, canonical_smiles

# Parse and canonicalize
mol = parse("C(C)CC")
print(canonical_smiles(mol))  # CCCC

# Substructure matching
from chiralipy.match import substructure_search
mol = parse("c1ccccc1CCO")
pattern = parse("[OH]")
matches = substructure_search(mol, pattern)  # [(7,)]

# BRICS decomposition
from chiralipy.decompose import brics_decompose
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
from chiralipy import parse, canonical_smiles, to_smiles

from chiralipy.match import substructure_search, has_substructure
from chiralipy.decompose import brics_decompose, get_scaffold
from chiralipy.rings import find_sssr
from chiralipy.transform import kekulize
```

## Benchmark: BRICS Decomposition

Comparison against RDKit (C++ implementation):

```
Molecule            Atoms  Bonds   RDKit ms  chiralipy ms    Ratio
----------------------------------------------------------------
small_ether             5      4     0.37       0.60      1.62x
medium_drug            15     15     0.88       1.89      2.13x
drug_like              37     41     3.54       5.05      1.43x
large_complex          84     98     5.04      13.33      2.64x

Average: ~2x slower than RDKit
```

## License

MIT
