# Chirpy 🐦

A pure Python library for SMILES/SMARTS parsing, canonicalization, and molecular representation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

## Features

- **SMILES Parsing**: Parse SMILES strings into molecular graph representations
- **SMARTS Support**: Parse SMARTS patterns with wildcards, atom lists, and query features
- **Substructure Matching**: RDKit-compatible SMARTS pattern matching
- **Canonical SMILES**: Generate deterministic canonical SMILES output
- **Aromaticity Perception**: Hückel-based aromaticity detection using electronic structure principles
- **Ring Detection**: SSSR (Smallest Set of Smallest Rings) detection
- **Stereochemistry**: Full support for tetrahedral chirality (`@`/`@@`) and E/Z isomerism (`/`/`\`)
- **Type-Safe**: Full type annotations with `py.typed` marker for static analysis
- **Zero Dependencies**: Pure Python implementation with no external dependencies

## Design Principles

Chirpy uses principled algorithms based on fundamental chemistry:

- **Aromaticity**: Based on Hückel's 4n+2 rule with π electron counting derived from
  outer electron configuration (periodic table groups), hybridization state, and formal charge
- **Canonicalization**: Uses iterative partition refinement with stable sorting to produce
  deterministic atom orderings
- **Element Properties**: Uses periodic table data (outer electrons, valence) rather than
  hard-coded element-specific rules

## Installation

### From source (recommended)

```bash
# Clone or download the repository
cd chirpy

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### With test dependencies

```bash
pip install -e ".[test]"
```

### With development dependencies

```bash
pip install -e ".[dev]"
```

### Manual installation

```bash
# Simply copy the chirpy package to your project
cp -r chirpy /path/to/your/project/
```

## Quick Start

```python
from chirpy import parse, canonical_smiles, to_smiles

# Parse a SMILES string
mol = parse("C(C)CC")
print(f"Atoms: {len(mol.atoms)}, Bonds: {len(mol.bonds)}")

# Get canonical SMILES
canon = canonical_smiles("C(C)CC")
print(canon)  # Output: CCCC

# Parse and convert back
mol = parse("c1ccc(cc1)O")
print(canonical_smiles(mol))  # Output: Oc1ccccc1

# Parse SMARTS patterns
pattern = parse("[NX3H2]")  # Primary amine pattern
print(pattern.atoms[0].connectivity_query)  # Output: 3
```

## API Reference

### Core Functions

#### `parse(smiles: str) -> Molecule`
Parse a SMILES string into a `Molecule` object.

```python
mol = parse("CCO")
```

#### `canonical_smiles(smiles_or_mol: str | Molecule) -> str`
Get the canonical SMILES string for a molecule. This is the main convenience function.

```python
canonical_smiles("C(C)C")  # Returns: "CCC"
canonical_smiles("[O-]C=O")  # Returns: "O=C[O-]"
```

#### `to_smiles(mol: Molecule, ranks: list[int] | None = None) -> str`
Convert a `Molecule` to a SMILES string with optional custom atom ordering.

### Classes

#### `Molecule`
Represents a molecular graph with atoms and bonds.

```python
mol = parse("CCO")
mol.atoms        # List of Atom objects
mol.bonds        # List of Bond objects
mol.num_atoms    # Number of atoms
mol.num_bonds    # Number of bonds
mol.connected_components()  # List of connected components
```

#### `Atom`
Represents an atom in a molecule.

```python
atom = mol.atoms[0]
atom.symbol              # Element symbol (e.g., "C", "N", "O")
atom.atomic_number       # Atomic number
atom.charge              # Formal charge
atom.is_aromatic         # Whether atom is aromatic
atom.chirality           # Chirality ("@", "@@", or None)
atom.isotope             # Isotope number or None
atom.explicit_hydrogens  # Number of explicit hydrogens
atom.bond_indices        # Indices of connected bonds

# SMARTS query properties
atom.is_wildcard         # True if wildcard atom (*)
atom.atom_list           # List of allowed elements [C,N,O]
atom.atom_list_negated   # True if negated [!C]
atom.ring_count          # Ring membership query (R, R0, R2)
atom.ring_size           # Ring size query (r5, r6)
atom.degree_query        # Degree query (D, D2)
atom.valence_query       # Valence query (v, v4)
atom.connectivity_query  # Connectivity query (X, X2)
atom.is_recursive        # True if recursive SMARTS
atom.recursive_smarts    # Recursive SMARTS content
```

#### `Bond`
Represents a bond between two atoms.

```python
bond = mol.bonds[0]
bond.atom1_idx    # Index of first atom
bond.atom2_idx    # Index of second atom
bond.order        # Bond order (1, 2, 3, 5=quadruple, 6=dative)
bond.is_aromatic  # Whether bond is aromatic
bond.stereo       # Stereo configuration ("/" or "\")
bond.is_dative    # True if dative/coordinate bond (->)
bond.is_any       # True if SMARTS any bond (~)
bond.other_atom(idx)  # Get the other atom in the bond
```

### Advanced Usage

#### Custom Canonicalization

```python
from chirpy import parse
from chirpy.canon import canonical_ranks
from chirpy.writer import SmilesWriter
from chirpy.aromaticity import perceive_aromaticity

mol = parse("c1ccccc1")
perceive_aromaticity(mol)
ranks = canonical_ranks(mol)
writer = SmilesWriter(mol, ranks)
smiles = writer.to_smiles()
```

#### Aromaticity Perception

```python
from chirpy import parse
from chirpy.aromaticity import perceive_aromaticity

mol = parse("C1=CC=CC=C1")  # Kekulé form
perceive_aromaticity(mol)
# Now mol.atoms have is_aromatic=True for the ring
```

## Package Structure

```
chirpy/                  # Repository root
├── pyproject.toml       # Package configuration
├── README.md            # This file
├── LICENSE              # MIT License
├── chirpy/              # Python package
│   ├── __init__.py      # Public API exports
│   ├── types.py         # Atom, Bond, Molecule dataclasses
│   ├── elements.py      # Periodic table and element constants
│   ├── parser.py        # SMILES/SMARTS parser
│   ├── writer.py        # SMILES writer
│   ├── canon.py         # Canonicalization algorithms
│   ├── aromaticity.py   # Aromaticity perception
│   ├── rings.py         # Ring detection (SSSR)
│   ├── match.py         # Substructure matching
│   ├── exceptions.py    # Custom exceptions
│   └── py.typed         # PEP 561 type marker
└── tests/               # Test suite
    ├── test_parser.py
    ├── test_writer.py
    ├── test_canon.py
    ├── test_aromaticity.py
    ├── test_smarts.py
    └── test_match.py
```

## Supported SMILES Features

| Feature | Example | Supported |
|---------|---------|-----------|
| Organic subset atoms | `C`, `N`, `O`, `S`, `P`, `F`, `Cl`, `Br`, `I` | ✅ |
| Bracket atoms | `[Cu]`, `[Fe+3]`, `[13C]` | ✅ |
| Aromatic atoms | `c`, `n`, `o`, `s` | ✅ |
| Single bonds | `CC` | ✅ |
| Double bonds | `C=C` | ✅ |
| Triple bonds | `C#C` | ✅ |
| Quadruple bonds | `C$C` | ✅ |
| Aromatic bonds | `c1ccccc1` | ✅ |
| Ring closures | `C1CC1`, `C%10CC%10` | ✅ |
| Branches | `C(C)C`, `C(C)(C)C` | ✅ |
| Charges | `[O-]`, `[NH4+]`, `[Fe+3]` | ✅ |
| Isotopes | `[13C]`, `[2H]` | ✅ |
| Tetrahedral chirality | `C[C@H](O)F` | ✅ |
| E/Z isomerism | `F/C=C/F`, `F/C=C\F` | ✅ |
| Disconnected structures | `[Na+].[Cl-]` | ✅ |
| Atom classes | `[C:1]` | ✅ |
| Explicit hydrogens | `[CH4]`, `[NH3]` | ✅ |
| Dative bonds | `N->B`, `B<-N` | ✅ |

## Supported SMARTS Features

| Feature | Example | Supported |
|---------|---------|-----------|
| Wildcard atom | `*`, `[*]` | ✅ |
| Atom lists | `[C,N,O]` | ✅ |
| Negation | `[!C]`, `[!N,O]` | ✅ |
| Any bond | `C~N` | ✅ |
| Ring membership | `[R]`, `[R0]`, `[R2]` | ✅ |
| Ring size | `[r5]`, `[r6]` | ✅ |
| Degree | `[D]`, `[D2]`, `[D3]` | ✅ |
| Valence | `[v]`, `[v4]` | ✅ |
| Connectivity | `[X]`, `[X2]`, `[X4]` | ✅ |
| Recursive SMARTS | `[$(C)]`, `[$(CC)]` | ✅ |
| Atomic number | `[#6]`, `[#7]` | ✅ |
| Combined queries | `[CRD2]`, `[NX3H2]` | ✅ |

### Common SMARTS Patterns

```python
from chirpy import parse

# Primary amine
pattern = parse("[NX3H2]")

# Secondary amine
pattern = parse("[NX3H1]")

# Carbonyl group
pattern = parse("[CX3]=[OX1]")

# Aromatic nitrogen
pattern = parse("[nR]")

# Carbon in 6-membered ring
pattern = parse("[Cr6]")

# Any atom in a ring
pattern = parse("[R]")
```

## Substructure Matching

Chirpy provides RDKit-compatible substructure matching using SMARTS patterns:

```python
from chirpy import parse
from chirpy.match import substructure_search, has_substructure, count_matches

# Find all matches of a pattern in a molecule
mol = parse("c1ccccc1CCO")  # phenylethanol
pattern = parse("[cR]")      # aromatic carbon in ring

matches = substructure_search(mol, pattern)
# Returns: [(0,), (1,), (2,), (3,), (4,), (5,)]  # 6 aromatic carbons

# Check if pattern exists
has_match = has_substructure(mol, pattern)  # True

# Count matches
num_matches = count_matches(mol, pattern)  # 6
```

### Pattern Matching Examples

```python
from chirpy import parse
from chirpy.match import substructure_search
from chirpy.aromaticity import perceive_aromaticity

# Hydroxyl group
mol = parse("CCO")
pattern = parse("[OH]")
matches = substructure_search(mol, pattern)  # [(2,)]

# Carbonyl
mol = parse("CC=O")
pattern = parse("[CX3]=[OX1]")
matches = substructure_search(mol, pattern)  # [(1, 2)]

# Primary amine
mol = parse("CCN")
pattern = parse("[NX3H2]")
matches = substructure_search(mol, pattern)  # [(2,)]

# Aromatic carbons in benzene
mol = parse("c1ccccc1")
perceive_aromaticity(mol)
pattern = parse("c")
matches = substructure_search(mol, pattern)  # 6 matches

# Atoms in fused ring systems
mol = parse("c1ccc2ccccc2c1")  # naphthalene
perceive_aromaticity(mol)
pattern = parse("[R2]")  # atoms in exactly 2 rings
matches = substructure_search(mol, pattern)  # [(4,), (9,)] - bridgehead atoms
```

### Match Options

```python
from chirpy.match import substructure_search

mol = parse("CCOCC")
pattern = parse("CC")

# Default: unique atom sets only
matches = substructure_search(mol, pattern)  # [(0, 1), (2, 3), (3, 4)]

# All permutations (like RDKit's uniquify=False)
matches = substructure_search(mol, pattern, uniquify=False)  # includes reversed tuples
```

## Exceptions

```python
from chirpy.exceptions import (
    ChemError,           # Base exception
    SmilesParseError,    # Invalid SMILES syntax
    InvalidAtomError,    # Unknown element symbol
    RingError,           # Unmatched ring closure
    ValenceError,        # Invalid valence
    AromaticityError,    # Aromaticity perception failed
)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome!
