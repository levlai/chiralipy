"""
SMILES/SMARTS string parser.

This module provides a robust parser that converts SMILES and SMARTS strings
into Molecule objects.

SMILES features:
    - All standard SMILES notation including aromatic atoms
    - Ring closures, branches, stereochemistry
    - Quadruple ($) and dative (->, <-) bonds

SMARTS features:
    - Wildcard atoms (*)
    - Any bond (~)
    - Atom lists [C,N,O]
    - Negated atoms [!C]
    - Ring membership (R, R0, R1, r5, r6)
    - Degree (D), valence (v), connectivity (X)
    - Recursive SMARTS ($(...))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from chirpy.elements import (
    AROMATIC_SUBSET,
    ORGANIC_SUBSET,
    TWO_LETTER_ORGANIC,
    Element,
    is_aromatic_symbol,
    BondOrder,
)
from chirpy.exceptions import ParseError, RingError
from chirpy.types import Atom, Bond, Molecule


class _Tokenizer:
    """Low-level SMILES tokenizer.
    
    Provides character-by-character access to a SMILES string with
    lookahead capability.
    """
    
    __slots__ = ("_string", "_pos")
    
    def __init__(self, string: str) -> None:
        self._string = string
        self._pos = 0
    
    @property
    def position(self) -> int:
        """Current position in the string."""
        return self._pos
    
    @property
    def remaining(self) -> str:
        """Remaining unparsed string."""
        return self._string[self._pos:]
    
    def peek(self, offset: int = 0) -> str | None:
        """Look at character at current position + offset without consuming.
        
        Args:
            offset: Positions ahead to look (default 0 = current).
        
        Returns:
            Character at position, or None if past end.
        """
        pos = self._pos + offset
        if pos >= len(self._string):
            return None
        return self._string[pos]
    
    def next(self) -> str | None:
        """Consume and return the next character.
        
        Returns:
            Next character, or None if at end.
        """
        if self._pos >= len(self._string):
            return None
        char = self._string[self._pos]
        self._pos += 1
        return char
    
    def skip(self, count: int = 1) -> None:
        """Skip forward by count characters."""
        self._pos += count
    
    def read_while(self, predicate) -> str:
        """Read characters while predicate is true.
        
        Args:
            predicate: Function(char) -> bool.
        
        Returns:
            String of consumed characters.
        """
        start = self._pos
        while self._pos < len(self._string) and predicate(self._string[self._pos]):
            self._pos += 1
        return self._string[start:self._pos]
    
    def read_number(self) -> int | None:
        """Read and return an integer, or None if no digits present."""
        digits = self.read_while(str.isdigit)
        return int(digits) if digits else None
    
    def is_eof(self) -> bool:
        """Check if at end of string."""
        return self._pos >= len(self._string)
    
    def expect(self, char: str) -> None:
        """Consume expected character or raise error.
        
        Args:
            char: Expected character.
        
        Raises:
            ParseError: If next character doesn't match.
        """
        actual = self.next()
        if actual != char:
            raise ParseError(
                f"Expected '{char}', got '{actual}'",
                self._string,
                self._pos - 1,
            )


@dataclass
class _ParserState:
    """Mutable state for the SMILES parser."""
    
    # Ring closure tracking: ring_index -> (atom_idx, pending_bond_order, pending_aromatic)
    open_rings: dict[int, tuple[int, int | None, bool | None]] = field(default_factory=dict)
    
    # Branch stack for parentheses
    branch_stack: list[int] = field(default_factory=list)
    
    # Current state
    prev_atom: int | None = None
    pending_bond_order: int | None = None
    pending_bond_aromatic: bool | None = None
    pending_bond_stereo: str | None = None
    pending_bond_dative: bool = False
    pending_bond_dative_dir: int = 0  # 1 = ->, -1 = <-
    pending_bond_any: bool = False  # SMARTS any bond
    
    # SMARTS bond query fields
    pending_bond_ring: bool | None = None  # True = must be ring bond (@)
    pending_bond_not_ring: bool = False    # True = must NOT be ring bond (!@)


class SmilesParser:
    """SMILES/SMARTS string parser.
    
    Parses SMILES (Simplified Molecular Input Line Entry System) and SMARTS
    (SMILES Arbitrary Target Specification) strings into Molecule objects.
    
    Supported SMILES features:
        - Organic subset atoms (B, C, N, O, P, S, F, Cl, Br, I)
        - Bracket atoms with charges, isotopes, hydrogens
        - Single, double, triple, quadruple bonds
        - Dative/coordinate bonds (-> and <-)
        - Aromatic atoms and bonds
        - Ring closures (1-9, %10-99, %(100+))
        - Branches (parentheses)
        - Multi-component molecules (dot separator)
        - Chirality markers (@ and @@)
        - Bond stereochemistry (/ and \\)
    
    Supported SMARTS features:
        - Wildcard atom (*)
        - Any bond (~)
        - Atom lists ([C,N,O])
        - Negated atoms ([!C])
        - Ring membership (R, R0, R1, R2)
        - Ring size (r5, r6)
        - Degree (D1, D2)
        - Valence (v1, v2)
        - Connectivity (X1, X2)
        - Recursive SMARTS ($(...))
    
    Example:
        >>> parser = SmilesParser("CCO")
        >>> mol = parser.parse()
        >>> len(mol.atoms)
        3
    
    For convenience, use the module-level `parse()` function:
        >>> from chirpy import parse
        >>> mol = parse("CCO")
    """
    
    # Bond character mapping: char -> (order, aromatic, is_dative, dative_direction, is_any)
    _BOND_CHARS: Final[dict[str, tuple[int | None, bool | None, bool, int, bool]]] = {
        "-": (1, False, False, 0, False),   # Single
        "=": (2, False, False, 0, False),   # Double
        "#": (3, False, False, 0, False),   # Triple
        "$": (5, False, False, 0, False),   # Quadruple (BondOrder.QUADRUPLE)
        ":": (1, True, False, 0, False),    # Aromatic
        "~": (0, None, False, 0, True),     # Any bond (BondOrder.ANY)
    }
    
    def __init__(self, smiles: str) -> None:
        """Initialize parser with a SMILES string.
        
        Args:
            smiles: SMILES string to parse.
        """
        self._smiles = smiles
        self._tokenizer = _Tokenizer(smiles)
        self._mol = Molecule()
        self._state = _ParserState()
    
    def parse(self) -> Molecule:
        """Parse the SMILES string into a Molecule.
        
        Returns:
            Parsed Molecule object.
        
        Raises:
            ParseError: If SMILES syntax is invalid.
            RingError: If ring closures are invalid.
        """
        tok = self._tokenizer
        
        while not tok.is_eof():
            char = tok.peek()
            
            if char is None:
                break
            
            if char == ".":
                # Component separator
                tok.next()
                self._state.prev_atom = None
                self._state.pending_bond_order = None
                self._state.pending_bond_aromatic = None
                self._state.pending_bond_dative = False
                self._state.pending_bond_dative_dir = 0
                self._state.pending_bond_any = False
                continue
            
            # Check for dative bonds -> or <- BEFORE regular bonds
            if char == "-" and tok.peek(1) == ">":
                tok.skip(2)  # consume "->"
                self._state.pending_bond_dative = True
                self._state.pending_bond_dative_dir = 1  # forward direction
                self._state.pending_bond_order = 6  # DATIVE order
                continue
            
            if char == "<" and tok.peek(1) == "-":
                tok.skip(2)  # consume "<-"
                self._state.pending_bond_dative = True
                self._state.pending_bond_dative_dir = -1  # reverse direction
                self._state.pending_bond_order = 6  # DATIVE order
                continue
            
            if char in "-=#:$~@" or char in "/\\":
                # Regular bond or stereo or ring bond constraint (@)
                self._parse_bond()
                continue
            
            if char == "(":
                tok.next()
                if self._state.prev_atom is not None:
                    self._state.branch_stack.append(self._state.prev_atom)
                continue
            
            if char == ")":
                tok.next()
                if self._state.branch_stack:
                    self._state.prev_atom = self._state.branch_stack.pop()
                continue
            
            if char.isdigit() or char == "%":
                self._parse_ring_closure()
                continue
            
            # Wildcard atom (SMARTS)
            if char == "*":
                tok.next()
                self._parse_wildcard_atom()
                continue
            
            if char == "[":
                self._parse_bracket_atom()
                continue
            
            # Must be an atom symbol
            if char.isalpha():
                self._parse_organic_atom()
                continue
            
            raise ParseError(
                f"Unexpected character: '{char}'",
                self._smiles,
                tok.position,
            )
        
        # Validate: no unclosed rings
        if self._state.open_rings:
            unclosed = sorted(self._state.open_rings.keys())
            raise RingError(
                f"Unclosed ring indices: {unclosed}",
                ring_index=unclosed[0],
            )
        
        return self._mol
    
    def _parse_bond(self) -> None:
        """Parse a bond symbol, including SMARTS bond expressions.
        
        SMARTS bond expressions can include:
        - Basic bonds: - = # : ~ $
        - Ring bond: @ (must be in ring)
        - Not ring bond: !@
        - AND operator: ; (e.g., -;!@ means single AND not ring)
        - Stereo markers: / \\
        """
        tok = self._tokenizer
        char = tok.next()
        
        # Reset bond query state
        self._state.pending_bond_ring = None
        self._state.pending_bond_not_ring = False
        
        if char in self._BOND_CHARS:
            order, aromatic, is_dative, dative_dir, is_any = self._BOND_CHARS[char]
            self._state.pending_bond_order = order
            self._state.pending_bond_aromatic = aromatic
            self._state.pending_bond_stereo = None
            self._state.pending_bond_dative = is_dative
            self._state.pending_bond_dative_dir = dative_dir
            self._state.pending_bond_any = is_any
            
            # Check for SMARTS bond expression continuation (;)
            self._parse_bond_expression()
            
        elif char == "@":
            # Ring bond constraint (standalone)
            self._state.pending_bond_ring = True
            self._parse_bond_expression()
            
        elif char in "/\\":
            # Stereochemistry marker - keep bond implicit
            self._state.pending_bond_stereo = char
    
    def _parse_bond_expression(self) -> None:
        """Parse remaining SMARTS bond expression after initial bond symbol.
        
        Handles: ;!@ ;@ !@ @ and combinations
        """
        tok = self._tokenizer
        
        while tok.peek() in ";!@":
            char = tok.peek()
            
            if char == ";":
                # AND operator - continue parsing
                tok.next()
                continue
            
            if char == "!":
                # Negation - check what follows
                tok.next()
                if tok.peek() == "@":
                    tok.next()
                    self._state.pending_bond_not_ring = True
                # Could extend for other negated bond properties
                continue
            
            if char == "@":
                # Ring bond constraint
                tok.next()
                self._state.pending_bond_ring = True
                continue
    
    def _parse_wildcard_atom(self) -> None:
        """Parse a wildcard atom (*)."""
        is_first = self._state.prev_atom is None
        
        atom_idx = self._mol.add_atom(
            symbol="*",
            is_aromatic=False,
        )
        self._mol.atoms[atom_idx].is_wildcard = True
        self._mol.atoms[atom_idx]._was_first_in_component = is_first
        
        # Add bond to previous atom
        self._add_bond_to_previous(atom_idx, False)
        
        self._state.prev_atom = atom_idx
    
    def _parse_ring_closure(self) -> None:
        """Parse a ring closure digit."""
        tok = self._tokenizer
        
        # Read ring index
        ring_idx = self._read_ring_index()
        
        if self._state.prev_atom is None:
            raise ParseError(
                "Ring closure without preceding atom",
                self._smiles,
                tok.position,
            )
        
        if ring_idx in self._state.open_rings:
            # Close the ring
            atom1, order1, aromatic1 = self._state.open_rings.pop(ring_idx)
            atom2 = self._state.prev_atom
            
            # Determine bond properties
            order = self._state.pending_bond_order or order1 or 1
            aromatic = self._state.pending_bond_aromatic
            if aromatic is None:
                aromatic = aromatic1
            if aromatic is None:
                # Infer from atoms
                aromatic = (
                    self._mol.atoms[atom1].is_aromatic and
                    self._mol.atoms[atom2].is_aromatic
                )
            
            self._mol.add_bond(
                atom1,
                atom2,
                order=1 if aromatic else order,
                is_aromatic=aromatic,
            )
        else:
            # Open new ring closure
            self._state.open_rings[ring_idx] = (
                self._state.prev_atom,
                self._state.pending_bond_order,
                self._state.pending_bond_aromatic,
            )
        
        # Reset pending bond
        self._state.pending_bond_order = None
        self._state.pending_bond_aromatic = None
    
    def _read_ring_index(self) -> int:
        """Read a ring closure index (1-9, %nn, %(n))."""
        tok = self._tokenizer
        
        if tok.peek() == "%":
            tok.next()  # consume '%'
            
            if tok.peek() == "(":
                # %(number) format
                tok.next()  # consume '('
                num = tok.read_number()
                if num is None:
                    raise ParseError(
                        "Empty ring index in %()",
                        self._smiles,
                        tok.position,
                    )
                tok.expect(")")
                return num
            
            # %nn format (two digits)
            d1 = tok.next()
            d2 = tok.next()
            if not (d1 and d1.isdigit() and d2 and d2.isdigit()):
                raise ParseError(
                    "Expected two digits after %",
                    self._smiles,
                    tok.position,
                )
            return int(d1 + d2)
        
        # Single digit
        char = tok.next()
        if not char or not char.isdigit():
            raise ParseError(
                "Invalid ring index",
                self._smiles,
                tok.position,
            )
        return int(char)
    
    def _parse_organic_atom(self) -> None:
        """Parse an organic subset atom (not in brackets)."""
        tok = self._tokenizer
        
        char1 = tok.next()
        assert char1 is not None
        
        symbol = char1
        
        # Check for two-letter symbol
        char2 = tok.peek()
        if char2 and char2.islower():
            candidate = char1 + char2
            # Accept Cl, Br, or aromatic two-letter (as, se)
            if candidate in TWO_LETTER_ORGANIC:
                tok.next()
                symbol = candidate
            elif candidate.lower() in AROMATIC_SUBSET:
                tok.next()
                symbol = candidate.lower()
        
        # Determine aromaticity from case
        aromatic = symbol.islower() and is_aromatic_symbol(symbol)
        
        # Track if first in component (for chirality)
        is_first = self._state.prev_atom is None
        
        # Add atom
        atom_idx = self._mol.add_atom(
            symbol=symbol,
            is_aromatic=aromatic,
        )
        self._mol.atoms[atom_idx]._was_first_in_component = is_first
        
        # Add bond to previous atom
        self._add_bond_to_previous(atom_idx, aromatic)
        
        self._state.prev_atom = atom_idx
    
    def _parse_bracket_atom(self) -> None:
        """Parse a bracket atom [...].
        
        Supports SMILES and SMARTS features including:
        - Basic: [C], [N+], [O-]
        - Isotopes: [13C], [2H]
        - Chirality: [C@H], [C@@H]
        - Wildcard: [*]
        - Atom lists: [C,N,O]
        - Negation: [!C], [!C,N]
        - Ring queries: [R], [R0], [R1], [r5], [r6]
        - Degree: [D], [D2], [D3]
        - Valence: [v], [v4]
        - Connectivity: [X], [X2], [X4]
        - Hydrogen count: [H], [H2]
        - Recursive SMARTS: [$(...)]
        """
        tok = self._tokenizer
        start_pos = tok.position
        
        tok.expect("[")
        
        # Check for wildcard
        if tok.peek() == "*":
            tok.next()
            tok.expect("]")
            is_first = self._state.prev_atom is None
            atom_idx = self._mol.add_atom(
                symbol="*",
                is_wildcard=True,
            )
            self._mol.atoms[atom_idx]._was_first_in_component = is_first
            self._add_bond_to_previous(atom_idx, False)
            self._state.prev_atom = atom_idx
            return
        
        # Check for atom list negation
        is_negated = False
        if tok.peek() == "!":
            tok.next()
            is_negated = True
        
        # Optional isotope
        isotope = tok.read_number()
        
        # SMARTS query attributes
        ring_count: int | None = None
        ring_size: int | None = None
        degree_query: int | None = None
        valence_query: int | None = None
        connectivity_query: int | None = None
        is_recursive = False
        recursive_smarts: str | None = None
        atom_list: list[str] = []
        atomic_number_list: list[int] = []
        charge_query: int | None = None
        
        # Check for recursive SMARTS $(...) 
        if tok.peek() == "$" and self._lookahead_recursive():
            tok.next()  # consume $
            tok.expect("(")
            # Read until matching )
            recursive_smarts = self._read_recursive_smarts()
            is_recursive = True
            # May have closing bracket
            while tok.peek() and tok.peek() != "]":
                tok.next()
            tok.expect("]")
            is_first = self._state.prev_atom is None
            atom_idx = self._mol.add_atom(
                symbol="*",
                is_recursive=True,
                recursive_smarts=recursive_smarts,
            )
            self._mol.atoms[atom_idx]._was_first_in_component = is_first
            self._add_bond_to_previous(atom_idx, False)
            self._state.prev_atom = atom_idx
            return
        
        # Try to parse element symbol or SMARTS query
        char1 = tok.peek()
        
        # Handle SMARTS queries that start with capital letters or special chars
        # Note: # is handled separately below to support atomic number lists [#0,#6,#7]
        if char1 in "RDXvr^":
            # These are SMARTS query primitives
            symbol, ring_count, ring_size, degree_query, valence_query, connectivity_query = \
                self._parse_smarts_queries()
            if symbol is None:
                symbol = "*"  # default wildcard if only queries
        elif char1 and char1.isalpha():
            tok.next()
            symbol = char1
            
            # Check for two-letter symbol
            char2 = tok.peek()
            if char2 and char2.islower():
                candidate = char1 + char2
                # Check periodic table
                if Element.from_symbol(candidate) is not None:
                    tok.next()
                    symbol = candidate
                elif candidate.lower() in AROMATIC_SUBSET:
                    tok.next()
                    symbol = candidate.lower()
            
            # Check for atom list (comma-separated)
            if tok.peek() == ",":
                atom_list.append(symbol)
                while tok.peek() == ",":
                    tok.next()
                    next_symbol = self._read_element_symbol_or_atomic_number()
                    if next_symbol:
                        atom_list.append(next_symbol)
                symbol = atom_list[0] if atom_list else "*"
        elif char1 == "#":
            # Atomic number specification [#6] = carbon or [#0,#6,#7] = list
            tok.next()
            atomic_num = tok.read_number()
            if atomic_num is not None:
                atomic_number_list.append(atomic_num)
                # Handle atomic number as symbol
                if atomic_num == 0:
                    symbol = "*"  # #0 = dummy atom / any
                else:
                    elem = Element.from_atomic_number(atomic_num)
                    symbol = elem.symbol if elem else "*"
                
                # Check for comma-separated atomic number list [#0,#6,#7]
                while tok.peek() == ",":
                    tok.next()
                    if tok.peek() == "#":
                        tok.next()
                        next_num = tok.read_number()
                        if next_num is not None:
                            atomic_number_list.append(next_num)
                    else:
                        # Could be element symbol mixed in
                        next_sym = self._read_element_symbol()
                        if next_sym:
                            atom_list.append(next_sym)
            else:
                symbol = "*"
        elif char1 in "+-":
            # Bare charge query [+], [-], [+2], [-3], etc.
            # This is a wildcard with charge constraint
            symbol = "*"
            # Don't consume the + or - here; it will be handled in the charge parsing loop below
        else:
            raise ParseError(
                "Expected element symbol or SMARTS query",
                self._smiles,
                tok.position,
            )
        
        # Aromaticity from lowercase
        aromatic = symbol.islower() if symbol != "*" else False
        
        # Parse remaining SMARTS primitives and standard SMILES attributes
        chirality: str | None = None
        hydrogens = 0
        charge = 0
        atom_class: int | None = None
        
        while tok.peek() and tok.peek() != "]":
            char = tok.peek()
            
            if char == "@":
                # Chirality
                tok.next()
                if tok.peek() == "@":
                    tok.next()
                    chirality = "@@"
                else:
                    chirality = "@"
            elif char == "H":
                # Hydrogen count
                tok.next()
                h_count = tok.read_number()
                hydrogens = h_count if h_count is not None else 1
            elif char in "+-":
                # Charge - could be +0 which means "charge must be 0"
                charge, is_explicit_zero = self._parse_charge_with_query()
                if is_explicit_zero:
                    charge_query = 0  # Explicit +0 or -0 means charge must be 0
            elif char == ":":
                # Atom class
                tok.next()
                atom_class = tok.read_number()
            elif char == "R":
                # Ring membership
                tok.next()
                ring_count = tok.read_number()
                if ring_count is None:
                    ring_count = -1  # -1 means "any ring"
            elif char == "r":
                # Ring size
                tok.next()
                ring_size = tok.read_number()
                if ring_size is None:
                    ring_size = -1  # any ring size
            elif char == "D":
                # Degree
                tok.next()
                degree_query = tok.read_number()
                if degree_query is None:
                    degree_query = -1  # any degree
            elif char == "X":
                # Connectivity (total connections including H)
                tok.next()
                connectivity_query = tok.read_number()
                if connectivity_query is None:
                    connectivity_query = -1
            elif char == "v":
                # Valence
                tok.next()
                valence_query = tok.read_number()
                if valence_query is None:
                    valence_query = -1
            elif char == "^":
                # Hybridization query: ^1=sp, ^2=sp2, ^3=sp3
                tok.next()
                hyb_num = tok.read_number()
                # Store as hybridization_query attribute (1=sp, 2=sp2, 3=sp3)
                # We'll add this to SMARTS attributes
                pass  # Currently parsed but not stored; can be extended
            elif char == ",":
                # More atom list entries
                tok.next()
                next_symbol = self._read_element_symbol()
                if next_symbol and next_symbol not in atom_list:
                    if not atom_list and symbol != "*":
                        atom_list.append(symbol)
                    atom_list.append(next_symbol)
            elif char == ";":
                # SMARTS AND operator - just continue parsing more attributes
                tok.next()
            elif char == "&":
                # SMARTS explicit AND operator - just continue parsing
                tok.next()
            elif char == "$":
                # Recursive SMARTS $(...) appearing after other attributes
                if self._lookahead_recursive():
                    tok.next()  # consume $
                    tok.expect("(")
                    recursive_smarts = self._read_recursive_smarts()
                    is_recursive = True
                else:
                    tok.next()  # Skip unrecognized $
            elif char == "!":
                # Negation - handle negated queries like !R, !D1, etc.
                tok.next()
                next_char = tok.peek()
                if next_char == "R":
                    tok.next()
                    num = tok.read_number()
                    ring_count = 0 if num is None else -num - 100  # Signal negated
                elif next_char == "D":
                    tok.next()
                    num = tok.read_number()
                    # Could store negated degree
                elif next_char == "$":
                    # Negated recursive SMARTS !$(...)
                    if self._lookahead_recursive():
                        tok.next()  # consume $
                        tok.expect("(")
                        neg_recursive = self._read_recursive_smarts()
                        # Could store negated recursive
                    else:
                        tok.next()
                # Skip other negated things for now
            elif char == "a":
                # Aromatic query - lowercase 'a' means any aromatic
                tok.next()
                aromatic = True
            elif char == "A":
                # Aliphatic query - uppercase 'A' means any aliphatic
                tok.next()
                aromatic = False
            else:
                # Skip unrecognized
                tok.next()
        
        tok.expect("]")
        
        # Track if first in component
        is_first = self._state.prev_atom is None
        
        # Add atom with all SMARTS properties
        atom_idx = self._mol.add_atom(
            symbol=symbol,
            charge=charge,
            explicit_hydrogens=hydrogens,
            is_aromatic=aromatic,
            isotope=isotope,
            chirality=chirality,
            atom_class=atom_class,
            is_wildcard=(symbol == "*"),
            atom_list=atom_list if atom_list else None,
            atom_list_negated=is_negated,
            atomic_number_list=atomic_number_list if atomic_number_list else None,
            ring_count=ring_count,
            ring_size=ring_size,
            degree_query=degree_query,
            valence_query=valence_query,
            connectivity_query=connectivity_query,
            is_recursive=is_recursive,
            recursive_smarts=recursive_smarts,
            charge_query=charge_query,
        )
        self._mol.atoms[atom_idx]._was_first_in_component = is_first
        
        # Add bond to previous atom
        self._add_bond_to_previous(atom_idx, aromatic)
        
        self._state.prev_atom = atom_idx
    
    def _lookahead_recursive(self) -> bool:
        """Check if next chars are $( for recursive SMARTS."""
        tok = self._tokenizer
        pos = tok.position
        if pos + 1 < len(self._smiles):
            return self._smiles[pos + 1] == "("
        return False
    
    def _read_recursive_smarts(self) -> str:
        """Read recursive SMARTS content until matching parenthesis.
        
        Properly handles nested brackets [...] within the recursive content.
        """
        tok = self._tokenizer
        paren_depth = 1
        bracket_depth = 0
        content = []
        
        while paren_depth > 0:
            char = tok.next()
            if char is None:
                raise ParseError(
                    "Unclosed recursive SMARTS",
                    self._smiles,
                    tok.position,
                )
            
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            elif char == "(" and bracket_depth == 0:
                paren_depth += 1
            elif char == ")" and bracket_depth == 0:
                paren_depth -= 1
                if paren_depth == 0:
                    break
            
            content.append(char)
        
        return "".join(content)
    
    def _read_element_symbol(self) -> str | None:
        """Read an element symbol (one or two letters)."""
        tok = self._tokenizer
        char1 = tok.peek()
        if not char1 or not char1.isalpha():
            return None
        tok.next()
        symbol = char1
        char2 = tok.peek()
        if char2 and char2.islower():
            candidate = char1 + char2
            if Element.from_symbol(candidate) is not None or candidate.lower() in AROMATIC_SUBSET:
                tok.next()
                symbol = candidate
        return symbol
    
    def _read_element_symbol_or_atomic_number(self) -> str | None:
        """Read an element symbol or atomic number specification (#N)."""
        tok = self._tokenizer
        char1 = tok.peek()
        
        if char1 == "#":
            # Atomic number specification
            tok.next()
            atomic_num = tok.read_number()
            if atomic_num is not None:
                if atomic_num == 0:
                    return "*"  # #0 = dummy atom
                elem = Element.from_atomic_number(atomic_num)
                return elem.symbol if elem else "*"
            return None
        elif char1 and char1.isalpha():
            return self._read_element_symbol()
        return None
    
    def _parse_smarts_queries(self) -> tuple:
        """Parse SMARTS query primitives at start of bracket atom.
        
        Note: # (atomic number) is NOT handled here - it's handled separately
        in _parse_bracket_atom to support atomic number lists [#0,#6,#7].
        
        Returns:
            Tuple of (symbol, ring_count, ring_size, degree, valence, connectivity)
        """
        tok = self._tokenizer
        symbol: str | None = None
        ring_count: int | None = None
        ring_size: int | None = None
        degree_query: int | None = None
        valence_query: int | None = None
        connectivity_query: int | None = None
        
        while tok.peek() in "RDXvr^":
            char = tok.peek()
            if char == "R":
                tok.next()
                ring_count = tok.read_number()
                if ring_count is None:
                    ring_count = -1
            elif char == "r":
                tok.next()
                ring_size = tok.read_number()
                if ring_size is None:
                    ring_size = -1
            elif char == "D":
                tok.next()
                degree_query = tok.read_number()
                if degree_query is None:
                    degree_query = -1
            elif char == "X":
                tok.next()
                connectivity_query = tok.read_number()
                if connectivity_query is None:
                    connectivity_query = -1
            elif char == "v":
                tok.next()
                valence_query = tok.read_number()
                if valence_query is None:
                    valence_query = -1
            elif char == "^":
                # Hybridization query: ^1=sp, ^2=sp2, ^3=sp3
                # Consume and ignore for now; can be stored if needed
                tok.next()
                tok.read_number()  # consume the number
        
        return symbol, ring_count, ring_size, degree_query, valence_query, connectivity_query
    
    def _parse_charge(self) -> int:
        """Parse optional charge (+, -, ++, --, +2, -3, etc.)."""
        charge, _ = self._parse_charge_with_query()
        return charge
    
    def _parse_charge_with_query(self) -> tuple[int, bool]:
        """Parse optional charge, returning (charge_value, is_explicit_zero).
        
        Returns:
            Tuple of (charge, is_explicit_zero) where is_explicit_zero is True
            if the pattern was +0 or -0 (SMARTS query for neutral).
        """
        tok = self._tokenizer
        
        char = tok.peek()
        if char not in "+-":
            return 0, False
        
        sign = 1 if char == "+" else -1
        
        # Count consecutive + or -
        count = 0
        while tok.peek() == char:
            tok.next()
            count += 1
        
        # Check for numeric charge after
        num = tok.read_number()
        if num is not None:
            # Explicit +0 or -0 means charge must be exactly 0
            is_explicit_zero = (num == 0)
            return sign * num, is_explicit_zero
        
        return sign * max(1, count), False
    
    def _add_bond_to_previous(self, atom_idx: int, is_aromatic: bool) -> None:
        """Add bond from previous atom to new atom."""
        if self._state.prev_atom is None:
            return
        
        prev_aromatic = self._mol.atoms[self._state.prev_atom].is_aromatic
        
        # Capture bond query state before resetting
        is_ring_bond = self._state.pending_bond_ring
        is_not_ring_bond = self._state.pending_bond_not_ring
        
        # Handle any bond (~) for SMARTS
        if self._state.pending_bond_any:
            self._mol.add_bond(
                self._state.prev_atom,
                atom_idx,
                order=0,  # ANY bond order
                is_aromatic=False,
                is_any=True,
                is_ring_bond=is_ring_bond,
                is_not_ring_bond=is_not_ring_bond,
            )
            self._reset_bond_state()
            return
        
        # Handle dative bonds (-> or <-)
        if self._state.pending_bond_dative:
            self._mol.add_bond(
                self._state.prev_atom,
                atom_idx,
                order=6,  # DATIVE bond order
                is_aromatic=False,
                is_dative=True,
                dative_direction=self._state.pending_bond_dative_dir,
                is_ring_bond=is_ring_bond,
                is_not_ring_bond=is_not_ring_bond,
            )
            self._reset_bond_state()
            return
        
        # Determine bond properties
        if self._state.pending_bond_order is not None or self._state.pending_bond_aromatic is not None:
            order = self._state.pending_bond_order or 1
            aromatic = bool(self._state.pending_bond_aromatic)
        else:
            # Infer aromatic bond between aromatic atoms
            aromatic = prev_aromatic and is_aromatic
            order = 1
        
        self._mol.add_bond(
            self._state.prev_atom,
            atom_idx,
            order=1 if aromatic else order,
            is_aromatic=aromatic,
            stereo=self._state.pending_bond_stereo,
            is_ring_bond=is_ring_bond,
            is_not_ring_bond=is_not_ring_bond,
        )
        
        self._reset_bond_state()
    
    def _reset_bond_state(self) -> None:
        """Reset all pending bond state."""
        self._state.pending_bond_order = None
        self._state.pending_bond_aromatic = None
        self._state.pending_bond_stereo = None
        self._state.pending_bond_dative = False
        self._state.pending_bond_dative_dir = None
        self._state.pending_bond_any = False
        self._state.pending_bond_ring = None
        self._state.pending_bond_not_ring = False


def parse(smiles: str) -> Molecule:
    """Parse a SMILES string into a Molecule.
    
    This is a convenience function that creates a SmilesParser and
    calls parse().
    
    Args:
        smiles: SMILES string to parse.
    
    Returns:
        Parsed Molecule object.
    
    Raises:
        ParseError: If SMILES syntax is invalid.
    
    Example:
        >>> mol = parse("CCO")
        >>> len(mol.atoms)
        3
    """
    return SmilesParser(smiles).parse()
