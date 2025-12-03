"""Tests for element and periodic table functionality."""

import pytest
from chiralipy.elements import (
    Element,
    ORGANIC_SUBSET,
    AROMATIC_SUBSET,
    get_atomic_number,
    get_default_valence,
    is_organic_symbol,
    is_aromatic_symbol,
)


class TestElement:
    """Test Element class."""

    def test_carbon(self):
        """Carbon element lookup."""
        elem = Element.from_symbol("C")
        assert elem is not None
        assert elem.symbol == "C"
        assert elem.atomic_number == 6

    def test_nitrogen(self):
        """Nitrogen element lookup."""
        elem = Element.from_symbol("N")
        assert elem is not None
        assert elem.symbol == "N"
        assert elem.atomic_number == 7

    def test_oxygen(self):
        """Oxygen element lookup."""
        elem = Element.from_symbol("O")
        assert elem is not None
        assert elem.symbol == "O"
        assert elem.atomic_number == 8

    def test_chlorine(self):
        """Chlorine (two-letter) element lookup."""
        elem = Element.from_symbol("Cl")
        assert elem is not None
        assert elem.symbol == "Cl"
        assert elem.atomic_number == 17

    def test_bromine(self):
        """Bromine element lookup."""
        elem = Element.from_symbol("Br")
        assert elem is not None
        assert elem.symbol == "Br"
        assert elem.atomic_number == 35

    def test_from_atomic_number(self):
        """Lookup element by atomic number."""
        elem = Element.from_atomic_number(6)
        assert elem is not None
        assert elem.symbol == "C"

    def test_invalid_symbol(self):
        """Invalid symbol should return None."""
        elem = Element.from_symbol("Xx")
        assert elem is None

    def test_lowercase_lookup(self):
        """Lowercase aromatic symbol lookup."""
        elem = Element.from_symbol("c")
        assert elem is not None
        assert elem.atomic_number == 6


class TestOrganicSubset:
    """Test organic subset constants."""

    def test_organic_subset_contains_common(self):
        """Organic subset should contain common elements."""
        assert "C" in ORGANIC_SUBSET
        assert "N" in ORGANIC_SUBSET
        assert "O" in ORGANIC_SUBSET
        assert "S" in ORGANIC_SUBSET
        assert "P" in ORGANIC_SUBSET

    def test_organic_subset_contains_halogens(self):
        """Organic subset should contain halogens."""
        assert "F" in ORGANIC_SUBSET
        assert "Cl" in ORGANIC_SUBSET
        assert "Br" in ORGANIC_SUBSET
        assert "I" in ORGANIC_SUBSET

    def test_organic_subset_excludes_metals(self):
        """Organic subset should not contain metals."""
        assert "Fe" not in ORGANIC_SUBSET
        assert "Cu" not in ORGANIC_SUBSET
        assert "Na" not in ORGANIC_SUBSET


class TestAromaticSubset:
    """Test aromatic subset constants."""

    def test_aromatic_subset_lowercase(self):
        """Aromatic subset should be lowercase."""
        for sym in AROMATIC_SUBSET:
            assert sym.islower()

    def test_aromatic_subset_contents(self):
        """Aromatic subset should contain expected elements."""
        assert "c" in AROMATIC_SUBSET
        assert "n" in AROMATIC_SUBSET
        assert "o" in AROMATIC_SUBSET
        assert "s" in AROMATIC_SUBSET


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_atomic_number(self):
        """get_atomic_number() function."""
        assert get_atomic_number("C") == 6
        assert get_atomic_number("N") == 7
        assert get_atomic_number("O") == 8
        assert get_atomic_number("Cl") == 17

    def test_get_atomic_number_invalid(self):
        """get_atomic_number() with invalid symbol."""
        result = get_atomic_number("Xx")
        assert result == 0

    def test_get_default_valence(self):
        """get_default_valence() function."""
        assert get_default_valence(6) == 4  # Carbon
        assert get_default_valence(7) == 3  # Nitrogen
        assert get_default_valence(8) == 2  # Oxygen
        assert get_default_valence(17) == 1  # Chlorine

    def test_is_organic_symbol(self):
        """is_organic_symbol() function."""
        assert is_organic_symbol("C")
        assert is_organic_symbol("N")
        assert is_organic_symbol("O")
        assert not is_organic_symbol("Fe")

    def test_is_aromatic_symbol(self):
        """is_aromatic_symbol() function."""
        assert is_aromatic_symbol("c")
        assert is_aromatic_symbol("n")
        assert not is_aromatic_symbol("C")
