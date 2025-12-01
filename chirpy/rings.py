"""
Ring detection algorithms.

This module provides functionality for finding rings (cycles) in molecular
structures, including the Smallest Set of Smallest Rings (SSSR).

These algorithms are used by both aromaticity perception and SMARTS
substructure matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Molecule


def find_sssr(mol: "Molecule", max_ring_size: int = 20) -> list[set[int]]:
    """Find Smallest Set of Smallest Rings (SSSR).
    
    Uses a DFS-based approach to find all simple cycles in the molecule.
    
    Args:
        mol: Molecule to analyze.
        max_ring_size: Maximum ring size to consider (default 20).
    
    Returns:
        List of rings, each as a set of atom indices.
    
    Example:
        >>> mol = parse("c1ccccc1")  # benzene
        >>> rings = find_sssr(mol)
        >>> len(rings)
        1
        >>> len(rings[0])
        6
    """
    n = len(mol.atoms)
    if n == 0:
        return []
    
    # Build adjacency list
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for bond in mol.bonds:
        adj[bond.atom1_idx].add(bond.atom2_idx)
        adj[bond.atom2_idx].add(bond.atom1_idx)
    
    # Find all cycles using DFS
    all_rings: list[tuple[int, ...]] = []
    
    def dfs_find_cycles(
        start: int,
        current: int,
        path: list[int],
        visited: set[int],
    ) -> None:
        """DFS to find cycles starting from start node."""
        if len(path) > max_ring_size + 1:
            return
        
        for neighbor in adj[current]:
            if neighbor == start and len(path) >= 3:
                # Found a cycle
                all_rings.append(tuple(path))
            elif neighbor not in visited and neighbor > start:
                # Continue search (only visit higher-indexed atoms to avoid duplicates)
                visited.add(neighbor)
                path.append(neighbor)
                dfs_find_cycles(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
    
    for start in range(n):
        dfs_find_cycles(start, start, [start], {start})
    
    # Remove duplicates and keep smallest unique rings
    unique_rings = _filter_unique_rings(all_rings)
    
    # Filter by size
    return [r for r in unique_rings if len(r) <= max_ring_size]


def _filter_unique_rings(rings: list[tuple[int, ...]]) -> list[set[int]]:
    """Filter to unique rings, preferring smaller ones.
    
    Args:
        rings: List of rings as tuples of atom indices.
    
    Returns:
        List of unique rings as sets.
    """
    if not rings:
        return []
    
    # Convert to canonical form (frozenset for comparison)
    seen: set[frozenset[int]] = set()
    unique: list[set[int]] = []
    
    # Sort by size to prefer smaller rings
    for ring in sorted(rings, key=len):
        ring_set = frozenset(ring)
        if ring_set not in seen:
            seen.add(ring_set)
            unique.append(set(ring))
    
    return unique


def get_ring_info(mol: "Molecule") -> tuple[dict[int, int], dict[int, set[int]]]:
    """Get ring membership and sizes for each atom.
    
    This is useful for SMARTS queries like [R] (in ring), [R2] (in 2 rings),
    [r5] (in 5-membered ring), etc.
    
    Uses a BFS-based algorithm to find all simple cycles.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        A tuple of (ring_count, ring_sizes) where:
        - ring_count: dict mapping atom index to number of rings it's in
        - ring_sizes: dict mapping atom index to set of ring sizes it's in
    
    Example:
        >>> mol = parse("c1ccc2ccccc2c1")  # naphthalene
        >>> ring_count, ring_sizes = get_ring_info(mol)
        >>> ring_count[4]  # bridgehead carbon
        2
        >>> ring_sizes[4]
        {6}
    """
    ring_count: dict[int, int] = {i: 0 for i in range(mol.num_atoms)}
    ring_sizes: dict[int, set[int]] = {i: set() for i in range(mol.num_atoms)}
    
    if mol.num_atoms == 0:
        return ring_count, ring_sizes
    
    # Build adjacency list
    adj: dict[int, set[int]] = {i: set() for i in range(mol.num_atoms)}
    for bond in mol.bonds:
        adj[bond.atom1_idx].add(bond.atom2_idx)
        adj[bond.atom2_idx].add(bond.atom1_idx)
    
    # Find all simple cycles using BFS from each edge
    # For each edge (start, first_nbr), find shortest path back to start
    found_rings: set[frozenset[int]] = set()
    
    for start in range(mol.num_atoms):
        for first_nbr in adj[start]:
            # Find shortest path from first_nbr back to start
            # without using the direct start-first_nbr edge
            visited: dict[int, int | None] = {first_nbr: None}
            queue: list[int] = [first_nbr]
            path_found = False
            
            while queue and not path_found:
                current = queue.pop(0)
                
                for nbr in adj[current]:
                    if nbr == start:
                        # Check we're not just going back on the original edge
                        if current != first_nbr:
                            # Found a cycle - reconstruct it
                            cycle: list[int] = [start]
                            node: int | None = current
                            while node is not None:
                                cycle.append(node)
                                node = visited[node]
                            
                            ring = frozenset(cycle)
                            if len(ring) >= 3:  # Valid ring
                                found_rings.add(ring)
                            path_found = True
                            break
                    
                    elif nbr not in visited:
                        visited[nbr] = current
                        queue.append(nbr)
    
    # Update ring counts and sizes
    for ring in found_rings:
        size = len(ring)
        for atom_idx in ring:
            ring_count[atom_idx] += 1
            ring_sizes[atom_idx].add(size)
    
    return ring_count, ring_sizes


def get_ring_bonds(mol: "Molecule") -> set[tuple[int, int]]:
    """Get all bonds that are part of a ring.
    
    A bond is a ring bond if both atoms are in the same ring.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        Set of (atom1_idx, atom2_idx) tuples for ring bonds,
        where atom1_idx < atom2_idx.
    
    Example:
        >>> mol = parse("c1ccccc1CC")  # toluene
        >>> ring_bonds = get_ring_bonds(mol)
        >>> len(ring_bonds)  # 6 bonds in benzene ring
        6
    """
    ring_bonds: set[tuple[int, int]] = set()
    rings = find_sssr(mol)
    
    for ring in rings:
        ring_list = list(ring)
        ring_set = set(ring_list)
        
        # Find all bonds where both atoms are in this ring
        for atom_idx in ring_set:
            atom = mol.atoms[atom_idx]
            for bond_idx in atom.bond_indices:
                bond = mol.bonds[bond_idx]
                other = bond.other_atom(atom_idx)
                if other in ring_set:
                    key = (min(atom_idx, other), max(atom_idx, other))
                    ring_bonds.add(key)
    
    return ring_bonds


def find_ring_systems(rings: list[set[int]]) -> list[list[set[int]]]:
    """Group rings into fused ring systems.
    
    Two rings are considered fused if they share at least 2 atoms (a bond).
    
    Args:
        rings: List of rings as atom index sets.
    
    Returns:
        List of ring systems, each containing fused rings.
    
    Example:
        >>> mol = parse("c1ccc2ccccc2c1")  # naphthalene
        >>> rings = find_sssr(mol)
        >>> systems = find_ring_systems(rings)
        >>> len(systems)  # naphthalene has one fused system
        1
        >>> len(systems[0])  # containing 2 rings
        2
    """
    if not rings:
        return []
    
    n = len(rings)
    parent = list(range(n))
    
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union rings that share at least 2 atoms (a bond)
    for i in range(n):
        for j in range(i + 1, n):
            if len(rings[i] & rings[j]) >= 2:
                union(i, j)
    
    # Group by parent
    systems: dict[int, list[set[int]]] = {}
    for i in range(n):
        p = find(i)
        if p not in systems:
            systems[p] = []
        systems[p].append(rings[i])
    
    return list(systems.values())
