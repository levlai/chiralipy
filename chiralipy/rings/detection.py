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
    from chiralipy.types import Molecule


def _find_ring_atoms_and_bonds_fast(mol: "Molecule") -> tuple[set[int], set[tuple[int, int]]]:
    """Fast detection of ring atoms and bonds using DFS cycle detection.
    
    This is O(V+E) and doesn't enumerate all rings, just identifies which
    atoms and bonds are part of ANY cycle.
    
    Returns:
        Tuple of (ring_atoms, ring_bonds) where ring_bonds are (min_idx, max_idx) tuples.
    """
    if mol.num_atoms == 0:
        return set(), set()
    
    # Build adjacency list with bond indices
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(mol.num_atoms)}
    for bond in mol.bonds:
        adj[bond.atom1_idx].append((bond.atom2_idx, bond.idx))
        adj[bond.atom2_idx].append((bond.atom1_idx, bond.idx))
    
    # DFS to find back edges (which indicate cycles)
    visited: set[int] = set()
    in_stack: set[int] = set()
    parent: dict[int, int] = {}
    parent_bond: dict[int, int] = {}
    back_edge_bonds: set[int] = set()  # Bond indices that are back edges
    
    def dfs(node: int) -> None:
        visited.add(node)
        in_stack.add(node)
        
        for neighbor, bond_idx in adj[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                parent_bond[neighbor] = bond_idx
                dfs(neighbor)
            elif neighbor in in_stack and parent.get(node) != neighbor:
                # Back edge found - this bond is part of a cycle
                back_edge_bonds.add(bond_idx)
        
        in_stack.remove(node)
    
    # Run DFS from each unvisited node
    for start in range(mol.num_atoms):
        if start not in visited:
            parent[start] = -1
            dfs(start)
    
    if not back_edge_bonds:
        return set(), set()
    
    # Now find all atoms and bonds that are part of cycles
    # A bond is a ring bond if removing it increases the number of connected components
    # But that's expensive. Instead, use the fact that ring bonds are those where 
    # both atoms can reach each other via another path.
    
    # Simpler approach: find all bonds in the "2-edge-connected" components
    # Using a different algorithm based on low-link values (Tarjan's bridge-finding)
    
    ring_bonds: set[tuple[int, int]] = set()
    ring_atoms: set[int] = set()
    
    # Find bridges using Tarjan's algorithm
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: set[int] = set()
    time_counter = [0]
    
    def find_bridges(node: int, parent_node: int, parent_bond_idx: int) -> None:
        discovery[node] = low[node] = time_counter[0]
        time_counter[0] += 1
        
        for neighbor, bond_idx in adj[node]:
            if neighbor not in discovery:
                find_bridges(neighbor, node, bond_idx)
                low[node] = min(low[node], low[neighbor])
                
                # If low[neighbor] > discovery[node], this is a bridge
                if low[neighbor] > discovery[node]:
                    bridges.add(bond_idx)
            elif bond_idx != parent_bond_idx:
                low[node] = min(low[node], discovery[neighbor])
    
    discovery.clear()
    for start in range(mol.num_atoms):
        if start not in discovery:
            find_bridges(start, -1, -1)
    
    # All bonds that are NOT bridges are ring bonds
    for bond in mol.bonds:
        if bond.idx not in bridges:
            key = (min(bond.atom1_idx, bond.atom2_idx), max(bond.atom1_idx, bond.atom2_idx))
            ring_bonds.add(key)
            ring_atoms.add(bond.atom1_idx)
            ring_atoms.add(bond.atom2_idx)
    
    return ring_atoms, ring_bonds


def find_sssr(mol: "Molecule", max_ring_size: int | None = None) -> list[set[int]]:
    """Find Smallest Set of Smallest Rings (SSSR).
    
    Uses a DFS-based approach to find all simple cycles in the molecule.
    
    Args:
        mol: Molecule to analyze.
        max_ring_size: Maximum ring size to consider (default None = no limit).
    
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
        if max_ring_size is not None and len(path) > max_ring_size + 1:
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
    
    # Filter by size if limit is set
    if max_ring_size is not None:
        return [r for r in unique_rings if len(r) <= max_ring_size]
    return unique_rings


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


def get_ring_membership(mol: "Molecule") -> dict[int, int]:
    """Fast ring membership detection (is atom in any ring?).
    
    Uses Tarjan's bridge algorithm for O(V+E) performance.
    For each atom, returns 1 if in a ring, 0 otherwise.
    
    Note: This doesn't count HOW MANY rings an atom is in, just whether
    it's in at least one. For accurate ring counts, use get_ring_info().
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        Dict mapping atom index to 1 (in ring) or 0 (not in ring).
    """
    ring_atoms, _ = _find_ring_atoms_and_bonds_fast(mol)
    return {i: (1 if i in ring_atoms else 0) for i in range(mol.num_atoms)}


def get_ring_info(
    mol: "Molecule",
    _ring_atoms: set[int] | None = None,
) -> tuple[dict[int, int], dict[int, set[int]]]:
    """Get ring membership and sizes for each atom.
    
    This is useful for SMARTS queries like [R] (in ring), [R2] (in 2 rings),
    [r5] (in 5-membered ring), etc.
    
    Uses a BFS-based algorithm to find all simple cycles.
    
    Args:
        mol: Molecule to analyze.
        _ring_atoms: Optional precomputed ring atoms (internal optimization).
    
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
    
    # Use precomputed ring atoms or compute them
    if _ring_atoms is None:
        ring_atoms, _ = _find_ring_atoms_and_bonds_fast(mol)
    else:
        ring_atoms = _ring_atoms
    
    if not ring_atoms:
        return ring_count, ring_sizes
    
    # Build adjacency list
    adj: dict[int, set[int]] = {i: set() for i in range(mol.num_atoms)}
    for bond in mol.bonds:
        adj[bond.atom1_idx].add(bond.atom2_idx)
        adj[bond.atom2_idx].add(bond.atom1_idx)
    
    # Find all simple cycles using BFS from each edge
    # For each edge (start, first_nbr), find shortest path back to start
    found_rings: set[frozenset[int]] = set()
    
    # Only start from ring atoms (optimization)
    for start in ring_atoms:
        for first_nbr in adj[start]:
            if first_nbr not in ring_atoms:
                continue  # Skip non-ring neighbors
                
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
    
    Uses Tarjan's bridge-finding algorithm for O(V+E) performance.
    A bond is a ring bond if it's not a bridge (removing it doesn't disconnect the graph).
    
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
    _, ring_bonds = _find_ring_atoms_and_bonds_fast(mol)
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
