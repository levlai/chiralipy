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
    
    The SSSR is a linearly independent basis of cycles where:
    - The number of rings equals the cyclomatic complexity (E - V + C)
    - Larger rings that can be expressed as combinations of smaller rings are excluded
    
    For example, naphthalene has cyclomatic complexity 2 (11 bonds - 10 atoms + 1),
    so its SSSR contains exactly 2 rings (the two 6-membered rings), not the
    10-membered envelope ring.
    
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
    
    # Remove duplicates
    unique_rings = _filter_unique_rings(all_rings)
    
    # Compute proper SSSR (linearly independent set)
    sssr = _compute_sssr(unique_rings, n, len(mol.bonds), adj)
    
    # Filter by size if limit is set
    if max_ring_size is not None:
        return [r for r in sssr if len(r) <= max_ring_size]
    return sssr


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


def _get_ring_bonds_from_atoms(ring: set[int], adj: dict[int, set[int]]) -> set[frozenset[int]]:
    """Get the bonds (as frozensets of 2 atoms) that form a ring.
    
    Args:
        ring: Set of atom indices in the ring.
        adj: Adjacency list for the molecule.
    
    Returns:
        Set of frozensets, each containing 2 atom indices representing a bond.
    """
    bonds: set[frozenset[int]] = set()
    for atom in ring:
        for neighbor in adj[atom]:
            if neighbor in ring:
                bonds.add(frozenset([atom, neighbor]))
    return bonds


def _compute_sssr(
    all_rings: list[set[int]], 
    num_atoms: int, 
    num_bonds: int,
    adj: dict[int, set[int]],
) -> list[set[int]]:
    """Compute the Smallest Set of Smallest Rings (SSSR).
    
    The SSSR is a linearly independent set of rings where:
    - The number of rings equals the cyclomatic complexity
    - Each ring cannot be expressed as a combination of other rings in the set
    
    We use a greedy algorithm that selects rings by size, checking linear
    independence using the bond-based representation (each ring is a vector
    of bonds, and rings are independent if their bond sets form a linearly
    independent set over GF(2) - the field with XOR as addition).
    
    Args:
        all_rings: All unique rings found in the molecule.
        num_atoms: Number of atoms in the molecule.
        num_bonds: Number of bonds in the molecule.
        adj: Adjacency list mapping atom index to neighbor set.
    
    Returns:
        The SSSR as a list of atom index sets.
    """
    if not all_rings:
        return []
    
    # Calculate cyclomatic complexity (number of independent cycles)
    # For a connected graph: mu = E - V + 1
    # For multiple components: mu = E - V + C
    # We'll compute the actual number of components
    
    # Find connected components
    visited: set[int] = set()
    num_components = 0
    
    def dfs_component(start: int) -> None:
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj[node] - visited)
    
    for atom in range(num_atoms):
        if atom not in visited and atom in adj:
            dfs_component(atom)
            num_components += 1
    
    # Cyclomatic complexity
    mu = num_bonds - num_atoms + num_components
    
    if mu <= 0:
        return []
    
    # Sort rings by size (prefer smaller rings)
    sorted_rings = sorted(all_rings, key=len)
    
    # Convert rings to bond sets for linear independence checking
    ring_bond_sets: list[tuple[set[int], set[frozenset[int]]]] = []
    for ring in sorted_rings:
        bonds = _get_ring_bonds_from_atoms(ring, adj)
        ring_bond_sets.append((ring, bonds))
    
    # Greedily select linearly independent rings
    # Using Gaussian elimination over GF(2) (XOR basis)
    sssr: list[set[int]] = []
    # Basis vectors represented as sets of bonds (XOR = symmetric difference)
    basis: list[set[frozenset[int]]] = []
    
    for ring, bonds in ring_bond_sets:
        if len(sssr) >= mu:
            break
        
        # Try to reduce this ring's bond set using existing basis
        reduced = set(bonds)
        for basis_vec in basis:
            # Check if XOR with this basis vector reduces the size
            xor_result = reduced.symmetric_difference(basis_vec)
            if len(xor_result) < len(reduced):
                reduced = xor_result
        
        # If reduced to non-empty, this ring is linearly independent
        if reduced:
            sssr.append(ring)
            basis.append(set(bonds))
    
    return sssr


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
    _use_sssr: bool = True,
) -> tuple[dict[int, int], dict[int, set[int]]]:
    """Get ring membership and sizes for each atom.
    
    This is useful for SMARTS queries like [R] (in ring), [R2] (in 2 rings),
    [r5] (in 5-membered ring), etc.
    
    Uses the SSSR (Smallest Set of Smallest Rings) for accurate counts when
    _use_sssr is True, otherwise uses fast BFS-based detection.
    
    Args:
        mol: Molecule to analyze.
        _ring_atoms: Optional precomputed ring atoms (internal optimization).
        _use_sssr: If True, use SSSR for accurate counts. If False, use
            fast approximation (may overcount for fused rings).
    
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
    
    if _use_sssr:
        # Use SSSR for accurate ring count/size information
        # This is more expensive but gives correct counts
        sssr = find_sssr(mol)
        
        # Update ring counts and sizes from SSSR
        for ring in sssr:
            size = len(ring)
            for atom_idx in ring:
                ring_count[atom_idx] += 1
                ring_sizes[atom_idx].add(size)
    else:
        # Fast approximation: just mark ring membership
        # This doesn't give accurate counts for fused rings
        for atom_idx in ring_atoms:
            ring_count[atom_idx] = 1
    
    return ring_count, ring_sizes


def get_ring_info_fast(mol: "Molecule") -> tuple[dict[int, int], dict[int, set[int]]]:
    """Fast ring detection without accurate counts.
    
    Returns ring_count with 1 for atoms in any ring, 0 otherwise.
    ring_sizes will be empty (no size information).
    
    Use this for SMARTS queries that only check [R] (in ring) or [R0] (not in ring),
    but don't need [R2] or [r5] type queries.
    
    Args:
        mol: Molecule to analyze.
    
    Returns:
        Tuple of (ring_count, ring_sizes) - ring_sizes will be empty sets.
    """
    ring_atoms, _ = _find_ring_atoms_and_bonds_fast(mol)
    ring_count = {i: (1 if i in ring_atoms else 0) for i in range(mol.num_atoms)}
    ring_sizes: dict[int, set[int]] = {i: set() for i in range(mol.num_atoms)}
    return ring_count, ring_sizes


def get_min_ring_sizes(mol: "Molecule", ring_atoms: set[int] | None = None) -> dict[int, int]:
    """Get minimum ring size for each atom efficiently using BFS.
    
    For each ring atom, finds the smallest ring it participates in using
    BFS from each atom. This is O(V * E) in the worst case but typically
    much faster for drug-like molecules.
    
    Args:
        mol: Molecule to analyze.
        ring_atoms: Optional precomputed set of ring atom indices.
    
    Returns:
        Dict mapping atom index to minimum ring size (0 if not in any ring).
    """
    n = mol.num_atoms
    if n == 0:
        return {}
    
    if ring_atoms is None:
        ring_atoms, _ = _find_ring_atoms_and_bonds_fast(mol)
    
    min_ring_size: dict[int, int] = {i: 0 for i in range(n)}
    
    if not ring_atoms:
        return min_ring_size
    
    # Build adjacency list
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for bond in mol.bonds:
        adj[bond.atom1_idx].append(bond.atom2_idx)
        adj[bond.atom2_idx].append(bond.atom1_idx)
    
    # For each ring atom, find shortest cycle through it using BFS
    for start in ring_atoms:
        # BFS to find shortest path back to start
        # dist[i] = distance from start to i (excluding direct edge)
        dist: dict[int, int] = {start: 0}
        parent: dict[int, int] = {start: -1}
        queue: list[int] = [start]
        min_cycle = float('inf')
        
        while queue:
            curr = queue.pop(0)
            curr_dist = dist[curr]
            
            # Early termination if we already found a small cycle
            if curr_dist >= min_cycle // 2:
                break
            
            for nbr in adj[curr]:
                if nbr not in dist:
                    dist[nbr] = curr_dist + 1
                    parent[nbr] = curr
                    queue.append(nbr)
                elif nbr != parent.get(curr, -1):
                    # Found a cycle - check if it goes through start
                    # The cycle length is dist[curr] + dist[nbr] + 1
                    cycle_len = dist[curr] + dist[nbr] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
        
        if min_cycle < float('inf'):
            min_ring_size[start] = int(min_cycle)
    
    return min_ring_size


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
