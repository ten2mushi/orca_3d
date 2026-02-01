"""
Comprehensive Test Suite for a_star.py

This module tests the A* path planning algorithm implementation.
Following the Yoneda philosophy, these tests serve as a complete behavioral
specification for the A* search algorithm.

Key Function Tested:
- a_star_search: Finds shortest path in visibility graph using A* algorithm

Mathematical Specification:
- g(v): Minimum path cost from start to v
- h(v): Heuristic estimate (Euclidean distance to goal) - must be admissible
- f(v) = g(v) + h(v): Priority function (lower is better)
- Algorithm guarantees optimal path when h(v) is admissible and consistent

Properties to Verify:
- Optimality: Returns shortest path when one exists
- Completeness: Finds path if one exists
- Correctness: Path connects start to goal
- Admissibility: Uses Euclidean heuristic which is admissible
"""

import sys
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from a_star import a_star_search
from utils import euclidean_distance_3d


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_path_length(path):
    """Compute total length of a path through waypoints."""
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += euclidean_distance_3d(path[i], path[i + 1])
    return total


def create_simple_graph():
    """
    Create a simple graph for testing:

    Vertices:
    0: (0,0,0) - Start
    1: (1,0,0)
    2: (2,0,0)
    3: (1,1,0)
    4: (2,2,0) - Goal

    Graph structure:
    0 -- 1 -- 2
         |
         3 -- 4
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [2.0, 0.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [2.0, 2.0, 0.0],  # 4
    ])

    # Build edges with costs
    edges = {
        0: [(1, 1.0)],
        1: [(0, 1.0), (2, 1.0), (3, 1.0)],
        2: [(1, 1.0)],
        3: [(1, 1.0), (4, np.sqrt(2.0))],
        4: [(3, np.sqrt(2.0))],
    }

    return vertices, edges


def create_grid_graph():
    """
    Create a 3x3 grid graph:

    6 -- 7 -- 8
    |    |    |
    3 -- 4 -- 5
    |    |    |
    0 -- 1 -- 2

    All horizontal/vertical edges have cost 1.0
    """
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],  # Row 0
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],  # Row 1
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0],  # Row 2
    ])

    edges = {i: [] for i in range(9)}

    # Horizontal edges
    for row in range(3):
        for col in range(2):
            i = row * 3 + col
            j = row * 3 + col + 1
            edges[i].append((j, 1.0))
            edges[j].append((i, 1.0))

    # Vertical edges
    for row in range(2):
        for col in range(3):
            i = row * 3 + col
            j = (row + 1) * 3 + col
            edges[i].append((j, 1.0))
            edges[j].append((i, 1.0))

    return vertices, edges


def create_3d_graph():
    """
    Create a 3D graph (2x2x2 cube vertices):

    Vertices at corners of unit cube plus center
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [0.0, 1.0, 1.0],  # 6
        [1.0, 1.0, 1.0],  # 7
    ])

    edges = {i: [] for i in range(8)}

    # Add edges for cube (each vertex connected to 3 neighbors)
    cube_edges = [
        (0, 1), (0, 2), (0, 4),  # From 0
        (1, 3), (1, 5),          # From 1
        (2, 3), (2, 6),          # From 2
        (3, 7),                  # From 3
        (4, 5), (4, 6),          # From 4
        (5, 7),                  # From 5
        (6, 7),                  # From 6
    ]

    for i, j in cube_edges:
        cost = euclidean_distance_3d(vertices[i], vertices[j])
        edges[i].append((j, cost))
        edges[j].append((i, cost))

    return vertices, edges


# =============================================================================
# BASIC CORRECTNESS TESTS
# =============================================================================

class TestAStarBasicCorrectness:
    """Tests for basic A* correctness."""

    def test_find_path_in_simple_graph(self):
        """A* should find a path in a simple connected graph."""
        vertices, edges = create_simple_graph()
        path, indices = a_star_search(vertices, edges, 0, 4)

        assert path is not None, "Path should be found"
        assert indices is not None, "Indices should be returned"
        assert len(path) >= 2, "Path should have at least start and goal"
        assert indices[0] == 0, "Path should start at start_idx"
        assert indices[-1] == 4, "Path should end at goal_idx"

    def test_path_to_same_node(self):
        """Path from node to itself should be just that node.

        BUG DETECTED: A* returns None when start_idx == goal_idx.
        The algorithm does not handle the trivial case where start equals goal.

        Expected: path = [start_position], indices = [0]
        Actual: path = None, indices = None

        Root Cause: The parent of goal_idx is never set because the node
        is immediately found and we break before processing.
        """
        vertices, edges = create_simple_graph()
        path, indices = a_star_search(vertices, edges, 0, 0)

        assert path is not None
        assert len(path) == 1, "Path to self should have length 1"
        assert indices == [0], "Indices should be just start"

    def test_path_to_adjacent_node(self):
        """Path to adjacent node should be direct."""
        vertices, edges = create_simple_graph()
        path, indices = a_star_search(vertices, edges, 0, 1)

        assert path is not None
        assert len(path) == 2, "Path to adjacent node should have 2 waypoints"
        assert indices == [0, 1], "Should go directly from 0 to 1"

    def test_find_path_in_grid_graph(self):
        """A* should find path in grid graph."""
        vertices, edges = create_grid_graph()
        # Path from bottom-left (0) to top-right (8)
        path, indices = a_star_search(vertices, edges, 0, 8)

        assert path is not None
        assert indices[0] == 0
        assert indices[-1] == 8

    def test_find_path_in_3d_graph(self):
        """A* should work in 3D graph."""
        vertices, edges = create_3d_graph()
        # Path from (0,0,0) to (1,1,1) - opposite corners
        path, indices = a_star_search(vertices, edges, 0, 7)

        assert path is not None
        assert indices[0] == 0
        assert indices[-1] == 7


# =============================================================================
# OPTIMALITY TESTS
# =============================================================================

class TestAStarOptimality:
    """Tests for A* optimality guarantee."""

    def test_finds_shortest_path_simple(self):
        """A* should find the shortest path in simple graph."""
        vertices, edges = create_simple_graph()
        path, indices = a_star_search(vertices, edges, 0, 4)

        # Optimal path: 0 -> 1 -> 3 -> 4
        # Length: 1.0 + 1.0 + sqrt(2) = 2 + sqrt(2) ~ 3.414
        path_length = compute_path_length(path)
        expected_length = 2.0 + np.sqrt(2.0)

        assert_allclose(path_length, expected_length, rtol=1e-10,
                       err_msg=f"Path length {path_length} != expected {expected_length}")

    def test_finds_shortest_path_grid(self):
        """A* should find shortest path in grid (Manhattan path)."""
        vertices, edges = create_grid_graph()
        path, indices = a_star_search(vertices, edges, 0, 8)

        # Shortest path from (0,0) to (2,2) in grid = 4 edges
        path_length = compute_path_length(path)
        assert_allclose(path_length, 4.0, rtol=1e-10)

    def test_shortest_path_3d_cube(self):
        """A* should find shortest path along cube edges."""
        vertices, edges = create_3d_graph()
        path, indices = a_star_search(vertices, edges, 0, 7)

        # Shortest path from (0,0,0) to (1,1,1) = 3 edges of length 1
        path_length = compute_path_length(path)
        assert_allclose(path_length, 3.0, rtol=1e-10)

    def test_prefers_shorter_path_over_longer(self):
        """A* should prefer shorter path when multiple paths exist."""
        # Create graph with two paths: short and long
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0: Start
            [1.0, 0.0, 0.0],  # 1: Direct path
            [0.5, 1.0, 0.0],  # 2: Detour waypoint
            [2.0, 0.0, 0.0],  # 3: Goal
        ])

        edges = {
            0: [(1, 1.0), (2, np.sqrt(1.25))],  # To 1: short, to 2: detour
            1: [(0, 1.0), (3, 1.0)],
            2: [(0, np.sqrt(1.25)), (3, np.sqrt(2.25))],
            3: [(1, 1.0), (2, np.sqrt(2.25))],
        }

        path, indices = a_star_search(vertices, edges, 0, 3)

        # Should choose direct path 0 -> 1 -> 3 (length = 2.0)
        # Not detour 0 -> 2 -> 3 (length ~ 2.62)
        path_length = compute_path_length(path)
        assert_allclose(path_length, 2.0, rtol=1e-10)
        assert indices == [0, 1, 3], f"Expected direct path [0,1,3], got {indices}"


# =============================================================================
# COMPLETENESS TESTS
# =============================================================================

class TestAStarCompleteness:
    """Tests for A* completeness (finds path if one exists)."""

    def test_no_path_returns_none(self):
        """A* should return None when no path exists."""
        # Disconnected graph
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [10.0, 0.0, 0.0], # 2 - Disconnected
        ])

        edges = {
            0: [(1, 1.0)],
            1: [(0, 1.0)],
            2: [],  # No connections
        }

        path, indices = a_star_search(vertices, edges, 0, 2)

        assert path is None, "Should return None for unreachable goal"
        assert indices is None, "Indices should also be None"

    def test_single_node_graph(self):
        """A* should work with single node graph.

        BUG DETECTED: Same issue as test_path_to_same_node.
        When start == goal, A* returns None instead of single-node path.

        Expected: path with single node
        Actual: None
        """
        vertices = np.array([[0.0, 0.0, 0.0]])
        edges = {0: []}

        path, indices = a_star_search(vertices, edges, 0, 0)

        assert path is not None
        assert len(path) == 1, "Path to self should have length 1"
        assert indices == [0], "Indices should be just start"

    def test_two_node_disconnected(self):
        """A* should handle two disconnected nodes."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        edges = {0: [], 1: []}

        path, indices = a_star_search(vertices, edges, 0, 1)

        assert path is None
        assert indices is None

    def test_finds_path_long_chain(self):
        """A* should find path in long chain graph."""
        n = 100
        vertices = np.array([[float(i), 0.0, 0.0] for i in range(n)])
        edges = {i: [] for i in range(n)}

        for i in range(n - 1):
            edges[i].append((i + 1, 1.0))
            edges[i + 1].append((i, 1.0))

        path, indices = a_star_search(vertices, edges, 0, n - 1)

        assert path is not None
        assert len(path) == n
        assert indices[0] == 0
        assert indices[-1] == n - 1


# =============================================================================
# PATH VALIDITY TESTS
# =============================================================================

class TestAStarPathValidity:
    """Tests that returned paths are valid."""

    def test_path_is_continuous(self):
        """Consecutive waypoints should be connected by edges."""
        vertices, edges = create_grid_graph()
        path, indices = a_star_search(vertices, edges, 0, 8)

        for i in range(len(indices) - 1):
            current = indices[i]
            next_node = indices[i + 1]
            neighbors = [n for n, _ in edges[current]]
            assert next_node in neighbors, f"Node {next_node} not connected to {current}"

    def test_path_waypoints_match_indices(self):
        """Path positions should match vertex positions at indices."""
        vertices, edges = create_simple_graph()
        path, indices = a_star_search(vertices, edges, 0, 4)

        for i, idx in enumerate(indices):
            assert_array_almost_equal(path[i], vertices[idx])

    def test_path_starts_at_start_position(self):
        """Path should start at the start vertex position."""
        vertices, edges = create_simple_graph()
        path, indices = a_star_search(vertices, edges, 0, 4)

        assert_array_almost_equal(path[0], vertices[0])

    def test_path_ends_at_goal_position(self):
        """Path should end at the goal vertex position."""
        vertices, edges = create_simple_graph()
        path, indices = a_star_search(vertices, edges, 0, 4)

        assert_array_almost_equal(path[-1], vertices[4])


# =============================================================================
# HEURISTIC TESTS
# =============================================================================

class TestAStarHeuristic:
    """Tests related to the Euclidean distance heuristic."""

    def test_heuristic_is_admissible(self):
        """Euclidean distance heuristic should never overestimate."""
        # This is a property of Euclidean distance - always less than actual path
        vertices, edges = create_simple_graph()

        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if i != j:
                    # Euclidean distance is always <= actual shortest path
                    h = euclidean_distance_3d(vertices[i], vertices[j])

                    # Find actual shortest path
                    path, _ = a_star_search(vertices, edges, i, j)
                    if path is not None:
                        actual = compute_path_length(path)
                        assert h <= actual + 1e-10, \
                            f"Heuristic {h} > actual {actual} for {i}->{j}"

    def test_heuristic_zero_at_goal(self):
        """Heuristic should be zero at the goal."""
        vertices, edges = create_simple_graph()
        goal_idx = 4

        h = euclidean_distance_3d(vertices[goal_idx], vertices[goal_idx])
        assert h == 0.0


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================

class TestAStarEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_edges_dict(self):
        """Graph with no edges should only find path to self.

        BUG DETECTED: Due to start == goal bug, path to self fails.
        """
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        edges = {0: [], 1: []}

        # Path to self should work
        path, indices = a_star_search(vertices, edges, 0, 0)
        assert path is not None
        assert len(path) == 1
        assert indices == [0]

        # Path to other fails (correctly - no edge exists)
        path, indices = a_star_search(vertices, edges, 0, 1)
        assert path is None

    def test_large_coordinate_values(self):
        """A* should handle large coordinate values."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1e6, 0.0, 0.0],
            [1e6, 1e6, 0.0],
        ])
        edges = {
            0: [(1, 1e6)],
            1: [(0, 1e6), (2, 1e6)],
            2: [(1, 1e6)],
        }

        path, indices = a_star_search(vertices, edges, 0, 2)
        assert path is not None
        assert indices == [0, 1, 2]

    def test_negative_coordinates(self):
        """A* should handle negative coordinates."""
        vertices = np.array([
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        dist01 = euclidean_distance_3d(vertices[0], vertices[1])
        dist12 = euclidean_distance_3d(vertices[1], vertices[2])

        edges = {
            0: [(1, dist01)],
            1: [(0, dist01), (2, dist12)],
            2: [(1, dist12)],
        }

        path, indices = a_star_search(vertices, edges, 0, 2)
        assert path is not None
        assert indices == [0, 1, 2]

    def test_multiple_equal_cost_paths(self):
        """A* should handle multiple paths of equal cost."""
        # Diamond graph: two paths of same length
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0: Start
            [1.0, 1.0, 0.0],  # 1: Top path
            [1.0, -1.0, 0.0], # 2: Bottom path
            [2.0, 0.0, 0.0],  # 3: Goal
        ])

        dist_side = np.sqrt(2.0)
        edges = {
            0: [(1, dist_side), (2, dist_side)],
            1: [(0, dist_side), (3, dist_side)],
            2: [(0, dist_side), (3, dist_side)],
            3: [(1, dist_side), (2, dist_side)],
        }

        path, indices = a_star_search(vertices, edges, 0, 3)

        # Should find one of the optimal paths
        assert path is not None
        path_length = compute_path_length(path)
        expected_length = 2 * np.sqrt(2.0)
        assert_allclose(path_length, expected_length, rtol=1e-10)


# =============================================================================
# NUMERICAL PRECISION TESTS
# =============================================================================

class TestAStarNumericalPrecision:
    """Tests for numerical precision."""

    def test_very_small_edge_costs(self):
        """A* should handle very small edge costs."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1e-10, 0.0, 0.0],
            [2e-10, 0.0, 0.0],
        ])

        edges = {
            0: [(1, 1e-10)],
            1: [(0, 1e-10), (2, 1e-10)],
            2: [(1, 1e-10)],
        }

        path, indices = a_star_search(vertices, edges, 0, 2)
        assert path is not None
        assert indices == [0, 1, 2]

    def test_mixed_scale_costs(self):
        """A* should handle mix of large and small costs."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1e6, 0.0, 0.0],
        ])

        edges = {
            0: [(1, 1.0), (2, 1e6)],
            1: [(0, 1.0), (2, 1e6 - 1)],
            2: [(0, 1e6), (1, 1e6 - 1)],
        }

        path, indices = a_star_search(vertices, edges, 0, 2)

        # Should prefer 0 -> 1 -> 2 (cost = 1 + 1e6-1 = 1e6) over 0 -> 2 (cost = 1e6)
        # Actually these are equal, but let's check path is found
        assert path is not None


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestAStarPerformance:
    """Performance and stress tests."""

    def test_large_graph_path_finding(self):
        """A* should handle larger graphs efficiently."""
        # Create a 10x10x10 grid (1000 nodes)
        n = 10
        vertices = []
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    vertices.append([float(x), float(y), float(z)])
        vertices = np.array(vertices)

        edges = {i: [] for i in range(len(vertices))}

        def idx(x, y, z):
            return x * n * n + y * n + z

        # Connect neighbors
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    i = idx(x, y, z)
                    if x + 1 < n:
                        j = idx(x + 1, y, z)
                        edges[i].append((j, 1.0))
                        edges[j].append((i, 1.0))
                    if y + 1 < n:
                        j = idx(x, y + 1, z)
                        edges[i].append((j, 1.0))
                        edges[j].append((i, 1.0))
                    if z + 1 < n:
                        j = idx(x, y, z + 1)
                        edges[i].append((j, 1.0))
                        edges[j].append((i, 1.0))

        # Find path from corner to opposite corner
        start = idx(0, 0, 0)
        goal = idx(n - 1, n - 1, n - 1)

        path, indices = a_star_search(vertices, edges, start, goal)

        assert path is not None
        # Optimal path length is 3*(n-1) = 27
        path_length = compute_path_length(path)
        assert_allclose(path_length, 3 * (n - 1), rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
