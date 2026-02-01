"""
Comprehensive Test Suite for visibility_graph.py

This module tests the 3D visibility graph construction.
Following the Yoneda philosophy, these tests serve as a complete behavioral
specification for the visibility graph builder.

Key Function Tested:
- build_visibility_graph: Constructs 3D visibility graph from obstacles

Mathematical Specification:
- Vertices V = union of obstacle vertices union {start, goal}
- Edges E: For each pair u,v in V, add edge if [u,v] does not intersect
           the interior of any obstacle
- Cost k(u,v) = ||u - v||_2 (Euclidean distance)

Properties to Verify:
- All obstacle vertices are included
- Start and goal are included
- Edges only exist between mutually visible vertices
- Edge costs are correct Euclidean distances
- Graph is undirected (symmetric edges)
"""

import sys
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visibility_graph import build_visibility_graph
from utils import euclidean_distance_3d


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def unit_cube_obstacle():
    """A unit cube from (0,0,0) to (1,1,1)."""
    return [{
        'vertices': np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top
        ], dtype=float),
        'faces': [
            [0, 1, 2, 3],  # Bottom
            [4, 7, 6, 5],  # Top
            [0, 4, 5, 1],  # Front
            [2, 6, 7, 3],  # Back
            [0, 3, 7, 4],  # Left
            [1, 5, 6, 2]   # Right
        ]
    }]


@pytest.fixture
def offset_cube_obstacle():
    """A cube from (2,2,0) to (3,3,1) - offset from origin."""
    return [{
        'vertices': np.array([
            [2, 2, 0], [3, 2, 0], [3, 3, 0], [2, 3, 0],
            [2, 2, 1], [3, 2, 1], [3, 3, 1], [2, 3, 1]
        ], dtype=float),
        'faces': [
            [0, 1, 2, 3], [4, 7, 6, 5],
            [0, 4, 5, 1], [2, 6, 7, 3],
            [0, 3, 7, 4], [1, 5, 6, 2]
        ]
    }]


@pytest.fixture
def empty_obstacles():
    """No obstacles."""
    return []


@pytest.fixture
def two_cubes_obstacles():
    """Two separated cubes."""
    return [
        {
            'vertices': np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ], dtype=float),
            'faces': [
                [0, 1, 2, 3], [4, 7, 6, 5],
                [0, 4, 5, 1], [2, 6, 7, 3],
                [0, 3, 7, 4], [1, 5, 6, 2]
            ]
        },
        {
            'vertices': np.array([
                [5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0],
                [5, 0, 1], [6, 0, 1], [6, 1, 1], [5, 1, 1]
            ], dtype=float),
            'faces': [
                [0, 1, 2, 3], [4, 7, 6, 5],
                [0, 4, 5, 1], [2, 6, 7, 3],
                [0, 3, 7, 4], [1, 5, 6, 2]
            ]
        }
    ]


# =============================================================================
# VERTEX INCLUSION TESTS
# =============================================================================

class TestVertexInclusion:
    """Tests for correct vertex inclusion in the graph."""

    def test_includes_start_and_goal(self, empty_obstacles):
        """Graph should include start and goal vertices."""
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 5.0, 5.0])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            empty_obstacles, start, goal
        )

        assert len(vertices) >= 2, "Should have at least start and goal"
        assert_array_almost_equal(vertices[start_idx], start)
        assert_array_almost_equal(vertices[goal_idx], goal)

    def test_includes_all_obstacle_vertices(self, unit_cube_obstacle):
        """Graph should include all obstacle vertices."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        # Unit cube has 8 vertices + start + goal = 10
        assert len(vertices) == 10

        # Check all cube vertices are present
        cube_verts = unit_cube_obstacle[0]['vertices']
        for cv in cube_verts:
            found = any(np.allclose(v, cv) for v in vertices)
            assert found, f"Cube vertex {cv} not found in graph"

    def test_includes_vertices_from_multiple_obstacles(self, two_cubes_obstacles):
        """Graph should include vertices from all obstacles."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([7.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            two_cubes_obstacles, start, goal
        )

        # Two cubes with 8 vertices each + start + goal = 18
        assert len(vertices) == 18

    def test_deduplicates_coincident_vertices(self):
        """Should not duplicate vertices at same position."""
        # Two obstacles sharing a vertex
        obstacles = [
            {
                'vertices': np.array([
                    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
                ], dtype=float),
                'faces': [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                         [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]
            },
            {
                'vertices': np.array([
                    [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0],  # Shares (1,0,0) and (1,1,0)
                    [1, 0, 1], [2, 0, 1], [2, 1, 1], [1, 1, 1]   # Shares (1,0,1) and (1,1,1)
                ], dtype=float),
                'faces': [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                         [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]
            }
        ]

        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([3.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            obstacles, start, goal
        )

        # 8 + 8 - 4 shared + 2 (start/goal) = 14
        # (4 vertices are shared between the cubes)
        assert len(vertices) == 14

    def test_start_at_obstacle_vertex(self, unit_cube_obstacle):
        """Start at an obstacle vertex should not duplicate."""
        start = np.array([0.0, 0.0, 0.0])  # Corner of cube
        goal = np.array([5.0, 5.0, 5.0])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        # 8 cube vertices + 1 goal = 9 (start is at cube vertex)
        assert len(vertices) == 9

    def test_goal_at_obstacle_vertex(self, unit_cube_obstacle):
        """Goal at an obstacle vertex should not duplicate."""
        start = np.array([-1.0, -1.0, -1.0])
        goal = np.array([0.0, 0.0, 0.0])  # Corner of cube

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        # 8 cube vertices + 1 start = 9 (goal is at cube vertex)
        assert len(vertices) == 9


# =============================================================================
# EDGE CONSTRUCTION TESTS
# =============================================================================

class TestEdgeConstruction:
    """Tests for correct edge construction."""

    def test_direct_visibility_no_obstacles(self, empty_obstacles):
        """Start and goal should be directly connected with no obstacles."""
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 5.0, 5.0])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            empty_obstacles, start, goal
        )

        # Check edge exists from start to goal
        neighbors = [n for n, _ in edges[start_idx]]
        assert goal_idx in neighbors, "Start should be connected to goal"

    def test_edges_are_symmetric(self, unit_cube_obstacle):
        """Graph edges should be undirected (symmetric)."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        for i in edges:
            for j, cost_ij in edges[i]:
                # Check reverse edge exists
                neighbors_j = [n for n, _ in edges[j]]
                assert i in neighbors_j, f"Edge ({i},{j}) exists but ({j},{i}) missing"

                # Check costs match
                cost_ji = None
                for n, c in edges[j]:
                    if n == i:
                        cost_ji = c
                        break
                assert_allclose(cost_ij, cost_ji, rtol=1e-10,
                               err_msg=f"Edge costs differ: ({i},{j})={cost_ij}, ({j},{i})={cost_ji}")

    def test_edge_costs_are_euclidean_distances(self, unit_cube_obstacle):
        """Edge costs should equal Euclidean distances."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        for i in edges:
            for j, cost in edges[i]:
                expected_cost = euclidean_distance_3d(vertices[i], vertices[j])
                assert_allclose(cost, expected_cost, rtol=1e-10,
                               err_msg=f"Edge ({i},{j}) cost {cost} != distance {expected_cost}")

    def test_no_edge_through_obstacle(self, offset_cube_obstacle):
        """No edge should pass through obstacle interior.

        BUG DETECTED: This test fails due to the line_obstacle_intersect_3d bug.
        The visibility graph incorrectly creates edges through obstacles because
        the intersection detection is faulty.

        Expected: Start and goal should NOT be directly connected
        Actual: They ARE connected because intersection is not detected

        Root Cause: line_obstacle_intersect_3d returns False even when line
        passes through cube interior.
        """
        # Start and goal on opposite sides of the cube
        start = np.array([0.0, 2.5, 0.5])  # Left of cube
        goal = np.array([5.0, 2.5, 0.5])   # Right of cube

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            offset_cube_obstacle, start, goal
        )

        # Start and goal should NOT be directly connected (cube in between)
        neighbors_of_start = [n for n, _ in edges[start_idx]]
        assert goal_idx not in neighbors_of_start, \
            "Start and goal should not be directly connected through obstacle"

    def test_edge_along_obstacle_surface(self, unit_cube_obstacle):
        """Edges along obstacle edges should exist."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        # Find cube corner vertices
        corner1_pos = np.array([0.0, 0.0, 0.0])
        corner2_pos = np.array([1.0, 0.0, 0.0])

        corner1_idx = None
        corner2_idx = None
        for i, v in enumerate(vertices):
            if np.allclose(v, corner1_pos):
                corner1_idx = i
            if np.allclose(v, corner2_pos):
                corner2_idx = i

        assert corner1_idx is not None and corner2_idx is not None

        # These corners should be connected (along cube edge)
        neighbors = [n for n, _ in edges[corner1_idx]]
        assert corner2_idx in neighbors, \
            "Adjacent cube corners should be connected"


# =============================================================================
# VISIBILITY TESTS
# =============================================================================

class TestVisibility:
    """Tests for visibility determination."""

    def test_vertices_around_obstacle_visible(self, unit_cube_obstacle):
        """Vertices around obstacle should see each other if clear path."""
        # Points on opposite sides but at different heights (can see over)
        start = np.array([-1.0, 0.5, 2.0])  # Above cube level
        goal = np.array([2.0, 0.5, 2.0])    # Also above cube level

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        # These should be directly visible (above the cube)
        neighbors = [n for n, _ in edges[start_idx]]
        assert goal_idx in neighbors, \
            "Points above obstacle should see each other"

    def test_visibility_blocked_by_obstacle(self, offset_cube_obstacle):
        """Line through obstacle interior should not create edge.

        BUG DETECTED: Same root cause as test_no_edge_through_obstacle.
        Visibility graph incorrectly shows direct visibility through obstacle.

        Expected: No direct edge between start and goal
        Actual: Direct edge exists
        """
        # Cube at (2,2,0) to (3,3,1)
        start = np.array([0.0, 2.5, 0.5])  # Left of cube
        goal = np.array([5.0, 2.5, 0.5])   # Right of cube

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            offset_cube_obstacle, start, goal
        )

        # Should NOT have direct edge (cube blocks view)
        neighbors = [n for n, _ in edges[start_idx]]
        assert goal_idx not in neighbors

    def test_path_exists_around_obstacle(self, offset_cube_obstacle):
        """Path should exist around obstacle even if blocked directly."""
        start = np.array([0.0, 2.5, 0.5])
        goal = np.array([5.0, 2.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            offset_cube_obstacle, start, goal
        )

        # Start should connect to some obstacle vertices
        assert len(edges[start_idx]) > 0, "Start should have some edges"

        # Goal should also have edges
        assert len(edges[goal_idx]) > 0, "Goal should have some edges"


# =============================================================================
# INDEX CORRECTNESS TESTS
# =============================================================================

class TestIndexCorrectness:
    """Tests for correct index handling."""

    def test_start_idx_points_to_start(self, empty_obstacles):
        """start_idx should point to start vertex."""
        start = np.array([1.0, 2.0, 3.0])
        goal = np.array([4.0, 5.0, 6.0])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            empty_obstacles, start, goal
        )

        assert_array_almost_equal(vertices[start_idx], start)

    def test_goal_idx_points_to_goal(self, empty_obstacles):
        """goal_idx should point to goal vertex."""
        start = np.array([1.0, 2.0, 3.0])
        goal = np.array([4.0, 5.0, 6.0])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            empty_obstacles, start, goal
        )

        assert_array_almost_equal(vertices[goal_idx], goal)

    def test_indices_are_valid(self, unit_cube_obstacle):
        """All indices in edges should be valid vertex indices."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        n = len(vertices)
        for i in edges:
            assert 0 <= i < n, f"Edge dict key {i} out of range"
            for j, _ in edges[i]:
                assert 0 <= j < n, f"Edge neighbor {j} out of range"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_start_equals_goal(self, empty_obstacles):
        """Start and goal at same position."""
        pos = np.array([1.0, 2.0, 3.0])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            empty_obstacles, pos, pos
        )

        # Should have only one vertex (start == goal)
        assert len(vertices) == 1
        assert start_idx == goal_idx

    def test_start_and_goal_very_close(self, empty_obstacles):
        """Start and goal very close together."""
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([1e-10, 1e-10, 1e-10])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            empty_obstacles, start, goal
        )

        # Both should be included (different positions)
        assert len(vertices) == 2

    def test_large_number_of_obstacle_vertices(self):
        """Handle obstacle with many vertices."""
        # Create obstacle with 100 vertices (a detailed polyhedron)
        n = 10
        vertices_list = []
        for i in range(n):
            for j in range(n):
                vertices_list.append([float(i), float(j), 0.0])

        obstacles = [{
            'vertices': np.array(vertices_list),
            'faces': []  # No faces - this is just for vertex testing
        }]

        start = np.array([-1.0, 5.0, 0.5])
        goal = np.array([11.0, 5.0, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            obstacles, start, goal
        )

        # 100 obstacle vertices + 2 (start/goal) = 102
        assert len(vertices) == 102


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

class TestConsistency:
    """Tests for internal consistency."""

    def test_all_vertices_have_edge_entries(self, unit_cube_obstacle):
        """Every vertex should have an entry in edges dict."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        for i in range(len(vertices)):
            assert i in edges, f"Vertex {i} missing from edges dict"

    def test_edge_dict_length_matches_vertices(self, unit_cube_obstacle):
        """Edge dict should have entries for all vertices."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        assert len(edges) == len(vertices)

    def test_no_self_loops(self, unit_cube_obstacle):
        """No vertex should have an edge to itself."""
        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            unit_cube_obstacle, start, goal
        )

        for i in edges:
            neighbors = [n for n, _ in edges[i]]
            assert i not in neighbors, f"Vertex {i} has self-loop"


# =============================================================================
# INTEGRATION TESTS WITH A*
# =============================================================================

class TestIntegrationWithAStar:
    """Tests that visibility graph works correctly with A*."""

    def test_astar_can_find_path_no_obstacles(self, empty_obstacles):
        """A* should find direct path with no obstacles."""
        from a_star import a_star_search

        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 5.0, 5.0])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            empty_obstacles, start, goal
        )

        path, indices = a_star_search(vertices, edges, start_idx, goal_idx)

        assert path is not None
        # Should be direct path (just start and goal)
        assert len(path) == 2

    def test_astar_can_find_path_around_obstacle(self, offset_cube_obstacle):
        """A* should find path around obstacle.

        BUG DETECTED: Due to line_obstacle_intersect_3d bug, the visibility
        graph creates a direct edge through the obstacle, so A* finds the
        direct path instead of going around.

        Expected: Path with > 2 waypoints (going around obstacle)
        Actual: Path with 2 waypoints (direct through obstacle)
        """
        from a_star import a_star_search

        start = np.array([0.0, 2.5, 0.5])
        goal = np.array([5.0, 2.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            offset_cube_obstacle, start, goal
        )

        path, indices = a_star_search(vertices, edges, start_idx, goal_idx)

        assert path is not None
        # Path should go around obstacle (more than 2 waypoints)
        assert len(path) > 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
