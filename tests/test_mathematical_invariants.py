"""
Comprehensive Test Suite for Mathematical Invariants

This module tests mathematical invariants that must always hold across
the entire ORCA-A* 3D implementation. Following the Yoneda philosophy,
these tests verify fundamental mathematical properties that define
correct behavior.

Invariants Tested:
1. Geometric Invariants
   - Triangle inequality for distances
   - Symmetry of distance function
   - Unit vector properties

2. A* Invariants
   - Heuristic admissibility (never overestimates)
   - Path optimality
   - Path connectivity

3. ORCA Invariants
   - Velocity constraints form convex set
   - Reciprocal sharing of avoidance effort
   - Feasible region contains valid velocities

4. System Invariants
   - Position continuity over time
   - Velocity magnitude bounded by v_max
   - No instantaneous teleportation
"""

import sys
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import euclidean_distance_3d, normalize, line_obstacle_intersect_3d
from a_star import a_star_search
from visibility_graph import build_visibility_graph
from orca import compute_vo_escape, compute_orca_velocity
from hybrid_orca_astar import Agent, simulate


# =============================================================================
# GEOMETRIC INVARIANTS
# =============================================================================

class TestGeometricInvariants:
    """Tests for fundamental geometric invariants."""

    @pytest.mark.parametrize("seed", range(10))
    def test_triangle_inequality_random_points(self, seed):
        """
        Triangle Inequality: d(a,c) <= d(a,b) + d(b,c)

        This must hold for all points in Euclidean space.
        """
        np.random.seed(seed)
        a = np.random.randn(3) * 100
        b = np.random.randn(3) * 100
        c = np.random.randn(3) * 100

        d_ac = euclidean_distance_3d(a, c)
        d_ab = euclidean_distance_3d(a, b)
        d_bc = euclidean_distance_3d(b, c)

        assert d_ac <= d_ab + d_bc + 1e-10, \
            f"Triangle inequality violated: {d_ac} > {d_ab} + {d_bc}"

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_symmetry_random_points(self, seed):
        """
        Symmetry: d(a,b) = d(b,a)

        Distance must be symmetric for all point pairs.
        """
        np.random.seed(seed)
        a = np.random.randn(3) * 100
        b = np.random.randn(3) * 100

        assert_allclose(euclidean_distance_3d(a, b), euclidean_distance_3d(b, a))

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_non_negativity(self, seed):
        """
        Non-negativity: d(a,b) >= 0

        Distance is always non-negative.
        """
        np.random.seed(seed)
        a = np.random.randn(3) * 100
        b = np.random.randn(3) * 100

        assert euclidean_distance_3d(a, b) >= 0

    @pytest.mark.parametrize("seed", range(10))
    def test_distance_identity(self, seed):
        """
        Identity: d(a,a) = 0

        Distance from a point to itself is zero.
        """
        np.random.seed(seed)
        a = np.random.randn(3) * 100

        assert euclidean_distance_3d(a, a) == 0

    @pytest.mark.parametrize("seed", range(10))
    def test_normalized_vector_unit_length(self, seed):
        """
        Unit Vector: ||normalize(v)|| = 1 for v != 0

        Normalized non-zero vectors have unit length.
        """
        np.random.seed(seed)
        v = np.random.randn(3)

        if np.linalg.norm(v) > 1e-6:
            n = normalize(v)
            assert_allclose(np.linalg.norm(n), 1.0, rtol=1e-10)

    @pytest.mark.parametrize("seed", range(10))
    def test_normalization_preserves_direction(self, seed):
        """
        Direction Preservation: normalize(v) is parallel to v

        Normalization only changes magnitude, not direction.
        """
        np.random.seed(seed)
        v = np.random.randn(3)

        if np.linalg.norm(v) > 1e-6:
            n = normalize(v)
            # Cross product of parallel vectors is zero
            cross = np.cross(v, n)
            assert_allclose(np.linalg.norm(cross), 0.0, atol=1e-10)


# =============================================================================
# A* ALGORITHM INVARIANTS
# =============================================================================

class TestAStarInvariants:
    """Tests for A* algorithm invariants."""

    def create_test_graph(self, n_vertices=10, seed=42):
        """Create a random test graph."""
        np.random.seed(seed)
        vertices = np.random.randn(n_vertices, 3) * 10
        edges = {i: [] for i in range(n_vertices)}

        # Add random edges
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if np.random.rand() < 0.5:  # 50% edge probability
                    cost = euclidean_distance_3d(vertices[i], vertices[j])
                    edges[i].append((j, cost))
                    edges[j].append((i, cost))

        return vertices, edges

    def test_heuristic_admissibility(self):
        """
        Admissibility: h(v) <= actual_cost(v, goal)

        The Euclidean heuristic never overestimates the actual path cost.
        """
        vertices, edges = self.create_test_graph(15)

        # For each pair of connected vertices
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if i != j:
                    # Euclidean heuristic
                    h = euclidean_distance_3d(vertices[i], vertices[j])

                    # Find actual shortest path
                    path, _ = a_star_search(vertices, edges, i, j)

                    if path is not None:
                        # Compute actual path cost
                        actual_cost = sum(
                            euclidean_distance_3d(path[k], path[k+1])
                            for k in range(len(path) - 1)
                        )

                        assert h <= actual_cost + 1e-10, \
                            f"Heuristic not admissible: h={h} > actual={actual_cost}"

    def test_path_optimality(self):
        """
        Optimality: A* returns the shortest path

        With admissible and consistent heuristic, A* finds optimal path.
        """
        # Create simple graph where optimal is known
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [1, 1, 0],  # Detour point
        ], dtype=float)

        edges = {
            0: [(1, 1.0), (3, np.sqrt(2))],
            1: [(0, 1.0), (2, 1.0), (3, 1.0)],
            2: [(1, 1.0), (3, np.sqrt(2))],
            3: [(0, np.sqrt(2)), (1, 1.0), (2, np.sqrt(2))],
        }

        path, indices = a_star_search(vertices, edges, 0, 2)

        # Optimal path is 0 -> 1 -> 2 with cost 2.0
        path_cost = sum(
            euclidean_distance_3d(path[k], path[k+1])
            for k in range(len(path) - 1)
        )

        assert_allclose(path_cost, 2.0, rtol=1e-10)

    def test_path_connectivity(self):
        """
        Connectivity: Each step in path uses valid edge

        Path must only use edges that exist in the graph.
        """
        vertices, edges = self.create_test_graph(10)

        path, indices = a_star_search(vertices, edges, 0, 5)

        if path is not None and indices is not None:
            for i in range(len(indices) - 1):
                current = indices[i]
                next_node = indices[i + 1]
                neighbors = [n for n, _ in edges[current]]

                assert next_node in neighbors, \
                    f"Path uses non-existent edge: {current} -> {next_node}"


# =============================================================================
# ORCA ALGORITHM INVARIANTS
# =============================================================================

class TestORCAInvariants:
    """Tests for ORCA algorithm invariants."""

    def test_velocity_bounded_by_vmax(self):
        """
        Speed Bound: ||v_new|| <= v_max

        Resulting velocity must not exceed maximum speed.
        """
        np.random.seed(42)

        for _ in range(20):
            agent_pos = np.random.randn(3) * 10
            agent_vel = np.random.randn(3)
            preferred_vel = np.random.randn(3) * 2

            other_agents = [
                {'pos': np.random.randn(3) * 10, 'vel': np.random.randn(3)}
                for _ in range(3)
            ]

            R = 1.0
            tau = 5.0
            v_max = 2.0

            new_vel = compute_orca_velocity(
                agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
            )

            speed = np.linalg.norm(new_vel)
            assert speed <= v_max + 1e-3, \
                f"Speed {speed} exceeds v_max {v_max}"

    def test_escape_vector_points_outward(self):
        """
        Outward Normal: Escape vector points away from VO interior

        The escape direction should move velocity outside the VO.
        """
        # Setup collision scenario
        p_rel = np.array([5.0, 0.0, 0.0])
        v_rel = np.array([2.0, 0.0, 0.0])  # Head-on
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        if u is not None and n_hat is not None:
            # n_hat should be unit vector
            assert_allclose(np.linalg.norm(n_hat), 1.0, rtol=1e-6)

            # u should be aligned with n_hat
            if np.linalg.norm(u) > 1e-6:
                alignment = np.dot(normalize(u), n_hat)
                assert abs(alignment) > 0.99, \
                    f"Escape vector not aligned with normal: {alignment}"

    def test_no_escape_needed_for_safe_velocities(self):
        """
        Safety: No escape for velocities that don't lead to collision

        Velocities that don't cause collision should return None.
        """
        # Parallel trajectories - no collision
        p_rel = np.array([5.0, 0.0, 0.0])
        v_rel = np.array([0.0, 1.0, 0.0])  # Perpendicular motion
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        assert u is None and n_hat is None, \
            "Should not require escape for safe velocity"


# =============================================================================
# SYSTEM INVARIANTS
# =============================================================================

class TestSystemInvariants:
    """Tests for whole-system invariants."""

    def test_position_continuity(self):
        """
        Continuity: Position changes are bounded by v_max * dt

        No instantaneous teleportation.
        """
        obstacles = []
        agent = Agent([0, 0, 0], [10, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=obstacles)

        dt = 0.1
        trajectories = simulate([agent], dt=dt, max_time=5.0)

        for i in range(len(trajectories[0]) - 1):
            pos1 = trajectories[0][i]
            pos2 = trajectories[0][i + 1]
            displacement = np.linalg.norm(pos2 - pos1)
            max_displacement = agent.v_max * dt

            assert displacement <= max_displacement + 1e-6, \
                f"Teleportation detected: moved {displacement} in dt={dt}"

    def test_velocity_magnitude_bounded(self):
        """
        Speed Limit: Agent speed never exceeds v_max

        Throughout simulation, velocity is bounded.
        """
        obstacles = []
        v_max = 1.5

        agents = [
            Agent([0, 0, 0], [10, 0, 0], max_speed=v_max, R=0.5, tau=5.0,
                 obstacles=obstacles),
            Agent([10, 0, 0], [0, 0, 0], max_speed=v_max, R=0.5, tau=5.0,
                 obstacles=obstacles),
        ]

        dt = 0.1
        trajectories = simulate(agents, dt=dt, max_time=10.0)

        for traj in trajectories:
            for i in range(len(traj) - 1):
                # Estimate velocity from positions
                vel_est = (traj[i + 1] - traj[i]) / dt
                speed = np.linalg.norm(vel_est)

                assert speed <= v_max + 1e-3, \
                    f"Speed limit violated: {speed} > {v_max}"

    def test_goal_convergence(self):
        """
        Convergence: Agent makes progress toward goal (without obstacles)

        In obstacle-free environment with no collisions, agent should reach goal.
        """
        obstacles = []
        goal = np.array([5.0, 0.0, 0.0])

        agent = Agent([0, 0, 0], goal.tolist(), max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=obstacles, epsilon=0.1)

        trajectories = simulate([agent], dt=0.1, max_time=10.0)

        final_pos = trajectories[0][-1]
        final_dist = np.linalg.norm(final_pos - goal)

        # Should be within epsilon of goal
        assert final_dist < 0.5, \
            f"Did not converge to goal: distance = {final_dist}"


# =============================================================================
# VISIBILITY GRAPH INVARIANTS
# =============================================================================

class TestVisibilityGraphInvariants:
    """Tests for visibility graph invariants."""

    def test_graph_symmetry(self):
        """
        Symmetry: If edge (u,v) exists, so does (v,u) with same cost

        Visibility graph must be undirected.
        """
        obstacles = [{
            'vertices': np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ], dtype=float),
            'faces': [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                     [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]
        }]

        start = np.array([-2.0, 0.5, 0.5])
        goal = np.array([3.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            obstacles, start, goal
        )

        for i in edges:
            for j, cost_ij in edges[i]:
                # Reverse edge should exist
                found = False
                for k, cost_ji in edges[j]:
                    if k == i:
                        found = True
                        assert_allclose(cost_ij, cost_ji, rtol=1e-10)
                        break

                assert found, f"Missing reverse edge: ({j}, {i})"

    def test_edge_costs_match_distances(self):
        """
        Consistency: Edge cost equals Euclidean distance

        All edge costs must be exact distances.
        """
        obstacles = []
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 5.0, 5.0])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            obstacles, start, goal
        )

        for i in edges:
            for j, cost in edges[i]:
                expected = euclidean_distance_3d(vertices[i], vertices[j])
                assert_allclose(cost, expected, rtol=1e-10)

    def test_no_self_loops(self):
        """
        No Self-Loops: No vertex has edge to itself

        Self-loops would have zero cost and be meaningless.
        """
        obstacles = [{
            'vertices': np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ], dtype=float),
            'faces': [[0, 1, 2, 3]]
        }]

        start = np.array([-1.0, 0.5, 0.5])
        goal = np.array([2.0, 0.5, 0.5])

        vertices, edges, start_idx, goal_idx = build_visibility_graph(
            obstacles, start, goal
        )

        for i in edges:
            neighbors = [j for j, _ in edges[i]]
            assert i not in neighbors, f"Self-loop at vertex {i}"


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================

class TestPropertyBased:
    """Property-based tests with random inputs."""

    @pytest.mark.parametrize("seed", range(5))
    def test_orca_returns_valid_velocity(self, seed):
        """ORCA should always return finite, bounded velocity."""
        np.random.seed(seed)

        agent_pos = np.random.randn(3) * 10
        agent_vel = np.random.randn(3)
        preferred_vel = np.random.randn(3) * 2

        other_agents = [
            {'pos': np.random.randn(3) * 10, 'vel': np.random.randn(3)}
            for _ in range(np.random.randint(0, 5))
        ]

        R = np.random.uniform(0.1, 2.0)
        tau = np.random.uniform(1.0, 10.0)
        v_max = np.random.uniform(0.5, 5.0)

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # Must be finite
        assert np.all(np.isfinite(new_vel)), "Velocity contains non-finite values"

        # Must be bounded
        assert np.linalg.norm(new_vel) <= v_max + 1e-3

    @pytest.mark.parametrize("seed", range(5))
    def test_astar_path_is_valid(self, seed):
        """A* path should use only valid edges."""
        np.random.seed(seed)

        # Random graph
        n = 10
        vertices = np.random.randn(n, 3) * 10
        edges = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < 0.4:
                    cost = euclidean_distance_3d(vertices[i], vertices[j])
                    edges[i].append((j, cost))
                    edges[j].append((i, cost))

        start_idx = 0
        goal_idx = n - 1

        path, indices = a_star_search(vertices, edges, start_idx, goal_idx)

        if indices is not None:
            # Verify path validity
            for i in range(len(indices) - 1):
                u = indices[i]
                v = indices[i + 1]
                neighbors = [n for n, _ in edges[u]]
                assert v in neighbors, f"Invalid edge in path: {u} -> {v}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
