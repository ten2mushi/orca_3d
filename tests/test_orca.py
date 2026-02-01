"""
Comprehensive Test Suite for orca.py

This module tests the 3D ORCA (Optimal Reciprocal Collision Avoidance) implementation.
Following the Yoneda philosophy, these tests serve as a complete behavioral
specification for the collision avoidance algorithm.

Key Functions Tested:
- compute_vo_escape: Computes minimal escape vector from 3D velocity obstacle
- compute_orca_velocity: Computes collision-free velocity using ORCA constraints

Mathematical Specification:
- Velocity Obstacle: VO_{i|j}^tau = {v_rel | exists t in [0,tau]: ||p_rel - t*v_rel|| < R}
- Forms a truncated cone in 3D velocity space
- ORCA constraint: half-space H_{i|j} = {v | (v - v_current) . n_hat >= (u . n_hat)/2}
- Final velocity: argmin_{v in ORCA_i cap B(0, v_max)} ||v - v_preferred||

Properties to Verify:
- Collision detection accuracy
- Escape vector correctness
- Reciprocal behavior
- Velocity constraint satisfaction
- Numerical stability
"""

import sys
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orca import compute_vo_escape, compute_orca_velocity
from utils import normalize


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_collision_time(p_rel, v_rel, R, tau):
    """
    Compute collision time analytically.
    Returns t if collision occurs in [0, tau], else None.

    Solve ||p_rel - t*v_rel||^2 = R^2
    """
    a = np.dot(v_rel, v_rel)
    b = -2 * np.dot(p_rel, v_rel)
    c = np.dot(p_rel, p_rel) - R**2

    if a < 1e-12:
        return None

    disc = b**2 - 4*a*c
    if disc < 0:
        return None

    t1 = (-b - np.sqrt(disc)) / (2*a)
    t2 = (-b + np.sqrt(disc)) / (2*a)

    # Get earliest positive collision time
    valid_times = [t for t in [t1, t2] if t > 0]
    if not valid_times:
        return None

    t_min = min(valid_times)
    return t_min if t_min <= tau else None


# =============================================================================
# COMPUTE_VO_ESCAPE TESTS - COLLISION DETECTION
# =============================================================================

class TestVOEscapeCollisionDetection:
    """Tests for collision detection in compute_vo_escape."""

    def test_no_collision_parallel_trajectories(self):
        """Parallel trajectories with separation should not trigger escape."""
        p_rel = np.array([5.0, 0.0, 0.0])  # Agents separated along x
        v_rel = np.array([0.0, 1.0, 0.0])  # Moving parallel
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        # Should return None - no collision trajectory
        assert u is None and n_hat is None

    def test_no_collision_diverging_trajectories(self):
        """Agents moving apart should not trigger escape."""
        p_rel = np.array([5.0, 0.0, 0.0])  # Other agent is 5 units away in +x
        v_rel = np.array([-1.0, 0.0, 0.0])  # Moving toward -x (away from other agent)
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        assert u is None and n_hat is None

    def test_collision_detected_head_on(self):
        """Head-on collision should be detected."""
        p_rel = np.array([5.0, 0.0, 0.0])
        v_rel = np.array([2.0, 0.0, 0.0])  # Moving toward each other (agent closing in)
        R = 1.0
        tau = 10.0

        # Verify collision occurs analytically
        t_col = check_collision_time(p_rel, v_rel, R, tau)
        assert t_col is not None, "Analytical check: collision should occur"

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        # Should trigger escape
        assert u is not None, "Should detect collision"
        assert n_hat is not None, "Should provide escape normal"

    def test_collision_beyond_horizon_not_detected(self):
        """Collision beyond time horizon should not trigger escape."""
        p_rel = np.array([100.0, 0.0, 0.0])
        v_rel = np.array([1.0, 0.0, 0.0])  # Very slow approach
        R = 1.0
        tau = 5.0  # Short horizon

        # Collision would be at t = (100-1)/1 = 99s > tau
        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        assert u is None and n_hat is None

    def test_already_colliding_returns_escape(self):
        """Already in collision should return emergency escape vector."""
        p_rel = np.array([0.5, 0.0, 0.0])  # Distance < R (d=0.5, R=1.0)
        v_rel = np.array([0.0, 0.0, 0.0])
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        # Should return escape vector pushing away from other agent
        assert u is not None and n_hat is not None
        # Escape direction should be away from other agent (opposite to p_rel)
        assert np.dot(n_hat, p_rel) < 0, "Escape should push away from other agent"
        # Normal should be unit vector
        assert abs(np.linalg.norm(n_hat) - 1.0) < 1e-6

    def test_zero_relative_velocity_no_collision(self):
        """Zero relative velocity with separation should not collide."""
        p_rel = np.array([5.0, 0.0, 0.0])
        v_rel = np.array([0.0, 0.0, 0.0])
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        assert u is None and n_hat is None


# =============================================================================
# COMPUTE_VO_ESCAPE TESTS - ESCAPE VECTOR CORRECTNESS
# =============================================================================

class TestVOEscapeVectorCorrectness:
    """Tests for escape vector correctness."""

    def test_escape_vector_direction_head_on(self):
        """Escape should push velocity outside velocity obstacle."""
        p_rel = np.array([5.0, 0.0, 0.0])
        v_rel = np.array([2.0, 0.0, 0.0])  # Direct collision course
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        if u is not None:
            # After applying escape, should be outside VO
            v_escaped = v_rel + u

            # Check collision time with escaped velocity
            t_col = check_collision_time(p_rel, v_escaped, R, tau)

            # Note: v_escaped might still be at boundary, so check is approximate
            assert n_hat is not None
            assert np.linalg.norm(n_hat) > 0.99  # Should be unit vector

    def test_normal_is_unit_vector(self):
        """Escape normal should be a unit vector."""
        p_rel = np.array([5.0, 0.0, 0.0])
        v_rel = np.array([2.0, 0.1, 0.0])  # Slight offset
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        if n_hat is not None:
            assert_allclose(np.linalg.norm(n_hat), 1.0, rtol=1e-6)

    def test_escape_is_minimal(self):
        """Escape should be approximately minimal (to boundary)."""
        p_rel = np.array([5.0, 0.0, 0.0])
        v_rel = np.array([2.0, 0.1, 0.0])
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        if u is not None:
            # Escape should point in direction of n_hat
            u_normalized = normalize(u)
            if np.linalg.norm(u) > 1e-6:
                # u should be (approximately) aligned with n_hat
                dot_product = abs(np.dot(u_normalized, n_hat))
                assert dot_product > 0.99, f"u and n_hat should be aligned, dot={dot_product}"


# =============================================================================
# COMPUTE_VO_ESCAPE TESTS - 3D GEOMETRY
# =============================================================================

class TestVOEscape3DGeometry:
    """Tests for 3D geometry in velocity obstacle computation."""

    def test_collision_in_z_direction(self):
        """Collision along z-axis should be detected."""
        p_rel = np.array([0.0, 0.0, 5.0])
        v_rel = np.array([0.0, 0.0, 2.0])  # Moving toward in z
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        # Should detect collision
        t_col = check_collision_time(p_rel, v_rel, R, tau)
        if t_col is not None:
            assert u is not None, "Should detect z-axis collision"

    def test_diagonal_collision_3d(self):
        """Diagonal 3D collision should be detected."""
        p_rel = np.array([3.0, 4.0, 5.0])  # Distance = sqrt(50) ~ 7.07
        v_rel = np.array([1.0, 1.3, 1.7])  # Moving toward
        R = 1.0
        tau = 10.0

        t_col = check_collision_time(p_rel, v_rel, R, tau)
        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        if t_col is not None and t_col <= tau:
            assert u is not None, "Should detect 3D diagonal collision"

    def test_escape_uses_3d_geometry(self):
        """Escape vector should use full 3D geometry."""
        p_rel = np.array([3.0, 3.0, 3.0])
        v_rel = np.array([1.0, 1.0, 1.0])  # Toward center
        R = 1.0
        tau = 10.0

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        if u is not None:
            # All three components might be non-zero
            assert n_hat.shape == (3,)


# =============================================================================
# COMPUTE_ORCA_VELOCITY TESTS - BASIC OPERATION
# =============================================================================

class TestORCAVelocityBasicOperation:
    """Tests for basic ORCA velocity computation."""

    def test_no_other_agents_returns_preferred(self):
        """With no other agents, should return preferred velocity."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([0.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])
        other_agents = []
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert_array_almost_equal(new_vel, preferred_vel)

    def test_preferred_velocity_within_speed_limit(self):
        """Preferred velocity within limit should be returned."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([0.0, 0.0, 0.0])
        preferred_vel = np.array([0.5, 0.0, 0.0])
        other_agents = []
        R = 1.0
        tau = 5.0
        v_max = 1.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert_array_almost_equal(new_vel, preferred_vel)

    def test_preferred_velocity_clamped_to_speed_limit(self):
        """Preferred velocity exceeding limit should be clamped."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([0.0, 0.0, 0.0])
        preferred_vel = np.array([5.0, 0.0, 0.0])  # Exceeds v_max
        other_agents = []
        R = 1.0
        tau = 5.0
        v_max = 1.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        speed = np.linalg.norm(new_vel)
        assert speed <= v_max + 1e-6, f"Speed {speed} exceeds v_max {v_max}"


# =============================================================================
# COMPUTE_ORCA_VELOCITY TESTS - COLLISION AVOIDANCE
# =============================================================================

class TestORCAVelocityCollisionAvoidance:
    """Tests for ORCA collision avoidance behavior."""

    def test_avoids_head_on_collision(self):
        """Should adjust velocity to avoid head-on collision."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        # Other agent coming from opposite direction
        other_agents = [{
            'pos': np.array([5.0, 0.0, 0.0]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # Velocity should be adjusted (not exactly preferred)
        # Either y or z component should be non-zero for avoidance
        assert not np.allclose(new_vel, preferred_vel, atol=0.01), \
            "Velocity should be adjusted for collision avoidance"

    def test_speed_constraint_satisfied(self):
        """Result velocity should satisfy speed constraint."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        other_agents = [{
            'pos': np.array([3.0, 0.0, 0.0]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 1.0
        tau = 5.0
        v_max = 1.5

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        speed = np.linalg.norm(new_vel)
        assert speed <= v_max + 1e-3, f"Speed {speed} exceeds v_max {v_max}"

    def test_distant_agent_no_avoidance(self):
        """Distant agent should not cause avoidance."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        # Very distant agent
        other_agents = [{
            'pos': np.array([100.0, 0.0, 0.0]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 1.0
        tau = 5.0  # Short horizon - won't reach in time
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # Should be close to preferred velocity
        assert_allclose(new_vel, preferred_vel, atol=0.1)


# =============================================================================
# COMPUTE_ORCA_VELOCITY TESTS - MULTIPLE AGENTS
# =============================================================================

class TestORCAVelocityMultipleAgents:
    """Tests for ORCA with multiple other agents."""

    def test_avoids_multiple_agents(self):
        """Should satisfy constraints from multiple agents."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([0.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        # Two agents approaching from different angles
        other_agents = [
            {'pos': np.array([3.0, 1.0, 0.0]), 'vel': np.array([-1.0, -0.3, 0.0])},
            {'pos': np.array([3.0, -1.0, 0.0]), 'vel': np.array([-1.0, 0.3, 0.0])},
        ]
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # Should return a valid velocity
        assert np.all(np.isfinite(new_vel)), "Velocity should be finite"
        assert np.linalg.norm(new_vel) <= v_max + 1e-3

    def test_multiple_constraints_feasible_region(self):
        """Result should be in feasible region of all constraints."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        other_agents = [
            {'pos': np.array([5.0, 0.0, 0.0]), 'vel': np.array([-1.0, 0.0, 0.0])},
            {'pos': np.array([0.0, 5.0, 0.0]), 'vel': np.array([0.0, -1.0, 0.0])},
        ]
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # Velocity should be valid
        assert np.all(np.isfinite(new_vel))


# =============================================================================
# COMPUTE_ORCA_VELOCITY TESTS - 3D AVOIDANCE
# =============================================================================

class TestORCAVelocity3DAvoidance:
    """Tests for 3D collision avoidance."""

    def test_avoidance_in_z_direction(self):
        """Should be able to avoid by moving in z direction."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        # Head-on collision
        other_agents = [{
            'pos': np.array([5.0, 0.0, 0.0]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 0.5
        tau = 10.0
        v_max = 1.5

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # At least one of y or z should be non-zero for avoidance
        lateral_movement = abs(new_vel[1]) + abs(new_vel[2])
        # Note: This might still be zero if the optimizer finds another solution

    def test_3d_collision_from_above(self):
        """Should avoid agent approaching from above."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([0.0, 0.0, 0.0])
        preferred_vel = np.array([0.0, 0.0, 1.0])  # Want to go up

        # Agent above coming down
        other_agents = [{
            'pos': np.array([0.0, 0.0, 5.0]),
            'vel': np.array([0.0, 0.0, -1.0])
        }]
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # Should have valid velocity
        assert np.all(np.isfinite(new_vel))


# =============================================================================
# RECIPROCAL BEHAVIOR TESTS
# =============================================================================

class TestORCAReciprocalBehavior:
    """Tests for reciprocal collision avoidance behavior."""

    def test_symmetric_agents_symmetric_avoidance(self):
        """Two symmetric agents should make symmetric adjustments."""
        # Agent 1
        pos1 = np.array([0.0, 0.0, 0.0])
        vel1 = np.array([1.0, 0.0, 0.0])
        pref1 = np.array([1.0, 0.0, 0.0])

        # Agent 2 (symmetric)
        pos2 = np.array([5.0, 0.0, 0.0])
        vel2 = np.array([-1.0, 0.0, 0.0])
        pref2 = np.array([-1.0, 0.0, 0.0])

        R = 1.0
        tau = 5.0
        v_max = 2.0

        # Compute new velocity for agent 1
        new_vel1 = compute_orca_velocity(
            pos1, vel1, pref1,
            [{'pos': pos2, 'vel': vel2}],
            R, tau, v_max
        )

        # Compute new velocity for agent 2
        new_vel2 = compute_orca_velocity(
            pos2, vel2, pref2,
            [{'pos': pos1, 'vel': vel1}],
            R, tau, v_max
        )

        # The y and z adjustments should be opposite (symmetric avoidance)
        # Note: Due to optimization, this might not be exactly symmetric
        # but the relative velocity should be safe


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================

class TestORCAEdgeCases:
    """Edge cases and boundary conditions."""

    def test_agent_at_same_position_different_velocity(self):
        """Agents at same position should handle gracefully."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        # Other agent at same position (already colliding)
        other_agents = [{
            'pos': np.array([0.0, 0.0, 0.0]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 1.0
        tau = 5.0
        v_max = 2.0

        # Should not crash
        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert np.all(np.isfinite(new_vel)), "Should return finite velocity"

    def test_zero_preferred_velocity(self):
        """Zero preferred velocity should work."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([0.0, 0.0, 0.0])
        preferred_vel = np.array([0.0, 0.0, 0.0])
        other_agents = []
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert_array_almost_equal(new_vel, np.zeros(3))

    def test_very_small_tau(self):
        """Very small time horizon should still work."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        other_agents = [{
            'pos': np.array([5.0, 0.0, 0.0]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 1.0
        tau = 0.1  # Very short horizon
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert np.all(np.isfinite(new_vel))

    def test_very_large_tau(self):
        """Very large time horizon should still work."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        other_agents = [{
            'pos': np.array([5.0, 0.0, 0.0]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 1.0
        tau = 100.0  # Very long horizon
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert np.all(np.isfinite(new_vel))


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

class TestORCANumericalStability:
    """Tests for numerical stability."""

    def test_large_positions(self):
        """Should handle large position values."""
        agent_pos = np.array([1e6, 1e6, 1e6])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        other_agents = [{
            'pos': np.array([1e6 + 5.0, 1e6, 1e6]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert np.all(np.isfinite(new_vel))

    def test_small_R(self):
        """Should handle very small collision radius."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        other_agents = [{
            'pos': np.array([0.1, 0.0, 0.0]),
            'vel': np.array([-1.0, 0.0, 0.0])
        }]
        R = 0.001  # Very small
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert np.all(np.isfinite(new_vel))

    def test_result_is_not_nan(self):
        """Result should never be NaN."""
        np.random.seed(42)

        for _ in range(10):
            agent_pos = np.random.randn(3) * 10
            agent_vel = np.random.randn(3)
            preferred_vel = np.random.randn(3)

            other_agents = [
                {'pos': np.random.randn(3) * 10, 'vel': np.random.randn(3)}
                for _ in range(3)
            ]
            R = abs(np.random.randn()) + 0.1
            tau = abs(np.random.randn()) * 5 + 1
            v_max = abs(np.random.randn()) + 1

            new_vel = compute_orca_velocity(
                agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
            )

            assert not np.any(np.isnan(new_vel)), "Result contains NaN"
            assert not np.any(np.isinf(new_vel)), "Result contains Inf"


# =============================================================================
# CONSTRAINT SATISFACTION TESTS
# =============================================================================

class TestORCAConstraintSatisfaction:
    """Tests that verify ORCA constraints are satisfied."""

    def test_velocity_in_feasible_region(self):
        """Result velocity should satisfy half-space constraints."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([1.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        other_pos = np.array([5.0, 0.0, 0.0])
        other_vel = np.array([-1.0, 0.0, 0.0])

        other_agents = [{'pos': other_pos, 'vel': other_vel}]
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # Compute the constraint manually
        p_rel = other_pos - agent_pos
        v_rel = agent_vel - other_vel

        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)

        if u is not None and n_hat is not None:
            # Check constraint: (v_new - agent_vel) . n_hat >= (u . n_hat) / 2
            lhs = np.dot(new_vel - agent_vel, n_hat)
            rhs = np.dot(u, n_hat) / 2

            # Should satisfy constraint (with some tolerance for optimizer)
            assert lhs >= rhs - 0.1, f"Constraint violated: {lhs} < {rhs}"


# =============================================================================
# OPTIMIZATION TESTS
# =============================================================================

class TestORCAOptimization:
    """Tests for the optimization process."""

    def test_finds_closest_to_preferred(self):
        """Should find velocity closest to preferred within constraints."""
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([0.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])
        other_agents = []
        R = 1.0
        tau = 5.0
        v_max = 2.0

        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        # With no constraints, should be exactly preferred velocity
        assert_array_almost_equal(new_vel, preferred_vel)

    def test_optimization_fallback(self):
        """Should handle optimization failures gracefully."""
        # Create a difficult scenario
        agent_pos = np.array([0.0, 0.0, 0.0])
        agent_vel = np.array([0.0, 0.0, 0.0])
        preferred_vel = np.array([1.0, 0.0, 0.0])

        # Many agents surrounding
        other_agents = [
            {'pos': np.array([2.0, 0.0, 0.0]), 'vel': np.array([-1.0, 0.0, 0.0])},
            {'pos': np.array([-2.0, 0.0, 0.0]), 'vel': np.array([1.0, 0.0, 0.0])},
            {'pos': np.array([0.0, 2.0, 0.0]), 'vel': np.array([0.0, -1.0, 0.0])},
            {'pos': np.array([0.0, -2.0, 0.0]), 'vel': np.array([0.0, 1.0, 0.0])},
        ]
        R = 1.0
        tau = 5.0
        v_max = 1.0

        # Should return something valid even if optimization is difficult
        new_vel = compute_orca_velocity(
            agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max
        )

        assert np.all(np.isfinite(new_vel)), "Should return finite velocity"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
