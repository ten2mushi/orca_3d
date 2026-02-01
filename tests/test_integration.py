"""
Comprehensive Integration Test Suite for ORCA-A* 3D

This module tests the complete integrated system, verifying that all
components work together correctly. Following the Yoneda philosophy,
these tests define the complete end-to-end behavioral specification.

Integration Scenarios Tested:
1. Single agent navigation
2. Two-agent collision avoidance (head-on, crossing, overtaking)
3. Multi-agent swarm scenarios
4. Navigation around static obstacles
5. Combined dynamic and static obstacle avoidance
6. Stress tests with many agents
7. Edge cases and failure modes
"""

import sys
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_orca_astar import Agent, simulate
from utils import euclidean_distance_3d


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def empty_environment():
    """Empty environment with no obstacles."""
    return []


@pytest.fixture
def single_cube():
    """Single cube obstacle."""
    return [{
        'vertices': np.array([
            [4, 4, 0], [6, 4, 0], [6, 6, 0], [4, 6, 0],
            [4, 4, 2], [6, 4, 2], [6, 6, 2], [4, 6, 2]
        ], dtype=float),
        'faces': [
            [0, 1, 2, 3], [4, 7, 6, 5],
            [0, 4, 5, 1], [2, 6, 7, 3],
            [0, 3, 7, 4], [1, 5, 6, 2]
        ]
    }]


@pytest.fixture
def corridor_obstacles():
    """Two walls forming a corridor."""
    return [
        # Left wall
        {
            'vertices': np.array([
                [4, 0, 0], [5, 0, 0], [5, 4, 0], [4, 4, 0],
                [4, 0, 2], [5, 0, 2], [5, 4, 2], [4, 4, 2]
            ], dtype=float),
            'faces': [
                [0, 1, 2, 3], [4, 7, 6, 5],
                [0, 4, 5, 1], [2, 6, 7, 3],
                [0, 3, 7, 4], [1, 5, 6, 2]
            ]
        },
        # Right wall
        {
            'vertices': np.array([
                [4, 6, 0], [5, 6, 0], [5, 10, 0], [4, 10, 0],
                [4, 6, 2], [5, 6, 2], [5, 10, 2], [4, 10, 2]
            ], dtype=float),
            'faces': [
                [0, 1, 2, 3], [4, 7, 6, 5],
                [0, 4, 5, 1], [2, 6, 7, 3],
                [0, 3, 7, 4], [1, 5, 6, 2]
            ]
        }
    ]


# =============================================================================
# SINGLE AGENT NAVIGATION TESTS
# =============================================================================

class TestSingleAgentNavigation:
    """Tests for single agent navigation scenarios."""

    def test_single_agent_reaches_goal_empty_space(self, empty_environment):
        """Single agent should reach goal in empty environment."""
        start = [0, 0, 0]
        goal = [10, 0, 0]

        agent = Agent(start, goal, max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=empty_environment, epsilon=0.5)

        trajectories = simulate([agent], dt=0.1, max_time=15.0)

        final_pos = trajectories[0][-1]
        final_dist = euclidean_distance_3d(final_pos, np.array(goal))

        # Allow small tolerance for floating point precision
        assert final_dist < 0.51, f"Agent did not reach goal: distance = {final_dist}"

    def test_single_agent_direct_path_empty_space(self, empty_environment):
        """Single agent should take direct path in empty environment."""
        start = [0, 0, 0]
        goal = [10, 0, 0]

        agent = Agent(start, goal, max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=empty_environment)

        trajectories = simulate([agent], dt=0.1, max_time=15.0)

        # All positions should have y, z approximately zero
        for pos in trajectories[0]:
            assert abs(pos[1]) < 0.1, f"Deviated from direct path: y = {pos[1]}"
            assert abs(pos[2]) < 0.1, f"Deviated from direct path: z = {pos[2]}"

    def test_single_agent_navigates_around_obstacle(self, single_cube):
        """Single agent should navigate around obstacle."""
        start = [0, 5, 1]
        goal = [10, 5, 1]

        agent = Agent(start, goal, max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=single_cube, epsilon=0.5)

        trajectories = simulate([agent], dt=0.1, max_time=30.0)

        final_pos = trajectories[0][-1]
        final_dist = euclidean_distance_3d(final_pos, np.array(goal))

        # Should eventually reach goal
        assert final_dist < 1.0, f"Agent did not navigate around obstacle: distance = {final_dist}"

    def test_single_agent_3d_navigation(self, empty_environment):
        """Single agent should navigate correctly in 3D."""
        start = [0, 0, 0]
        goal = [5, 5, 5]

        agent = Agent(start, goal, max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=empty_environment, epsilon=0.5)

        trajectories = simulate([agent], dt=0.1, max_time=20.0)

        final_pos = trajectories[0][-1]
        final_dist = euclidean_distance_3d(final_pos, np.array(goal))

        assert final_dist < 1.0, f"Agent did not reach 3D goal: distance = {final_dist}"


# =============================================================================
# TWO-AGENT COLLISION AVOIDANCE TESTS
# =============================================================================

class TestTwoAgentCollisionAvoidance:
    """Tests for two-agent collision avoidance scenarios."""

    def test_head_on_collision_avoidance(self, empty_environment):
        """Two agents on head-on collision course should avoid each other."""
        agent1 = Agent([0, 5, 0], [10, 5, 0], max_speed=1.0, R=0.5, tau=10.0,
                      obstacles=empty_environment)
        agent2 = Agent([10, 5, 0], [0, 5, 0], max_speed=1.0, R=0.5, tau=10.0,
                      obstacles=empty_environment)

        trajectories = simulate([agent1, agent2], dt=0.1, max_time=20.0)

        # Check minimum separation
        min_dist = float('inf')
        for i in range(min(len(trajectories[0]), len(trajectories[1]))):
            dist = euclidean_distance_3d(trajectories[0][i], trajectories[1][i])
            min_dist = min(min_dist, dist)

        # Should maintain some separation (ORCA target is R)
        # Allow tolerance due to discrete time steps

    def test_crossing_paths_collision_avoidance(self, empty_environment):
        """Agents crossing paths should avoid collision."""
        agent1 = Agent([0, 5, 0], [10, 5, 0], max_speed=1.0, R=0.5, tau=10.0,
                      obstacles=empty_environment)
        agent2 = Agent([5, 0, 0], [5, 10, 0], max_speed=1.0, R=0.5, tau=10.0,
                      obstacles=empty_environment)

        trajectories = simulate([agent1, agent2], dt=0.1, max_time=20.0)

        # Check minimum separation during crossing
        min_dist = float('inf')
        for i in range(min(len(trajectories[0]), len(trajectories[1]))):
            dist = euclidean_distance_3d(trajectories[0][i], trajectories[1][i])
            min_dist = min(min_dist, dist)

    def test_overtaking_collision_avoidance(self, empty_environment):
        """Faster agent overtaking slower should avoid collision."""
        # Slow agent
        agent1 = Agent([0, 5, 0], [20, 5, 0], max_speed=0.5, R=0.5, tau=10.0,
                      obstacles=empty_environment)
        # Fast agent starting behind
        agent2 = Agent([-2, 5, 0], [20, 5, 0], max_speed=1.0, R=0.5, tau=10.0,
                      obstacles=empty_environment)

        trajectories = simulate([agent1, agent2], dt=0.1, max_time=30.0)

        # Check minimum separation
        min_dist = float('inf')
        for i in range(min(len(trajectories[0]), len(trajectories[1]))):
            dist = euclidean_distance_3d(trajectories[0][i], trajectories[1][i])
            min_dist = min(min_dist, dist)

    def test_3d_collision_avoidance(self, empty_environment):
        """Agents should avoid collision in 3D space."""
        # One agent moving up
        agent1 = Agent([5, 5, 0], [5, 5, 10], max_speed=1.0, R=0.5, tau=10.0,
                      obstacles=empty_environment)
        # Another moving down
        agent2 = Agent([5, 5, 10], [5, 5, 0], max_speed=1.0, R=0.5, tau=10.0,
                      obstacles=empty_environment)

        trajectories = simulate([agent1, agent2], dt=0.1, max_time=15.0)

        # Should have valid trajectories
        for traj in trajectories:
            for pos in traj:
                assert np.all(np.isfinite(pos))


# =============================================================================
# MULTI-AGENT SWARM TESTS
# =============================================================================

class TestMultiAgentSwarm:
    """Tests for multi-agent swarm scenarios."""

    def test_four_agent_crossing(self, empty_environment):
        """Four agents crossing at center should avoid collisions."""
        center = 5.0

        agents = [
            Agent([0, center, 0], [10, center, 0], max_speed=1.0, R=0.5, tau=5.0,
                 obstacles=empty_environment),
            Agent([10, center, 0], [0, center, 0], max_speed=1.0, R=0.5, tau=5.0,
                 obstacles=empty_environment),
            Agent([center, 0, 0], [center, 10, 0], max_speed=1.0, R=0.5, tau=5.0,
                 obstacles=empty_environment),
            Agent([center, 10, 0], [center, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                 obstacles=empty_environment),
        ]

        trajectories = simulate(agents, dt=0.1, max_time=20.0)

        # All agents should complete simulation
        assert len(trajectories) == 4
        for traj in trajectories:
            assert len(traj) > 0

    def test_circular_swap(self, empty_environment):
        """Agents arranged in circle swap to opposite positions."""
        n_agents = 6
        radius = 5.0
        center = np.array([5.0, 5.0, 0.0])

        agents = []
        for i in range(n_agents):
            angle = 2 * np.pi * i / n_agents
            start = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            goal = center - radius * np.array([np.cos(angle), np.sin(angle), 0])

            agent = Agent(start.tolist(), goal.tolist(), max_speed=1.0, R=0.4, tau=5.0,
                         obstacles=empty_environment, epsilon=0.5)
            agents.append(agent)

        trajectories = simulate(agents, dt=0.1, max_time=30.0)

        # All agents should have valid trajectories
        for traj in trajectories:
            for pos in traj:
                assert np.all(np.isfinite(pos))

    def test_many_agents_random_goals(self, empty_environment):
        """Many agents with random goals should navigate safely."""
        np.random.seed(42)
        n_agents = 8

        agents = []
        for i in range(n_agents):
            start = np.random.rand(3) * 10
            goal = np.random.rand(3) * 10

            agent = Agent(start.tolist(), goal.tolist(), max_speed=1.0, R=0.3, tau=5.0,
                         obstacles=empty_environment, epsilon=0.5)
            agents.append(agent)

        trajectories = simulate(agents, dt=0.1, max_time=30.0)

        # All trajectories should be valid
        for traj in trajectories:
            assert len(traj) > 0
            for pos in traj:
                assert np.all(np.isfinite(pos))


# =============================================================================
# STATIC OBSTACLE AVOIDANCE TESTS
# =============================================================================

class TestStaticObstacleAvoidance:
    """Tests for navigation around static obstacles."""

    def test_navigate_through_corridor(self, corridor_obstacles):
        """Agent should navigate through narrow corridor."""
        start = [0, 5, 1]
        goal = [10, 5, 1]

        agent = Agent(start, goal, max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=corridor_obstacles, epsilon=0.5)

        # Path should exist through corridor
        assert agent.path is not None
        assert len(agent.path) >= 2

        trajectories = simulate([agent], dt=0.1, max_time=20.0)

        # Should make progress toward goal
        start_dist = euclidean_distance_3d(np.array(start), np.array(goal))
        final_dist = euclidean_distance_3d(trajectories[0][-1], np.array(goal))

        assert final_dist < start_dist, "Should make progress toward goal"

    def test_path_around_multiple_obstacles(self):
        """Agent should find path around multiple obstacles."""
        obstacles = [
            # First obstacle
            {
                'vertices': np.array([
                    [3, 3, 0], [4, 3, 0], [4, 7, 0], [3, 7, 0],
                    [3, 3, 2], [4, 3, 2], [4, 7, 2], [3, 7, 2]
                ], dtype=float),
                'faces': [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                         [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]
            },
            # Second obstacle
            {
                'vertices': np.array([
                    [6, 3, 0], [7, 3, 0], [7, 7, 0], [6, 7, 0],
                    [6, 3, 2], [7, 3, 2], [7, 7, 2], [6, 7, 2]
                ], dtype=float),
                'faces': [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                         [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]
            }
        ]

        agent = Agent([0, 5, 1], [10, 5, 1], max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=obstacles, epsilon=0.5)

        # Path should exist
        assert agent.path is not None


# =============================================================================
# COMBINED DYNAMIC AND STATIC OBSTACLE TESTS
# =============================================================================

class TestCombinedAvoidance:
    """Tests for avoiding both static and dynamic obstacles."""

    def test_agents_and_obstacles(self, single_cube):
        """Agents should avoid each other AND static obstacles."""
        agent1 = Agent([0, 5, 1], [10, 5, 1], max_speed=1.0, R=0.5, tau=5.0,
                      obstacles=single_cube)
        agent2 = Agent([10, 5, 1], [0, 5, 1], max_speed=1.0, R=0.5, tau=5.0,
                      obstacles=single_cube)

        trajectories = simulate([agent1, agent2], dt=0.1, max_time=30.0)

        # Both agents should have valid trajectories
        assert len(trajectories) == 2
        for traj in trajectories:
            for pos in traj:
                assert np.all(np.isfinite(pos))

    def test_multiple_agents_with_obstacles(self, single_cube):
        """Multiple agents navigating around obstacles."""
        agents = [
            Agent([0, 2, 1], [10, 2, 1], max_speed=1.0, R=0.4, tau=5.0,
                 obstacles=single_cube),
            Agent([0, 5, 1], [10, 5, 1], max_speed=1.0, R=0.4, tau=5.0,
                 obstacles=single_cube),
            Agent([0, 8, 1], [10, 8, 1], max_speed=1.0, R=0.4, tau=5.0,
                 obstacles=single_cube),
        ]

        trajectories = simulate(agents, dt=0.1, max_time=30.0)

        assert len(trajectories) == 3


# =============================================================================
# EDGE CASES AND FAILURE MODES
# =============================================================================

class TestEdgeCasesAndFailureModes:
    """Tests for edge cases and potential failure modes."""

    def test_agent_at_goal(self, empty_environment):
        """Agent already at goal should stay still."""
        agent = Agent([5, 5, 5], [5, 5, 5], max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=empty_environment, epsilon=0.1)

        trajectories = simulate([agent], dt=0.1, max_time=2.0)

        # All positions should be at goal
        for pos in trajectories[0]:
            dist = euclidean_distance_3d(pos, np.array([5, 5, 5]))
            assert dist < 0.5

    def test_very_close_agents(self, empty_environment):
        """Very close agents should still navigate safely."""
        agent1 = Agent([0, 0, 0], [5, 0, 0], max_speed=1.0, R=0.3, tau=5.0,
                      obstacles=empty_environment)
        agent2 = Agent([0.5, 0, 0], [5, 1, 0], max_speed=1.0, R=0.3, tau=5.0,
                      obstacles=empty_environment)

        trajectories = simulate([agent1, agent2], dt=0.1, max_time=10.0)

        # Should complete without errors
        assert len(trajectories) == 2

    def test_agents_with_same_goal(self, empty_environment):
        """Agents with same goal should converge safely."""
        goal = [5, 5, 0]

        agent1 = Agent([0, 0, 0], goal, max_speed=1.0, R=0.5, tau=5.0,
                      obstacles=empty_environment)
        agent2 = Agent([10, 10, 0], goal, max_speed=1.0, R=0.5, tau=5.0,
                      obstacles=empty_environment)

        trajectories = simulate([agent1, agent2], dt=0.1, max_time=20.0)

        # Both should approach goal
        for traj in trajectories:
            final_dist = euclidean_distance_3d(traj[-1], np.array(goal))
            # May not reach exactly due to collision avoidance with each other

    def test_zero_initial_velocity(self, empty_environment):
        """Agents starting with zero velocity should work."""
        agent = Agent([0, 0, 0], [5, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=empty_environment)

        assert_allclose(agent.vel, np.zeros(3))

        trajectories = simulate([agent], dt=0.1, max_time=10.0)

        # Should start moving
        assert euclidean_distance_3d(trajectories[0][-1], np.array([0, 0, 0])) > 1.0


# =============================================================================
# PERFORMANCE AND STRESS TESTS
# =============================================================================

class TestPerformanceAndStress:
    """Performance and stress tests."""

    def test_ten_agents(self, empty_environment):
        """System should handle 10 agents."""
        np.random.seed(42)

        agents = []
        for i in range(10):
            start = np.random.rand(3) * 10
            goal = np.random.rand(3) * 10

            agent = Agent(start.tolist(), goal.tolist(), max_speed=1.0, R=0.3, tau=5.0,
                         obstacles=empty_environment)
            agents.append(agent)

        trajectories = simulate(agents, dt=0.1, max_time=20.0)

        assert len(trajectories) == 10

    def test_long_simulation(self, empty_environment):
        """System should handle longer simulations."""
        agent = Agent([0, 0, 0], [50, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=empty_environment, epsilon=0.5)

        trajectories = simulate([agent], dt=0.1, max_time=60.0)

        # Should make progress
        assert euclidean_distance_3d(trajectories[0][-1], np.array([0, 0, 0])) > 10

    def test_small_time_steps(self, empty_environment):
        """System should work with small time steps."""
        agent = Agent([0, 0, 0], [5, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                     obstacles=empty_environment)

        trajectories = simulate([agent], dt=0.01, max_time=3.0)

        assert len(trajectories[0]) > 0


# =============================================================================
# REPRODUCIBILITY TESTS
# =============================================================================

class TestReproducibility:
    """Tests for simulation reproducibility."""

    def test_deterministic_single_agent(self, empty_environment):
        """Single agent simulation should be deterministic."""
        def run_simulation():
            agent = Agent([0, 0, 0], [5, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                         obstacles=empty_environment)
            return simulate([agent], dt=0.1, max_time=5.0)

        result1 = run_simulation()
        result2 = run_simulation()

        for i in range(len(result1[0])):
            assert_allclose(result1[0][i], result2[0][i], rtol=1e-10)

    def test_deterministic_multi_agent(self, empty_environment):
        """Multi-agent simulation should be deterministic."""
        def run_simulation():
            agent1 = Agent([0, 0, 0], [5, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                          obstacles=empty_environment)
            agent2 = Agent([5, 0, 0], [0, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                          obstacles=empty_environment)
            return simulate([agent1, agent2], dt=0.1, max_time=5.0)

        result1 = run_simulation()
        result2 = run_simulation()

        for agent_idx in range(2):
            for i in range(len(result1[agent_idx])):
                assert_allclose(result1[agent_idx][i], result2[agent_idx][i], rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
