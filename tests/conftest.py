"""
pytest configuration and shared fixtures for ORCA-A* 3D tests.

This file provides common fixtures and configuration used across all test modules.
"""

import sys
import os
import pytest
import numpy as np

# Add src directory to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# COMMON OBSTACLE FIXTURES
# =============================================================================

@pytest.fixture
def empty_obstacles():
    """Empty environment with no obstacles."""
    return []


@pytest.fixture
def unit_cube():
    """Unit cube from (0,0,0) to (1,1,1)."""
    return [{
        'vertices': np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float),
        'faces': [
            [0, 1, 2, 3], [4, 7, 6, 5],
            [0, 4, 5, 1], [2, 6, 7, 3],
            [0, 3, 7, 4], [1, 5, 6, 2]
        ]
    }]


@pytest.fixture
def offset_cube():
    """Cube from (2,2,0) to (3,3,1)."""
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
def two_cubes():
    """Two separated cube obstacles."""
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


@pytest.fixture
def wall_obstacle():
    """A wall blocking direct path."""
    return [{
        'vertices': np.array([
            [4, 0, 0], [5, 0, 0], [5, 10, 0], [4, 10, 0],
            [4, 0, 3], [5, 0, 3], [5, 10, 3], [4, 10, 3]
        ], dtype=float),
        'faces': [
            [0, 1, 2, 3], [4, 7, 6, 5],
            [0, 4, 5, 1], [2, 6, 7, 3],
            [0, 3, 7, 4], [1, 5, 6, 2]
        ]
    }]


# =============================================================================
# COMMON GRAPH FIXTURES
# =============================================================================

@pytest.fixture
def simple_graph():
    """Simple 5-node graph for A* testing."""
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [2.0, 0.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [2.0, 2.0, 0.0],  # 4
    ])

    edges = {
        0: [(1, 1.0)],
        1: [(0, 1.0), (2, 1.0), (3, 1.0)],
        2: [(1, 1.0)],
        3: [(1, 1.0), (4, np.sqrt(2.0))],
        4: [(3, np.sqrt(2.0))],
    }

    return vertices, edges


@pytest.fixture
def grid_graph_3x3():
    """3x3 grid graph."""
    from utils import euclidean_distance_3d

    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0],
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


# =============================================================================
# AGENT FIXTURES
# =============================================================================

@pytest.fixture
def simple_agent(empty_obstacles):
    """Simple agent moving from origin to (5,0,0)."""
    from hybrid_orca_astar import Agent
    return Agent([0, 0, 0], [5, 0, 0], max_speed=1.0, R=0.5, tau=5.0,
                obstacles=empty_obstacles)


@pytest.fixture
def head_on_agents(empty_obstacles):
    """Two agents on head-on collision course."""
    from hybrid_orca_astar import Agent
    agent1 = Agent([0, 0, 0], [10, 0, 0], max_speed=1.0, R=0.5, tau=10.0,
                  obstacles=empty_obstacles)
    agent2 = Agent([10, 0, 0], [0, 0, 0], max_speed=1.0, R=0.5, tau=10.0,
                  obstacles=empty_obstacles)
    return [agent1, agent2]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_path_length(path):
    """Compute total length of a path through waypoints."""
    if path is None or len(path) < 2:
        return 0.0
    from utils import euclidean_distance_3d
    total = 0.0
    for i in range(len(path) - 1):
        total += euclidean_distance_3d(path[i], path[i + 1])
    return total


def check_collision(pos1, pos2, R):
    """Check if two positions are in collision (distance < R)."""
    from utils import euclidean_distance_3d
    return euclidean_distance_3d(pos1, pos2) < R


def find_min_separation(trajectory1, trajectory2):
    """Find minimum separation between two trajectories over time."""
    from utils import euclidean_distance_3d
    min_dist = float('inf')
    for i in range(min(len(trajectory1), len(trajectory2))):
        dist = euclidean_distance_3d(trajectory1[i], trajectory2[i])
        min_dist = min(min_dist, dist)
    return min_dist


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# =============================================================================
# AUTOUSE FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
