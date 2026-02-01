"""
Comprehensive Test Suite for utils.py

This module tests the geometric utility functions that form the mathematical
foundation for the ORCA-A* 3D algorithm. Following the Yoneda philosophy,
these tests serve as a complete behavioral specification.

Key Functions Tested:
- euclidean_distance_3d: 3D Euclidean distance computation
- normalize: Vector normalization to unit length
- line_obstacle_intersect_3d: 3D line-segment intersection with polyhedral obstacles

Mathematical Invariants:
- Distance is always non-negative
- Distance is symmetric: d(a,b) = d(b,a)
- Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
- Normalized vectors have unit length (or zero for zero vectors)
- Ray-triangle intersection uses Moller-Trumbore algorithm
"""

import sys
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import euclidean_distance_3d, normalize, line_obstacle_intersect_3d


# =============================================================================
# EUCLIDEAN DISTANCE 3D TESTS
# =============================================================================

class TestEuclideanDistance3D:
    """
    Tests for euclidean_distance_3d function.

    Mathematical Definition:
    d(p1, p2) = ||p1 - p2||_2 = sqrt(sum((p1[i] - p2[i])^2 for i in [0,1,2]))
    """

    # -------------------------------------------------------------------------
    # Basic Correctness Tests
    # -------------------------------------------------------------------------

    def test_distance_between_same_point_is_zero(self):
        """Distance from a point to itself should be exactly zero."""
        p = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance_3d(p, p) == 0.0

    def test_distance_between_origin_points(self):
        """Distance from origin to origin should be zero."""
        origin = np.array([0.0, 0.0, 0.0])
        assert euclidean_distance_3d(origin, origin) == 0.0

    def test_distance_along_x_axis(self):
        """Distance along x-axis should equal absolute difference."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([5.0, 0.0, 0.0])
        assert euclidean_distance_3d(p1, p2) == 5.0

    def test_distance_along_y_axis(self):
        """Distance along y-axis should equal absolute difference."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 7.0, 0.0])
        assert euclidean_distance_3d(p1, p2) == 7.0

    def test_distance_along_z_axis(self):
        """Distance along z-axis should equal absolute difference."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 3.0])
        assert euclidean_distance_3d(p1, p2) == 3.0

    def test_distance_3d_diagonal(self):
        """Test 3D diagonal distance (unit cube diagonal)."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 1.0])
        expected = np.sqrt(3.0)
        assert_allclose(euclidean_distance_3d(p1, p2), expected, rtol=1e-10)

    def test_distance_3_4_5_triangle(self):
        """Test using 3-4-5 Pythagorean triple in 2D plane (z=0)."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        assert euclidean_distance_3d(p1, p2) == 5.0

    def test_distance_3d_pythagorean_triple(self):
        """Test 3D using 3^2 + 4^2 + 12^2 = 169 = 13^2."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 12.0])
        assert_allclose(euclidean_distance_3d(p1, p2), 13.0, rtol=1e-10)

    # -------------------------------------------------------------------------
    # Mathematical Properties Tests
    # -------------------------------------------------------------------------

    def test_distance_is_symmetric(self):
        """Distance should be symmetric: d(a,b) = d(b,a)."""
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])
        assert euclidean_distance_3d(p1, p2) == euclidean_distance_3d(p2, p1)

    def test_distance_is_non_negative(self):
        """Distance should always be non-negative."""
        # Test with various point combinations
        points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, -2.0, 3.0]),
            np.array([-5.0, -5.0, -5.0]),
            np.array([100.0, 200.0, 300.0]),
        ]
        for p1 in points:
            for p2 in points:
                dist = euclidean_distance_3d(p1, p2)
                assert dist >= 0.0, f"Distance should be >= 0, got {dist}"

    def test_triangle_inequality(self):
        """Triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([2.0, 0.0, 0.0])

        d_ac = euclidean_distance_3d(a, c)
        d_ab = euclidean_distance_3d(a, b)
        d_bc = euclidean_distance_3d(b, c)

        assert d_ac <= d_ab + d_bc + 1e-10  # Allow small numerical tolerance

    def test_triangle_inequality_non_collinear(self):
        """Triangle inequality with non-collinear points."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0])
        c = np.array([2.0, 0.0, 1.0])

        d_ac = euclidean_distance_3d(a, c)
        d_ab = euclidean_distance_3d(a, b)
        d_bc = euclidean_distance_3d(b, c)

        # Strict inequality for non-collinear points
        assert d_ac < d_ab + d_bc

    # -------------------------------------------------------------------------
    # Edge Cases and Boundary Conditions
    # -------------------------------------------------------------------------

    def test_distance_with_negative_coordinates(self):
        """Distance with negative coordinates should work correctly."""
        p1 = np.array([-1.0, -2.0, -3.0])
        p2 = np.array([-4.0, -5.0, -6.0])
        expected = np.sqrt(3**2 + 3**2 + 3**2)
        assert_allclose(euclidean_distance_3d(p1, p2), expected, rtol=1e-10)

    def test_distance_with_very_small_values(self):
        """Distance with very small values should maintain precision."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1e-10, 1e-10, 1e-10])
        expected = np.sqrt(3 * 1e-20)
        assert_allclose(euclidean_distance_3d(p1, p2), expected, rtol=1e-5)

    def test_distance_with_very_large_values(self):
        """Distance with very large values should not overflow."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1e6, 1e6, 1e6])
        expected = np.sqrt(3) * 1e6
        assert_allclose(euclidean_distance_3d(p1, p2), expected, rtol=1e-10)

    def test_distance_with_mixed_sign_coordinates(self):
        """Distance with mixed positive and negative coordinates."""
        p1 = np.array([1.0, -2.0, 3.0])
        p2 = np.array([-4.0, 5.0, -6.0])
        expected = np.sqrt(5**2 + 7**2 + 9**2)
        assert_allclose(euclidean_distance_3d(p1, p2), expected, rtol=1e-10)


# =============================================================================
# NORMALIZE TESTS
# =============================================================================

class TestNormalize:
    """
    Tests for normalize function.

    Mathematical Definition:
    normalize(v) = v / ||v||_2 if ||v||_2 > epsilon, else zero vector

    Properties:
    - ||normalize(v)||_2 = 1 for non-zero v
    - normalize(v) is parallel to v (same direction)
    - normalize(zero) = zero (or similar small vector)
    """

    # -------------------------------------------------------------------------
    # Basic Correctness Tests
    # -------------------------------------------------------------------------

    def test_normalize_unit_x_vector(self):
        """Normalizing a unit vector should return the same vector."""
        v = np.array([1.0, 0.0, 0.0])
        result = normalize(v)
        assert_array_almost_equal(result, v)

    def test_normalize_unit_y_vector(self):
        """Normalizing unit y vector."""
        v = np.array([0.0, 1.0, 0.0])
        result = normalize(v)
        assert_array_almost_equal(result, v)

    def test_normalize_unit_z_vector(self):
        """Normalizing unit z vector."""
        v = np.array([0.0, 0.0, 1.0])
        result = normalize(v)
        assert_array_almost_equal(result, v)

    def test_normalize_scaled_vector(self):
        """Normalizing a scaled vector should give unit vector in same direction."""
        v = np.array([5.0, 0.0, 0.0])
        result = normalize(v)
        expected = np.array([1.0, 0.0, 0.0])
        assert_array_almost_equal(result, expected)

    def test_normalize_diagonal_vector(self):
        """Normalizing diagonal vector (1,1,1)."""
        v = np.array([1.0, 1.0, 1.0])
        result = normalize(v)
        expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        assert_array_almost_equal(result, expected)

    def test_normalize_arbitrary_vector(self):
        """Normalizing arbitrary 3D vector."""
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        expected = np.array([0.6, 0.8, 0.0])
        assert_array_almost_equal(result, expected)

    # -------------------------------------------------------------------------
    # Mathematical Properties Tests
    # -------------------------------------------------------------------------

    def test_normalized_vector_has_unit_length(self):
        """Normalized non-zero vectors should have unit length."""
        test_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([3.0, 4.0, 0.0]),
            np.array([-1.0, 2.0, -3.0]),
            np.array([100.0, 200.0, 300.0]),
        ]
        for v in test_vectors:
            result = normalize(v)
            length = np.linalg.norm(result)
            assert_allclose(length, 1.0, rtol=1e-10,
                           err_msg=f"normalize({v}) has length {length}, expected 1.0")

    def test_normalized_vector_same_direction(self):
        """Normalized vector should be in the same direction as original."""
        v = np.array([3.0, 4.0, 5.0])
        result = normalize(v)
        # Dot product of parallel vectors is positive
        dot = np.dot(v, result)
        assert dot > 0, "Normalized vector should point in same direction"
        # Cross product should be zero (parallel vectors)
        cross = np.cross(v, result)
        assert_allclose(np.linalg.norm(cross), 0.0, atol=1e-10)

    # -------------------------------------------------------------------------
    # Edge Cases and Boundary Conditions
    # -------------------------------------------------------------------------

    def test_normalize_zero_vector_returns_zero(self):
        """Normalizing zero vector should return zero (avoid division by zero)."""
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        assert_array_almost_equal(result, v)

    def test_normalize_near_zero_vector(self):
        """Normalizing near-zero vector should handle gracefully."""
        v = np.array([1e-8, 0.0, 0.0])
        result = normalize(v)
        # Should either be unit or zero depending on threshold
        length = np.linalg.norm(result)
        assert length == 0.0 or abs(length - 1.0) < 1e-5

    def test_normalize_negative_vector(self):
        """Normalizing vector with negative components."""
        v = np.array([-3.0, -4.0, 0.0])
        result = normalize(v)
        expected = np.array([-0.6, -0.8, 0.0])
        assert_array_almost_equal(result, expected)

    def test_normalize_very_large_vector(self):
        """Normalizing very large vector should not cause overflow."""
        v = np.array([1e10, 1e10, 1e10])
        result = normalize(v)
        assert_allclose(np.linalg.norm(result), 1.0, rtol=1e-10)


# =============================================================================
# LINE OBSTACLE INTERSECT 3D TESTS
# =============================================================================

class TestLineObstacleIntersect3D:
    """
    Tests for line_obstacle_intersect_3d function.

    This function uses the Moller-Trumbore algorithm to detect if a line segment
    intersects the interior of a polyhedral obstacle.

    Mathematical Definition:
    - Line parameterization: p(t) = p1 + t*(p2-p1), t in (0,1)
    - Uses ray-triangle intersection for each triangulated face
    - Returns True if segment strictly intersects any face interior
    """

    @pytest.fixture
    def unit_cube_obstacle(self):
        """A unit cube centered at origin (vertices from 0 to 1)."""
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
    def large_cube_obstacle(self):
        """A larger cube from (2,2,0) to (3,3,1)."""
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

    # -------------------------------------------------------------------------
    # Non-Intersection Tests
    # -------------------------------------------------------------------------

    def test_line_completely_outside_cube(self, unit_cube_obstacle):
        """Line segment entirely outside obstacle should not intersect."""
        p1 = np.array([5.0, 5.0, 5.0])
        p2 = np.array([6.0, 6.0, 6.0])
        assert not line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)

    def test_line_parallel_to_cube_face(self, unit_cube_obstacle):
        """Line parallel to but outside cube face should not intersect."""
        p1 = np.array([0.5, 2.0, 0.5])  # Outside cube
        p2 = np.array([0.5, 3.0, 0.5])
        assert not line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)

    def test_line_along_cube_edge(self, unit_cube_obstacle):
        """Line along cube edge should not intersect interior."""
        p1 = np.array([0.0, 0.0, 0.0])  # Corner
        p2 = np.array([1.0, 0.0, 0.0])  # Adjacent corner
        assert not line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)

    def test_line_touching_corner_only(self, unit_cube_obstacle):
        """Line touching only a corner should not intersect interior."""
        p1 = np.array([-1.0, -1.0, -1.0])
        p2 = np.array([0.0, 0.0, 0.0])  # Corner of cube
        # This is boundary, not interior
        assert not line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)

    def test_empty_obstacles_no_intersection(self):
        """Empty obstacle list should never intersect."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 1.0])
        assert not line_obstacle_intersect_3d(p1, p2, [])

    # -------------------------------------------------------------------------
    # Intersection Tests
    # -------------------------------------------------------------------------

    def test_line_through_cube_center(self, unit_cube_obstacle):
        """Line passing through cube center should intersect.

        BUG DETECTED: The line_obstacle_intersect_3d function fails to detect
        intersection when a line passes through the interior of a polyhedral
        obstacle. The current implementation only checks for ray-triangle
        intersection with the faces, but the face winding order or the
        triangulation may be causing misses.

        Expected: True (line passes through cube)
        Actual: False (intersection not detected)
        """
        p1 = np.array([-1.0, 0.5, 0.5])
        p2 = np.array([2.0, 0.5, 0.5])
        result = line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)
        assert result == True

    def test_line_through_cube_diagonal(self, unit_cube_obstacle):
        """Line through cube along diagonal should intersect.

        BUG DETECTED: Same issue as test_line_through_cube_center.
        The implementation fails to detect diagonal lines through the cube.

        Expected: True
        Actual: False
        """
        p1 = np.array([-1.0, -1.0, -1.0])
        p2 = np.array([2.0, 2.0, 2.0])
        result = line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)
        assert result == True

    def test_line_enters_and_exits_cube(self, unit_cube_obstacle):
        """Line that enters and exits cube should intersect.

        BUG DETECTED: Same issue - line through obstacle not detected.

        Expected: True
        Actual: False
        """
        p1 = np.array([0.5, -1.0, 0.5])  # Below cube
        p2 = np.array([0.5, 2.0, 0.5])   # Above cube
        result = line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)
        assert result == True

    # -------------------------------------------------------------------------
    # Edge Cases and Boundary Conditions
    # -------------------------------------------------------------------------

    def test_line_starts_on_face_goes_outside(self, unit_cube_obstacle):
        """Line starting on face and going outward should not intersect interior."""
        p1 = np.array([0.5, 0.0, 0.5])  # On front face
        p2 = np.array([0.5, -1.0, 0.5])  # Going away
        assert not line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)

    def test_zero_length_line_outside(self, unit_cube_obstacle):
        """Zero-length line (point) outside should not intersect."""
        p = np.array([5.0, 5.0, 5.0])
        assert not line_obstacle_intersect_3d(p, p, unit_cube_obstacle)

    def test_line_grazing_cube_face(self, unit_cube_obstacle):
        """Line grazing cube face at one point should not intersect interior."""
        p1 = np.array([0.5, 0.0, -1.0])  # Approach from below
        p2 = np.array([0.5, 0.0, 2.0])   # Exit above
        # This line touches the z=0 face at exactly (0.5, 0.0, 0.0)
        # Depending on implementation, this may or may not count
        # The implementation uses strict interior check
        result = line_obstacle_intersect_3d(p1, p2, unit_cube_obstacle)
        # This is a boundary case - the result depends on implementation

    # -------------------------------------------------------------------------
    # Multiple Obstacles Tests
    # -------------------------------------------------------------------------

    def test_line_misses_first_hits_second_obstacle(self):
        """Line that misses first obstacle but hits second should intersect.

        BUG DETECTED: Same underlying issue with line_obstacle_intersect_3d.
        Lines through obstacle interiors are not detected.

        Expected: True
        Actual: False
        """
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
                    [5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0],
                    [5, 0, 1], [6, 0, 1], [6, 1, 1], [5, 1, 1]
                ], dtype=float),
                'faces': [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                         [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]
            }
        ]
        # Line from (3, 0.5, 0.5) to (7, 0.5, 0.5) misses first, hits second
        p1 = np.array([3.0, 0.5, 0.5])
        p2 = np.array([7.0, 0.5, 0.5])
        result = line_obstacle_intersect_3d(p1, p2, obstacles)
        assert result == True

    def test_line_hits_both_obstacles(self):
        """Line that hits both obstacles should still return True.

        BUG DETECTED: Same underlying issue.

        Expected: True
        Actual: False
        """
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
                    [3, 0, 0], [4, 0, 0], [4, 1, 0], [3, 1, 0],
                    [3, 0, 1], [4, 0, 1], [4, 1, 1], [3, 1, 1]
                ], dtype=float),
                'faces': [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
                         [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]
            }
        ]
        # Line from (-1, 0.5, 0.5) to (5, 0.5, 0.5) hits both
        p1 = np.array([-1.0, 0.5, 0.5])
        p2 = np.array([5.0, 0.5, 0.5])
        result = line_obstacle_intersect_3d(p1, p2, obstacles)
        assert result == True


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================

class TestUtilsProperties:
    """Property-based tests for utility functions."""

    def test_distance_symmetry_random(self):
        """Distance symmetry with random points."""
        np.random.seed(42)
        for _ in range(100):
            p1 = np.random.randn(3) * 100
            p2 = np.random.randn(3) * 100
            assert_allclose(
                euclidean_distance_3d(p1, p2),
                euclidean_distance_3d(p2, p1),
                rtol=1e-10
            )

    def test_normalize_idempotent_for_unit_vectors(self):
        """Normalizing a normalized vector should return the same vector."""
        np.random.seed(42)
        for _ in range(100):
            v = np.random.randn(3)
            if np.linalg.norm(v) > 1e-6:
                n1 = normalize(v)
                n2 = normalize(n1)
                assert_array_almost_equal(n1, n2)

    def test_distance_matches_numpy_norm(self):
        """Our distance function should match numpy's norm."""
        np.random.seed(42)
        for _ in range(100):
            p1 = np.random.randn(3) * 100
            p2 = np.random.randn(3) * 100
            our_dist = euclidean_distance_3d(p1, p2)
            np_dist = np.linalg.norm(p1 - p2)
            assert_allclose(our_dist, np_dist, rtol=1e-10)


# =============================================================================
# NUMERICAL PRECISION TESTS
# =============================================================================

class TestNumericalPrecision:
    """Tests for numerical precision and stability."""

    def test_distance_precision_small_differences(self):
        """Distance with very small differences should be accurate."""
        p1 = np.array([1.0, 1.0, 1.0])
        p2 = np.array([1.0 + 1e-10, 1.0, 1.0])
        dist = euclidean_distance_3d(p1, p2)
        assert_allclose(dist, 1e-10, rtol=1e-5)

    def test_normalize_preserves_direction_large_scale(self):
        """Normalization should preserve direction regardless of scale."""
        v_small = np.array([1e-5, 2e-5, 3e-5])
        v_large = np.array([1e5, 2e5, 3e5])

        n_small = normalize(v_small)
        n_large = normalize(v_large)

        # Should be the same direction
        if np.linalg.norm(n_small) > 0.5:  # Not zeroed out
            assert_allclose(n_small, n_large, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
