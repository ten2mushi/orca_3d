import numpy as np

def euclidean_distance_3d(p1, p2):
    return np.linalg.norm(p1 - p2)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else np.zeros_like(v)

def line_obstacle_intersect_3d(p1, p2, obstacles):
    """
    Checks if line segment [p1, p2] intersects the interior of any obstacle.

    Mathematical:
    - Parameterize line: p(t) = p1 + t (p2 - p1), t in (0,1).
    - For each face (triangulated), use Moller-Trumbore to find t, u, v.
    - Intersect interior if 0 < t < 1 and point is on or inside triangle.
    """
    EPS = 1e-6
    dir_vec = p2 - p1
    for ob in obstacles:
        for face in ob['faces']:
            # Triangulate polygon (fan)
            for i in range(len(face) - 2):
                v0 = ob['vertices'][face[0]]
                v1 = ob['vertices'][face[i + 1]]
                v2 = ob['vertices'][face[i + 2]]
                # Moller-Trumbore
                edge1 = v1 - v0
                edge2 = v2 - v0
                h = np.cross(dir_vec, edge2)
                a = np.dot(edge1, h)
                if abs(a) < EPS:
                    continue
                f = 1.0 / a
                s = p1 - v0
                u = f * np.dot(s, h)
                if u < -EPS or u > 1 + EPS:
                    continue
                q = np.cross(s, edge1)
                v = f * np.dot(dir_vec, q)
                if v < -EPS or u + v > 1 + EPS:
                    continue
                t = f * np.dot(edge2, q)
                # t must be strictly interior (not at endpoints)
                # u,v can be on edge (>= -EPS) as long as point is in triangle
                if t > EPS and t < 1 - EPS and u >= -EPS and v >= -EPS and u + v <= 1 + EPS:
                    return True
    return False