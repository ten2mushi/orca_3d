import heapq
from utils import euclidean_distance_3d

def a_star_search(vertices, edges, start_idx, goal_idx):
    """
    Computes the shortest path from start to goal using A* on the visibility graph.

    Inputs:
    - vertices: np.array shape (num_vertices, 3).
    - edges: dict of lists (neighbor_idx, cost).
    - start_idx: int.
    - goal_idx: int.

    Outputs:
    - path: np.array shape (num_waypoints, 3) of waypoints if found, else None.
    - path_indices: list of int indices if found, else None.

    Mathematical Responsibilities:
    - g(v): min path cost from start to v, g(start) = 0, others inf.
    - h(v) = ||v - goal||_2.
    - f(v) = g(v) + h(v).
    - Priority queue for min f.
    - Update if g_new = g(u) + k(u, v) < g(v).
    - Reconstruct via parents.
    """
    # Handle trivial case: already at goal
    if start_idx == goal_idx:
        return vertices[[start_idx]], [start_idx]

    n = len(vertices)
    g = [float('inf')] * n
    g[start_idx] = 0
    parent = [-1] * n
    pq = []
    heapq.heappush(pq, (euclidean_distance_3d(vertices[start_idx], vertices[goal_idx]), start_idx))
    visited = [False] * n

    while pq:
        f_curr, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        if u == goal_idx:
            break
        for v, cost in edges.get(u, []):
            g_new = g[u] + cost
            if g_new < g[v]:
                g[v] = g_new
                parent[v] = u
                h = euclidean_distance_3d(vertices[v], vertices[goal_idx])
                heapq.heappush(pq, (g_new + h, v))

    if parent[goal_idx] == -1:
        return None, None

    path_indices = []
    curr = goal_idx
    while curr != -1:
        path_indices.append(curr)
        curr = parent[curr]
    path_indices.reverse()
    path = vertices[path_indices]

    return path, path_indices