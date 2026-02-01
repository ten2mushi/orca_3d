import numpy as np
from utils import euclidean_distance_3d, line_obstacle_intersect_3d

def build_visibility_graph(obstacles, start, goal):
    """
    Constructs the 3D visibility graph for path planning around static polyhedral obstacles.

    Inputs:
    - obstacles: List of dicts, each with 'vertices' (np.array shape (num_vertices, 3)) and 'faces' (list of lists of vertex indices).
    - start: np.array shape (3,) for agent start position.
    - goal: np.array shape (3,) for agent goal position.

    Outputs:
    - vertices: np.array shape (num_vertices, 3) of all unique vertices including start and goal.
    - edges: dict where keys are vertex indices, values are lists of (neighbor_idx, cost) tuples.
    - start_idx: int index of start in vertices.
    - goal_idx: int index of goal in vertices.

    Mathematical Responsibilities:
    - Vertices \mathcal{V} = union of obstacle vertices union {start, goal}, using tuple hashing for uniqueness.
    - Edges \mathcal{E}: For each pair u, v, add if [u, v] does not intersect interior of any O_m (using line_obstacle_intersect_3d).
    - Cost k(u, v) = ||u - v||_2.
    """
    vertices = []
    vertex_map = {}
    idx = 0

    # Add obstacle vertices
    for ob in obstacles:
        for v in ob['vertices']:
            vt = tuple(v)
            if vt not in vertex_map:
                vertex_map[vt] = idx
                vertices.append(v)
                idx += 1

    # Add start
    st = tuple(start)
    if st not in vertex_map:
        vertex_map[st] = idx
        vertices.append(start)
        idx += 1

    # Add goal
    gt = tuple(goal)
    if gt not in vertex_map:
        vertex_map[gt] = idx
        vertices.append(goal)
        idx += 1

    vertices = np.array(vertices)

    # Build edges
    edges = {i: [] for i in range(len(vertices))}
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            u, v = vertices[i], vertices[j]
            if not line_obstacle_intersect_3d(u, v, obstacles):
                cost = euclidean_distance_3d(u, v)
                edges[i].append((j, cost))
                edges[j].append((i, cost))

    start_idx = vertex_map[st]
    goal_idx = vertex_map[gt]

    return vertices, edges, start_idx, goal_idx