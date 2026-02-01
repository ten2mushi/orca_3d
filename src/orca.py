import numpy as np
from scipy.optimize import minimize
from utils import normalize

def compute_vo_escape(p_rel, v_rel, R, tau):
    """
    Computes the minimal escape vector u and normal n_hat for escaping the 3D velocity obstacle.

    Mathematical Model:
    VO_rel^τ = {v_rel | ∃t ∈ [0, τ]: ||p_rel - t*v_rel||_2 < R}

    This forms a truncated cone in 3D velocity space:
    - Cone axis: -p_rel direction
    - Half-angle: arcsin(R / ||p_rel||)
    - Truncation: at distance ||p_rel||/τ + R/τ from origin

    Inputs:
    - p_rel: np.array (3,) relative position vector
    - v_rel: np.array (3,) relative velocity vector
    - R: float >0 collision radius
    - tau: float >0 time horizon

    Outputs:
    - u: np.array (3,) minimal escape vector or None
    - n_hat: np.array (3,) outward unit normal or None
    """
    d = np.linalg.norm(p_rel)
    if d <= R:
        # Already colliding - compute emergency escape vector to push apart
        if d < 1e-12:
            # Agents at same position - pick arbitrary escape direction
            escape_dir = np.array([1.0, 0.0, 0.0])
        else:
            # Push away from other agent (opposite direction of p_rel)
            escape_dir = -p_rel / d
        # Escape by enough to get outside collision radius plus margin
        escape_magnitude = (R - d) + 0.1 * R
        u = escape_magnitude * escape_dir
        n_hat = escape_dir
        return u, n_hat

    # Check if collision occurs within time horizon
    # Solve ||p_rel - t * v_rel||² = R² for t
    a = np.dot(v_rel, v_rel)
    b = -2 * np.dot(p_rel, v_rel)
    c = np.dot(p_rel, p_rel) - R**2
    disc = b**2 - 4 * a * c

    if disc < 0:
        return None, None
    if a < 1e-12:  # No relative motion
        return None, None

    t1 = (-b - np.sqrt(disc)) / (2 * a)
    t2 = (-b + np.sqrt(disc)) / (2 * a)

    # Get earliest positive collision time
    valid_times = [t for t in [t1, t2] if t >= 1e-12]
    if not valid_times:
        return None, None

    t_min = min(valid_times)
    if t_min > tau:
        return None, None

    # Velocity obstacle geometry
    # The cone points in the direction of p_rel (towards the other agent)
    # VO = {v_rel | ∃t ∈ [0, τ]: ||p_rel - t*v_rel|| < R}
    # This means we're looking for v_rel such that the agent moves towards collision
    cone_axis = p_rel / d  # Unit vector pointing towards the other agent
    cone_half_angle = np.arcsin(R / d)

    # Check if v_rel is inside the truncated cone
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm < 1e-12:
        # Zero relative velocity - escape in any direction
        # Choose arbitrary direction perpendicular to cone axis
        if abs(cone_axis[0]) < 0.9:
            perp = np.cross(cone_axis, np.array([1, 0, 0]))
        else:
            perp = np.cross(cone_axis, np.array([0, 1, 0]))
        perp = normalize(perp)

        # Minimal escape
        u = 0.01 * perp  # Small escape in perpendicular direction
        n_hat = normalize(u)
        return u, n_hat

    # Project v_rel onto cone axis
    v_axial = np.dot(v_rel, cone_axis)
    v_radial_vec = v_rel - v_axial * cone_axis
    v_radial = np.linalg.norm(v_radial_vec)

    # For VO: we need v_rel pointing towards the other agent (positive axial component)
    # and within the cone (radial component < axial * tan(half_angle))
    if v_axial <= 0:
        return None, None

    required_radial = v_axial * np.tan(cone_half_angle)
    if v_radial >= required_radial:
        # Outside cone mantle
        return None, None

    # Check truncation: for collision within tau, we need t_min <= tau (already checked above)
    # The geometric truncation is more complex, but since we already verified collision within tau,
    # we can proceed with the cone geometry
    max_speed_for_collision = d / tau + R / tau  # This is an upper bound

    # v_rel is inside the truncated cone - find closest point on boundary

    # Candidate 1: Closest point on cone mantle
    # For a point inside the cone, project to the mantle surface
    if v_radial > 1e-12:
        # Non-zero radial component - project radially outward to mantle
        radial_unit = v_radial_vec / v_radial
        # On the mantle: v_radial = v_axial * tan(cone_half_angle)
        mantle_radial = v_axial * np.tan(cone_half_angle)
        mantle_point = v_axial * cone_axis + mantle_radial * radial_unit
        mantle_dist = np.linalg.norm(mantle_point - v_rel)
    else:
        # On cone axis - project to mantle edge
        # Choose arbitrary radial direction
        if abs(cone_axis[0]) < 0.9:
            radial_unit = normalize(np.cross(cone_axis, np.array([1, 0, 0])))
        else:
            radial_unit = normalize(np.cross(cone_axis, np.array([0, 1, 0])))

        mantle_radial = v_axial * np.tan(cone_half_angle)
        mantle_point = v_axial * cone_axis + mantle_radial * radial_unit
        mantle_dist = np.linalg.norm(mantle_point - v_rel)

    # Candidate 2: Closest point on truncation cap (if applicable)
    # Cap is a circle at distance max_speed_for_collision from origin
    if v_rel_norm < max_speed_for_collision:
        # Project v_rel to the cap sphere
        cap_point = v_rel * (max_speed_for_collision / v_rel_norm)
        cap_dist = np.linalg.norm(cap_point - v_rel)
    else:
        cap_dist = float('inf')

    # Choose escape direction
    # When v_rel is nearly on the cone axis (head-on collision), prefer mantle escape
    # because cap escape would tell agent to speed up into collision, which is wrong
    # The mantle escape deflects sideways, which is the correct avoidance behavior
    on_axis = v_radial < 0.1 * v_axial  # Nearly head-on if radial << axial

    if on_axis or mantle_dist <= cap_dist:
        closest_point = mantle_point
    else:
        closest_point = cap_point

    # Compute escape vector and normal
    u = closest_point - v_rel
    if np.linalg.norm(u) < 1e-12:
        return None, None

    n_hat = normalize(u)

    return u, n_hat

def compute_orca_velocity(agent_pos, agent_vel, preferred_vel, other_agents, R, tau, v_max):
    """
    Computes the new collision-free velocity using ORCA in 3D.

    Inputs:
    - agent_pos: np.array (3,).
    - agent_vel: np.array (3,).
    - preferred_vel: np.array (3,).
    - other_agents: list of dicts {'pos': np.array(3,), 'vel': np.array(3,)}.
    - R, tau, v_max: floats.

    Output:
    - new_vel: np.array (3,).

    Mathematical:
    - For each j, p_rel = p_j - p_i, v_rel = agent_vel - v_j.
    - Compute u, n_hat.
    - Half-plane H_i|j: (v - agent_vel) . n_hat >= (u . n_hat) / 2
    - ORCA_i = intersection H, cap ||v|| <= v_max.
    - v_new = argmin ||v - preferred_vel||_2 in ORCA_i, using minimize with SLSQP.
    """
    constraints = []
    current_vel = agent_vel.copy()  # Fix closure issue

    for j, other in enumerate(other_agents):
        p_rel = other['pos'] - agent_pos
        v_rel = agent_vel - other['vel']
        u, n_hat = compute_vo_escape(p_rel, v_rel, R, tau)
        if u is not None:
            dot_un = np.dot(u, n_hat)
            # Fix lambda closure bug by capturing values explicitly
            cons = {'type': 'ineq',
                    'fun': lambda v, n=n_hat.copy(), d=dot_un, curr_vel=current_vel:
                           np.dot(v - curr_vel, n) - d / 2}
            constraints.append(cons)

    # Speed constraint
    constraints.append({'type': 'ineq',
                        'fun': lambda v: v_max**2 - np.dot(v, v)})

    # Objective
    def obj(v):
        return np.sum((v - preferred_vel)**2)

    # No collision constraints, return preferred velocity clamped to max speed
    if len(constraints) == 1:  # Only speed constraint
        if np.linalg.norm(preferred_vel) <= v_max:
            return preferred_vel
        else:
            return v_max * preferred_vel / np.linalg.norm(preferred_vel)

    # Optimize
    res = minimize(obj, preferred_vel, method='SLSQP', constraints=constraints, tol=1e-6)
    if res.success:
        return res.x

    # Fallback: try with zero initial guess
    res2 = minimize(obj, np.zeros(3), method='SLSQP', constraints=constraints, tol=1e-6)
    if res2.success:
        return res2.x

    # Constraints are likely infeasible (ORCA deadlock)
    # Find velocity that minimizes maximum constraint violation
    # This gives the "least bad" velocity when perfect avoidance is impossible
    orca_constraints = constraints[:-1]  # All except speed constraint

    if not orca_constraints:
        # Only speed constraint - just clamp preferred velocity
        if np.linalg.norm(preferred_vel) <= v_max:
            return preferred_vel
        return v_max * preferred_vel / np.linalg.norm(preferred_vel)

    def max_violation(v):
        # Compute maximum constraint violation (negative = violated)
        violations = [c['fun'](v) for c in orca_constraints]
        return -min(violations)  # Minimize the maximum violation

    # Speed constraint only
    speed_constraint = {'type': 'ineq', 'fun': lambda v: v_max**2 - np.dot(v, v)}

    # Try to minimize max violation while respecting speed limit
    res3 = minimize(max_violation, preferred_vel, method='SLSQP',
                    constraints=[speed_constraint], tol=1e-6)
    if res3.success and np.linalg.norm(res3.x) <= v_max + 1e-6:
        return res3.x

    # Final fallback: move away from the average constraint direction
    # This provides some avoidance even in deadlock
    avg_normal = np.zeros(3)
    for c in orca_constraints:
        # Extract normal from constraint (it's captured in the closure)
        # Approximate by evaluating gradient
        eps = 1e-6
        v0 = np.zeros(3)
        grad = np.array([(c['fun'](v0 + eps*np.eye(3)[i]) - c['fun'](v0)) / eps for i in range(3)])
        avg_normal += grad

    if np.linalg.norm(avg_normal) > 1e-6:
        # Move perpendicular to average constraint direction
        avg_normal = avg_normal / np.linalg.norm(avg_normal)
        # Find perpendicular direction closest to preferred velocity
        perp = preferred_vel - np.dot(preferred_vel, avg_normal) * avg_normal
        if np.linalg.norm(perp) > 1e-6:
            return v_max * perp / np.linalg.norm(perp)

    return np.zeros(3)
