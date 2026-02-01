import numpy as np
from visibility_graph import build_visibility_graph
from a_star import a_star_search
from orca import compute_orca_velocity
from utils import euclidean_distance_3d, line_obstacle_intersect_3d, normalize

class Agent:
    def __init__(self, start_pos, goal_pos, max_speed, R, tau, obstacles, epsilon=1e-5):
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.zeros(3)
        self.goal = np.array(goal_pos, dtype=float)
        self.v_max = max_speed
        self.R = R
        self.tau = tau
        self.obstacles = obstacles
        self.epsilon = epsilon
        self.path = None
        self.current_waypoint_idx = 0
        self._plan_path()

    def _plan_path(self):
        vertices, edges, start_idx, goal_idx = build_visibility_graph(self.obstacles, self.pos, self.goal)
        self.path, path_indices = a_star_search(vertices, edges, start_idx, goal_idx)
        self.current_waypoint_idx = 1  # start at 0, next 1

    def get_preferred_velocity(self):
        if self.path is None or self.current_waypoint_idx >= len(self.path):
            return np.zeros(3)

        waypoint = self.path[self.current_waypoint_idx]
        dir_to = waypoint - self.pos
        dist = np.linalg.norm(dir_to)
        if dist < self.epsilon:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.path):
                return np.zeros(3)
            waypoint = self.path[self.current_waypoint_idx]
            dir_to = waypoint - self.pos
            dist = np.linalg.norm(dir_to)
        return self.v_max * dir_to / dist

    def update_velocity(self, other_agents):
        pref_vel = self.get_preferred_velocity()
        self.vel = compute_orca_velocity(self.pos, self.vel, pref_vel, 
                                         [{'pos': a.pos, 'vel': a.vel} for a in other_agents],
                                         self.R, self.tau, self.v_max)

    def step(self, dt, other_agents):
        self.update_velocity(other_agents)
        self.pos += dt * self.vel

        # Check replan
        if self.path is None or self.current_waypoint_idx >= len(self.path):
            return
        waypoint = self.path[self.current_waypoint_idx]
        if line_obstacle_intersect_3d(self.pos, waypoint, self.obstacles):
            self._plan_path()
            self.current_waypoint_idx = 1

def simulate(agents, dt, max_time):
    """
    Runs the multi-agent simulation.

    Inputs:
    - agents: list of Agent.
    - dt: float time step.
    - max_time: float.

    Outputs:
    - trajectories: list of lists of np.array positions for each agent.
    """
    trajectories = [[] for _ in agents]
    for t in np.arange(0, max_time, dt):
        # Fix: Correctly append each agent's position to its trajectory
        for i, agent in enumerate(agents):
            trajectories[i].append(agent.pos.copy())
        
        # Update agents
        for i, agent in enumerate(agents):
            other_agents = [a for j, a in enumerate(agents) if j != i]
            agent.step(dt, other_agents)
            
        # Check if all reached goals
        if all(np.linalg.norm(a.pos - a.goal) < a.epsilon for a in agents):
            break
    return trajectories