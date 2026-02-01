#!/usr/bin/env python3
"""
multi-agent testing example
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os
import time

# Add src directory to path to import ORCA modules
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

from hybrid_orca_astar import Agent, simulate

class MultiAgentTest:
    """
    Testing environment for ORCA-A* with multiple challenging scenarios.
    """
    
    def __init__(self, scenario='star_formation'):
        # Simulation parameters
        self.dt = 0.1  # Time step (seconds)
        self.max_time = 60.0  # Maximum simulation time
        self.scenario = scenario
        
        # Agent parameters (varied for different testing scenarios)
        self.base_separation = 0.4  # Base minimum separation between agents
        self.base_speed = 1.2  # Base speed for most agents
        self.time_horizon = 4.0  # ORCA time horizon
        
        # Visualization parameters
        self.trail_length = 100  # Number of past positions to show
        
        # No obstacles - focus on pure multi-agent collision avoidance
        self.obstacles = []
        
        # Setup the selected scenario
        self.setup_scenario()
        
        # Run complete simulation to collect trajectory data
        self.run_complete_simulation()
        
        # Initialize visualization
        self.setup_visualization()
        
    def setup_scenario(self):
        """Setup agents based on selected scenario."""
        
        if self.scenario == 'star_formation':
            self.setup_star_formation()
        elif self.scenario == 'cross_pattern':
            self.setup_cross_pattern()
        elif self.scenario == 'swarm_clustering':
            self.setup_swarm_clustering()
        elif self.scenario == 'circular_exchange':
            self.setup_circular_exchange()
        elif self.scenario == 'multi_level_traffic':
            self.setup_multi_level_traffic()
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
    
    def setup_star_formation(self):
        """
        Star Formation Test: 8 agents arranged in a circle, all converge to center,
        then spread out to opposite positions. Tests convergence and divergence patterns.
        """
        print("=== Star Formation Test ===")
        print("8 agents: Circle → Center → Opposite positions")
        
        num_agents = 8
        radius = 6.0
        center = np.array([0.0, 0.0, 2.0])
        
        self.agents = []
        self.trajectories = []
        
        for i in range(num_agents):
            angle = 2 * np.pi * i / num_agents
            
            # Start positions: arranged in circle
            start_x = center[0] + radius * np.cos(angle)
            start_y = center[1] + radius * np.sin(angle)
            start_z = center[2] + 0.2 * np.sin(3 * angle)  # Slight z variation
            start_pos = np.array([start_x, start_y, start_z])
            
            # Goal positions: opposite side of circle (through center)
            goal_x = center[0] - radius * np.cos(angle) 
            goal_y = center[1] - radius * np.sin(angle)
            goal_z = center[2] - 0.2 * np.sin(3 * angle)
            goal_pos = np.array([goal_x, goal_y, goal_z])
            
            # Vary agent parameters for more realistic testing
            speed = self.base_speed * (0.8 + 0.4 * np.random.random())
            separation = self.base_separation * (0.8 + 0.4 * np.random.random())
            
            agent = Agent(
                start_pos=start_pos,
                goal_pos=goal_pos,
                max_speed=speed,
                R=separation,
                tau=self.time_horizon,
                obstacles=self.obstacles,
                epsilon=0.2
            )
            
            self.agents.append(agent)
            self.trajectories.append([])
            
            print(f"Agent {i+1}: {start_pos} → {goal_pos} (speed: {speed:.2f})")
    
    def setup_cross_pattern(self):
        """
        Cross Pattern Test: 12 agents in 3D crossing patterns at different altitudes.
        Tests complex multi-directional collision avoidance.
        """
        print("=== Cross Pattern Test ===")
        print("12 agents: Complex 3D crossing patterns")
        
        self.agents = []
        self.trajectories = []
        
        patterns = [
            # Horizontal cross (Z=1.5)
            {'starts': [[-8, 0, 1.5], [8, 0, 1.5], [0, -8, 1.5], [0, 8, 1.5]],
             'goals': [[8, 0, 1.5], [-8, 0, 1.5], [0, 8, 1.5], [0, -8, 1.5]]},
            
            # Diagonal cross (Z=2.5) 
            {'starts': [[-6, -6, 2.5], [6, 6, 2.5], [-6, 6, 2.5], [6, -6, 2.5]],
             'goals': [[6, 6, 2.5], [-6, -6, 2.5], [6, -6, 2.5], [-6, 6, 2.5]]},
             
            # Vertical transitions
            {'starts': [[-4, -4, 0.5], [4, 4, 0.5], [-4, 4, 3.5], [4, -4, 3.5]],
             'goals': [[-4, -4, 3.5], [4, 4, 3.5], [-4, 4, 0.5], [4, -4, 0.5]]}
        ]
        
        agent_id = 0
        for pattern in patterns:
            for start, goal in zip(pattern['starts'], pattern['goals']):
                # Vary speeds to create more complex interactions
                speed = self.base_speed * (0.7 + 0.6 * np.random.random())
                separation = self.base_separation
                
                agent = Agent(
                    start_pos=np.array(start),
                    goal_pos=np.array(goal),
                    max_speed=speed,
                    R=separation,
                    tau=self.time_horizon * (0.8 + 0.4 * np.random.random()),
                    obstacles=self.obstacles,
                    epsilon=0.15
                )
                
                self.agents.append(agent)
                self.trajectories.append([])
                
                print(f"Agent {agent_id+1}: {start} → {goal} (speed: {speed:.2f})")
                agent_id += 1
    
    def setup_swarm_clustering(self):
        """
        Swarm Clustering Test: 10 agents move from random scattered positions 
        to form tight formation. Tests high-density collision avoidance.
        """
        print("=== Swarm Clustering Test ===")
        print("10 agents: Scattered → Tight formation")
        
        self.agents = []
        self.trajectories = []
        
        # Formation center
        formation_center = np.array([0.0, 0.0, 2.0])
        
        # Tight formation positions (rectangular grid)
        formation_positions = []
        spacing = 0.8  # Tight spacing to test close-proximity avoidance
        for i in range(-2, 3):
            for j in range(-1, 2):
                if len(formation_positions) < 10:
                    pos = formation_center + np.array([i * spacing, j * spacing, 0.1 * i * j])
                    formation_positions.append(pos)
        
        for i in range(10):
            # Random scattered start positions
            angle = 2 * np.pi * np.random.random()
            radius = 8 + 4 * np.random.random()
            height = 0.5 + 3 * np.random.random()
            
            start_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle), 
                height
            ])
            
            goal_pos = formation_positions[i]
            
            # Varied agent characteristics
            speed = self.base_speed * (0.6 + 0.8 * np.random.random())
            separation = self.base_separation * (0.7 + 0.6 * np.random.random())
            
            agent = Agent(
                start_pos=start_pos,
                goal_pos=goal_pos,
                max_speed=speed,
                R=separation,
                tau=self.time_horizon,
                obstacles=self.obstacles,
                epsilon=0.1
            )
            
            self.agents.append(agent)
            self.trajectories.append([])
            
            print(f"Agent {i+1}: {start_pos} → {goal_pos} (speed: {speed:.2f})")
    
    def setup_circular_exchange(self):
        """
        Circular Exchange Test: 8 agents move in circular patterns with 
        clockwise and counter-clockwise directions. Tests sustained interaction.
        """
        print("=== Circular Exchange Test ===")
        print("8 agents: Circular patterns (CW and CCW)")
        
        self.agents = []
        self.trajectories = []
        
        radius = 5.0
        center = np.array([0.0, 0.0, 2.0])
        
        for i in range(8):
            start_angle = 2 * np.pi * i / 8
            
            # Alternating clockwise and counter-clockwise
            if i % 2 == 0:
                # Clockwise: move to next position clockwise
                goal_angle = start_angle + 2 * np.pi / 8
                direction = "CW"
            else:
                # Counter-clockwise: move to next position counter-clockwise  
                goal_angle = start_angle - 2 * np.pi / 8
                direction = "CCW"
            
            start_pos = np.array([
                center[0] + radius * np.cos(start_angle),
                center[1] + radius * np.sin(start_angle),
                center[2] + 0.3 * np.sin(4 * start_angle)
            ])
            
            goal_pos = np.array([
                center[0] + radius * np.cos(goal_angle),
                center[1] + radius * np.sin(goal_angle),
                center[2] + 0.3 * np.sin(4 * goal_angle)
            ])
            
            # Slightly different speeds for more complex interactions
            speed = self.base_speed * (0.8 + 0.4 * np.random.random())
            
            agent = Agent(
                start_pos=start_pos,
                goal_pos=goal_pos,
                max_speed=speed,
                R=self.base_separation,
                tau=self.time_horizon,
                obstacles=self.obstacles,
                epsilon=0.2
            )
            
            self.agents.append(agent)
            self.trajectories.append([])
            
            print(f"Agent {i+1} ({direction}): {start_pos} → {goal_pos}")
    
    def setup_multi_level_traffic(self):
        """
        Multi-Level Traffic Test: 9 agents at three altitude levels with 
        inter-level transitions. Tests 3D collision avoidance thoroughly.
        """
        print("=== Multi-Level Traffic Test ===")
        print("9 agents: 3-level traffic with vertical transitions")
        
        self.agents = []
        self.trajectories = []
        
        levels = [0.8, 2.0, 3.2]  # Three altitude levels
        
        # Level 1: Horizontal movements
        level1_starts = [[-8, -3, levels[0]], [-8, 0, levels[0]], [-8, 3, levels[0]]]
        level1_goals = [[8, -3, levels[1]], [8, 0, levels[2]], [8, 3, levels[0]]]  # Mixed level goals
        
        # Level 2: Diagonal movements  
        level2_starts = [[-6, -6, levels[1]], [0, 0, levels[1]], [6, 6, levels[1]]]
        level2_goals = [[6, 6, levels[0]], [-6, 6, levels[1]], [-6, -6, levels[2]]]
        
        # Level 3: Circular pattern
        level3_starts = [[4, 0, levels[2]], [0, 4, levels[2]], [-4, 0, levels[2]]]
        level3_goals = [[0, 4, levels[2]], [-4, 0, levels[0]], [4, 0, levels[1]]]
        
        all_starts = level1_starts + level2_starts + level3_starts
        all_goals = level1_goals + level2_goals + level3_goals
        
        for i, (start, goal) in enumerate(zip(all_starts, all_goals)):
            # Vary speeds significantly to test different dynamics
            speed = self.base_speed * (0.5 + 1.0 * np.random.random())
            separation = self.base_separation * (0.8 + 0.4 * np.random.random())
            
            agent = Agent(
                start_pos=np.array(start),
                goal_pos=np.array(goal),
                max_speed=speed,
                R=separation,
                tau=self.time_horizon,
                obstacles=self.obstacles,
                epsilon=0.15
            )
            
            self.agents.append(agent)
            self.trajectories.append([])
            
            level_start = [l for l in levels if abs(start[2] - l) < 0.1][0]
            level_goal = [l for l in levels if abs(goal[2] - l) < 0.1][0]
            print(f"Agent {i+1}: L{levels.index(level_start)+1} → L{levels.index(level_goal)+1} "
                  f"({start} → {goal}, speed: {speed:.2f})")
    
    def run_complete_simulation(self):
        """Run the complete simulation to collect all trajectory data."""
        print(f"\nRunning {self.scenario} simulation...")
        print(f"Total agents: {len(self.agents)}")
        
        current_time = 0.0
        step_count = 0
        max_steps = int(self.max_time / self.dt)
        
        # Track performance metrics
        start_sim_time = time.time()
        collision_count = 0
        min_separation = float('inf')
        
        while current_time < self.max_time and step_count < max_steps:
            # Check goal completion
            completed_agents = 0
            for i, agent in enumerate(self.agents):
                distance_to_goal = np.linalg.norm(agent.pos - agent.goal)
                if distance_to_goal < agent.epsilon:
                    completed_agents += 1
            
            if completed_agents >= len(self.agents) * 0.8:  # 80% completion
                print(f"Simulation completed early: {completed_agents}/{len(self.agents)} agents reached goals")
                break
            
            # Store current positions
            for i, agent in enumerate(self.agents):
                self.trajectories[i].append(agent.pos.copy())
            
            # Track minimum separation for collision analysis
            for i in range(len(self.agents)):
                for j in range(i + 1, len(self.agents)):
                    separation = np.linalg.norm(self.agents[i].pos - self.agents[j].pos)
                    min_separation = min(min_separation, separation)
                    
                    # Count potential collisions
                    if separation < max(self.agents[i].R, self.agents[j].R):
                        collision_count += 1
            
            # Update each agent using ORCA
            for i, agent in enumerate(self.agents):
                other_agents = [other for j, other in enumerate(self.agents) if j != i]
                agent.step(self.dt, other_agents)
            
            current_time += self.dt
            step_count += 1
            
            # Progress reporting
            if step_count % 100 == 0:
                avg_dist_to_goal = np.mean([
                    np.linalg.norm(agent.pos - agent.goal) for agent in self.agents
                ])
                print(f"Step {step_count}: Time={current_time:.1f}s, "
                      f"Avg distance to goal={avg_dist_to_goal:.2f}m, "
                      f"Min separation={min_separation:.3f}m")
        
        # Add final positions
        for i, agent in enumerate(self.agents):
            self.trajectories[i].append(agent.pos.copy())
        
        sim_duration = time.time() - start_sim_time
        
        print(f"\nSimulation completed in {sim_duration:.2f} seconds")
        print(f"Total steps: {len(self.trajectories[0])}")
        print(f"Minimum separation achieved: {min_separation:.3f}m")
        print(f"Collision events: {collision_count}")
    
    def setup_visualization(self):
        """Initialize matplotlib visualization."""
        self.fig = plt.figure(figsize=(18, 8))
        
        # 3D trajectory view
        self.ax_3d = self.fig.add_subplot(131, projection='3d')
        self.ax_3d.set_title(f'3D Multi-Agent Trajectories\n{self.scenario.replace("_", " ").title()}', 
                            fontsize=12, fontweight='bold')
        self.ax_3d.set_xlabel('X (meters)')
        self.ax_3d.set_ylabel('Y (meters)')
        self.ax_3d.set_zlabel('Z (meters)')
        
        # Auto-scale based on trajectory extents
        all_positions = []
        for traj in self.trajectories:
            all_positions.extend(traj)
        
        if len(all_positions) > 0:
            all_positions = np.array(all_positions)
            margin = 1.0
            
            self.ax_3d.set_xlim(all_positions[:, 0].min() - margin, 
                               all_positions[:, 0].max() + margin)
            self.ax_3d.set_ylim(all_positions[:, 1].min() - margin,
                               all_positions[:, 1].max() + margin)
            self.ax_3d.set_zlim(all_positions[:, 2].min() - margin,
                               all_positions[:, 2].max() + margin)
        
        # Top-down view (X-Y)
        self.ax_2d = self.fig.add_subplot(132)
        self.ax_2d.set_title('Top-Down View (X-Y Plane)', fontsize=12, fontweight='bold')
        self.ax_2d.set_xlabel('X (meters)')
        self.ax_2d.set_ylabel('Y (meters)')
        self.ax_2d.grid(True, alpha=0.3)
        self.ax_2d.set_aspect('equal')
        
        if len(all_positions) > 0:
            self.ax_2d.set_xlim(all_positions[:, 0].min() - margin,
                               all_positions[:, 0].max() + margin)
            self.ax_2d.set_ylim(all_positions[:, 1].min() - margin,
                               all_positions[:, 1].max() + margin)
        
        # Analysis plots (separation distances over time)
        self.ax_analysis = self.fig.add_subplot(133)
        self.ax_analysis.set_title('Minimum Separation Over Time', fontsize=12, fontweight='bold')
        self.ax_analysis.set_xlabel('Time (seconds)')
        self.ax_analysis.set_ylabel('Minimum Distance (meters)')
        self.ax_analysis.grid(True, alpha=0.3)
        
        # Generate color palette for agents
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.agents)))
        
        # Initialize plot elements
        self.drone_points_3d = []
        self.drone_points_2d = []
        self.trail_lines_3d = []
        self.trail_lines_2d = []
        
        for i in range(len(self.agents)):
            # 3D elements
            point_3d, = self.ax_3d.plot([], [], [], 'o', color=self.colors[i], 
                                       markersize=8, label=f'Agent {i+1}')
            trail_3d, = self.ax_3d.plot([], [], [], '-', color=self.colors[i], 
                                       alpha=0.7, linewidth=1.5)
            
            # 2D elements
            point_2d, = self.ax_2d.plot([], [], 'o', color=self.colors[i], 
                                       markersize=8, label=f'Agent {i+1}')
            trail_2d, = self.ax_2d.plot([], [], '-', color=self.colors[i], 
                                       alpha=0.7, linewidth=1.5)
            
            self.drone_points_3d.append(point_3d)
            self.drone_points_2d.append(point_2d)
            self.trail_lines_3d.append(trail_3d)
            self.trail_lines_2d.append(trail_2d)
        
        # Plot start and goal positions
        for i, agent in enumerate(self.agents):
            start = self.trajectories[i][0] if self.trajectories[i] else agent.pos
            goal = agent.goal
            
            # 3D markers
            self.ax_3d.plot([start[0]], [start[1]], [start[2]], 's', 
                           color=self.colors[i], markersize=6, alpha=0.8)
            self.ax_3d.plot([goal[0]], [goal[1]], [goal[2]], '*', 
                           color=self.colors[i], markersize=10, alpha=0.8)
            
            # 2D markers
            self.ax_2d.plot([start[0]], [start[1]], 's', 
                           color=self.colors[i], markersize=6, alpha=0.8)
            self.ax_2d.plot([goal[0]], [goal[1]], '*', 
                           color=self.colors[i], markersize=10, alpha=0.8)
        
        # Legends (only show first few agents to avoid clutter)
        if len(self.agents) <= 6:
            self.ax_3d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            self.ax_2d.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Status text
        self.status_text = self.fig.suptitle(f'{self.scenario.replace("_", " ").title()} - Frame 0', 
                                           fontsize=14)
        
        plt.tight_layout()
        
        # Store frame count
        self.total_frames = len(self.trajectories[0]) if self.trajectories else 0
    
    def animate_frame(self, frame):
        """Animation function for creating frames."""
        if frame >= self.total_frames:
            frame = self.total_frames - 1
        
        # Clear previous positions
        for i in range(len(self.agents)):
            self.drone_points_3d[i].set_data([], [])
            self.drone_points_3d[i].set_3d_properties([])
            self.drone_points_2d[i].set_data([], [])
            self.trail_lines_3d[i].set_data([], [])
            self.trail_lines_3d[i].set_3d_properties([])
            self.trail_lines_2d[i].set_data([], [])
        
        # Calculate minimum separation at current frame
        current_time = frame * self.dt
        min_separation = float('inf')
        
        if frame < min(len(traj) for traj in self.trajectories):
            positions = [np.array(traj[frame]) for traj in self.trajectories]
            
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    sep = np.linalg.norm(positions[i] - positions[j])
                    min_separation = min(min_separation, sep)
        
        # Update status
        status = (f'{self.scenario.replace("_", " ").title()} - '
                 f'Frame {frame+1}/{self.total_frames} | '
                 f'Time: {current_time:.1f}s | '
                 f'Min Sep: {min_separation:.2f}m')
        self.status_text.set_text(status)
        
        # Update agent positions and trails
        for i in range(len(self.agents)):
            if frame < len(self.trajectories[i]):
                # Current position
                pos = np.array(self.trajectories[i][frame])
                
                # Update 3D position
                self.drone_points_3d[i].set_data([pos[0]], [pos[1]])
                self.drone_points_3d[i].set_3d_properties([pos[2]])
                
                # Update 2D position
                self.drone_points_2d[i].set_data([pos[0]], [pos[1]])
                
                # Update trails
                trail_start = max(0, frame - self.trail_length)
                trail_end = frame + 1
                trail = np.array(self.trajectories[i][trail_start:trail_end])
                
                if len(trail) > 1:
                    # 3D trail
                    self.trail_lines_3d[i].set_data(trail[:, 0], trail[:, 1])
                    self.trail_lines_3d[i].set_3d_properties(trail[:, 2])
                    
                    # 2D trail
                    self.trail_lines_2d[i].set_data(trail[:, 0], trail[:, 1])
        
        # Update analysis plot (separation over time)
        if frame > 0:
            times = np.arange(0, min(frame + 1, self.total_frames)) * self.dt
            min_separations = []
            
            for f in range(min(frame + 1, self.total_frames)):
                if f < min(len(traj) for traj in self.trajectories):
                    positions = [np.array(traj[f]) for traj in self.trajectories]
                    frame_min_sep = float('inf')
                    
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            sep = np.linalg.norm(positions[i] - positions[j])
                            frame_min_sep = min(frame_min_sep, sep)
                    
                    min_separations.append(frame_min_sep)
            
            self.ax_analysis.clear()
            self.ax_analysis.set_title('Minimum Separation Over Time', fontsize=12, fontweight='bold')
            self.ax_analysis.set_xlabel('Time (seconds)')
            self.ax_analysis.set_ylabel('Minimum Distance (meters)')
            self.ax_analysis.grid(True, alpha=0.3)
            
            if min_separations:
                self.ax_analysis.plot(times[:len(min_separations)], min_separations, 'g-', linewidth=2)
                self.ax_analysis.axhline(y=self.base_separation, color='r', linestyle='--', 
                                       alpha=0.7, label=f'Min Req: {self.base_separation:.1f}m')
                self.ax_analysis.legend()
        
        return [self.status_text] + self.drone_points_3d + self.drone_points_2d + \
               self.trail_lines_3d + self.trail_lines_2d
    
    def create_gif(self, filename=None, fps=8):
        """Create animated GIF of the simulation."""
        if filename is None:
            filename = f"{self.scenario}_test.gif"
        
        print(f"\nGenerating GIF: {filename}")
        print(f"Total frames: {self.total_frames}")
        
        # Limit frames for reasonable GIF size
        frame_skip = max(1, self.total_frames // 120)
        frames = range(0, self.total_frames, frame_skip)
        
        print(f"Using every {frame_skip} frame(s), GIF frames: {len(frames)}")
        
        def animate_gif_frame(frame_idx):
            actual_frame = frames[frame_idx] if frame_idx < len(frames) else frames[-1]
            return self.animate_frame(actual_frame)
        
        animation = FuncAnimation(
            self.fig,
            animate_gif_frame,
            frames=len(frames),
            interval=1000//fps,
            blit=False,
            repeat=True
        )
        
        try:
            writer = PillowWriter(fps=fps)
            animation.save(filename, writer=writer)
            print(f"✅ GIF saved: {filename}")
            
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f"File size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ Error saving GIF: {e}")
            plt.show()
        
        return animation
    
    def print_analysis(self):
        """Print detailed analysis of the simulation results."""
        print("\n" + "="*60)
        print(f"ANALYSIS: {self.scenario.upper().replace('_', ' ')}")
        print("="*60)
        
        if not self.trajectories or not any(self.trajectories):
            print("No trajectory data available for analysis.")
            return
        
        # Calculate performance metrics
        total_time = len(self.trajectories[0]) * self.dt
        
        # Goal completion analysis
        completed_agents = 0
        final_distances = []
        for i, agent in enumerate(self.agents):
            final_pos = np.array(self.trajectories[i][-1]) if self.trajectories[i] else agent.pos
            distance_to_goal = np.linalg.norm(final_pos - agent.goal)
            final_distances.append(distance_to_goal)
            
            if distance_to_goal < agent.epsilon:
                completed_agents += 1
        
        completion_rate = completed_agents / len(self.agents) * 100
        
        # Collision analysis
        min_separation_overall = float('inf')
        collision_violations = 0
        total_interactions = 0
        
        for frame in range(len(self.trajectories[0])):
            if frame < min(len(traj) for traj in self.trajectories):
                positions = [np.array(traj[frame]) for traj in self.trajectories]
                
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        separation = np.linalg.norm(positions[i] - positions[j])
                        min_separation_overall = min(min_separation_overall, separation)
                        total_interactions += 1
                        
                        if separation < max(self.agents[i].R, self.agents[j].R):
                            collision_violations += 1
        
        # Path efficiency analysis
        path_lengths = []
        direct_distances = []
        
        for i, traj in enumerate(self.trajectories):
            if len(traj) > 1:
                # Calculate total path length
                path_length = sum(np.linalg.norm(np.array(traj[j+1]) - np.array(traj[j])) 
                                for j in range(len(traj)-1))
                path_lengths.append(path_length)
                
                # Calculate direct distance
                direct_dist = np.linalg.norm(np.array(traj[-1]) - np.array(traj[0]))
                direct_distances.append(direct_dist)
        
        avg_efficiency = np.mean([direct / path for direct, path in 
                                zip(direct_distances, path_lengths) if path > 0]) * 100
        
        # Print results
        print(f"Simulation Duration: {total_time:.1f} seconds ({len(self.trajectories[0])} steps)")
        print(f"Number of Agents: {len(self.agents)}")
        print()
        
        print("GOAL COMPLETION:")
        print(f"  Agents reached goal: {completed_agents}/{len(self.agents)} ({completion_rate:.1f}%)")
        print(f"  Average final distance to goal: {np.mean(final_distances):.3f}m")
        print(f"  Best final distance: {min(final_distances):.3f}m")
        print(f"  Worst final distance: {max(final_distances):.3f}m")
        print()
        
        print("COLLISION AVOIDANCE:")
        print(f"  Minimum separation achieved: {min_separation_overall:.3f}m")
        print(f"  Required separation: {self.base_separation:.3f}m")
        print(f"  Collision violations: {collision_violations}")
        print(f"  Safety margin: {min_separation_overall - self.base_separation:.3f}m")
        
        print("PATH EFFICIENCY:")
        print(f"  Average path efficiency: {avg_efficiency:.1f}%")
        print(f"  Average path length: {np.mean(path_lengths):.2f}m")
        print(f"  Average direct distance: {np.mean(direct_distances):.2f}m")
        print()


def main():
    """Main function to run ORCA testing scenarios."""
    scenarios = [
        'star_formation',
        'cross_pattern',
        'swarm_clustering',
        'circular_exchange',
        'multi_level_traffic'
    ]

    print("=== ORCA-A* MULTI-AGENT TESTING ===")
    print("\nAvailable scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario.replace('_', ' ').title()}")

    try:
        try:
            choice = input(f"\nSelect scenario (1-{len(scenarios)}, default=1): ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = '1'

        if choice in ['1', '2', '3', '4', '5']:
            scenario = scenarios[int(choice) - 1]
        else:
            scenario = scenarios[0]

        test = MultiAgentTest(scenario)

        print("\nChoose output:")
        print("1. Generate GIF (default)")
        print("2. Show live animation")

        try:
            output_choice = input("Enter choice (1-2, default=1): ").strip()
        except (EOFError, KeyboardInterrupt):
            output_choice = '1'

        if output_choice == '2':
            animation = FuncAnimation(
                test.fig,
                test.animate_frame,
                frames=test.total_frames,
                interval=100,
                blit=False,
                repeat=True
            )
            plt.show()
        else:
            test.create_gif()

        test.print_analysis()

    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()