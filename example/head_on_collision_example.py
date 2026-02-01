#!/usr/bin/env python3
"""

two drones starting at opposite positions and attempting to reach each other's starting location.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os

# Add src directory to path to import ORCA modules
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)  # Insert at beginning to prioritize our modules

from hybrid_orca_astar import Agent, simulate

class HeadOnCollisionDemo:
    """
    Demonstrates head-on collision avoidance between two drones using ORCA algorithm.
    """
    
    def __init__(self):
        # Simulation parameters
        self.dt = 0.1  # Time step (seconds)
        self.max_time = 20.0  # Maximum simulation time
        self.separation_distance = 0.5  # Minimum separation between drones (meters)
        self.max_speed = 1.0  # Maximum drone speed (m/s)
        self.time_horizon = 5.0  # ORCA time horizon for collision prediction
        
        # Visualization parameters
        self.trail_length = 50  # Number of past positions to show in trail
        
        # Setup drones in head-on collision scenario
        self.setup_drones()
        
        # Run simulation first to collect all trajectory data
        self.run_complete_simulation()
        
        # Initialize visualization with complete data
        self.setup_visualization()
        
    def setup_drones(self):
        """
        Create two drones positioned for head-on collision.
        
        Drone 1: Starts at left, moves right
        Drone 2: Starts at right, moves left
        Both at same altitude and y-coordinate for direct collision course.
        """
        # No obstacles - clean environment to focus on collision avoidance
        obstacles = []
        
        # Starting positions (head-on collision course)
        start1 = np.array([0.0, 0.0, 1.0])   # Left side
        goal1 = np.array([10.0, 0.0, 1.0])   # Right side
        
        start2 = np.array([10.0, 0.0, 1.0])  # Right side  
        goal2 = np.array([0.0, 0.0, 1.0])    # Left side
        
        print("=== Head-on Collision Avoidance Demo ===")
        print(f"Drone 1: {start1} → {goal1}")
        print(f"Drone 2: {start2} → {goal2}")
        print(f"Separation distance: {self.separation_distance}m")
        print(f"Max speed: {self.max_speed}m/s")
        print(f"Time horizon: {self.time_horizon}s")
        print("=" * 45)
        
        # Create agents with ORCA collision avoidance
        self.agent1 = Agent(
            start_pos=start1,
            goal_pos=goal1,
            max_speed=self.max_speed,
            R=self.separation_distance,
            tau=self.time_horizon,
            obstacles=obstacles,
            epsilon=0.1  # Goal tolerance
        )
        
        self.agent2 = Agent(
            start_pos=start2,
            goal_pos=goal2,
            max_speed=self.max_speed,
            R=self.separation_distance,
            tau=self.time_horizon,
            obstacles=obstacles,
            epsilon=0.1
        )
        
        self.agents = [self.agent1, self.agent2]
        
        # Initialize trajectory storage
        self.trajectories = [[], []]
        
    def run_complete_simulation(self):
        """
        Run the complete simulation to collect all trajectory data.
        """
        print("\nRunning simulation to collect trajectory data...")
        
        current_time = 0.0
        step_count = 0
        max_steps = int(self.max_time / self.dt)
        
        while current_time < self.max_time and step_count < max_steps:
            # Check if any agent reached its goal
            simulation_complete = False
            for i, agent in enumerate(self.agents):
                distance_to_goal = np.linalg.norm(agent.pos - agent.goal)
                if distance_to_goal < agent.epsilon:
                    print(f"Drone {i+1} reached goal! Distance: {distance_to_goal:.3f}m")
                    simulation_complete = True
                    break
            
            if simulation_complete:
                break
                
            # Store current positions
            for i, agent in enumerate(self.agents):
                self.trajectories[i].append(agent.pos.copy())
            
            # Update each agent using ORCA
            for i, agent in enumerate(self.agents):
                other_agents = [other for j, other in enumerate(self.agents) if j != i]
                agent.step(self.dt, other_agents)
                
            current_time += self.dt
            step_count += 1
            
            # Print progress every 50 steps
            if step_count % 50 == 0:
                distance = np.linalg.norm(self.agents[0].pos - self.agents[1].pos)
                print(f"Step {step_count}: Time={current_time:.1f}s, Distance={distance:.2f}m")
        
        # Add final positions
        for i, agent in enumerate(self.agents):
            self.trajectories[i].append(agent.pos.copy())
            
        print(f"Simulation completed. Total steps: {len(self.trajectories[0])}")
        
    def setup_visualization(self):
        """
        Initialize matplotlib figure and 3D axes for GIF generation.
        """
        self.fig = plt.figure(figsize=(15, 6))
        
        # Create 3D subplot for trajectory
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_3d.set_title('3D Drone Collision Avoidance', fontsize=14, fontweight='bold')
        self.ax_3d.set_xlabel('X (meters)')
        self.ax_3d.set_ylabel('Y (meters)')
        self.ax_3d.set_zlabel('Z (meters)')
        
        # Set fixed axes limits
        self.ax_3d.set_xlim(-1, 11)
        self.ax_3d.set_ylim(-2, 2)
        self.ax_3d.set_zlim(0.5, 1.5)
        
        # Create 2D top-down view
        self.ax_2d = self.fig.add_subplot(122)
        self.ax_2d.set_title('Top-Down View (X-Y Plane)', fontsize=14, fontweight='bold')
        self.ax_2d.set_xlabel('X (meters)')
        self.ax_2d.set_ylabel('Y (meters)')
        self.ax_2d.set_xlim(-1, 11)
        self.ax_2d.set_ylim(-2, 2)
        self.ax_2d.grid(True, alpha=0.3)
        self.ax_2d.set_aspect('equal')
        
        # Initialize plot elements (trajectories already populated from simulation)
        self.drone_points_3d = []
        self.drone_points_2d = []
        self.trail_lines_3d = []
        self.trail_lines_2d = []
        
        # Colors for each drone
        self.colors = ['red', 'blue']
        self.drone_names = ['Drone 1', 'Drone 2']
        
        # Initialize plot elements for each drone
        for i in range(2):
            # 3D plot elements
            point_3d, = self.ax_3d.plot([], [], [], 'o', color=self.colors[i], 
                                       markersize=10, label=self.drone_names[i])
            trail_3d, = self.ax_3d.plot([], [], [], '-', color=self.colors[i], 
                                       alpha=0.6, linewidth=2)
            
            # 2D plot elements
            point_2d, = self.ax_2d.plot([], [], 'o', color=self.colors[i], 
                                       markersize=10, label=self.drone_names[i])
            trail_2d, = self.ax_2d.plot([], [], '-', color=self.colors[i], 
                                       alpha=0.6, linewidth=2)
            
            self.drone_points_3d.append(point_3d)
            self.drone_points_2d.append(point_2d)
            self.trail_lines_3d.append(trail_3d)
            self.trail_lines_2d.append(trail_2d)
        
        # Add legends
        self.ax_3d.legend()
        self.ax_2d.legend()
        
        # Plot start and goal positions (use initial trajectory points for starts)
        starts = [np.array(self.trajectories[0][0]), np.array(self.trajectories[1][0])]
        goals = [self.agent1.goal, self.agent2.goal]
        
        for i, (start, goal) in enumerate(zip(starts, goals)):
            # 3D markers
            self.ax_3d.plot([start[0]], [start[1]], [start[2]], 's', 
                           color=self.colors[i], markersize=8, alpha=0.7, label=f'Start {i+1}')
            self.ax_3d.plot([goal[0]], [goal[1]], [goal[2]], '*', 
                           color=self.colors[i], markersize=12, alpha=0.7, label=f'Goal {i+1}')
            
            # 2D markers
            self.ax_2d.plot([start[0]], [start[1]], 's', 
                           color=self.colors[i], markersize=8, alpha=0.7)
            self.ax_2d.plot([goal[0]], [goal[1]], '*', 
                           color=self.colors[i], markersize=12, alpha=0.7)
        
        # Status text
        self.status_text = self.fig.suptitle('ORCA Collision Avoidance - Frame 0', fontsize=16)
        
        plt.tight_layout()
        
        # Store total number of frames for animation
        self.total_frames = len(self.trajectories[0])
        
    def animate_frame(self, frame):
        """
        Animation function for creating GIF frames.
        
        Args:
            frame: Current frame number
            
        Returns:
            list: Updated plot elements
        """
        if frame >= self.total_frames:
            frame = self.total_frames - 1
            
        # Clear previous drone positions and trails
        for i in range(2):
            self.drone_points_3d[i].set_data([], [])
            self.drone_points_3d[i].set_3d_properties([])
            self.drone_points_2d[i].set_data([], [])
            self.trail_lines_3d[i].set_data([], [])
            self.trail_lines_3d[i].set_3d_properties([])
            self.trail_lines_2d[i].set_data([], [])
        
        # Calculate current positions and distances
        current_time = frame * self.dt
        distance_between = 0.0
        
        if frame < len(self.trajectories[0]) and frame < len(self.trajectories[1]):
            pos1 = np.array(self.trajectories[0][frame])
            pos2 = np.array(self.trajectories[1][frame])
            distance_between = np.linalg.norm(pos1 - pos2)
        
        # Update status
        collision_warning = ""
        if distance_between < self.separation_distance * 2 and distance_between > 0:
            collision_warning = " ⚠️ CLOSE APPROACH"
        
        status = (f"ORCA Collision Avoidance - Frame {frame+1}/{self.total_frames} | "
                 f"Time: {current_time:.1f}s | Distance: {distance_between:.2f}m{collision_warning}")
        self.status_text.set_text(status)
        
        # Update drone positions and trails
        updated_elements = [self.status_text]
        
        for i in range(2):
            if frame < len(self.trajectories[i]):
                # Current position
                pos = np.array(self.trajectories[i][frame])
                
                # Update 3D position
                self.drone_points_3d[i].set_data([pos[0]], [pos[1]])
                self.drone_points_3d[i].set_3d_properties([pos[2]])
                
                # Update 2D position
                self.drone_points_2d[i].set_data([pos[0]], [pos[1]])
                
                # Update trails (show trail up to current frame)
                trail_start = max(0, frame - self.trail_length)
                trail_end = frame + 1
                trail = np.array(self.trajectories[i][trail_start:trail_end])
                
                if len(trail) > 1:
                    # 3D trail
                    self.trail_lines_3d[i].set_data(trail[:, 0], trail[:, 1])
                    self.trail_lines_3d[i].set_3d_properties(trail[:, 2])
                    
                    # 2D trail
                    self.trail_lines_2d[i].set_data(trail[:, 0], trail[:, 1])
                
                updated_elements.extend([
                    self.drone_points_3d[i], self.drone_points_2d[i],
                    self.trail_lines_3d[i], self.trail_lines_2d[i]
                ])
        
        return updated_elements
        
    def create_gif(self, filename="head_on_collision.gif", fps=10):
        """
        Create animated GIF of the collision avoidance simulation.
        
        Args:
            filename: Output filename for the GIF
            fps: Frames per second for the animation
        """
        print(f"\nGenerating animated GIF: {filename}")
        print(f"Total frames: {self.total_frames}")
        print(f"Animation duration: {self.total_frames/fps:.1f} seconds")
        
        # Create animation with reduced frame rate for GIF
        frame_skip = max(1, self.total_frames // 100)  # Limit to ~100 frames max
        frames = range(0, self.total_frames, frame_skip)
        
        print(f"Using every {frame_skip} frame(s), total frames in GIF: {len(frames)}")
        
        def animate_gif_frame(frame_idx):
            actual_frame = frames[frame_idx] if frame_idx < len(frames) else frames[-1]
            return self.animate_frame(actual_frame)
        
        # Create animation
        animation = FuncAnimation(
            self.fig,
            animate_gif_frame,
            frames=len(frames),
            interval=1000//fps,  # Convert fps to interval
            blit=False,
            repeat=True
        )
        
        # Save as GIF
        try:
            writer = PillowWriter(fps=fps)
            animation.save(filename, writer=writer)
            print(f"✅ GIF saved successfully: {filename}")
            
            # Calculate file size
            import os
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"File size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ Error saving GIF: {e}")
            print("Trying to display animation instead...")
            plt.show()
        
        return animation
        
    def plot_final_trajectories(self):
        """
        Plot the complete trajectories after simulation.
        """
        fig = plt.figure(figsize=(15, 5))
        
        # 3D plot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_title('3D Trajectories')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 2D X-Y plot
        ax2 = fig.add_subplot(132)
        ax2.set_title('Top View (X-Y)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Distance over time plot
        ax3 = fig.add_subplot(133)
        ax3.set_title('Distance Between Drones')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Distance (m)')
        ax3.grid(True, alpha=0.3)
        
        # Plot trajectories
        for i, traj in enumerate(self.trajectories):
            if len(traj) > 0:
                traj = np.array(traj)
                color = self.colors[i]
                label = self.drone_names[i]
                
                # 3D trajectory
                ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                        color=color, linewidth=2, label=label)
                ax1.plot([traj[0, 0]], [traj[0, 1]], [traj[0, 2]], 
                        's', color=color, markersize=8, label=f'Start {i+1}')
                ax1.plot([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]], 
                        '*', color=color, markersize=12, label=f'End {i+1}')
                
                # 2D trajectory
                ax2.plot(traj[:, 0], traj[:, 1], 
                        color=color, linewidth=2, label=label)
                ax2.plot([traj[0, 0]], [traj[0, 1]], 
                        's', color=color, markersize=8)
                ax2.plot([traj[-1, 0]], [traj[-1, 1]], 
                        '*', color=color, markersize=12)
        
        # Distance plot
        if len(self.trajectories[0]) > 0 and len(self.trajectories[1]) > 0:
            min_len = min(len(self.trajectories[0]), len(self.trajectories[1]))
            times = np.arange(min_len) * self.dt
            distances = []
            
            for j in range(min_len):
                dist = np.linalg.norm(
                    np.array(self.trajectories[0][j]) - np.array(self.trajectories[1][j])
                )
                distances.append(dist)
            
            ax3.plot(times, distances, 'g-', linewidth=2, label='Distance')
            ax3.axhline(y=self.separation_distance, color='r', linestyle='--', 
                       label=f'Min separation ({self.separation_distance}m)')
            ax3.legend()
            
            # Find minimum distance
            min_distance = min(distances)
            min_time = times[distances.index(min_distance)]
            ax3.plot([min_time], [min_distance], 'ro', markersize=8, 
                    label=f'Min: {min_distance:.2f}m at {min_time:.1f}s')
        
        # Add legends
        ax1.legend()
        ax2.legend()
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
    def print_final_stats(self):
        """
        Print simulation statistics.
        """
        print("\n" + "="*50)
        print("SIMULATION COMPLETED")
        print("="*50)
        
        if len(self.trajectories[0]) > 0 and len(self.trajectories[1]) > 0:
            final_time = len(self.trajectories[0]) * self.dt
            print(f"Total simulation time: {final_time:.1f} seconds")
            print(f"Total steps: {len(self.trajectories[0])}")
            
            # Calculate minimum distance achieved
            min_len = min(len(self.trajectories[0]), len(self.trajectories[1]))
            min_distance = float('inf')
            min_time = 0
            
            for j in range(min_len):
                dist = np.linalg.norm(
                    np.array(self.trajectories[0][j]) - np.array(self.trajectories[1][j])
                )
                if dist < min_distance:
                    min_distance = dist
                    min_time = j * self.dt
            
            print(f"Minimum distance achieved: {min_distance:.3f}m at {min_time:.1f}s")
            print(f"Required separation: {self.separation_distance:.3f}m")
            
            if min_distance >= self.separation_distance:
                print("✅ SUCCESS: No collision occurred!")
            else:
                print("❌ WARNING: Separation violated!")
            
            # Final positions
            for i, agent in enumerate(self.agents):
                distance_to_goal = np.linalg.norm(agent.pos - agent.goal)
                print(f"Drone {i+1} final distance to goal: {distance_to_goal:.3f}m")


def main():
    """
    Main function to run the head-on collision avoidance demonstration.
    """
    demo = HeadOnCollisionDemo()
    
    print("Choose output mode:")
    print("1. Generate animated GIF (default)")
    print("2. Show static trajectory plots")
    print("3. Display live animation (requires GUI)")
    
    try:
        choice = input("Enter choice (1, 2, or 3, default=1): ").strip()
        
        if choice == '2':
            demo.plot_final_trajectories()
        elif choice == '3':
            # Create live animation
            animation = FuncAnimation(
                demo.fig,
                demo.animate_frame,
                frames=demo.total_frames,
                interval=100,
                blit=False,
                repeat=True
            )
            plt.show()
        else:
            # Generate GIF (default)
            demo.create_gif("head_on_collision.gif", fps=10)
            
        # Always print final statistics
        demo.print_final_stats()
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to static trajectory plots...")
        demo.plot_final_trajectories()
        demo.print_final_stats()


if __name__ == "__main__":
    main()