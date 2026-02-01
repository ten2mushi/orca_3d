import numpy as np
import matplotlib.pyplot as plt
from hybrid_orca_astar import Agent, simulate

def main():
    # Example setup
    obstacles = [
        # Cube obstacle
        {
            'vertices': np.array([
                [2, 2, 0], [3, 2, 0], [3, 3, 0], [2, 3, 0],
                [2, 2, 1], [3, 2, 1], [3, 3, 1], [2, 3, 1]
            ]),
            'faces': [
                [0, 1, 2, 3], [0, 3, 7, 4], [3, 2, 6, 7],
                [2, 1, 5, 6], [1, 0, 4, 5], [4, 7, 6, 5]
            ]
        }
    ]

    # Create a head-on collision scenario
    start1 = [0, 2.5, 0.5]
    goal1 = [5, 2.5, 0.5]
    start2 = [5, 2.5, 0.5]
    goal2 = [0, 2.5, 0.5]

    agent1 = Agent(start1, goal1, max_speed=1.0, R=0.5, tau=10.0, obstacles=obstacles)
    agent2 = Agent(start2, goal2, max_speed=1.0, R=0.5, tau=10.0, obstacles=obstacles)

    agents = [agent1, agent2]

    dt = 0.1
    max_time = 100.0
    trajectories = simulate(agents, dt, max_time)

    # Plot (optional)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Agent {i+1}')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()