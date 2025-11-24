import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


class MinigridVisualizer:
    def __init__(self, env, solution=None, collision_steps=None):
        self.env = env
        self.solution = solution
        self.collision_steps = set(collision_steps) if collision_steps else set()
        self.agent_colors = ['red', 'blue', 'green', 'orange', 'purple']

    def render_static(self, show_paths=True, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 8))

        self._draw_grid(ax)

        self._draw_obstacles(ax)

        self._draw_goals(ax)

        if show_paths and self.solution:
            self._draw_paths(ax)

        self._draw_agents(ax)

        ax.set_xlim(-0.5, self.env.width - 0.5)
        ax.set_ylim(-0.5, self.env.height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Multi-Agent Minigrid ({self.env.num_agents} agents)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Static render saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def animate_solution(self, save_path=None, fps=2, show_plot=True):
        if not self.solution:
            raise ValueError("No solution available to animate")

        max_steps = max(len(path) for path in self.solution.values())

        fig, ax = plt.subplots(figsize=(8, 8))

        agent_artists = []
        path_artists = []

        self._draw_grid(ax)
        self._draw_obstacles(ax)
        self._draw_goals(ax)

        for i in range(self.env.num_agents):
            circle = plt.Circle(
                (0, 0),
                0.3,
                color=self.agent_colors[i % len(self.agent_colors)],
                alpha=0.8,
                zorder=10
            )
            ax.add_patch(circle)
            agent_artists.append(circle)

            text = ax.text(0, 0, f'{i}', ha='center', va='center',
                          color='white', fontweight='bold', zorder=11)
            agent_artists.append(text)

        for i in range(self.env.num_agents):
            line, = ax.plot([], [], '--',
                           color=self.agent_colors[i % len(self.agent_colors)],
                           alpha=0.3, linewidth=2)
            path_artists.append(line)

        ax.set_xlim(-0.5, self.env.width - 0.5)
        ax.set_ylim(-0.5, self.env.height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        def init():
            return agent_artists + path_artists

        def animate(frame):
            ax.set_title(f'Multi-Agent Pathfinding - Step {frame}/{max_steps-1}')

            for i in range(self.env.num_agents):
                agent_name = f'agent{i}'
                if agent_name in self.solution:
                    path = self.solution[agent_name]

                    if frame < len(path):
                        pos = path[frame]
                        x, y = pos['x'], pos['y']
                    else:
                        pos = path[-1]
                        x, y = pos['x'], pos['y']

                    agent_artists[i * 2].center = (x, y)

                    agent_artists[i * 2 + 1].set_position((x, y))

                    traveled_path = path[:min(frame + 1, len(path))]
                    xs = [p['x'] for p in traveled_path]
                    ys = [p['y'] for p in traveled_path]
                    path_artists[i].set_data(xs, ys)

            self._highlight_collisions(ax, frame, agent_artists)

            return agent_artists + path_artists

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=max_steps,
            interval=1000 / fps,
            blit=True,
            repeat=True
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.suffix == '.gif':
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
                print(f"Animation saved to {save_path}")
            elif save_path.suffix == '.mp4':
                anim.save(save_path, writer='ffmpeg', fps=fps)
                print(f"Animation saved to {save_path}")
            else:
                print(f"Unsupported format: {save_path.suffix}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return anim

    def _draw_grid(self, ax):
        for i in range(self.env.width + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for j in range(self.env.height + 1):
            ax.axhline(j - 0.5, color='gray', linewidth=0.5, alpha=0.3)

    def _draw_obstacles(self, ax):
        obstacles = self.env._get_obstacles()
        for x, y in obstacles:
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                linewidth=1,
                edgecolor='black',
                facecolor='gray',
                alpha=0.8
            )
            ax.add_patch(rect)

    def _draw_goals(self, ax):
        for i, goal in enumerate(self.env.agent_goal_positions):
            if goal:
                x, y = goal
                rect = patches.Rectangle(
                    (x - 0.4, y - 0.4), 0.8, 0.8,
                    linewidth=2,
                    edgecolor=self.agent_colors[i % len(self.agent_colors)],
                    facecolor=self.agent_colors[i % len(self.agent_colors)],
                    alpha=0.2
                )
                ax.add_patch(rect)
                ax.text(x, y, 'G' + str(i), ha='center', va='center',
                       fontweight='bold', fontsize=10,
                       color=self.agent_colors[i % len(self.agent_colors)])

    def _draw_agents(self, ax):
        for i, pos in enumerate(self.env.agent_positions):
            if pos:
                x, y = pos
                circle = plt.Circle(
                    (x, y), 0.3,
                    color=self.agent_colors[i % len(self.agent_colors)],
                    alpha=0.8,
                    zorder=10
                )
                ax.add_patch(circle)
                ax.text(x, y, str(i), ha='center', va='center',
                       color='white', fontweight='bold', zorder=11)

    def _draw_paths(self, ax):
        for i in range(self.env.num_agents):
            agent_name = f'agent{i}'
            if agent_name in self.solution:
                path = self.solution[agent_name]
                xs = [p['x'] for p in path]
                ys = [p['y'] for p in path]
                ax.plot(xs, ys, '--',
                       color=self.agent_colors[i % len(self.agent_colors)],
                       alpha=0.4, linewidth=2)

    def _highlight_collisions(self, ax, frame, agent_artists):
        positions = {}
        for i in range(self.env.num_agents):
            agent_name = f'agent{i}'
            if agent_name in self.solution:
                path = self.solution[agent_name]
                if frame < len(path):
                    pos = (path[frame]['x'], path[frame]['y'])
                else:
                    pos = (path[-1]['x'], path[-1]['y'])
                positions[i] = pos

        collision_agents = set()
        
        for i in positions:
            for j in positions:
                if i < j and positions[i] == positions[j]:
                    collision_agents.add(i)
                    collision_agents.add(j)
        
        if frame in self.collision_steps:
            for i in range(self.env.num_agents):
                if f'agent{i}' in self.solution:
                    collision_agents.add(i)

        for i in range(self.env.num_agents):
            if i in collision_agents:
                agent_artists[i * 2].set_edgecolor('red')
                agent_artists[i * 2].set_linewidth(3)
            else:
                agent_artists[i * 2].set_edgecolor('none')
                agent_artists[i * 2].set_linewidth(1)

    def save_trajectory_plot(self, save_path):
        if not self.solution:
            raise ValueError("No solution available")

        fig, ax = plt.subplots(figsize=(10, 10))

        self._draw_grid(ax)
        self._draw_obstacles(ax)
        self._draw_goals(ax)

        for i in range(self.env.num_agents):
            agent_name = f'agent{i}'
            if agent_name in self.solution:
                path = self.solution[agent_name]
                xs = [p['x'] for p in path]
                ys = [p['y'] for p in path]

                ax.plot(xs, ys, '-',
                       color=self.agent_colors[i % len(self.agent_colors)],
                       linewidth=2, alpha=0.7, label=f'Agent {i}')
                ax.plot(xs[0], ys[0], 'o',
                       color=self.agent_colors[i % len(self.agent_colors)],
                       markersize=10, label=f'Agent {i} start')
                ax.plot(xs, ys, 'o',
                       color=self.agent_colors[i % len(self.agent_colors)],
                       markersize=4, alpha=0.5)

        ax.set_xlim(-0.5, self.env.width - 0.5)
        ax.set_ylim(-0.5, self.env.height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('Agent Trajectories')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
        plt.close()
