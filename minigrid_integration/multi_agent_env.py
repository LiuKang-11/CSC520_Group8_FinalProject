import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class MultiAgentMinigrid(MiniGridEnv):
    def __init__(
        self,
        width=8,
        height=8,
        num_agents=2,
        agent_start_pos=None,
        agent_goal_pos=None,
        max_steps=100,
        see_through_walls=True,
        **kwargs
    ):
        self.num_agents = num_agents
        self.agent_start_positions = agent_start_pos or []
        self.agent_goal_positions = agent_goal_pos or []

        self.agent_colors = ['red', 'blue', 'green'][:num_agents]

        self.agent_positions = [None] * num_agents
        self.agent_dirs = [0] * num_agents

        self.goal_objects = []

        from minigrid.core.mission import MissionSpace
        mission_space = MissionSpace(mission_func=lambda: "Reach your goal")

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        self.goal_objects = []
        for i, goal_pos in enumerate(self.agent_goal_positions):
            if goal_pos:
                x, y = goal_pos
                goal = Goal()
                self.grid.set(x, y, goal)
                self.goal_objects.append((x, y))

        if self.agent_start_positions:
            for i, start_pos in enumerate(self.agent_start_positions):
                if start_pos and i < self.num_agents:
                    self.agent_positions[i] = start_pos
        else:
            for i in range(self.num_agents):
                self.place_agent(agent_id=i)

    def place_agent(self, agent_id=0, top=None, size=None, rand_dir=True, max_tries=100):
        if top is None:
            top = (1, 1)
        if size is None:
            size = (self.width - 2, self.height - 2)

        for _ in range(max_tries):
            x = self._rand_int(top[0], top[0] + size[0])
            y = self._rand_int(top[1], top[1] + size[1])

            if self.grid.get(x, y) is None:
                occupied = False
                for j, pos in enumerate(self.agent_positions):
                    if j != agent_id and pos == (x, y):
                        occupied = True
                        break

                if not occupied:
                    self.agent_positions[agent_id] = (x, y)
                    if rand_dir:
                        self.agent_dirs[agent_id] = self._rand_int(0, 4)
                    return

        raise ValueError(f"Could not place agent {agent_id}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        self.step_count = 0

        self._gen_grid(self.width, self.height)

        if not self.agent_start_positions:
            self.agent_positions = [None] * self.num_agents
            for i in range(self.num_agents):
                self.place_agent(agent_id=i)
        else:
            self.agent_positions = [tuple(pos) for pos in self.agent_start_positions]
            for i in range(self.num_agents):
                self.agent_dirs[i] = 0

        if self.agent_positions[0]:
            self.agent_pos = np.array(self.agent_positions[0])
            self.agent_dir = self.agent_dirs[0]
        else:
            self.agent_pos = np.array([1, 1])
            self.agent_dir = 0

        obs = self.gen_obs()
        return obs, {}

    def step(self, actions):
        rewards = [0] * self.num_agents
        terminated = False
        truncated = False

        old_positions = self.agent_positions.copy()

        for agent_id, action in enumerate(actions):
            if action is not None:
                reward = self._execute_action(agent_id, action)
                rewards[agent_id] = reward

        collision = self._check_collisions(old_positions)
        if collision:
            self.agent_positions = old_positions
            rewards = [-1] * self.num_agents

        all_at_goal = all(
            self.agent_positions[i] == self.agent_goal_positions[i]
            for i in range(self.num_agents)
            if self.agent_goal_positions[i] is not None
        )

        if all_at_goal:
            terminated = True
            rewards = [10] * self.num_agents

        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        info = {
            'agent_positions': self.agent_positions.copy(),
            'collision': collision,
            'all_at_goal': all_at_goal
        }

        return obs, sum(rewards), terminated, truncated, info

    def _execute_action(self, agent_id, action):
        reward = -0.1

        x, y = self.agent_positions[agent_id]
        direction = self.agent_dirs[agent_id]

        if action == self.actions.left:
            self.agent_dirs[agent_id] = (direction - 1) % 4

        elif action == self.actions.right:
            self.agent_dirs[agent_id] = (direction + 1) % 4

        elif action == self.actions.forward:
            if direction == 0:
                new_x, new_y = x + 1, y
            elif direction == 1:
                new_x, new_y = x, y + 1
            elif direction == 2:
                new_x, new_y = x - 1, y
            else:
                new_x, new_y = x, y - 1

            cell = self.grid.get(new_x, new_y)
            if cell is None or cell.can_overlap():
                self.agent_positions[agent_id] = (new_x, new_y)
            else:
                reward = -0.5

        return reward

    def _check_collisions(self, old_positions):
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if self.agent_positions[i] == self.agent_positions[j]:
                    return True
                
                if old_positions and self.agent_positions[i] == old_positions[j] and \
                   self.agent_positions[j] == old_positions[i]:
                    return True
        return False

    def gen_obs(self):
        obs = {
            'grid_size': (self.width, self.height),
            'agent_positions': self.agent_positions.copy(),
            'agent_goals': self.agent_goal_positions.copy(),
            'obstacles': self._get_obstacles()
        }
        return obs

    def _get_obstacles(self):
        obstacles = []
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if cell is not None and isinstance(cell, Wall):
                    obstacles.append((x, y))
        return obstacles

    def render(self):
        img = super().render()
        return img

    def get_grid_state(self):
        return {
            'dimension': [self.width, self.height],
            'agents': [
                {
                    'name': f'agent{i}',
                    'start': list(self.agent_positions[i]) if self.agent_positions[i] else None,
                    'goal': list(self.agent_goal_positions[i]) if i < len(self.agent_goal_positions) else None
                }
                for i in range(self.num_agents)
            ],
            'obstacles': self._get_obstacles()
        }
