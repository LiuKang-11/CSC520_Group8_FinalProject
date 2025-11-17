import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml

from cbs.cbs import CBS, Environment
from sipp.multi_sipp import MultiSIPP


class MinigridAdapter:

    def __init__(self, env, algorithm='cbs'):
        self.env = env
        self.algorithm = algorithm.lower()
        self.solution = None
        self.current_step = 0

    def plan_paths(self):
        grid_state = self.env.get_grid_state()

        if self.algorithm == 'cbs':
            self.solution = self._plan_with_cbs(grid_state)
        elif self.algorithm == 'sipp':
            self.solution = self._plan_with_sipp(grid_state)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return self.solution

    def _plan_with_cbs(self, grid_state):
        env = Environment(
            dimension=grid_state['dimension'],
            agents=grid_state['agents'],
            obstacles=grid_state['obstacles']
        )

        cbs = CBS(env)

        solution = cbs.search()

        if not solution:
            print("CBS failed to find solution!")
            return None

        return solution

    def _plan_with_sipp(self, grid_state):
        sipp = MultiSIPP(
            dimension=grid_state['dimension'],
            agents=grid_state['agents'],
            obstacles=grid_state['obstacles']
        )

        sipp_solution = sipp.search()

        if not sipp_solution:
            print("SIPP failed to find solution!")
            return None

        solution = {}
        for agent_name, path in sipp_solution.items():
            schedule = []
            for t, pos in enumerate(path):
                schedule.append({
                    't': t,
                    'x': pos[0],
                    'y': pos[1]
                })
            solution[agent_name] = schedule

        return solution

    def execute_step(self):
        if self.solution is None:
            raise RuntimeError("No solution available. Call plan_paths() first.")

        current_positions = self.env.agent_positions.copy()

        actions = []
        for i in range(self.env.num_agents):
            agent_name = f'agent{i}'

            if agent_name not in self.solution:
                actions.append(None)
                continue

            schedule = self.solution[agent_name]

            if self.current_step >= len(schedule):
                actions.append(None)
                continue

            next_pos = schedule[self.current_step]
            target_x, target_y = next_pos['x'], next_pos['y']
            current_x, current_y = current_positions[i]

            action = self._position_to_action(
                current=(current_x, current_y),
                target=(target_x, target_y),
                agent_id=i
            )
            actions.append(action)

        obs, reward, terminated, truncated, info = self.env.step(actions)

        self.current_step += 1

        return obs, reward, terminated, truncated, info

    def _position_to_action(self, current, target, agent_id):
        current_x, current_y = current
        target_x, target_y = target

        if current_x == target_x and current_y == target_y:
            return None

        dx = target_x - current_x
        dy = target_y - current_y

        if dx > 0:
            required_dir = 0
        elif dx < 0:
            required_dir = 2
        elif dy > 0:
            required_dir = 1
        else:
            required_dir = 3

        current_dir = self.env.agent_dirs[agent_id]

        if current_dir != required_dir:
            diff = (required_dir - current_dir) % 4
            if diff == 1 or diff == -3:
                return self.env.actions.right
            elif diff == 3 or diff == -1:
                return self.env.actions.left
            else:
                return self.env.actions.right
        else:
            return self.env.actions.forward

    def execute_full_plan(self, max_steps=None):
        if self.solution is None:
            raise RuntimeError("No solution available. Call plan_paths() first.")

        if max_steps is None:
            max_steps = max(len(schedule) for schedule in self.solution.values())

        self.current_step = 0
        total_reward = 0
        collision_count = 0
        success = False

        trajectory = {f'agent{i}': [] for i in range(self.env.num_agents)}

        for step in range(max_steps):
            obs, reward, terminated, truncated, info = self.execute_step()

            total_reward += reward
            if info.get('collision', False):
                collision_count += 1

            for i, pos in enumerate(info['agent_positions']):
                trajectory[f'agent{i}'].append(pos)

            if terminated:
                success = True
                break
            if truncated:
                break

        stats = {
            'success': success,
            'total_reward': total_reward,
            'steps_taken': self.current_step,
            'collision_count': collision_count,
            'trajectory': trajectory
        }

        return stats

    def save_solution(self, filepath):
        if self.solution is None:
            raise RuntimeError("No solution to save. Call plan_paths() first.")

        output = {
            'schedule': self.solution,
            'cost': sum(len(schedule) for schedule in self.solution.values())
        }

        with open(filepath, 'w') as f:
            yaml.dump(output, f, default_flow_style=None)

        print(f"Solution saved to {filepath}")

    @staticmethod
    def load_solution(filepath):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return data.get('schedule', {})

    def reset(self):
        self.current_step = 0
        self.solution = None

    def get_planned_path_length(self, agent_id):
        if self.solution is None:
            return 0

        agent_name = f'agent{agent_id}'
        if agent_name not in self.solution:
            return 0

        return len(self.solution[agent_name])

    def get_optimal_path_length(self, agent_id):
        if agent_id >= len(self.env.agent_start_positions) or \
           agent_id >= len(self.env.agent_goal_positions):
            return 0

        start = self.env.agent_start_positions[agent_id]
        goal = self.env.agent_goal_positions[agent_id]

        if start is None or goal is None:
            return 0

        return abs(goal[0] - start[0]) + abs(goal[1] - start[1])
