import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml

from cbs.cbs import CBS, Environment
from cbs.cbs_dijkstra import CBS as CBSDijkstra
from cbs.ecbs import ECBS
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
        elif self.algorithm == 'cbs_dijkstra':
            self.solution = self._plan_with_cbs_dijkstra(grid_state)
        elif self.algorithm == 'ecbs':
            self.solution = self._plan_with_ecbs(grid_state)
        elif self.algorithm == 'independent':
            self.solution = self._plan_with_independent_astar(grid_state)
        elif self.algorithm == 'greedy':
            self.solution = self._plan_with_greedy(grid_state)
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

    def _plan_with_independent_astar(self, grid_state):
        env = Environment(
            dimension=grid_state['dimension'],
            agents=grid_state['agents'],
            obstacles=grid_state['obstacles']
        )

        solution = {}
        for agent in env.agent_dict.keys():
            path = env.a_star.search(agent)
            if not path:
                print(f"Independent A* failed for {agent}")
                return None
            
            path_dict_list = [{'t': state.time, 'x': state.location.x, 'y': state.location.y} for state in path]
            solution[agent] = path_dict_list
            
        return solution

    def _plan_with_greedy(self, grid_state):
        agents = grid_state['agents']
        obstacles = set(grid_state['obstacles'])
        dimensions = grid_state['dimension']
        
        current_positions = {a['name']: (a['start'][0], a['start'][1]) for a in agents}
        goals = {a['name']: (a['goal'][0], a['goal'][1]) for a in agents}
        
        solution = {a['name']: [{'t': 0, 'x': a['start'][0], 'y': a['start'][1]}] for a in agents}
        
        max_steps = 100 # Safety break
        finished = {a['name']: False for a in agents}
        
        for t in range(1, max_steps + 1):
            if all(finished.values()):
                break
                
            next_positions = {}
            for agent in agents:
                name = agent['name']
                if finished[name]:
                    next_positions[name] = current_positions[name]
                    continue
                
                curr = current_positions[name]
                goal = goals[name]
                
                best_move = curr
                min_dist = abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])
                
                candidates = [
                    (curr[0]+1, curr[1]), (curr[0]-1, curr[1]),
                    (curr[0], curr[1]+1), (curr[0], curr[1]-1),
                    curr # Wait
                ]
                
                for cand in candidates:
                    if not (0 <= cand[0] < dimensions[0] and 0 <= cand[1] < dimensions[1]):
                        continue
                    if cand in obstacles:
                        continue
                        
                    dist = abs(cand[0] - goal[0]) + abs(cand[1] - goal[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_move = cand
                    elif dist == min_dist and cand == goal: # Prefer goal
                        best_move = cand
                        
                next_positions[name] = best_move

            final_positions = {}
            occupied = set()
            
            for agent in agents:
                name = agent['name']
                proposed = next_positions[name]
                
                if proposed in occupied:
                    final_positions[name] = current_positions[name]
                else:
                    final_positions[name] = proposed
                    occupied.add(proposed)
            
            for name, pos in final_positions.items():
                current_positions[name] = pos
                solution[name].append({'t': t, 'x': pos[0], 'y': pos[1]})
                if pos == goals[name]:
                    finished[name] = True
                    
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

    def _plan_with_cbs_dijkstra(self, grid_state):
        env = Environment(
            dimension=grid_state['dimension'],
            agents=grid_state['agents'],
            obstacles=grid_state['obstacles']
        )

        cbs = CBSDijkstra(env)

        solution = cbs.search()

        if not solution:
            print("CBS (Dijkstra) failed to find solution!")
            return None

        return solution

    def _plan_with_ecbs(self, grid_state):
        env = Environment(
            dimension=grid_state['dimension'],
            agents=grid_state['agents'],
            obstacles=grid_state['obstacles']
        )

        ecbs = ECBS(env, w_low=1.5, w_high=1.5)

        solution_internal = ecbs.search()

        if not solution_internal:
            print("ECBS failed to find solution!")
            return None
            
        solution = ecbs.generate_plan(solution_internal)

        return solution

    def execute_step(self):
        if self.solution is None:
            raise RuntimeError("No solution available. Call plan_paths() first.")

        if not hasattr(self, 'agent_plan_indices'):
            self.agent_plan_indices = [0] * self.env.num_agents

        current_positions = self.env.agent_positions.copy()
        
        actions = []
        for i in range(self.env.num_agents):
            agent_name = f'agent{i}'

            if agent_name not in self.solution:
                actions.append(None)
                continue

            schedule = self.solution[agent_name]
            idx = self.agent_plan_indices[i]
            
            if idx >= len(schedule):
                actions.append(None)
                continue
                
            target_node = schedule[idx]
            target_pos = (target_node['x'], target_node['y'])
            current_pos = current_positions[i]
            
            if current_pos == target_pos:
                self.agent_plan_indices[i] += 1
                idx = self.agent_plan_indices[i]
                if idx >= len(schedule):
                    actions.append(None)
                    continue
                target_node = schedule[idx]
                target_pos = (target_node['x'], target_node['y'])
            
            target_occupied = False
            for j, other_pos in enumerate(current_positions):
                if i != j and other_pos == target_pos:
                    target_occupied = True
                    break
            
            if target_occupied:
                print(f"Agent {i} waiting for {target_pos} to clear")
                actions.append(None)
                continue

            action = self._position_to_action(
                current=current_pos,
                target=target_pos,
                agent_id=i
            )
            print(f"Agent {i} at {current_pos} targeting {target_pos} -> Action {action}")
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
        if abs(dx) + abs(dy) > 1:
            if abs(dx) > abs(dy):
                dy = 0
            else:
                dx = 0

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

        max_plan_len = max(len(schedule) for schedule in self.solution.values())
        
        if max_steps is None:
            max_steps = max_plan_len * 5

        self.current_step = 0
        total_reward = 0
        collision_count = 0
        success = False

        trajectory = {f'agent{i}': [] for i in range(self.env.num_agents)}
        
        self.agent_plan_indices = [0] * self.env.num_agents
        collision_steps = set()
        execution_step_count = 0
        target_plan_time = 1
        
        while target_plan_time < max_plan_len and execution_step_count < max_steps:
            all_reached = True
            current_positions = self.env.agent_positions
            
            for i in range(self.env.num_agents):
                agent_name = f'agent{i}'
                if agent_name not in self.solution:
                    continue
                
                schedule = self.solution[agent_name]
                if target_plan_time < len(schedule):
                    target_pos = (schedule[target_plan_time]['x'], schedule[target_plan_time]['y'])
                    if current_positions[i] != target_pos:
                        all_reached = False
                        break
            
            if all_reached:
                target_plan_time += 1
                if target_plan_time >= max_plan_len:
                    success = True
                    break
                continue

            actions = []
            for i in range(self.env.num_agents):
                agent_name = f'agent{i}'
                if agent_name not in self.solution:
                    actions.append(None)
                    continue

                schedule = self.solution[agent_name]
                
                if target_plan_time >= len(schedule):
                    actions.append(None)
                    continue
                    
                target_node = schedule[target_plan_time]
                target_pos = (target_node['x'], target_node['y'])
                current_pos = current_positions[i]
                
                if current_pos == target_pos:
                    actions.append(None)
                else:
                    target_occupied = False
                    for j, other_pos in enumerate(current_positions):
                        if i != j and other_pos == target_pos:
                            target_occupied = True
                            break
                    
                    if target_occupied:
                        actions.append(None)
                    else:
                        action = self._position_to_action(
                            current=current_pos,
                            target=target_pos,
                            agent_id=i
                        )
                        actions.append(action)

            obs, reward, terminated, truncated, info = self.env.step(actions)
            
            total_reward += reward
            if info.get('collision', False):
                collision_count += 1
                collision_steps.add(execution_step_count)
                for i, pos in enumerate(info['agent_positions']):
                    trajectory[f'agent{i}'].append(pos)
                execution_step_count += 1
                self.current_step = execution_step_count
                success = False
                break

            for i, pos in enumerate(info['agent_positions']):
                trajectory[f'agent{i}'].append(pos)
                
            execution_step_count += 1
            self.current_step = execution_step_count
            
            if terminated:
                success = True
                break

        stats = {
            'success': success,
            'total_reward': total_reward,
            'steps_taken': self.current_step,
            'collision_count': collision_count,
            'collision_steps': list(collision_steps),
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
