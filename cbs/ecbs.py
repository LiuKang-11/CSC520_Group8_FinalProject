import sys
sys.path.insert(0, '../')

import argparse
import yaml
import heapq
from itertools import combinations, count
from copy import deepcopy

from cbs.cbs import Environment, Constraints


class WeightedAStar(object):
    def __init__(self, environment, w_low=1.5):
        self.env = environment
        self.w_low = max(1.0, w_low)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def search(self, agent_name):
        start = self.env.agent_dict[agent_name]["start"]

        open_heap = []
        tie_breaker = count()
        came_from = {}
        g_score = {start: 0}
        h_start = self.env.admissible_heuristic(start, agent_name)
        f_start = g_score[start] + self.w_low * h_start
        heapq.heappush(open_heap, (f_start, g_score[start], next(tie_breaker), start))

        closed = set()

        while open_heap:
            f_curr, g_curr, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if self.env.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            for neighbor in self.env.get_neighbors(current):
                tentative_g = g_curr + 1
                if neighbor in g_score and tentative_g >= g_score[neighbor]:
                    continue

                g_score[neighbor] = tentative_g
                came_from[neighbor] = current

                h = self.env.admissible_heuristic(neighbor, agent_name)
                f_neighbor = tentative_g + self.w_low * h
                heapq.heappush(open_heap, (f_neighbor, tentative_g, next(tie_breaker), neighbor))

        return None


class ECBSNode(object):
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0
        self.num_conflicts = 0

    def _constraint_signature(self):
        items = []
        for agent in sorted(self.constraint_dict.keys()):
            vc = frozenset(self.constraint_dict[agent].vertex_constraints)
            ec = frozenset(self.constraint_dict[agent].edge_constraints)
            items.append((agent, vc, ec))
        return tuple(items)

    def __eq__(self, other):
        if not isinstance(other, ECBSNode):
            return NotImplemented
        return self._constraint_signature() == other._constraint_signature()

    def __hash__(self):
        return hash(self._constraint_signature())


def count_conflicts(env, solution):
    if not solution:
        return 0

    conflict_set = set()
    max_t = max(len(plan) for plan in solution.values())
    agents = list(solution.keys())

    for t in range(max_t):
        for agent_1, agent_2 in combinations(agents, 2):
            state_1 = env.get_state(agent_1, solution, t)
            state_2 = env.get_state(agent_2, solution, t)
            if state_1.is_equal_except_time(state_2):
                ag = tuple(sorted([agent_1, agent_2]))
                loc = (state_1.location.x, state_1.location.y)
                conflict_set.add(('vertex', ag, t, loc))

    if max_t > 1:
        for t in range(max_t - 1):
            for agent_1, agent_2 in combinations(agents, 2):
                state_1a = env.get_state(agent_1, solution, t)
                state_1b = env.get_state(agent_1, solution, t + 1)

                state_2a = env.get_state(agent_2, solution, t)
                state_2b = env.get_state(agent_2, solution, t + 1)

                if state_1a.is_equal_except_time(state_2b) and \
                   state_1b.is_equal_except_time(state_2a):
                    ag = tuple(sorted([agent_1, agent_2]))
                    loc1 = (state_1a.location.x, state_1a.location.y)
                    loc2 = (state_1b.location.x, state_1b.location.y)
                    locs = tuple(sorted([loc1, loc2]))
                    conflict_set.add(('edge', ag, t, locs))

    return len(conflict_set)


class ECBS(object):
    def __init__(self, environment, w_low=1.5, w_high=1.5):
        self.env = environment
        self.w_low = max(1.0, w_low)
        self.w_high = max(1.0, w_high)

    def _compute_solution_for_node(self, constraint_dict):
        solution = {}
        w_astar = WeightedAStar(self.env, self.w_low)

        for agent in self.env.agent_dict.keys():
            self.env.constraints = constraint_dict.setdefault(agent, Constraints())
            self.env.constraint_dict = constraint_dict

            local_solution = w_astar.search(agent)
            if not local_solution:
                return None
            solution[agent] = local_solution

        return solution

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [
                {
                    't': state.time,
                    'x': state.location.x,
                    'y': state.location.y
                }
                for state in path
            ]
            plan[agent] = path_dict_list
        return plan

    def search(self):
        root = ECBSNode()
        root.constraint_dict = {
            agent: Constraints() for agent in self.env.agent_dict.keys()
        }

        root.solution = self._compute_solution_for_node(root.constraint_dict)
        if root.solution is None:
            return None

        root.cost = self.env.compute_solution_cost(root.solution)
        root.num_conflicts = count_conflicts(self.env, root.solution)

        open_heap = []
        counter = count()
        heapq.heappush(open_heap, (root.cost, next(counter), root))

        open_dict = {root: root.cost}
        closed_set = set()

        while open_heap:
            while open_heap and open_heap[0][2] not in open_dict:
                heapq.heappop(open_heap)
            
            if not open_heap:
                break

            best_cost = open_heap[0][0]
            bound = self.w_high * best_cost

            focal = []
            for cost, _, node in open_heap:
                if node in open_dict and cost <= bound:
                    focal.append(node)

            if not focal:
                _, _, P = heapq.heappop(open_heap)
                if P not in open_dict:
                    continue
            else:
                P = min(focal, key=lambda n: (n.num_conflicts, n.cost))

            del open_dict[P]
            closed_set.add(P)

            self.env.constraint_dict = P.constraint_dict
            conflict = self.env.get_first_conflict(P.solution)
            if not conflict:
                print("Solution found")
                return P.solution

            new_constraints_per_agent = self.env.create_constraints_from_conflict(conflict)

            for agent, added_constraints in new_constraints_per_agent.items():
                child = deepcopy(P)
                child.constraint_dict[agent].add_constraint(added_constraints)

                self.env.constraint_dict = child.constraint_dict
                child.solution = self._compute_solution_for_node(child.constraint_dict)
                if child.solution is None:
                    continue

                child.cost = self.env.compute_solution_cost(child.solution)
                child.num_conflicts = count_conflicts(self.env, child.solution)

                if child not in closed_set and child not in open_dict:
                    heapq.heappush(open_heap, (child.cost, next(counter), child))
                    open_dict[child] = child.cost

        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param", help="input file containing map and obstacles")
    parser.add_argument("output", help="output file with the schedule")
    parser.add_argument(
        "--w_low",
        type=float,
        default=1.5,
        help="low-level weight (Weighted A*), >= 1.0"
    )
    parser.add_argument(
        "--w_high",
        type=float,
        default=1.5,
        help="high-level ECBS focal weight, >= 1.0"
    )
    args = parser.parse_args()

    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            return

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param["agents"]

    env = Environment(dimension, agents, obstacles)

    ecbs = ECBS(env, w_low=args.w_low, w_high=args.w_high)
    solution_internal = ecbs.search()
    if solution_internal is None:
        print("Solution not found")
        return

    schedule = ecbs.generate_plan(solution_internal)
    total_cost = env.compute_solution_cost(solution_internal)

    output = {
        "schedule": schedule,
        "cost": total_cost
    }

    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)


if __name__ == "__main__":
    main()