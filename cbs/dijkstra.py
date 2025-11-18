"""
Dijstra search (Dijkstra’s algorithm used as low-level planner)
"""

class Dijkstra:
    def __init__(self, env):
        # 
        self.agent_dict = env.agent_dict
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):
        """
        low level search with Dijkstra algorithm
        (A* with heuristic = 0)
        """
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1

        closed_set = set()
        open_set = {initial_state}

        came_from = {}

        # g_score: best known cost from start to this state
        g_score = {}
        g_score[initial_state] = 0

        while open_set:
            # min g_score 
            temp_dict = {open_item: g_score.setdefault(open_item, float("inf")) 
                         for open_item in open_set}
            current = min(temp_dict, key=temp_dict.get)

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            open_set -= {current}
            closed_set |= {current}

            neighbor_list = self.get_neighbors(current)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                if neighbor not in open_set:
                    open_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue

                
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score

        
        return False
