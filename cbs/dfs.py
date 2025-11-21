"""
DFS search (Depth-First Search used as low-level planner)
OPTIMIZED VERSION
"""

class DFS:
    def __init__(self, env):
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
        OPTIMIZED DFS with iterative deepening and depth limit
        """
        initial_state = self.agent_dict[agent_name]["start"]
        goal_state = self.agent_dict[agent_name]["goal"]
        
        # Calculate reasonable depth limit based on Manhattan distance
        start_loc = initial_state.location
        goal_loc = goal_state.location
        manhattan_dist = abs(goal_loc.x - start_loc.x) + abs(goal_loc.y - start_loc.y)
        max_depth = manhattan_dist + 10  # Add buffer for obstacles
        
        # Use iterative deepening DFS for completeness with reasonable depth
        for depth_limit in range(1, max_depth + 1):
            stack = [(initial_state, 0)]  # (state, current_depth)
            visited = set()
            came_from = {}
            
            visited.add(initial_state)
            
            while stack:
                current, depth = stack.pop()
                
                if self.is_at_goal(current, agent_name):
                    return self.reconstruct_path(came_from, current)
                
                # Skip if we've reached depth limit
                if depth >= depth_limit:
                    continue
                
                # Get neighbors in reverse order to explore more promising directions first
                neighbor_list = self.get_neighbors(current)
                # Reverse to explore towards goal first (heuristic ordering)
                neighbor_list.reverse()
                
                for neighbor in neighbor_list:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        came_from[neighbor] = current
                        stack.append((neighbor, depth + 1))
        
        return False