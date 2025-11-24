"""

Extension of SIPP to multi-robot scenarios

author: Ashwin Bose (@atb033)

See the article: 10.1109/ICRA.2011.5980306

"""

import argparse
import yaml
from math import fabs
from sipp.graph_generation import SippGraph, State
from sipp.sipp import SippPlanner


class MultiSIPP:

    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.agents = agents
        self.obstacles = obstacles

    def search(self):
        map_data = {
            'map': {
                'dimensions': self.dimension,
                'obstacles': self.obstacles
            },
            'agents': self.agents,
            'dynamic_obstacles': {}
        }

        solution = {}

        for i, agent in enumerate(self.agents):
            sipp_planner = SippPlanner(map_data, i)

            if sipp_planner.compute_plan():
                plan = sipp_planner.get_plan()
                agent_name = agent['name']

                if agent_name in plan:
                    path = [(state['x'], state['y']) for state in plan[agent_name]]
                    solution[agent_name] = path

                    map_data['dynamic_obstacles'][agent_name] = plan[agent_name]
            else:
                return None

        return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="input file containing map and dynamic obstacles")
    parser.add_argument("output", help="output file with the schedule")
    
    args = parser.parse_args()
    
    # Read Map
    with open(args.map, 'r') as map_file:
        try:
            map = yaml.load(map_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    # Output file
    output = dict()
    output["schedule"] = dict()

    for i in range(len(map["agents"])):
        sipp_planner = SippPlanner(map,i)
    
        if sipp_planner.compute_plan():
            plan = sipp_planner.get_plan()
            output["schedule"].update(plan)
            map["dynamic_obstacles"].update(plan)

            with open(args.output, 'w') as output_yaml:
                yaml.safe_dump(output, output_yaml)  
        else: 
            print("Plan not found")


if __name__ == "__main__":
    main()
