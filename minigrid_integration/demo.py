import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import json
from minigrid_integration.multi_agent_env import MultiAgentMinigrid
from minigrid_integration.minigrid_adapter import MinigridAdapter
from minigrid_integration.minigrid_visualizer import MinigridVisualizer
from minigrid.core.world_object import Wall


def run_scenario(scenario_name, algorithm='cbs', save_metrics=True):
    scenarios = {
        '1': simple_crossing,
        '2': collision_avoidance,
        '3': complex_obstacles,
        '4': congested_intersection
    }

    if scenario_name in scenarios:
        metrics = scenarios[scenario_name](algorithm)

        if save_metrics and metrics:
            save_metrics_to_file(metrics, scenario_name, algorithm)

        return metrics
    else:
        print(f"Unknown scenario: {scenario_name}")
        return None


def simple_crossing(algorithm='cbs'):
    print("\n" + "="*60)
    print(f"SCENARIO 1: Simple Crossing ({algorithm})")
    print("="*60)

    env = MultiAgentMinigrid(
        width=8, height=8, num_agents=2,
        agent_start_pos=[(1, 1), (6, 6)],
        agent_goal_pos=[(6, 1), (1, 6)],
        max_steps=100
    )
    env.reset()

    print("  - 2 agents with crossing diagonal paths")
    print("  - Empty grid, no obstacles")

    adapter = MinigridAdapter(env, algorithm=algorithm)

    start_time = time.time()
    solution = adapter.plan_paths()
    planning_time = (time.time() - start_time) * 1000

    if solution:
        stats = adapter.execute_full_plan()
        metrics = collect_metrics(adapter, env, planning_time, stats, scenario='simple', algorithm=algorithm)
        print_solution_summary(adapter, env, stats)
        visualize(env, solution, f'simple_{algorithm}', stats=stats)
        return metrics
    else:
        print("No solution found")
        return None


def collision_avoidance(algorithm='cbs'):
    print("\n" + "="*60)
    print(f"SCENARIO 2: Collision Avoidance ({algorithm})")
    print("="*60)

    starts = [(2, 3), (5, 3)]
    goals = [(5, 3), (2, 3)]

    env = MultiAgentMinigrid(
        width=8, height=8, num_agents=2,
        agent_start_pos=starts,
        agent_goal_pos=goals,
        max_steps=50
    )
    env.reset()

    print(f"  Agent 0 (RED):  {starts[0]} -> {goals[0]}")
    print(f"  Agent 1 (BLUE): {starts[1]} -> {goals[1]}")
    print("  Direct paths would collide")

    adapter = MinigridAdapter(env, algorithm=algorithm)

    start_time = time.time()
    solution = adapter.plan_paths()
    planning_time = (time.time() - start_time) * 1000

    if solution:
        print(f"\n{algorithm} found solution:")
        stats = adapter.execute_full_plan()
        metrics = collect_metrics(adapter, env, planning_time, stats, scenario='collision', algorithm=algorithm)
        print_solution_summary(adapter, env, stats)
        visualize(env, solution, f'collision_{algorithm}', stats=stats, fps=1)
        return metrics
    else:
        print("No solution found")
        return None


def complex_obstacles(algorithm='cbs'):
    print("\n" + "="*60)
    print(f"SCENARIO 3: Complex with Obstacles ({algorithm})")
    print("="*60)

    starts = [(1, 1), (6, 1), (1, 6)]
    goals = [(6, 6), (1, 6), (6, 1)]

    env = MultiAgentMinigrid(
        width=8, height=8, num_agents=3,
        agent_start_pos=starts,
        agent_goal_pos=goals,
        max_steps=150
    )
    env.reset()

    obstacles = [
        (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 3), (5, 3),
        (2, 4), (5, 5)
    ]

    print(f"  - 3 agents with crossing paths")
    print(f"  - {len(obstacles)} obstacles creating maze")

    for obs_x, obs_y in obstacles:
        env.grid.set(obs_x, obs_y, Wall())

    adapter = MinigridAdapter(env, algorithm=algorithm)

    start_time = time.time()
    solution = adapter.plan_paths()
    planning_time = (time.time() - start_time) * 1000

    if solution:
        stats = adapter.execute_full_plan()
        metrics = collect_metrics(adapter, env, planning_time, stats, scenario='complex', algorithm=algorithm)
        print_solution_summary(adapter, env, stats)
        visualize(env, solution, f'complex_{algorithm}', stats=stats, fps=3)
        return metrics
    else:
        print("No solution found")
        return None


def print_paths(solution):
    for agent_name, path in solution.items():
        print(f"\n{agent_name}:")
        for t in range(len(path)):
            pos = path[t]
            if t > 0 and pos['x'] == path[t-1]['x'] and pos['y'] == path[t-1]['y']:
                print(f"  t={t}: ({pos['x']}, {pos['y']}) WAIT")
            else:
                print(f"  t={t}: ({pos['x']}, {pos['y']})")


def print_solution_summary(adapter, env, stats):
    print("\nSolution Summary:")
    total_planned = 0
    total_optimal = 0

    for i in range(env.num_agents):
        planned = adapter.get_planned_path_length(i)
        optimal = adapter.get_optimal_path_length(i)
        ratio = planned / optimal if optimal > 0 else 0

        total_planned += planned
        total_optimal += optimal

        print(f"  Agent {i}: {planned} steps ({ratio:.2f}x optimal)")

    avg_ratio = total_planned / total_optimal if total_optimal > 0 else 0
    print(f"  APLR: {avg_ratio:.2f}x")
    print(f"  Success: {stats['success']}")
    print(f"  Collisions: {stats['collision_count']}")


def visualize(env, solution, name, stats=None, fps=2):
    print(f"\nGenerating visualizations...")

    import os
    output_dir = 'minigrid_integration/output'
    os.makedirs(output_dir, exist_ok=True)

    visualizer_plan = MinigridVisualizer(env, solution)
    visualizer_plan.render_static(
        show_paths=True,
        save_path=f'{output_dir}/{name}_static.png'
    )

    if stats and 'trajectory' in stats:
        exec_solution = {}
        for agent_name, path in stats['trajectory'].items():
            exec_path = []
            for t, pos in enumerate(path):
                exec_path.append({'t': t, 'x': pos[0], 'y': pos[1]})
            exec_solution[agent_name] = exec_path
        
        collision_steps = stats.get('collision_steps', [])
        visualizer_exec = MinigridVisualizer(env, exec_solution, collision_steps=collision_steps)
        visualizer_exec.animate_solution(
            save_path=f'{output_dir}/{name}_animation.gif',
            fps=fps,
            show_plot=False
        )
        visualizer_exec.save_trajectory_plot(f'{output_dir}/{name}_trajectories.png')
    else:
        visualizer_plan.animate_solution(
            save_path=f'{output_dir}/{name}_animation.gif',
            fps=fps,
            show_plot=False
        )
        visualizer_plan.save_trajectory_plot(f'{output_dir}/{name}_trajectories.png')

    print(f"  Saved: output/{name}_animation.gif")
    print(f"  Saved: output/{name}_static.png")
    print(f"  Saved: output/{name}_trajectories.png")


def collect_metrics(adapter, env, planning_time, stats, scenario='unknown', algorithm='unknown'):
    num_agents = env.num_agents

    path_lengths = []
    optimal_lengths = []

    for i in range(num_agents):
        planned = adapter.get_planned_path_length(i)
        optimal = adapter.get_optimal_path_length(i)

        path_lengths.append(planned)
        optimal_lengths.append(optimal)

    total_planned = sum(path_lengths)
    total_optimal = sum(optimal_lengths)
    aplr = total_planned / total_optimal if total_optimal > 0 else 0

    agent_ratios = [
        path_lengths[i] / optimal_lengths[i] if optimal_lengths[i] > 0 else 0
        for i in range(num_agents)
    ]

    metrics = {
        'scenario': scenario,
        'algorithm': algorithm,
        'num_agents': num_agents,
        'success_rate': 1.0 if stats['success'] else 0.0,
        'collision_count': stats['collision_count'],
        'aplr': round(aplr, 3),
        'planning_time_ms': round(planning_time, 2),
        'per_agent_metrics': [
            {
                'agent_id': i,
                'path_length': path_lengths[i],
                'optimal_length': optimal_lengths[i],
                'ratio': round(agent_ratios[i], 3)
            }
            for i in range(num_agents)
        ],
        'total_path_length': total_planned,
        'total_optimal_length': total_optimal
    }

    return metrics


def save_metrics_to_file(metrics, scenario_name, algorithm):
    output_dir = 'minigrid_integration/output'
    os.makedirs(output_dir, exist_ok=True)

    filename = f'{output_dir}/metrics_{scenario_name}_{algorithm}.json'

    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Metrics saved: {filename}")


def save_combined_metrics(results):
    output_dir = 'minigrid_integration/output'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f'{output_dir}/combined_metrics.json'
    
    valid_results = [r for r in results if r is not None]
    
    with open(filename, 'w') as f:
        json.dump(valid_results, f, indent=2)
        
    print(f"\n  Combined metrics saved: {filename}")


def congested_intersection(algorithm='cbs'):
    print("\n" + "="*60)
    print(f"SCENARIO 4: Congested Intersection ({algorithm})")
    print("="*60)
    
    starts = [(3, 1), (1, 3), (6, 3)]
    goals = [(3, 6), (6, 3), (1, 3)]

    env = MultiAgentMinigrid(
        width=8,
        height=8,
        num_agents=3,
        agent_start_pos=starts,
        agent_goal_pos=goals,
        max_steps=50
    )
    env.reset()

    print(f"  Agent 0 (RED):  {starts[0]} -> {goals[0]}")
    print(f"  Agent 1 (BLUE): {starts[1]} -> {goals[1]}")
    print(f"  Agent 2 (GREEN): {starts[2]} -> {goals[2]}")
    print("  Conflicts at (3,3) and along row 3")
    
    adapter = MinigridAdapter(env, algorithm=algorithm)
    
    start_time = time.time()
    solution = adapter.plan_paths()
    planning_time = (time.time() - start_time) * 1000
    
    if solution:
        print(f"\n{algorithm} found solution:")
        stats = adapter.execute_full_plan()
        metrics = collect_metrics(adapter, env, planning_time, stats, scenario="congested", algorithm=algorithm)
        print_solution_summary(adapter, env, stats)
        visualize(env, solution, f'congested_{algorithm}', stats=stats, fps=3)
        return metrics
    else:
        print("No solution found")
        return None


def main():
    algorithms = ['cbs', 'cbs_dijkstra', 'ecbs', 'independent', 'greedy']
    results = []

    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        if len(sys.argv) > 2:
            algo = sys.argv[2]
            if algo in algorithms:
                results.append(run_scenario(scenario, algo))
            else:
                print(f"Unknown algorithm: {algo}")
        elif scenario == 'all':
            for i in range(1, 5):
                for algo in algorithms:
                    results.append(run_scenario(str(i), algo))
        else:
            for algo in algorithms:
                results.append(run_scenario(scenario, algo))
    else:
        for i in range(1, 5):
            for algo in algorithms:
                results.append(run_scenario(str(i), algo))

    save_combined_metrics(results)




if __name__ == '__main__':
    import sys
    import time
    import json
    from minigrid_integration.multi_agent_env import MultiAgentMinigrid
    from minigrid_integration.minigrid_adapter import MinigridAdapter
    from minigrid_integration.minigrid_visualizer import MinigridVisualizer
    from minigrid.core.world_object import Wall

    main()
