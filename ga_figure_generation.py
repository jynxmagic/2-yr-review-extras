import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import heapq
import random
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# Set a random seed for reproducibility (optional)
random.seed(42)
np.random.seed(42)
mpl.rc('font', family='serif')  # Use a serif font (Computer Modern)
mpl.rcParams['font.size'] = 11
# Parameters for the grid
grid_size = (22, 22)  # Grid size remains slightly larger for visualization
start = (0, 0)
goal = (11, 11)  # Central goal position, not at the edge
population_size = 100
mutation_rate = 0.1  # Increased mutation rate for diversity
num_generations = 1000  # Increased number of generations
max_chromosome_length = 200  # Maximum number of bits in a chromosome
min_chromosome_length = 10   # Minimum number of bits in a chromosome

# Fixed obstacles
obstacles = [
    (5, 5), (5, 6), (5, 7), (5, 8),
    (6, 5), (7, 5), (8, 5), (9, 5),
    (10, 10), (10, 11), (10, 12),
    (15, 15), (16, 15), (17, 15),
    (2, 2), (2, 3), (2, 4),
    (3, 2), (3, 3),
    (7, 10), (8, 10), (9, 10),
    (12, 14), (13, 14), (14, 14),
    (17, 5), (17, 6), (17, 7),
    (12, 2), (13, 2), (14, 2),
    (0, 18), (1, 18), (2, 18), (3, 18), (4, 18),
    (10, 17), (11, 17), (12, 17),
    (19, 10), (18, 10), (17, 10),
    (6, 15), (7, 15), (8, 15),
    (14, 7), (14, 8), (14, 9),
    (15, 3), (16, 3), (17, 3),
    # Remove obstacles that block the goal or make it unreachable
]

# Remove any obstacles at the goal position and adjust nearby if needed
for pos in [goal, (10, 11), (11, 10), (10, 10)]:
    if pos in obstacles:
        obstacles.remove(pos)

# Generate the grid with obstacles
grid = np.zeros(grid_size)
for x, y in obstacles:
    grid[x, y] = 1

# Movement direction mapping
direction_map = {
    '00': (-1, 0),  # Up
    '01': (0, 1),   # Right
    '10': (1, 0),   # Down
    '11': (0, -1),  # Left
}

# Decode a binary chromosome into a path
def decode_chromosome(chromosome):
    path = [start]
    idx = 0
    while idx + 2 <= len(chromosome):
        move_bits = chromosome[idx:idx+2]
        move_str = ''.join(str(bit) for bit in move_bits)
        if move_str in direction_map:
            dx, dy = direction_map[move_str]
            x, y = path[-1]
            new_pos = (x + dx, y + dy)
            # Check within grid bounds and avoid obstacles
            if (0 <= new_pos[0] < grid_size[0] and
                0 <= new_pos[1] < grid_size[1] and
                grid[new_pos] == 0):
                path.append(new_pos)
                # Stop if goal is reached
                if new_pos == goal:
                    break
            else:
                break  # Stop if out of bounds or hitting an obstacle
        else:
            break  # Invalid move, stop the path
        idx += 2
    return path

# Fitness function for binary chromosomes
def fitness_binary(chromosome):
    path = decode_chromosome(chromosome)
    last_pos = path[-1]
    distance_to_goal = heuristic(last_pos, goal)
    fitness = -distance_to_goal
    if last_pos == goal:
        fitness += 10000  # Bonus for reaching the goal
    return fitness

# Generate initial population of binary chromosomes
def generate_initial_population():
    population = []
    for _ in range(population_size):
        chromosome_length = random.randrange(min_chromosome_length, max_chromosome_length + 1, 2)
        chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(chromosome)
    return population

# Tournament selection
def tournament_selection_binary(population, k=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, k)
        tournament_fitness = [fitness_binary(individual) for individual in tournament]
        winner = tournament[np.argmax(tournament_fitness)]
        selected.append(winner)
    return selected

# Crossover function for binary chromosomes
def crossover_binary(parent1, parent2):
    # Ensure crossover at even index to not split a move
    min_len = min(len(parent1), len(parent2))
    if min_len < 4:
        return parent1.copy(), parent2.copy()  # Skip crossover if too short
    crossover_point = random.randrange(2, min_len - 1, 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation function for binary chromosomes
def mutate_binary(chromosome):
    mutated_chromosome = chromosome.copy()
    i = 0
    while i < len(mutated_chromosome):
        if random.random() < mutation_rate:
            mutation_type = random.choice(['flip', 'delete', 'insert'])
            if mutation_type == 'flip':
                mutated_chromosome[i] = 1 - mutated_chromosome[i]
                i += 1
            elif mutation_type == 'delete' and len(mutated_chromosome) > min_chromosome_length:
                del mutated_chromosome[i:i+2]
            elif mutation_type == 'insert' and len(mutated_chromosome) < max_chromosome_length:
                new_move = [random.randint(0, 1), random.randint(0, 1)]
                mutated_chromosome[i:i] = new_move
                i += 2  # Skip the inserted move
            else:
                i += 1
        else:
            i += 1
    return mutated_chromosome

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Plotting function for paths decoded from binary chromosomes
def plot_path_from_chromosome(chromosome, algorithm_name):
    # Check if the input is a chromosome or already a list of coordinates
    if isinstance(chromosome[0], tuple):
        path = chromosome  # Directly use the path if it's a list of coordinates (for A*)
    else:
        path = decode_chromosome(chromosome)  # Decode if it's a binary chromosome (for GA)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1, grid_size[1])
    ax.set_ylim(-1, grid_size[0])
    ax.set_xticks(np.arange(-1, grid_size[1]+1, 1))
    ax.set_yticks(np.arange(-1, grid_size[0]+1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_aspect('equal')

    # Plot obstacles with hatching
    for (x, y), value in np.ndenumerate(grid):
        if value == 1:
            rect = Rectangle((y, grid_size[0] - x - 1), 1, 1, facecolor='white', hatch='////', edgecolor='black')
            ax.add_patch(rect)
            
    if algorithm_name == 'Genetic Algorithm':
        line_style = 'dashed'
        marker_style = 's'
        title_text = 'Genetic Algorithm Path'
    else:
        line_style = 'solid'
        marker_style = 'o'
        title_text = 'Optimal Path'
    ax.set_title(title_text, fontsize=28)

    # Plot the path
    if len(path) > 1:
        path_x, path_y = zip(*path)
        path_x = [grid_size[0] - x - 1 for x in path_x]
        path_y = [y for y in path_y]
        line_style = 'dashed' if algorithm_name == 'Genetic Algorithm' else 'solid'
        marker_style = 's' if algorithm_name == 'Genetic Algorithm' else 'o'
        ax.plot(path_y, path_x, linestyle=line_style, marker=marker_style, color='black', linewidth=1.5, markersize=5, label=f'{algorithm_name} Path')

    # Plot start and goal positions
    start_x, start_y = grid_size[0] - start[0] - 1, start[1]
    goal_x, goal_y = grid_size[0] - goal[0] - 1, goal[1]
    ax.scatter(start_y, start_x, marker='^', color='black', s=100, label='Start')
    ax.scatter(goal_y, goal_x, marker='*', color='black', s=150, label='Goal')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], linestyle='none', marker='^', color='black', label='Start'),
        Line2D([0], [0], linestyle='none', marker='*', color='black', label='Goal'),
        Line2D([0], [0], linestyle=line_style, marker=marker_style, color='black', label=f'{algorithm_name} Path'),
        Rectangle((0, 0), 1, 1, facecolor='white', hatch='////', edgecolor='black', label='Obstacle')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(title_text, fontsize=28, family='serif')
    plt.tight_layout()
    if algorithm_name == 'Genetic Algorithm':
        plt.savefig('ga_search.pdf')
    else:
        plt.savefig('a_star_search.pdf')
    # plt.show()

# A* Algorithm for the entire grid
def a_star_full_grid(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < grid_size[0] and
                0 <= neighbor[1] < grid_size[1] and
                grid[neighbor] == 0):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Main GA loop
def run_genetic_algorithm():
    population = generate_initial_population()
    elite_size = int(0.1 * population_size)  # Top 10% are carried over
    best_fitness = float('-inf')
    best_chromosome = None

    for generation in range(num_generations):
        population_fitness = [(individual, fitness_binary(individual)) for individual in population]
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        population = [individual for individual, fitness in population_fitness]

        current_best_fitness = population_fitness[0][1]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = population[0]

        best_path = decode_chromosome(best_chromosome)
        if best_path[-1] == goal:
            print(f"Solution found at generation {generation}.")
            break

        elite = population[:elite_size]
        selected_population = tournament_selection_binary(population, k=3)
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            child1, child2 = crossover_binary(parent1, parent2)
            child1 = mutate_binary(child1)
            child2 = mutate_binary(child2)
            new_population.extend([child1, child2])

        population = new_population[:population_size]

        if generation % 100 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness}")

    if best_chromosome:
        plot_path_from_chromosome(best_chromosome, "Genetic Algorithm")

def run_a_star():
    a_star_path = a_star_full_grid(grid, start, goal)
    if a_star_path:
        plot_path_from_chromosome(a_star_path, "A* Algorithm")

# Execute both algorithms and plot
run_a_star()
run_genetic_algorithm()