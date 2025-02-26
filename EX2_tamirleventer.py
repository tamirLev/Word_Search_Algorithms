import heapq
import os
import random
import math

# Define the set of vowels
vowels = {'a', 'e', 'i', 'o', 'u'}

# Heuristic function: calculates the heuristic cost between two words
def calculate_heuristic_improved(word1, word2):
    common_length = min(len(word1), len(word2))
    cost = 0

    # Compare characters in the common part
    for i in range(common_length):
        if word1[i] != word2[i]:
            cost += 0.25 if word2[i] in vowels else 1

    # Handle size differences: Calculate the cost for the extra/missing characters
    extra_letters_word1 = word1[common_length:]
    extra_letters_word2 = word2[common_length:]

    for letter in extra_letters_word1:  # Extra letters in word1
        cost += 0.25 if letter in vowels else 1
    for letter in extra_letters_word2:  # Extra letters in word2
        cost += 0.25 if letter in vowels else 1

    return cost


# Load dictionary from file
def load_dictionary():
    file_path = os.path.join(os.path.dirname(__file__), 'dictionary.txt')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return set(word.strip().lower() for word in file.readlines())
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return set()

# Generate child node based on an action
def child_node(dictionary, node, action):
    current_word = node['state']
    index, change_type, new_letter = action

    if change_type == "replace":
        new_word = current_word[:index] + new_letter + current_word[index + 1:]
    elif change_type == "add":
        new_word = current_word[:index] + new_letter + current_word[index:]
    elif change_type == "remove":
        new_word = current_word[:index] + current_word[index + 1:]
    else:
        return None

    if change_type == "replace":
        cost = 0.25 if new_letter in vowels else 1
    elif change_type == "add":
        cost = 0.25 if new_letter in vowels else 1
    elif change_type == "remove":
        removed_letter = current_word[index]
        cost = 0.25 if removed_letter in vowels else 1

    if new_word in dictionary:
        total_cost = node['path_cost'] + cost
        return {'state': new_word, 'path_cost': total_cost, 'parent': node}
    return None

# Check if the goal is reached
def goal_test(state, goal_word):
    return state == goal_word

def get_neighbors(dictionary, word):
    neighbors = set()
    for i in range(len(word)):
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if letter != word[i]:
                new_word = word[:i] + letter + word[i + 1:]
                if new_word in dictionary:
                    neighbors.add(new_word)

    # Generate neighbors by adding one letter
    for i in range(len(word) + 1):
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            new_word = word[:i] + letter + word[i:]
            if new_word in dictionary:
                neighbors.add(new_word)

    # Generate neighbors by removing one letter
    for i in range(len(word)):
        new_word = word[:i] + word[i + 1:]
        if new_word in dictionary:
            neighbors.add(new_word)

    return list(neighbors)



# A* Search algorithm
def a_star_search(dictionary, starting_word, goal_word, detail_output):
    start_node = {'state': starting_word, 'path_cost': 0}
    frontier = []
    heapq.heappush(frontier, (
        start_node['path_cost'] + calculate_heuristic_improved(start_node['state'], goal_word), starting_word, start_node))
    explored = set()
    frontier_dict = {starting_word: start_node}

    first_transformation = True  # Track the first transformation

    while frontier:
        _, state, node = heapq.heappop(frontier)
        frontier_dict.pop(state, None)

        if goal_test(node['state'], goal_word):
            solution = []
            while node:
                solution.append(node['state'])
                node = node.get('parent', None)
            solution.reverse()
            return solution

        explored.add(node['state'])

        for i in range(len(node['state'])):
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                if letter != node['state'][i]:  # Replace letter
                    action = (i, "replace", letter)
                    child = child_node(dictionary, node, action)
                    if child and child['state'] not in explored and child['state'] not in frontier_dict:
                        if detail_output and first_transformation:
                            print(f"Heuristic: {calculate_heuristic_improved(child['state'], goal_word)}")
                            first_transformation = False
                        heapq.heappush(frontier, (
                            child['path_cost'] + calculate_heuristic_improved(child['state'], goal_word), child['state'], child))
                        frontier_dict[child['state']] = child

            # Remove letter
            action = (i, "remove", None)
            child = child_node(dictionary, node, action)
            if child and child['state'] not in explored and child['state'] not in frontier_dict:
                if detail_output and first_transformation:
                    print(f"Heuristic: {calculate_heuristic_improved(child['state'], goal_word)}")
                    first_transformation = False
                heapq.heappush(frontier, (
                    child['path_cost'] + calculate_heuristic_improved(child['state'], goal_word), child['state'], child))
                frontier_dict[child['state']] = child

        # Add a new letter
        for i in range(len(node['state']) + 1):
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                action = (i, "add", letter)
                child = child_node(dictionary, node, action)
                if child and child['state'] not in explored and child['state'] not in frontier_dict:
                    if detail_output and first_transformation:
                        print(f"Heuristic: {calculate_heuristic_improved(child['state'], goal_word)}")
                        first_transformation = False
                    heapq.heappush(frontier, (
                        child['path_cost'] + calculate_heuristic_improved(child['state'], goal_word), child['state'], child))
                    frontier_dict[child['state']] = child

    return "No path found"


def hill_climbing_search(dictionary, starting_word, goal_word, detail_output):
    current = starting_word
    path = [current]
    first_transformation = True  # Track if we've done the first transformation

    while True:
        neighbors = get_neighbors(dictionary, current)  # Get valid single-letter transformations

        if not neighbors:
            return "No path found"  # Return this only if there are no neighbors

        # Evaluate the neighbors based on costs using the heuristic
        neighbor_costs = {neighbor: calculate_heuristic_improved(neighbor, goal_word) for neighbor in neighbors}

        # Find the best neighbor based on heuristic value
        best_neighbor = min(neighbor_costs, key=neighbor_costs.get)

        # If the best neighbor is better than the current
        if neighbor_costs[best_neighbor] < calculate_heuristic_improved(current, goal_word):
            # Move to the best neighbor
            if detail_output and first_transformation:
                print(current)  # Print the current word before transformation
                print(best_neighbor)  # Print the new best neighbor found
                first_transformation = False  # Ensure we only print once

            current = best_neighbor
            path.append(current)

            # Check if the goal has been reached
            if current == goal_word:
                return "\n".join(path)  # Return the full path if the goal is reached
        else:
            return "No path found"  # If no better neighbor is found, we did not find a path


def temperature_schedule(t):
        return 100 / (1 + 0.1 * math.log(t + 1))


def simulated_annealing(dictionary, starting_word, goal_word, detail_output):
    max_iterations = 100000
    current = starting_word
    current_cost = calculate_heuristic_improved(current, goal_word)
    path = [current]
    t = 0  # Starting time step
    first_step = True  # Flag to track the first transformation

    for _ in range(max_iterations):  # Limit the number of iterations
        T = temperature_schedule(t)

        if T <= 0:  # Allow T to reach 0
            return "No path found"

        neighbors = get_neighbors(dictionary, current)

        if not neighbors:
            return "No path found"

        # Select a random neighbor
        next_state = random.choice(neighbors)
        next_cost = calculate_heuristic_improved(next_state, goal_word)
        delta_E = next_cost - current_cost  # Change in energy

        # Only show details for the first step after the initial state
        if detail_output and first_step:
            print(f"Current state: {current}")
            print(f"Considering neighbor: {next_state}")
            print(f"Heuristic value of neighbor: {next_cost}")
            probability = math.exp(delta_E / T)  # Probability of accepting the worse state
            print(f"Acceptance probability: {min(probability, 1):.4f}")
            first_step = False  # Reset the flag after the first step
        else:
            probability = math.exp(delta_E / T)
            probability = min(probability, 1)

        if delta_E > 0 or probability > random.random():  # Accept current if better or probabilistically
            current = next_state
            current_cost = next_cost
            path.append(current)

            if goal_test(current, goal_word):  # Check if the goal is reached
                return "\n".join(path)

        t += 1  # Increment time step

    return "No path found"  # Return if the max iteration is reached without finding a path


def local_beam_search(dictionary, starting_word, goal_word, detail_output):
    k = 3  # Number of beams
    current_beam = [starting_word]
    paths = {starting_word: [starting_word]}  # Track paths
    initial_candidates = []  # To store the first generated successors

    while True:
        new_beam = {}
        next_candidates = []  # To store generated successors from current beam

        # Generate successors for each state in the current beam
        for state in current_beam:
            successors = get_neighbors(dictionary, state)

            for successor in successors:
                if successor in dictionary and successor != state:
                    # Calculate the heuristic value for the successor
                    new_beam[successor] = calculate_heuristic_improved(successor, goal_word)

                    if successor not in paths:
                        paths[successor] = paths[state] + [successor]

                    next_candidates.append(successor)  # Gather all candidates for the next beam

        # Check if the goal word is in the new states
        if goal_word in new_beam:
            return "\n".join(paths[goal_word])  # Return the path to the goal word

        if not new_beam:
            return "No path found"

        # Print the initial generated candidates if it's the first iteration
        if detail_output and not initial_candidates:
            print(f"Bag of actions considered: {next_candidates}")  # Bag of actions considered
            initial_candidates = next_candidates[:]  # Store for consistency across iterations

        # Sort based on the heuristic values of new states and pick the best k
        best_successors = sorted(new_beam.items(), key=lambda item: item[1])[:k]

        # Print the chosen actions and their associated paths at the first stage
        if detail_output and not initial_candidates:
            chosen_actions = [bs[0] for bs in best_successors]
            print(f"Chosen actions: {chosen_actions}")

        # If detail_output is True, check for a change in first letter
        if detail_output and len(best_successors) > 1:
            previous_first_letter = best_successors[0][0][0]  # First letter of best successor
            for bs in best_successors:
                current_first_letter = bs[0][0]  # First letter of the current successor
                if current_first_letter != previous_first_letter:
                    print("Three chosen words in beams:", [b[0] for b in best_successors[:3]])  # Print first three words
                    break  # Exit the loop after the first change

        # Update the beam with the selected successors
        current_beam = [bs[0] for bs in best_successors]

        # Check if there are new successors from these states
        local_maxima = True  # Assume we hit a local maximum
        for state in current_beam:
            successors = get_neighbors(dictionary, state)
            for successor in successors:
                if successor in dictionary and successor not in paths:
                    new_heuristic = calculate_heuristic_improved(successor, goal_word)
                    if new_heuristic < new_beam[current_beam[0]]:  # Check for a promising successor
                        local_maxima = False
                        break

        # If we're at local maxima
        if local_maxima:
            return "No path found"  # Exit if stuck


POPULATION_SIZE = 10
MUTATION_RATE = 0.1
GENERATIONS = 100


def fitness(individual, goal_word):
    last_word = individual[-1]
    matches = sum(1 for i in range(min(len(last_word), len(goal_word))) if last_word[i] == goal_word[i])
    length_difference = abs(len(last_word) - len(goal_word))
    return matches - (length_difference * 0.5)  # Adjust the weight based on experimentation


def initialize_population(starting_word, goal_word, dictionary):
    population = []
    for _ in range(POPULATION_SIZE):
        # Start individual with the starting word
        individual = [starting_word]

        while len(individual) < 5:
            last_word = individual[-1]
            neighbors = get_neighbors(dictionary, last_word)  # Get valid transitions
            if neighbors:
                individual.append(random.choice(neighbors))  # Append a valid neighbor
            else:
                break  # No more valid transformations

        population.append(individual)

    return population


def random_selection(population, fitness_fn):
    total_fitness = sum(fitness_fn(ind) for ind in population)
    if total_fitness == 0:
        return random.choice(population)  # Select a random individual to avoid division by zero

    selection_probs = [fitness_fn(ind) / total_fitness for ind in population]
    return random.choices(population, weights=selection_probs, k=1)[0]


def reproduce(parent1, parent2, dictionary):
    # Now ensure the children are valid transformations by checking each crossover
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    new_individual = parent1[:crossover_point] + parent2[crossover_point:]

    # Check that the new individual is a valid sequence of words
    if all(word in dictionary for word in new_individual):
        return new_individual

    # If not valid, fallback to one of the parents
    return random.choice([parent1, parent2])


def mutate(individual, dictionary):
    if random.random() < MUTATION_RATE:
        current_word = individual[-1]
        # Randomly choose to either mutate a letter or replace the last word with a valid neighbor
        if random.choice([True, False]):
            mutation_point = random.randint(0, len(current_word) - 1)
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                if letter != current_word[mutation_point]:
                    new_word = current_word[:mutation_point] + letter + current_word[mutation_point + 1:]
                    if new_word in dictionary:
                        individual.append(new_word)
                        return individual  # Return after successful mutation
        else:
            # Replace the last word with a valid neighbor
            neighbors = get_neighbors(dictionary, current_word)
            if neighbors:
                new_word = random.choice(neighbors)
                individual.append(new_word)
    return individual  # If no mutation occurred, return original


def genetic_algorithm(starting_word, goal_word, dictionary, detail_output):
    population = initialize_population(starting_word, goal_word, dictionary)

    for generation in range(GENERATIONS):
        next_population = []

        while len(next_population) < POPULATION_SIZE:
            parent1 = random_selection(population, lambda ind: fitness(ind, goal_word))
            parent2 = random_selection(population, lambda ind: fitness(ind, goal_word))

            child = reproduce(parent1, parent2, dictionary)
            child = mutate(child, dictionary)

            next_population.append(child)

        population = next_population

        # Find the best individual in the current generation
        best_individual = max(population, key=lambda ind: fitness(ind, goal_word))

        # Check if we have reached the goal word
        if goal_word in best_individual:
            return "\n".join(best_individual)  # Return the successful path

        # Print the population after the first generation
        if generation == 0 and detail_output:
            print("New population after the first step:")
            for ind in population:
                print(" -> ".join(ind))  # Print each individual's path

    return "No path found"  # If no solution was found after all generations


def find_word_path(starting_word, goal_word, search_method, detail_output):
    dictionary_data = load_dictionary()

    if starting_word not in dictionary_data or goal_word not in dictionary_data:
        print("One of the words is not in the dictionary.")
        return

    # Execute the chosen search method
    if search_method == 1:
        path = a_star_search(dictionary_data, starting_word, goal_word, detail_output)
    elif search_method == 2:
        path = hill_climbing_search(dictionary_data, starting_word, goal_word, detail_output)
    elif search_method == 3:
        path = simulated_annealing(dictionary_data, starting_word, goal_word, detail_output)
    elif search_method == 4:
        path = local_beam_search(dictionary_data, starting_word, goal_word, detail_output)
    elif search_method == 5:
        path = genetic_algorithm(starting_word, goal_word, dictionary_data, detail_output)
    else:
        print("Unsupported search method.")
        return

    # Directly print the output
    if isinstance(path, list):
        print("\n".join(path))  # Join and print the path directly
    else:
        print(path)  # If it's already a string, print it directly
