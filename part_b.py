import numpy as np
import math
import matplotlib.pyplot as plt
import random
import re


# Define a function to create a permutation of a list and return it
def permutation(list_to_permute):
    # Create copy of list, so we don't modify the original list that was passed in (object
    # reference).
    copy_of_list_to_permute = list_to_permute.copy()
    permutation = []

    while len(copy_of_list_to_permute) > 0:
        # Randomly select distinct elements from list to append to new list, and remove
        # them from the original list copy so that we don't append duplicate elements to
        # the new list
        rand_index_of_element = random.randint(0, len(copy_of_list_to_permute) - 1)
        rand_element = copy_of_list_to_permute[rand_index_of_element]

        permutation.append(rand_element)
        copy_of_list_to_permute.remove(rand_element)

    return permutation


# Helper function using code from the fitness function to calculate the number of bins
# corresponding to a given solution
def calculate_num_bins(pop_member):
    # Keep track of the number of bins being used. We begin with one bin, and start
    # logically packing items.
    number_of_bins = 1

    # Keep track of the total weight of items logically placed in the 'current bin' via their
    # permutation position in the 'pop_member' list of items
    current_bin_weight = 0
    for j in range(len(pop_member)):
        # If the current item (pop_member[j]) can fit in this logical bin
        if capacity_of_bins - current_bin_weight >= pop_member[j][1]:
            # Add to current logical bin
            current_bin_weight += pop_member[j][1]
        else:
            # Increment because we are stopping filling this bin and moving on to a new bin
            number_of_bins += 1
            # current_bin_weight now contains the weight of the first item being placed
            # in the next bin
            current_bin_weight = pop_member[j][1]

    return number_of_bins

# Calculate the fitness of a particular bin packing solution i.e., a permutation of items.
# Remember that pop_member is a tuple of format (item_id, item_weight)
def fitness(pop_member):
    fitness_score = 0

    # Keep track of the number of bins being used. We begin with one bin, and start
    # logically packing items.
    number_of_bins = 1

    # Keep track of the total weight of items logically placed in the 'current bin' via their
    # permutation position in the 'pop_member' list of items
    current_bin_weight = 0
    for j in range(len(pop_member)):
        # If the current item (pop_member[j]) can fit in this logical bin
        if capacity_of_bins - current_bin_weight >= pop_member[j][1]:
            # Add to current logical bin
            current_bin_weight += pop_member[j][1]
        else:
            # Greater fitness score is appended for bins that leave minimal empty space in them
            # i.e., efficient packing
            empty_space_in_current_bin = capacity_of_bins - current_bin_weight
            if empty_space_in_current_bin == 0:
                fitness_score += 10
            else:
                fitness_score += (1 / empty_space_in_current_bin)**2

            # Increment because we are stopping filling this bin and moving on to a new bin
            number_of_bins += 1
            # current_bin_weight now contains the weight of the first item being placed
            # in the next bin
            current_bin_weight = pop_member[j][1]

    # Greater fitness score is appended when fewer bins are used as defined by the logical
    # packing of items in bins defined by the pop_member list (a single genotype)
    fitness_score += (1 / number_of_bins) * 100

    return fitness_score


# Define an ordered crossover function where a subset of the first parent's genes are preserved
# in a continguous 'chunk', while the remaining distinct genes from the second parent are filled
# in the child in the order they appear in the second parent
def crossover(parent_1, parent_2):
    chromosome_length = len(parent_1)

    # For array values x1, x2, ... , xn, crossover point is at a random position from '|' to '|':
    # [x1 | x2 ... xn-1  | xn]
    crossover_point_1 = random.randint(1, chromosome_length - 1)
    crossover_point_2 = crossover_point_1
    # We want two distinct indices i..e, two distinct crossover points in parent_1
    while crossover_point_2 == crossover_point_1:
        crossover_point_2 = random.randint(1, chromosome_length - 1)

    # Initiate child with elements in parent_1 from crossover_point_1 up to (but not including)
    # crossover_point_2
    child = parent_1[crossover_point_1:crossover_point_2]

    # Then, any items in parent_2 and not already in the first set of ordered items (genes)
    # placed in child are placed in child in the order in which they appear in parent_2
    remaining_items = [item for item in parent_2 if item not in child]

    child += remaining_items

    return child


# Define function to mutate a genotype which is a list of item tuples of format
# (item_id, item_weight). This mutation function will swap the positions of two
# of these item tuples.
def mutate(pop_member):
    pop_member_length = len(pop_member)

    # We are going to swap the position of two randpom item tuples, creating
    # a new permutation - and thus a new population member
    swap_index_1 = random.randint(0, pop_member_length - 1)
    swap_index_2 = swap_index_1
    # We want two distinct indices for item tuples to swap
    while swap_index_2 == swap_index_1:
        swap_index_2 = random.randint(0, pop_member_length - 1)

    # Create copy so we don't alter original passed-in list (object reference)
    mutated_pop_member = pop_member.copy()

    # Swap the posiitons of the two randomly-selected item tuples
    mutated_pop_member[swap_index_1], mutated_pop_member[swap_index_2] = (
        mutated_pop_member[swap_index_2], mutated_pop_member[swap_index_1])

    return mutated_pop_member


init_population_size = 500
max_iterations = 2000
min_fitness = 130
rate_of_elitism = 0.01
rate_of_crossover = 0.8
rate_of_mutation = 0.01

bins_items_file = open("Binpacking-2.txt")

five_bin_packing_problems = bins_items_file.read().split("BPP")[1:]

# In each of these outermost loop iterations, we run the GA for one of the five bin
# packing problems and output the results
for i in range(len(five_bin_packing_problems)):
    # Convert each problem string (read from file) into a list of strings (split
    # via newline character).
    # Remove the first line which is a remainder of the header line
    # e.g, "      1" from "BPP      1".
    # Additionally, remove last split line which is an empty string "".
    five_bin_packing_problems[i] = five_bin_packing_problems[i].split("\n")[1:-1]

    print(f"\nProblem {i + 1}")

    current_bin_packing_problem = five_bin_packing_problems[i]
    max_weight = []

    number_m_different_item_weights = int(current_bin_packing_problem[0])
    capacity_of_bins = int(current_bin_packing_problem[1])

    # Now that we have cached number_m_different_item_weights and capacity_of_bins,
    # we can remove these from the current_bin_packing_problem list
    current_bin_packing_problem = current_bin_packing_problem[2:]

    number_of_items_to_pack = 0
    for j in range(len(current_bin_packing_problem)):
        # Split using the RegEx module. Split on one or more whitespace
        # character occurrences.
        # Remove empty string element from each jth list.
        current_bin_packing_problem[j] = re.split("\s+", current_bin_packing_problem[j])[1:]
        # Now, convert each of the two elements in current_bin_packing_problem (1st is weight, 2nd is
        # number of items) to an int
        for k in range(len(current_bin_packing_problem[j])):
            current_bin_packing_problem[j][k] = int(current_bin_packing_problem[j][k])

    # This is a list of the read-in items for the current bin packing problem, where each element
    # is the weight of that item
    item_weights_list = []

    for g in range(len(current_bin_packing_problem)):
        # For specified no. of items with certain weight
        for p in range(current_bin_packing_problem[g][1]):
            # Append the weight value as the item's representation in items_list
            item_weights_list.append(current_bin_packing_problem[g][0])

    # Create a list of tuples of format (item_id, item_weight) for keeping track of distinct
    # items during genetic operations. item_id is being set as the list positions of the item
    # weights in item_weights_list for identification purposes. The generated tuples here
    # will be shuffled around during population member generation and modification.
    item_tuples_list = []
    for item_tuple in enumerate(item_weights_list):
        item_tuples_list.append(item_tuple)

    print(f"item_tuples_list of size {len(item_tuples_list)}: {item_tuples_list}")

    # Population members will each be represented, genotypically, as a permutation of items
    # (i.e., their weight values), where the order of the items determines their placement
    # in bins i.e., the items in the first few positions will be logically placed in the first
    # bin until the total weight exceeds capacity_of_bins. The item causing the bin capacity
    # to be exceeded will instead be placed in the second bin, and so on.
    population_members = []

    # Create our initial random population members
    for j in range(init_population_size):
        population_members.append(permutation(item_tuples_list))

    # List to store fitness of current generation population members
    population_fitness = []

    # List to record average fitness of each generation as the algorithm iterates (for use in plotting average
    # fitness over time)
    average_population_fitness_vals = []

    # List to store the number of bins for the best solution in each generation
    # List to store the number of bins for the best solution in each generation
    best_solution_number_bins_vals = []

    # Calculate initial population fitness
    for k in range(len(population_members)):
        population_fitness.append(fitness(population_members[k]))

    # Calculate the average fitness (single value) of the current population
    # and append it onto the average_population_fitness_vals array
    average_population_fitness_vals.append(sum(population_fitness) / len(population_fitness))

    # List to record best fitness of each generation as the algorithm iterates (for use in plotting the  best fitness
    # over time)
    max_population_fitness_vals = [max(population_fitness)]

    sort_indices = np.argsort(population_fitness)
    index_of_best_solution = sort_indices[-1:][0]

    # Append the number of bins for the best fitness solution achieved in this initial generation
    best_solution_number_bins_vals.append(
        calculate_num_bins(population_members[index_of_best_solution]))

    relative_pop_fitness = []
    total_pop_fitness = sum(population_fitness)

    # Calculate relative fitness of each population member i.e., normalise their fitness to a value in the range [0, 1]
    # These relative fitness values willl be used in our roulette wheel selection scheme
    for p in range(len(population_fitness)):
        relative_pop_fitness.append(population_fitness[p] / total_pop_fitness)

    num_iterations = 0

    while num_iterations < max_iterations and min_fitness not in population_fitness:
        num_iterations += 1

        parents = []
        new_generation = []
        # Keep note of current population size as we want to maintain a constant population size through each generation
        population_size = len(population_members)

        # Create parent indices for selection below as np.random.choice() does not work
        # with lists with more than one dimension (population_members is a list of tuples)
        pop_member_indices = []
        for h in range(population_size):
            pop_member_indices.append(h)

        # Choose population size/2 number of parents according to a roulette wheel selection scheme.
        # The probability of a population member being selected as a parent is equal to their relative fitness which is
        # a value in the range [0, 1]
        for b in range(math.ceil(population_size / 2)):
            selected_parent_index = np.random.choice(pop_member_indices, p=relative_pop_fitness)
            parents.append(population_members[selected_parent_index])

        # Begin creating the new generation by incorporating elitism at a rate of (rate_of_elitism*100)% of the
        # population (rounded up to the nearest int, so at least one elite population member is propogated to the new
        # generation, regardless of the population size, in order to allow good genetic material to remain regardless
        # of subsequent stochastic crossover and mutation operations
        num_elites_to_keep = math.ceil(population_size * rate_of_elitism)

        sort_indices = np.argsort(population_fitness)
        indices_of_elites = sort_indices[-num_elites_to_keep:]

        for j in indices_of_elites:
            new_generation.append(population_members[j])

        # Only create enough children as needed to maintain the previous generation population size
        num_children_to_create = population_size - num_elites_to_keep

        for m in range(num_children_to_create):
            num_parents = len(parents)
            parent_1_index = random.randint(0, num_parents - 1)
            parent_2_index = parent_1_index  # Initialise to have scope outside of following while loop
            # We want two distinct indices i.e. two distinct parents
            while parent_2_index == parent_1_index:
                parent_2_index = random.randint(0, num_parents - 1)

            new_child = None

            # Produce a new child from two parents via crossover at a rate of (rate_of_crossover*100)%
            if random.random() <= rate_of_crossover:
                new_child = crossover(parents[parent_1_index], parents[parent_2_index])
            # Otherwise, select one of the selected parents (unchanged) to be a new child,
            # at a rate of ((1-rate_of_crossover)*100)%.
            # Note that, in each case, the new child is still subject to mutation below at a rate of
            # (rate_of_mutation*100)%
            else:
                new_child = parents[parent_1_index]

            # Mutate the child produced via crossover/parent selection at a rate of (rate_of_mutation*100)%
            if random.random() <= rate_of_mutation:
                new_child = mutate(new_child)

            new_generation.append(new_child)

        # We now have our new population generation of size population_size.
        # We will assign it to be the new current population, evaluate the fitness of each member and continue to
        # the next iteration
        population_members = new_generation
        for k in range(len(population_members)):
            population_fitness[k] = fitness(population_members[k])

        # Calculate the average fitness (single value) of the current population
        # and append it onto the average_population_fitness_vals array
        average_population_fitness_vals.append(sum(population_fitness) / len(population_fitness))

        # Append the best fitness achieved in this generation
        max_population_fitness_vals.append(max(population_fitness))

        sort_indices = np.argsort(population_fitness)
        index_of_best_solution = sort_indices[-1:][0]

        # Append the number of bins for the best fitness solution achieved in this generation
        best_solution_number_bins_vals.append(
            calculate_num_bins(population_members[index_of_best_solution]))

        # Calculate relative fitness of each population member
        total_pop_fitness = sum(population_fitness)
        for p in range(len(population_fitness)):
            relative_pop_fitness[p] = population_fitness[p] / total_pop_fitness

        print(f"Bin packing problem {i + 1}: Iteration {num_iterations}."
              f" Max fitness at end of iteration: {max(population_fitness)}, with"
              f" {calculate_num_bins(population_members[index_of_best_solution])} bins.")

    print(f"\nFinished GA iterations for bin packing problem {i + 1}")
    sort_indices = np.argsort(population_fitness)
    index_of_best_solution = sort_indices[-1:][0]
    print(f"* Number of iterations: {num_iterations} (max_iterations = {max_iterations})")
    print(f"* Fitness of best solution: {population_fitness[index_of_best_solution]} (min_fitness = {min_fitness})")
    print(f"* Best evolved solution, packed into {calculate_num_bins(population_members[index_of_best_solution])} bins):"
          f" {population_members[index_of_best_solution]}")

    print(f"\nCreating plot of average and best population fitness over {num_iterations} generations.")
    # Increase xlim by 5% to make it obvious in cases where algorithm stops when min_fitness is achieved in the
    # population
    plt.xlim([0, num_iterations + math.ceil(num_iterations * 0.05)])
    plt.ylim([0, min_fitness + 5])
    plt.xlabel("Generation Number")
    plt.ylabel("Population Fitness")

    x_vals = []
    min_fitness_y_vals = []

    # Before the first iteration begins (i.e., when num_iterations == 0), we have already generated the first generation
    # and evaluated their fitnesses. So, the total number of generations produced is num_iterations + 1.
    # Object returned by range() produces sequential ints including the first param up to, but not including,
    # the second param. So, pass in (1, num_generations + 2).
    for x in range(1, num_iterations + 2):
        x_vals.append(x)
        min_fitness_y_vals.append(min_fitness)

    plt.plot(x_vals, average_population_fitness_vals, color="black", label="Average Fitness of Population")
    plt.plot(x_vals, max_population_fitness_vals, color="blue", label="Best Fitness of Population")

    plt.title(
        f"Average and best population fitness over sequential generations for bin packing"
        f" problem {i + 1}.")
    plt.legend()
    plt.show()

    # Clear the current axis of the plot for the next plot that will plot the number of bins
    # of the best solution in each generation over all generations
    plt.cla()

    plt.xlim([0, num_iterations + math.ceil(num_iterations * 0.05)])
    y_ticks = []
    for w in range(23):
        y_ticks.append(w)
    plt.yticks(y_ticks)
    plt.ylim([0, 22])
    plt.xlabel("Generation Number")
    plt.ylabel("Number of Bins")
    plt.plot(x_vals, best_solution_number_bins_vals, color="red", label="Number "
                                                                        "of Bins of Best Solution")
    plt.title(
        f"Number of bins for best solution in generation over sequential generations for"
        f" bin packing problem {i + 1}.")
    plt.legend()
    plt.show()
