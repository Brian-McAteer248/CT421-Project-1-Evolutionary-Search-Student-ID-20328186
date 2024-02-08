import numpy as np
import math
import matplotlib.pyplot as plt
import random


# Define fitness function for population members for the one-max problem
def one_max_fitness(pop_member):
    return pop_member.count("1")


# Define fitness function for population members for the target string problem.
# Note: target_string is defined before the relevant GA function call below
def target_string_fitness(pop_member):
    pop_member_length = len(pop_member)
    fitness = 0
    curr_char_index = 0
    for char in pop_member:
        if char == target_string[curr_char_index]:
            fitness += 1

        curr_char_index += 1
        if curr_char_index >= pop_member_length:
            return fitness


# Define fitness function for population members for the deceptive landscape problem.
def deceptive_landscape_fitness(pop_member):
    pop_member_length = len(pop_member)
    if pop_member.count("0") == pop_member_length:
        return 2 * pop_member_length
    else:
        return pop_member.count("1")


# Define the genetic algorithm for Part A.
# Population member representation is constant throughout Part A, so only the fitness function need be changed (as well
# as the other standard GA parameters that can be tuned).
def part_a_genetic_algorithm(init_population_size, pop_member_length, fitness_func, rate_of_elitism, rate_of_crossover,
                             rate_of_mutation, max_iterations, min_fitness):
    # Define helper function that creates a random population member.
    # Nested definition as this function does not change across Part A.
    def rand_binary_str(length):
        binary_str = ""

        for i in range(length):
            binary_str += str(random.randint(0, 1))

        return binary_str

    # Define single-point crossover function that takes two parents as inputs and returns a single child as output.
    # Nested definition as this function does not change across Part A.
    def crossover(parent_1, parent_2):
        chromosome_length = len(parent_1)

        # For array values x1, x2, ... , xn, crossover point is at a random position from '|' to '|':
        # [x1 | x2 ... xn-1  | xn]
        crossover_point = random.randint(1, chromosome_length - 1)

        section_1 = parent_1[:crossover_point]
        section_2 = parent_2[crossover_point:chromosome_length]

        # Swap chromosome sections from respective parents combine to create new child
        return section_2 + section_1

    # Define mutation function that mutates a single gene of the input population member and returns this mutated
    # individual.
    # Nested definition as this function does not change across Part A.
    def mutate(pop_member):
        pop_member_len = len(pop_member)

        # Obtain index of random gene to mutate
        gene_index = random.randint(0, pop_member_len - 1)

        if pop_member[gene_index] == "0":
            mutated_pop_member = pop_member[:gene_index] + "1" + pop_member[gene_index + 1:pop_member_len]
        else:
            mutated_pop_member = pop_member[:gene_index] + "0" + pop_member[gene_index + 1:pop_member_len]

        return mutated_pop_member

    population_members = []
    # Create our initial random population members
    for j in range(init_population_size):
        population_members.append(rand_binary_str(pop_member_length))

    # List to store fitness of current generation population members
    population_fitness = []

    # List to record average fitness of each generation as the algorithm iterates (for use in plotting average
    # fitness over time)
    average_population_fitness_vals = []

    # Calculate initial population fitness
    for k in range(len(population_members)):
        population_fitness.append(fitness_func(population_members[k]))

    # Calculate the average fitness (single value) of the current population
    # and append it onto the average_population_fitness_vals array
    average_population_fitness_vals.append(sum(population_fitness) / len(population_fitness))

    # List to record best fitness of each generation as the algorithm iterates (for use in plotting the  best fitness
    # over time)
    max_population_fitness_vals = [max(population_fitness)]

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

        # Choose population size/2 number of parents according to a roulette wheel selection scheme.
        # The probability of a population member being selected as a parent is equal to their relative fitness which is
        # a value in the range [0, 1]
        for i in range(math.ceil(population_size / 2)):
            parents.append(np.random.choice(population_members, p=relative_pop_fitness))

        # Begin creating the new generation by incorporating elitism at a rate of (rate_of_elitism*100)% of the
        # population (rounded up to the nearest int, so at least one elite population member is propogated to the new
        # generation, regardless of the population size, in order to allow good genetic material to remain regardless
        # of subsequent stochastic crossover and mutation operations
        num_elites_to_keep = math.ceil(population_size * rate_of_elitism)

        sort_indices = np.argsort(population_fitness)
        indices_of_elites = sort_indices[-num_elites_to_keep:]

        elites = np.array(population_members)[indices_of_elites].tolist()
        for j in range(len(elites)):
            new_generation.append(elites[j])

        # Only create enough children as needed to maintain the previous generation population size
        num_children_to_create = population_size - num_elites_to_keep

        for m in range(num_children_to_create):
            num_parents = len(parents)
            parent_1_index = random.randint(0, num_parents - 1)
            parent_2_index = parent_1_index
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
            population_fitness[k] = fitness_func(population_members[k])

        # Calculate the average fitness (single value) of the current population
        # and append it onto the average_population_fitness_vals array
        average_population_fitness_vals.append(sum(population_fitness) / len(population_fitness))

        # Append the best fitness achieved in this generation
        max_population_fitness_vals.append(max(population_fitness))

        # Calculate relative fitness of each population member
        total_pop_fitness = sum(population_fitness)
        for p in range(len(population_fitness)):
            relative_pop_fitness[p] = population_fitness[p] / total_pop_fitness

        print(f"part_a_genetic_algorithm() using {fitness_func.__name__}: Iteration {num_iterations}."
              f" Max fitness at end of iteration: {max(population_fitness)}")

    print("\npart_a_genetic_algorithm() Finished:")
    sort_indices = np.argsort(population_fitness)
    index_of_best_solution = sort_indices[-1:][0]
    print(f"* Number of iterations: {num_iterations} (max_iterations = {max_iterations})")
    print(f"* Fitness of best solution: {population_fitness[index_of_best_solution]} (min_fitness = {min_fitness})")
    print(f"* Best evolved solution: {population_members[index_of_best_solution]}")

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
    plt.plot(x_vals, min_fitness_y_vals, color="red", linestyle="dashed", label="Minimum Desired Fitness")

    plt.title(f"Average and best population fitness over sequential generations.\nFitness function: {fitness_func.__name__}")
    plt.legend()
    plt.show()

    print("\nReturning the final population.\n")
    return population_members


final_population_one_max = part_a_genetic_algorithm(init_population_size=200, pop_member_length=30,
                                                    fitness_func=one_max_fitness, rate_of_elitism=0.01,
                                                    rate_of_crossover=0.8, rate_of_mutation=0.01, max_iterations=500,
                                                    min_fitness=30)


# This target string is accessed in the target_string_fitness function
target_string = "110100010100011100010110101011"
# Note: using a higher rate_of_mutation causes quicker convergence for this fitness landscape
final_population_target_string = part_a_genetic_algorithm(init_population_size=200, pop_member_length=30,
                                                          fitness_func=target_string_fitness, rate_of_elitism=0.01,
                                                          rate_of_crossover=0.8, rate_of_mutation=0.5,
                                                          max_iterations=500, min_fitness=30)


final_population_deceptive_landscape = part_a_genetic_algorithm(init_population_size=200, pop_member_length=30,
                                                                fitness_func=deceptive_landscape_fitness,
                                                                rate_of_elitism=0.01, rate_of_crossover=0.8,
                                                                rate_of_mutation=0.01, max_iterations=500,
                                                                min_fitness=60)
