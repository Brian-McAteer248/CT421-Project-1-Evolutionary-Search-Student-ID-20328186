# CT421-Project-1-Evolutionary-Search-Student-ID-20328186
Repository for 'CT421 Project 1 - Evolutionary Search' for Brian McAteer (Student ID 20328186). Created Feb 2024.

## Code Structure
### Part A
`part-a.py` contains the Python code for Part A of this project.

In this file, there is a `part_a_genetic_algorithm()` function that itself has some nested function definitions
such as `crossover()` and `mutate()`, which remain constant across Part A.

The fitness functions, however, are defined outside of `part_a_genetic_algorithm()` and are passed in as a
to `part_a_genetic_algorithm()` (param name `fitness_func`). A different fitness function can be passed in
to carry out search on a different landscape.

There are a number of other standard parameters for `part_a_genetic_algorithm()`, such as `init_population_size`,
`rate_of_mutation`, `rate_of_elitism` and `max_iterations`, among others.

The code in `part_a_genetic_algorithm()` follows these high level genetic algorithm steps:

* Produce initial population.
* Evaluate fitness of all population members.
* While `min_fitness` not reached or current iteration < `max_iterations`:
  * Select fitter individuals as parents via roulette wheel scheme.
  * Keep the best performing individuals at a rate of `rate_of_elitism`.
  * Carry out crossover at a rate of `rate_of_crossover`.
  * Carry out mutation at a rate of `rate_of_mutation`.
  * Assign result as new generation.
  * Calculate fitness of new generation.
 
The results are plotted using the `matplotlib` library.

### Part B
`part-b.py` contains the Python code for Part B of this project.

The bin packing problem specification information is read in from the local `Binpacking-2.txt` file.

Much of the code structure is the same as Part A, but it has not been wrapped into a single callable function
like `part_a_genetic_algorithm()` due to the absence of a need to dynamically change/pass in the fitness function.

Parameters such as `rate_of_mutation` and `max_iterations` are specified once as module-scope variables before initiating the
genetic algorithm.

The high level algorithmic steps are the same as the code in `part_a.py`.

Similarly to `part-a.py`, the results are plotted using the `matplotlib` library.
