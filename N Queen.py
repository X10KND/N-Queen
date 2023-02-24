import numpy as np

BOARD_SIZE = 8
POPULATION_SIZE = 1000
GEN_LIMIT = 1000
MUTATION_RATE = 0.3
TARGET = (BOARD_SIZE * (BOARD_SIZE - 1)) / 2


def fitness(population):

    fitness_scores = []

    max_fitness = None
    min_fitness = None
    total_fitness = 0

    best_choromosome = None

    for p in population:

        fitness_score = TARGET - conflict(p)

        total_fitness += fitness_score
        
        if max_fitness == None or min_fitness == None:
            max_fitness = fitness_score
            min_fitness = fitness_score
            best_choromosome = p

        elif fitness_score > max_fitness:
            max_fitness = fitness_score
            best_choromosome = p

        elif fitness_score < min_fitness:
            min_fitness = fitness_score
        
        fitness_scores.append(fitness_score)

    average_fitness = total_fitness / len(population)

    return fitness_scores, max_fitness, average_fitness, min_fitness, best_choromosome


def conflict(board):
    
    attack = 0
    values = [0 for x in range(len(board))]

    for x in board:
        values[x - 1] += 1

    for x in values:
        if x > 1:
            attack += x * (x - 1)
            
    for row, col in enumerate(board):
        for j in range(len(board)):
            if j == row:
                continue
            if board[j] == col + abs(row - j) or board[j] == col - abs(row - j):
                attack += 1

    return (attack / 2)


def select(population, fit):

    a = np.arange(len(population))
    fit = fit - np.min(fit)
    fit = fit / np.sum(fit)

    new_pop_index = np.random.choice(a, len(population), True, fit)

    return new_pop_index


def crossover(x, y):

    crossover_point = np.random.randint(1, len(x))

    child_1 = np.concatenate((x[:crossover_point], y[crossover_point:]))
    child_2 = np.concatenate((y[:crossover_point], x[crossover_point:]))

    return child_1, child_2

def mutate(child):

    r = np.random.randint(0, len(child))
    val = np.random.randint(1, BOARD_SIZE + 1)

    child[r] = val

    return child


def show_board(board):
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i] == j + 1:
                print("X ", end="")
            else:
                print(". ", end="")
        print()
    print()


population = np.random.randint(1, BOARD_SIZE + 1, (POPULATION_SIZE, BOARD_SIZE))

limit_reached = False
gens_needed = 0

for g in range(GEN_LIMIT):
    
    gens_needed += 1
    fitness_scores, max_fitness, average_fitness, min_fitness, best_choromosome = fitness(population)
    
    print(f"Generation: {g + 1}")
    print("__________________________")
    print(f"Max Fitness: {max_fitness}")
    print(f"Average Fitness: {average_fitness}")
    print(f"Min Fitness: {min_fitness}")
    print("__________________________")
    print()

    # Uncomment the line below to show best choromosome of each generation
    #show_board(best_choromosome)

    print(best_choromosome)
    print("__________________________")
    print()
    
    if max_fitness == TARGET:
        break

    new_pop_index = select(population, fitness_scores)
    selected = population[new_pop_index]

    children = []
    for i in range(0, len(selected), 2):
        child_1, child_2 = crossover(selected[i], selected[i + 1])
        children.append(child_1)
        children.append(child_2)

    population = []
    for c in children:
        r = np.random.rand()
        if r < MUTATION_RATE:
            c = mutate(c)

        population.append(c)

    population = np.array(population)
    print()

    if g == GEN_LIMIT - 1:
        limit_reached = True


if limit_reached:
    print("\nGENERATION LIMIT REACH - INCREASE GEN_LIMIT VALUE")

else:
    print(f"\nSolution to {BOARD_SIZE} X {BOARD_SIZE} is {best_choromosome}")
    print(f"At Generation {gens_needed}\n")
    show_board(best_choromosome)
