import pandas as pd
from deap import base, creator, tools
import random
from pathlib import Path


def _build_data_file_path(path_to_data, file_name):
    return Path(path_to_data) / file_name

def student_group_df(path_to_data, file_name):
    df = pd.read_csv(_build_data_file_path(path_to_data, file_name))
    df["Asignaturas"] = df["Asignaturas"].apply(lambda x: list(map(int, x.split(","))))
    return df

def get_subjects_df(path_to_data, file_name):
    df = pd.read_csv(_build_data_file_path(path_to_data, file_name))
    df["Profesores"] = df["Profesores"].apply(lambda x: list(map(int, x.split(","))))
    return df

def get_professors_df(path_to_data, file_name):
    df = pd.read_csv(_build_data_file_path(path_to_data, file_name))
    return df

def get_time_slots_df(path_to_data, file_name):
    df = pd.read_csv(_build_data_file_path(path_to_data, file_name))
    return df

def get_classroom_df(path_to_data, file_name):
    df = pd.read_csv(_build_data_file_path(path_to_data, file_name))
    return df


if __name__ == "__main__":

    POPULATION_SIZE = 100
    NUMBER_OF_GENERATIONS = 50
    MUTATION_PROBABILITY = 0.2
    CROSSOVER_PROBABILITY = 0.5
    TOURNAMENT_SIZE = 3
    ELITISM = True


    path_to_data = Path(__file__).resolve().parent / "data"
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolBox = base.Toolbox()

    subjects_df = get_subjects_df(path_to_data, "asignaturas.csv")
    classroom_df = get_classroom_df(path_to_data, "aulas.csv")
    professors_df = get_professors_df(path_to_data, "profesores.csv")
    time_slots_df = get_time_slots_df(path_to_data, "franjas_horarias.csv")    
    student_group_df = student_group_df(path_to_data, "grupos_estudiantes.csv")

    # Create lists of valid IDs for each entity to facilitate random selection during gene creation and mutation
    valid_subjects = subjects_df["Id"].tolist()
    valid_classrooms = classroom_df["Id"].tolist()
    valid_professors = professors_df["Id"].tolist()
    valid_time_slots = time_slots_df["Id"].tolist()

    # Precompute classroom capacities and group sizes for quick access during fitness evaluation
    classroom_capacities = {row["Id"]: row["Capacidad"] for _, row in classroom_df.iterrows()}
    group_sizes = {row["Id"]: row["Numero_de_Estudiantes"] for _, row in student_group_df.iterrows()}

    def create_gene():
        subject_id = random.choice(valid_subjects)
        classroom_id = random.choice(valid_classrooms)
        professor_id = random.choice(valid_professors)
        time_slot_id = random.choice(valid_time_slots)
        return (subject_id, classroom_id, professor_id, time_slot_id)

    # Each student group has a list of subjects they MUST take.
    # We create a list of tuples (group_id, subject_id) for each group-subject combination that must exist in the schedule
    group_subject_assignments = []
    for group_id, subjects in zip(student_group_df["Id"], student_group_df["Asignaturas"]):
        for subject_id in subjects:
            group_subject_assignments.append((group_id, subject_id))
    
    # To make sure every group takes all the subjects they must,
    # The ith gene of each individual will be associated to the ith element of the group_subject_assignments list
    toolBox.register("individual", tools.initRepeat, creator.Individual, create_gene, n=len(group_subject_assignments))
    toolBox.register("population", tools.initRepeat, list, toolBox.individual)


    # fitness function
    def evaluate(individual):
        penalty = 0
        professor_time_slots = {}
        classroom_time_slots = {}     

        for i, class_assignment in enumerate(individual):
            subject_id, classroom_id, professor_id, time_slot_id = class_assignment

            if professor_time_slots.get(professor_id) is None:
                professor_time_slots[professor_id] = set()
            
            if classroom_time_slots.get(classroom_id) is None:
                classroom_time_slots[classroom_id] = set()

            # Same professor teaching two classes in the same time slot
            if time_slot_id in professor_time_slots[professor_id]:
                penalty += 1 
            
            # Same classroom hosting two classes in the same time slot
            if time_slot_id in classroom_time_slots[classroom_id]:
                penalty += 1

            professor_time_slots[professor_id].add(time_slot_id)
            classroom_time_slots[classroom_id].add(time_slot_id)

            # Check if the assigned subject matches the required subject for the student group
            if group_subject_assignments[i][1] != subject_id:
                penalty += 1

            classroom_capacity = classroom_capacities.get(classroom_id, -1)
            if classroom_capacity == -1:
                raise ValueError(f"Classroom ID {classroom_id} not found in classroom capacities.")

            group_size = group_sizes.get(group_subject_assignments[i][0], -1)
            if group_size == -1:
                raise ValueError(f"Group ID {group_subject_assignments[i][0]} not found in group sizes.")

            # Check if the assigned classroom can accommodate the student group
            if group_size > classroom_capacity:
                penalty += 1

        return penalty,
    

    # Flip mutation: choose a random gene and change one of its components randomly
    def mutate(individual):
        i = random.randint(0, len(individual) - 1)
        subject_id, classroom_id, professor_id, time_slot_id = individual[i]

        # randomly decide what to mutate
        choice = random.choice(["subject", "classroom", "time slot", "professor"])

        if choice == "subject":
            subject_id = random.choice(valid_subjects)
        elif choice == "classroom":
            classroom_id = random.choice(valid_classrooms)
        elif choice == "time slot":
            time_slot_id = random.choice(valid_time_slots)
        else:
            professor_id = random.choice(valid_professors)

        individual[i] = (subject_id, classroom_id, professor_id, time_slot_id)

        return (individual,)

    
    toolBox.register("evaluate", evaluate)
    toolBox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolBox.register("mate", tools.cxTwoPoint, )
    toolBox.register("mutate", mutate)

    # Initial population
    population = toolBox.population(n=POPULATION_SIZE)
    for individual in population:
        individual.fitness.values = toolBox.evaluate(individual)

    # Evolutionary loop
    for generation in range(NUMBER_OF_GENERATIONS):
        offspring = toolBox.select(population, len(population))
        offspring = list(map(toolBox.clone, offspring))

        for i in range(0, len(offspring), 2):
            if random.random() < CROSSOVER_PROBABILITY:
                toolBox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_PROBABILITY:
                toolBox.mutate(mutant)
                del mutant.fitness.values

        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_individuals:
            ind.fitness.values = toolBox.evaluate(ind)
        
        if ELITISM:
            elite = tools.selBest(population, 1)
            elite = list(map(toolBox.clone, elite))
            population[:] = elite + offspring[:-1]
            print(f"Generation {generation}: Best Fitness = {elite[0].fitness.values[0]}")
        else:
            population[:] = offspring
            print(f"Generation {generation}: Best = {min(ind.fitness.values[0] for ind in population)}")
        
    
    best_individual = tools.selBest(population, 1)[0]
    print("Best Individual:")
    for i, class_assignment in enumerate(best_individual):
        subject_id, classroom_id, professor_id, time_slot_id = class_assignment
        group_id = group_subject_assignments[i][0]
        print(f"Group {group_id} - Subject {subject_id} - Classroom {classroom_id} - Professor {professor_id} - Time Slot {time_slot_id}")