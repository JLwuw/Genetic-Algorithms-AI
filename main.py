import pandas as pd
from deap import base, creator, tools
import random
from pathlib import Path


def _build_data_file_path(path_to_data, file_name):
    return Path(path_to_data) / file_name

def get_student_groups_df(path_to_data, file_name):
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
    student_groups_df = get_student_groups_df(path_to_data, "grupos_estudiantes.csv")

    # Create lists of valid IDs for each entity to facilitate random selection during gene creation and mutation
    valid_classrooms = classroom_df["Id"].tolist()
    valid_professors = professors_df["Id"].tolist()
    valid_time_slots = time_slots_df["Id"].tolist()

    # Precompute classroom capacities, group sizes, and subject-professor mappings for quick access during fitness evaluation
    classroom_capacities = {row["Id"]: row["Capacidad"] for _, row in classroom_df.iterrows()}
    group_sizes = {row["Id"]: row["Numero_de_Estudiantes"] for _, row in student_groups_df.iterrows()}
    subject_professors = {row["Id"]: row["Profesores"] for _, row in subjects_df.iterrows()}

    # Each student group has a list of subjects they MUST take.
    # We create a list of tuples (group_id, subject_id) for each group-subject combination that must exist in the schedule
    group_subject_assignments = []
    for group_id, subjects in zip(student_groups_df["Id"], student_groups_df["Asignaturas"]):
        for subject_id in subjects:
            group_subject_assignments.append((group_id, subject_id))

    def create_individual():
        genes = []
        for _, required_subject_id in group_subject_assignments:
            classroom_id = random.choice(valid_classrooms)
            professor_options = subject_professors.get(required_subject_id, valid_professors)
            professor_id = random.choice(professor_options)
            time_slot_id = random.choice(valid_time_slots)
            genes.append((classroom_id, professor_id, time_slot_id))
        return creator.Individual(genes)
    
    # To make sure every group takes all the subjects they must,
    # The ith gene of each individual will be associated to the ith element of the group_subject_assignments list
    toolBox.register("individual", create_individual)
    toolBox.register("population", tools.initRepeat, list, toolBox.individual)


    # fitness function
    def evaluate(individual):
        penalty = 0
        professor_time_slots = {}
        classroom_time_slots = {}
        group_time_slots = {}

        for i, class_assignment in enumerate(individual):
            classroom_id, professor_id, time_slot_id = class_assignment
            group_id, required_subject_id = group_subject_assignments[i]

            if professor_time_slots.get(professor_id) is None:
                professor_time_slots[professor_id] = set()
            
            if classroom_time_slots.get(classroom_id) is None:
                classroom_time_slots[classroom_id] = set()

            if group_time_slots.get(group_id) is None:
                group_time_slots[group_id] = set()

            # Same professor teaching two classes in the same time slot
            if time_slot_id in professor_time_slots[professor_id]:
                penalty += 1 
            
            # Same classroom hosting two classes in the same time slot
            if time_slot_id in classroom_time_slots[classroom_id]:
                penalty += 1

            # Same student group taking two classes in the same time slot
            if time_slot_id in group_time_slots[group_id]:
                penalty += 1

            professor_time_slots[professor_id].add(time_slot_id)
            classroom_time_slots[classroom_id].add(time_slot_id)
            group_time_slots[group_id].add(time_slot_id)

            # Check if the assigned professor is allowed to teach the subject
            allowed_professors = subject_professors.get(required_subject_id, [])
            if professor_id not in allowed_professors:
                penalty += 1

            classroom_capacity = classroom_capacities.get(classroom_id, -1)
            if classroom_capacity == -1:
                raise ValueError(f"Classroom ID {classroom_id} not found in classroom capacities.")

            group_size = group_sizes.get(group_id, -1)
            if group_size == -1:
                raise ValueError(f"Group ID {group_id} not found in group sizes.")

            # Check if the assigned classroom can accommodate the student group
            if group_size > classroom_capacity:
                penalty += 1

        return penalty,
    

    # Flip mutation: choose a random gene and change one of its components randomly
    def mutate(individual):
        i = random.randint(0, len(individual) - 1)
        classroom_id, professor_id, time_slot_id = individual[i]
        _, required_subject_id = group_subject_assignments[i]

        # randomly decide what to mutate
        choice = random.choice(["classroom", "time slot", "professor"])

        if choice == "classroom":
            classroom_id = random.choice(valid_classrooms)
        elif choice == "time slot":
            time_slot_id = random.choice(valid_time_slots)
        else:
            professor_id = random.choice(subject_professors.get(required_subject_id, valid_professors))

        individual[i] = (classroom_id, professor_id, time_slot_id)

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

        for i in range(0, len(offspring) - 1, 2):
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
        
    
    subject_names = {row["Id"]: row["Abreviacion"] for _, row in subjects_df.iterrows()}
    classroom_names = {row["Id"]: row["Aula"] for _, row in classroom_df.iterrows()}
    professor_names = {row["Id"]: row["Nombres"] for _, row in professors_df.iterrows()}
    time_slot_details = {row["Id"]: (row["Dia"], row["Hora"]) for _, row in time_slots_df.iterrows()}

    best_individual = tools.selBest(population, 1)[0]
    print("Best Individual:")
    for i, class_assignment in enumerate(best_individual):
        classroom_id, professor_id, time_slot_id = class_assignment
        group_id, subject_id = group_subject_assignments[i]
        subject_name = subject_names.get(subject_id, f"Unknown Subject {subject_id}")
        classroom_name = classroom_names.get(classroom_id, f"Unknown Classroom {classroom_id}")
        professor_name = professor_names.get(professor_id, f"Unknown Professor {professor_id}")
        day, hour = time_slot_details.get(time_slot_id, (f"Unknown Day ({time_slot_id})", "Unknown Hour"))
        print(
            f"{subject_name} - {professor_name} - {group_id} - "
            f"{classroom_name} - {day} - {hour}"
        )