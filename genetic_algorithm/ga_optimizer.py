import numpy as np
import importlib
from core.optimizer import Optimizer
from .individual import Individual


class GeneticAlgorithm(Optimizer):
    """
    All parameters are configured via config.json.
    """
    
    def __init__(self, problem, population_size=100, num_generations=200,
                 representation='permutation',
                 selection_method='tournament',
                 crossover_method='pmx',
                 mutation_method='swap',
                 crossover_rate=0.8,
                 mutation_rate=0.2,
                 tournament_size=3,
                 elitism=2,
                 multi_parent=False,
                 num_parents=2
                 ):
        
        super().__init__(problem, population_size, num_generations)
        
        # Get problem size
        self.problem_size = getattr(problem, 'size', 
                                   getattr(problem, 'dimension',
                                          getattr(problem, 'num_dimensions', 10)))
        
        # Store parameters (from config.json)
        self.representation = representation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.multi_parent = bool(multi_parent)
        self.num_parents = int(num_parents)

        # Load operators dynamically
        self.selection_func = self._load_operator("selection", selection_method)
        self.crossover_func = self._load_operator("crossover", crossover_method)
        self.mutation_func = self._load_operator("mutation", mutation_method)
        
        # History
        self.history = []
    
    def _load_operator(self, folder, name):
        """Load operator function dynamically."""
        try:
            module = importlib.import_module(f"genetic_algorithm.{folder}.{name}")
            return getattr(module, name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot load operator '{name}' from {folder}: {e}")
    
    def optimize(self):
        """Main GA loop."""
        # Initialize population
        self.population = [
            Individual(representation_type=self.representation, size=self.problem_size)
            for _ in range(self.population_size)
        ]
        
        # Evaluate initial population
        for individual in self.population:
            individual.fitness = self._evaluate(individual)
            self._update_global_best(individual.chromosome.copy(), individual.fitness, 0)
        
        # Evolution loop
        for gen in range(self.num_generations):
            new_population = []
            
            # Elitism: keep best individuals
            if self.elitism > 0:
                sorted_pop = sorted(self.population, key=lambda ind: ind.fitness)
                elites = [ind.copy() for ind in sorted_pop[:self.elitism]]
                new_population.extend(elites)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._select()
                parent2 = self._select()
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    if self.multi_parent and self.num_parents >= 3:
                        parent3 = self._select()
                        child1, child2 = self.crossover_func(parent1, parent2, parent3)
                    else:
                        child1, child2 = self.crossover_func(parent1, parent2)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    child1 = self.mutation_func(child1)
                if np.random.rand() < self.mutation_rate:
                    child2 = self.mutation_func(child2)
                
                # Evaluate
                child1.fitness = self._evaluate(child1)
                child2.fitness = self._evaluate(child2)
                
                # Update best
                self._update_global_best(child1.chromosome.copy(), child1.fitness, gen + 1)
                self._update_global_best(child2.chromosome.copy(), child2.fitness, gen + 1)
                
                new_population.extend([child1, child2])
            
            # Replace population
            self.population = new_population[:self.population_size]
            
            # History
            sol = self.gbest_position
            if isinstance(sol, np.ndarray):
                sol = sol.tolist()
            elif hasattr(sol, "copy"):
                # if it's a list etc
                sol = sol.copy()

            self.history.append({
                "generation": int(gen + 1),
                "fitness": float(self.gbest_fitness),
                "solution": sol
            })
            
            print(f"Gen {gen + 1}/{self.num_generations} | Best: {self.gbest_fitness:.6f}")
        
        return self.gbest_position, self.gbest_fitness
    
    def _select(self):
        """Select parent using configured method."""
        if self.selection_method == 'tournament':
            return self.selection_func(
                population=self.population,
                tournament_size=self.tournament_size,
                minimize=self.problem.is_minimization
            )
        else:
            return self.selection_func(
                population=self.population,
                minimize=self.problem.is_minimization
            )
    
    def _evaluate(self, individual):
        result = self.problem.evaluate(individual.chromosome)
        if isinstance(result, tuple):
            result = result[0]
        # convert numpy scalars -> python numbers
        if isinstance(result, (np.integer, np.floating)):
            return result.item()
        return result