import numpy as np
import importlib


class Individual:
    """
    Individual whose chromosome initialization is loaded from:
      genetic_algorithm/representation/<representation_type>.py

    Each representation module must define a function with the same name,
    e.g. representation/permutation.py must define def permutation(size, **kwargs).
    """

    def __init__(self, representation_type='permutation', size=None, values=None, rep_kwargs=None):
        self.representation_type = representation_type
        self.fitness = float('inf')
        self.rep_kwargs = rep_kwargs or {}

        if values is not None:
            self.chromosome = np.array(values)
        elif size is not None:
            self.chromosome = self._initialize_chromosome(size)
        else:
            raise ValueError("Either 'size' or 'values' must be provided")

    def _load_representation(self, name):
        try:
            module = importlib.import_module(f"genetic_algorithm.representation.{name}")
            fn = getattr(module, name)
            return fn
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot load representation '{name}': {e}")

    def _initialize_chromosome(self, size):
        gen = self._load_representation(self.representation_type)
        return gen(size=size, **self.rep_kwargs)

    def copy(self):
        new_individual = Individual(
            representation_type=self.representation_type,
            values=self.chromosome.copy(),
            rep_kwargs=self.rep_kwargs
        )
        new_individual.fitness = self.fitness
        return new_individual

    def __repr__(self):
        return f"Individual(chromosome={self.chromosome.tolist()}, fitness={self.fitness:.4f})"

    def __len__(self):
        return len(self.chromosome)