"""CircuitEvolver - Adaptation of RecursiveSelfImprovementEngine for quantum circuits.

Evolves rewrite sequences to optimize circuit compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import random
import copy

from .rewrites import RewriteLibrary
from .circuit_graph import CircuitGraph


@dataclass
class RewritePolicy:
    """Mutable policy for circuit optimization - like SolverHeuristicPolicy"""
    policy_id: str
    rewrite_sequence: List[str] = field(default_factory=list)
    aggressiveness: float = 0.5
    prefer_cancellation: bool = True
    prefer_fusion: bool = True
    fitness_score: float = 0.0
    generation: int = 0
    gate_count: int = 0
    is_valid: bool = True


class CircuitEvolver:
    """
    Adapted from RecursiveSelfImprovementEngine.
    Evolves rewrite sequences through mutation and selection.
    """
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation_counter = 0
        self.elite_archive: List[RewritePolicy] = []
        self.rewrite_lib = RewriteLibrary()
        self.graph_analyzer = CircuitGraph()
        
        # Available rewrites
        self.available_rewrites = [
            'cancel_inverses', 'commute_gates', 'fuse_rotations', 
            'remove_identity', 'decompose_to_target'
        ]
    
    def initialize_population(self) -> List[RewritePolicy]:
        """Create initial random population of rewrite strategies"""
        population = []
        for i in range(self.population_size):
            # Random rewrite sequence
            seq_len = random.randint(2, 6)
            seq = random.choices(self.available_rewrites, k=seq_len)
            # Remove duplicates of same consecutive rewrite sometimes
            policy = RewritePolicy(
                policy_id=f"gen0_p{i}",
                rewrite_sequence=seq,
                aggressiveness=random.uniform(0.3, 0.8),
                prefer_cancellation=random.choice([True, False]),
                prefer_fusion=random.choice([True, False]),
                generation=0
            )
            population.append(policy)
        return population
    
    def mutate_policy(self, parent: RewritePolicy, new_id: str) -> RewritePolicy:
        """Mutate a policy - like mutate_policy in RSI engine"""
        child_seq = list(parent.rewrite_sequence)
        
        # Random mutations
        if random.random() < self.mutation_rate and child_seq:
            # Mutate: change random element
            idx = random.randint(0, len(child_seq) - 1)
            child_seq[idx] = random.choice(self.available_rewrites)
        
        if random.random() < self.mutation_rate:
            # Add random rewrite
            child_seq.append(random.choice(self.available_rewrites))
        
        if random.random() < self.mutation_rate and len(child_seq) > 2:
            # Remove random rewrite
            idx = random.randint(0, len(child_seq) - 1)
            child_seq.pop(idx)
        
        if random.random() < self.mutation_rate and len(child_seq) >= 2:
            # Swap two elements
            i, j = random.sample(range(len(child_seq)), 2)
            child_seq[i], child_seq[j] = child_seq[j], child_seq[i]
        
        return RewritePolicy(
            policy_id=new_id,
            rewrite_sequence=child_seq,
            aggressiveness=max(0.1, min(1.0, parent.aggressiveness + random.gauss(0, 0.1))),
            prefer_cancellation=parent.prefer_cancellation if random.random() > 0.2 else not parent.prefer_cancellation,
            prefer_fusion=parent.prefer_fusion if random.random() > 0.2 else not parent.prefer_fusion,
            generation=parent.generation + 1
        )
    
    def crossover_policies(self, parent1: RewritePolicy, parent2: RewritePolicy, new_id: str) -> RewritePolicy:
        """Crossover two policies"""
        # Take parts from both parents
        if len(parent1.rewrite_sequence) >= 2 and len(parent2.rewrite_sequence) >= 2:
            cut1 = random.randint(1, len(parent1.rewrite_sequence)-1)
            cut2 = random.randint(1, len(parent2.rewrite_sequence)-1)
            child_seq = parent1.rewrite_sequence[:cut1] + parent2.rewrite_sequence[cut2:]
        else:
            child_seq = random.choice([parent1.rewrite_sequence, parent2.rewrite_sequence]).copy()
        
        return RewritePolicy(
            policy_id=new_id,
            rewrite_sequence=child_seq,
            aggressiveness=(parent1.aggressiveness + parent2.aggressiveness) / 2,
            prefer_cancellation=parent1.prefer_cancellation,
            prefer_fusion=parent2.prefer_fusion,
            generation=max(parent1.generation, parent2.generation) + 1
        )
    
    def evaluate_policy_fitness(self, policy: RewritePolicy, circuit, evaluator) -> float:
        """
        Evaluate fitness of a policy on a circuit.
        Fitness = correctness_score * (1.0 / gate_count) or similar
        """
        try:
            # Apply rewrite sequence
            opt_circuit = copy.deepcopy(circuit)
            for rewrite_name in policy.rewrite_sequence:
                opt_circuit = self.rewrite_lib.apply(opt_circuit, rewrite_name)
            opt_circuit = self.rewrite_lib.apply_all(opt_circuit)
            
            # Evaluate
            result = evaluator.evaluate(circuit, opt_circuit)
            policy.is_valid = result['equivalent']
            policy.gate_count = result['opt_gates']
            orig_gates = result['orig_gates']
            
            if not result['equivalent']:
                policy.fitness_score = 0.0
                return 0.0
            
            # Fitness rewards correctness and gate reduction
            # Heavily penalize 2-qubit gates if possible
            reduction = max(0, orig_gates - result['opt_gates'])
            fitness = (reduction / max(1, orig_gates)) * 100 + 1.0  # +1 for being valid
            policy.fitness_score = fitness
            return fitness
        except Exception as e:
            policy.fitness_score = 0.0
            policy.is_valid = False
            return 0.0
    
    def evolve(self, circuit, evaluator, generations: int = 10) -> RewritePolicy:
        """
        Evolve best rewrite strategy for a circuit - like execute_rsi_cycle
        """
        population = self.initialize_population()
        
        for gen in range(generations):
            self.generation_counter += 1
            # Evaluate all
            for policy in population:
                self.evaluate_policy_fitness(policy, circuit, evaluator)
            
            # Sort by fitness
            population.sort(key=lambda p: p.fitness_score, reverse=True)
            elite = population[0]
            self.elite_archive.append(copy.deepcopy(elite))
            
            # Selection - keep top half
            survivors = population[:max(4, self.population_size // 2)]
            next_generation = list(copy.deepcopy(survivors))
            
            # Breed
            child_idx = 0
            while len(next_generation) < self.population_size:
                if random.random() < 0.7 and len(survivors) >= 2:
                    # Crossover
                    p1, p2 = random.sample(survivors, 2)
                    child = self.crossover_policies(p1, p2, f"gen{self.generation_counter}_c{child_idx}")
                else:
                    # Mutation
                    parent = random.choice(survivors)
                    child = self.mutate_policy(parent, f"gen{self.generation_counter}_c{child_idx}")
                next_generation.append(child)
                child_idx += 1
            
            population = next_generation
        
        self.elite_archive.sort(key=lambda p: p.fitness_score, reverse=True)
        return self.elite_archive[0] if self.elite_archive else population[0]
    
    def apply_best(self, circuit, best_policy: RewritePolicy) -> Any:
        """Apply best policy to circuit"""
        opt_circuit = copy.deepcopy(circuit)
        for rewrite_name in best_policy.rewrite_sequence:
            opt_circuit = self.rewrite_lib.apply(opt_circuit, rewrite_name)
        opt_circuit = self.rewrite_lib.apply_all(opt_circuit)
        return opt_circuit
