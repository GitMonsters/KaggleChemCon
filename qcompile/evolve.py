"""CircuitEvolver - Adaptation of RecursiveSelfImprovementEngine for quantum circuits.

Evolves rewrite sequences to optimize circuit compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Any
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

    def __init__(self, population_size: int = 40, mutation_rate: float = 0.4):
        self.population_size = max(20, population_size)
        self.mutation_rate = mutation_rate
        self.generation_counter = 0
        self.elite_archive: List[RewritePolicy] = []
        self.rewrite_lib = RewriteLibrary()
        self.graph_analyzer = CircuitGraph()

        # Available rewrites
        self.available_rewrites = [
            "cancel_inverses",
            "commute_gates",
            "fuse_rotations",
            "remove_identity",
            "peephole_opt",
            "synthesise_tket",
            "kak_decompose",
            "full_opt",
            "decompose_to_target",
        ]

    def initialize_population(self) -> List[RewritePolicy]:
        """Create initial random population of rewrite strategies"""
        population = []
        for i in range(self.population_size):
            # Random rewrite sequence with varied lengths
            seq_len = random.randint(3, 10)
            seq = random.choices(self.available_rewrites, k=seq_len)
            # Add some known-good patterns
            if random.random() < 0.3:
                seq = ["full_opt"] * random.randint(1, 2) + seq[:3]
            if random.random() < 0.3:
                seq = ["peephole_opt", "kak_decompose"] + seq
            policy = RewritePolicy(
                policy_id=f"gen0_p{i}",
                rewrite_sequence=seq,
                aggressiveness=random.uniform(0.4, 0.9),
                prefer_cancellation=random.choice([True, False]),
                prefer_fusion=random.choice([True, False]),
                generation=0,
            )
            population.append(policy)
        return population

    def mutate_policy(self, parent: RewritePolicy, new_id: str) -> RewritePolicy:
        """Mutate a policy - like mutate_policy in RSI engine"""
        child_seq = list(parent.rewrite_sequence)

        # Random mutations
        if random.random() < self.mutation_rate and child_seq:
            idx = random.randint(0, len(child_seq) - 1)
            child_seq[idx] = random.choice(self.available_rewrites)

        if random.random() < self.mutation_rate and len(child_seq) < 16:
            child_seq.append(random.choice(self.available_rewrites))

        if random.random() < self.mutation_rate and len(child_seq) > 2:
            idx = random.randint(0, len(child_seq) - 1)
            child_seq.pop(idx)

        if random.random() < self.mutation_rate and len(child_seq) >= 2:
            i, j = random.sample(range(len(child_seq)), 2)
            child_seq[i], child_seq[j] = child_seq[j], child_seq[i]

        # Insert random block
        if random.random() < self.mutation_rate and len(child_seq) < 12:
            idx = random.randint(0, len(child_seq))
            child_seq.insert(idx, random.choice(["full_opt", "peephole_opt", "kak_decompose"]))

        return RewritePolicy(
            policy_id=new_id,
            rewrite_sequence=child_seq,
            aggressiveness=max(0.1, min(1.0, parent.aggressiveness + random.gauss(0, 0.15))),
            prefer_cancellation=parent.prefer_cancellation if random.random() > 0.15 else not parent.prefer_cancellation,
            prefer_fusion=parent.prefer_fusion if random.random() > 0.15 else not parent.prefer_fusion,
            generation=parent.generation + 1,
        )

    def crossover_policies(self, parent1: RewritePolicy, parent2: RewritePolicy, new_id: str) -> RewritePolicy:
        """Crossover two policies"""
        if len(parent1.rewrite_sequence) >= 2 and len(parent2.rewrite_sequence) >= 2:
            cut1 = random.randint(1, len(parent1.rewrite_sequence) - 1)
            cut2 = random.randint(1, len(parent2.rewrite_sequence) - 1)
            child_seq = parent1.rewrite_sequence[:cut1] + parent2.rewrite_sequence[cut2:]
        else:
            child_seq = random.choice([parent1.rewrite_sequence, parent2.rewrite_sequence]).copy()

        # Sometimes add strong optimizer
        if random.random() < 0.2:
            child_seq = ["full_opt"] + child_seq

        return RewritePolicy(
            policy_id=new_id,
            rewrite_sequence=child_seq,
            aggressiveness=(parent1.aggressiveness + parent2.aggressiveness) / 2,
            prefer_cancellation=parent1.prefer_cancellation,
            prefer_fusion=parent2.prefer_fusion,
            generation=max(parent1.generation, parent2.generation) + 1,
        )

    def _copy_circuit(self, circuit):
        try:
            return copy.deepcopy(circuit)
        except Exception:
            try:
                return copy.copy(circuit)
            except Exception:
                pass
        # Try serialization
        try:
            if hasattr(circuit, 'save'):
                import tempfile, os
                f = tempfile.NamedTemporaryFile(delete=False, suffix='.kmb')
                f.close()
                circuit.save(f.name)
                # Try to reload
                try:
                    import kickmix as km
                    if hasattr(km.Circuit, 'read'):
                        c2 = km.Circuit.read(f.name)
                    elif hasattr(km, 'read_circuit'):
                        c2 = km.read_circuit(f.name)
                    else:
                        c2 = circuit
                    os.unlink(f.name)
                    return c2
                except Exception:
                    os.unlink(f.name)
                    pass
        except Exception:
            pass
        return circuit

    def evaluate_policy_fitness(self, policy: RewritePolicy, circuit, evaluator) -> float:
        """
        Evaluate fitness of a policy on a circuit.
        """
        try:
            opt_circuit = self._copy_circuit(circuit)
            # Apply sequence
            for rewrite_name in policy.rewrite_sequence:
                try:
                    opt_circuit = self.rewrite_lib.apply(opt_circuit, rewrite_name)
                except Exception:
                    continue
            try:
                opt_circuit = self.rewrite_lib.apply_all(opt_circuit)
            except Exception:
                pass

            result = evaluator.evaluate(circuit, opt_circuit)
            policy.is_valid = result["equivalent"]
            policy.gate_count = result["opt_gates"]
            orig_gates = result["orig_gates"]

            if not result["equivalent"]:
                policy.fitness_score = 0.0
                return 0.0

            reduction = max(0, orig_gates - result["opt_gates"])
            cost_reduction = max(0, result["orig_cost"] - result["opt_cost"])
            # Favor both gate and cost reduction
            fitness = (reduction / max(1, orig_gates)) * 60.0 + (cost_reduction / max(1e-9, result["orig_cost"])) * 40.0 + 5.0
            policy.fitness_score = fitness
            return fitness
        except Exception:
            policy.fitness_score = 0.0
            policy.is_valid = False
            return 0.0

    def evolve(self, circuit, evaluator, generations: int = 50) -> RewritePolicy:
        """
        Evolve best rewrite strategy for a circuit
        """
        population = self.initialize_population()

        for gen in range(generations):
            self.generation_counter += 1
            for policy in population:
                self.evaluate_policy_fitness(policy, circuit, evaluator)

            population.sort(key=lambda p: p.fitness_score, reverse=True)
            elite = population[0]
            self.elite_archive.append(copy.deepcopy(elite))

            survivors = population[: max(6, self.population_size // 2)]
            next_generation = list(copy.deepcopy(survivors))

            child_idx = 0
            while len(next_generation) < self.population_size:
                if random.random() < 0.75 and len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    child = self.crossover_policies(p1, p2, f"gen{self.generation_counter}_c{child_idx}")
                else:
                    parent = random.choice(survivors)
                    child = self.mutate_policy(parent, f"gen{self.generation_counter}_c{child_idx}")
                next_generation.append(child)
                child_idx += 1

            population = next_generation

        self.elite_archive.sort(key=lambda p: p.fitness_score, reverse=True)
        return self.elite_archive[0] if self.elite_archive else population[0]

    def apply_best(self, circuit, best_policy: RewritePolicy) -> Any:
        """Apply best policy to circuit"""
        opt_circuit = self._copy_circuit(circuit)
        for rewrite_name in best_policy.rewrite_sequence:
            try:
                opt_circuit = self.rewrite_lib.apply(opt_circuit, rewrite_name)
            except Exception:
                continue
        try:
            opt_circuit = self.rewrite_lib.apply_all(opt_circuit)
        except Exception:
            pass
        return opt_circuit
