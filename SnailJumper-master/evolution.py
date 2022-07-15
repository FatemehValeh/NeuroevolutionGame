import copy
import random

from player import Player
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import heapq

THRESHOLD = 1

class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.min_population = []
        self.max_population = []
        self.avg_population = []

    def calculate_probabilities(self, players):
        total = 0
        for player in players:
            total += player.fitness

        probabilities = []
        for i in range(len(players)):
            if i == 0:
                probabilities.append(players[i].fitness / total)
            else:
                probabilities.append(players[i].fitness / total)

        return probabilities

    def roulette_wheel(self, players, num_parents):
        probabilities = self.calculate_probabilities(players)
        selected = []

        for random_number in np.random.uniform(low=0, high=1, size=num_parents):
            for i in range(len(probabilities)):
                if random_number <= probabilities[i]:
                    selected.append(self.clone_player(players[i]))
                    break

        return selected

    def sus(self, players, num_players):
        total_fitness = 0
        for player in players:
            total_fitness += player.fitness
        step = total_fitness / num_players
        begin = np.random.uniform(0, step, 1)

        middles = []
        for i in range(num_players):
            middles.append(begin + (i * step))
        next_generation = []
        for m in middles:
            fitness = i = 0
            while fitness < m:
                fitness += players[i].fitness
                i += 1
            next_generation.append(players[i - 1])

        return next_generation

    def q_tournament(self, players, num_players, q):
        next_generation = []
        for i in range(num_players):
            selected = np.random.choice(players, q)
            next_generation.append(max(selected, key=lambda p: p.fitness))
        return next_generation

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        players.sort(key=lambda x: x.fitness, reverse=True)
        players = players[: num_players]
        # for player in players:
        #    print(player.fitness, end=' ')
        # print()

        # TODO (Additional: Implement roulette wheel here)
        # players = self.roulette_wheel(players, num_players)  # BETTER RESULT

        # TODO (Additional: Implement SUS here)
        # players = self.sus(players, num_players)

        # players = self.q_tournament(players, num_players, 2)

        # TODO (Additional: Learning curve)
        self.min_population.append(min(players, key=lambda x: x.fitness).fitness)
        self.max_population.append(max(players, key=lambda x: x.fitness).fitness)
        self.avg_population.append(sum(player.fitness for player in players) / len(players))

        plt.plot(np.array(self.min_population), label='min')
        plt.plot(np.array(self.max_population), label='max')
        plt.plot(np.array(self.avg_population), label='avg')
        plt.plot()
        plt.show()

        return players

    def crossover(self, child1_array, child2_array, parent1_array, parent2_array):

        row_size, column_size = child1_array.shape
        section_1, section_2, section_3 = int(row_size / 3), int(2 * row_size / 3), row_size

        random_number = np.random.uniform(0, 1, 1)
        if random_number > 0.5:
            child1_array[:section_1, :] = parent1_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent1_array[section_2:, :]

            child2_array[:section_1, :] = parent2_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent2_array[section_2:, :]
        else:
            child1_array[:section_1, :] = parent2_array[:section_1:, :]
            child1_array[section_1:section_2, :] = parent1_array[section_1:section_2, :]
            child1_array[section_2:, :] = parent2_array[section_2:, :]

            child2_array[:section_1, :] = parent1_array[:section_1:, :]
            child2_array[section_1:section_2, :] = parent2_array[section_1:section_2, :]
            child2_array[section_2:, :] = parent1_array[section_2:, :]

    def mutation(self, player):
        for count in range(len(player.nn.weights)):
            player.nn.weights[count] += np.random.normal(0, 0.8, player.nn.weights[count].shape)
            player.nn.biases[count] += np.random.normal(0, 0.8, player.nn.biases[count].shape)
        return player

    def reproduction(self, parent1, parent2):
        child1 = self.clone_player(parent1)
        child2 = self.clone_player(parent2)

        random_number = random.uniform(0, 1)
        # random_number = 0.7
        if random_number <= THRESHOLD:
            self.crossover(child1.nn.weights[0], child2.nn.weights[0], parent1.nn.weights[0], parent2.nn.weights[0])
            self.crossover(child1.nn.weights[1], child2.nn.weights[1], parent1.nn.weights[1], parent2.nn.weights[1])

        # self.crossover(child1.nn.biases[0], child2.nn.biases[0], parent1.nn.biases[0], parent2.nn.biases[0], 'biases')
        # self.crossover(child1.nn.biases[1], child2.nn.biases[1], parent1.nn.biases[1], parent2.nn.biases[1], 'biases')

            self.mutation(child1)
            self.mutation(child2)
        return child1, child2

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """

        first_generation = prev_players is None

        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]

        else:
            new_players = []

            # RW BETTER RESULT
            parents = self.sus(prev_players, num_players)
            # parents = self.q_tournament(prev_players, num_players, 2)

            for i in range(0, len(parents), 2):
                child1, child2 = self.reproduction(parents[i], parents[i + 1])
                new_players.append(child1)
                new_players.append(child2)

            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
