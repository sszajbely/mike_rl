import numpy as np
from .agent import Agent
from .util import make_move, board_full, has_valid_move, calculate_next_player, unpadded_board, BOARD_SIZE
import time
import matplotlib.pyplot as plt

AGENTS_REFRESH_FREQUENCY = 100

class MikeWorld():
    def __init__(self, agents: list[Agent], n_rounds: int, has_player: bool, verbose: bool, training: bool, spectator: bool, evaluation: bool) -> None:
        assert len(agents) == 4
        self.agents = agents
        self.n_rounds = n_rounds
        self.has_player = has_player
        self.verbose = verbose
        self.spectator = spectator
        self.evaluation = evaluation
        self.training = training

    def set_next_player(self):
        self.next_player = calculate_next_player(self.next_player)

    def new_game(self):
        self.next_player = 1
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if i < 3:
                    if j < 3:
                        self.board[i, j] = 1
                    elif j > 5:
                        self.board[i, j] = 2
                elif i > 5:
                    if j < 3:
                        self.board[i, j] = 3
                    elif j > 5:
                        self.board[i, j] = 4
        self.board = np.pad(self.board, 2, constant_values=-1)
    
    def run(self):
        training_agent_players = []
        start_time = time.time_ns()
        scores = [[] for _ in range(4)]
        for i, agent in enumerate(self.agents):
            agent.setup(i + 1)
        for round in range(1, self.n_rounds + 1):
            self.new_game()
            # Randomize the order of the players
            training_agent_players = []
            self.agents: list[Agent] = [self.agents[permutation] for permutation in np.random.permutation(4)]
            for i in range(4):
                self.agents[i].set_player(i + 1)
                if self.agents[i].training:
                    training_agent_players.append(i + 1)

            # Refresh the knowledge for the agents that are not training
            # if round % AGENTS_REFRESH_FREQUENCY == 0 and self.training:
            #     # Refresh the knowledge for the agent training first, as it is the one that will be saved
            #     for agent in sorted(self.agents, key=lambda agent: agent.training, reverse=True):
            #         agent.refresh_knowledge()

            waypoint = 0
            old_notified_waypoint = [0, 0, 0, 0]
            new_notified_waypoint = [0, 0, 0, 0]
            old_boards = [None for _ in range(4)]
            old_actions = [None for _ in range(4)]
            new_boards = [None for _ in range(4)]
            steps = 0
            while not board_full(self.board):
                if self.has_player or self.verbose:
                    print("--------------------")
                    print(f"In loop, next is {self.next_player, self.agents[self.next_player - 1].__class__.__name__}")
                    print(unpadded_board(self.board))
                    if self.has_player:
                        plt.xticks(np.arange(BOARD_SIZE), np.arange(1, BOARD_SIZE + 1))
                        plt.yticks(np.arange(BOARD_SIZE), np.arange(1, BOARD_SIZE + 1))
                        plt.imshow(unpadded_board(self.board))
                        plt.show()
                if not has_valid_move(self.board, self.next_player):
                    if self.has_player:
                        print("No valid moves for player", self.next_player)
                        print("Skipping turn")
                    self.set_next_player()
                    continue
                new_boards[self.next_player - 1] = self.board.copy()
                if old_actions[self.next_player - 1] is not None and self.agents[self.next_player - 1].training:
                    self.agents[self.next_player - 1].train(
                        old_boards, 
                        old_actions, 
                        new_boards, 
                        [{ "waypoint": old_notified_waypoint[i] < waypoint } for i in range(4)],
                        [{ "waypoint": new_notified_waypoint[i] < waypoint } for i in range(4)]
                    )
                old_notified_waypoint[self.next_player - 1] = new_notified_waypoint[self.next_player - 1]
                new_notified_waypoint[self.next_player - 1] = waypoint
                old_boards[self.next_player - 1] = self.board.copy()
                to_field, from_field = self.agents[self.next_player - 1].act(self.board.copy())
                if self.verbose:
                    print(f"Player {self.next_player} moves {from_field} -> {to_field}")
                old_actions[self.next_player - 1] = (to_field, from_field)
                try:
                    make_move(self.board, self.next_player, to_field, from_field)
                except ValueError as e:
                    print(e)
                    continue
                num_empty_field = np.sum(self.board == 0)
                occupied_percentage = (BOARD_SIZE ** 2 - 36 - num_empty_field) / (BOARD_SIZE ** 2 - 36) # 36 is the number of initial pieces
                if occupied_percentage > 0.25:
                    waypoint = 25
                if occupied_percentage > 0.5:
                    waypoint = 50
                if occupied_percentage > 0.75:
                    waypoint = 75
                self.set_next_player()
                if len(training_agent_players) < 4 and self.training:
                    all_training_agents_lost = True
                    for training_agent in training_agent_players:
                        if (self.board == training_agent).sum() > 0:
                            all_training_agents_lost = False
                            break
                    if all_training_agents_lost:
                        print(f"All training agents lost after step {steps}")
                        break
                steps += 1
                if self.spectator:
                    input()

                if steps > 1000:
                    print("Game is taking too long")
                    break
            game_scores = [np.sum(self.board == i + 1) for i in range(4)]
            max_score = max(game_scores)
            if self.has_player or self.verbose:
                print("--------------------")
                print("Scores:")
                for i, agent in enumerate(self.agents):
                    print(f"Player {agent.player} ({agent.__class__.__name__}, id: {agent.id}): {game_scores[i]}")
            for i, agent in enumerate(self.agents):
                if agent.training:
                    end_board = self.board.copy()
                    agent.train(old_boards, old_actions, [end_board for i in range(4)], [{} for i in range(4)], [{ "game_end": game_scores[i] == max_score } for i in range(4)])
                    agent.round_end(end_board, old_actions[agent.player - 1], round)
                scores[agent.index].append(game_scores[agent.player - 1])
            elapsed_time = (time.time_ns() - start_time) / 1e9
            estimated_time = elapsed_time / round * self.n_rounds
            remaining_time = estimated_time - elapsed_time
            elapsed_units = "s"
            remaining_units = "s"
            if elapsed_time > 60:
                elapsed_time /= 60
                elapsed_units = "m"
            if remaining_time > 60:
                remaining_time /= 60
                remaining_units = "m"
            if elapsed_time > 60:
                elapsed_time /= 60
                elapsed_units = "h"
            if remaining_time > 60:
                remaining_time /= 60
                remaining_units = "h"
            print(f"Round {round}/{self.n_rounds} done. Elapsed time: {elapsed_time:.2f}{elapsed_units}. Estimated time left: {remaining_time:.2f}{remaining_units}")

        for agent in self.agents:
            agent.dispose()
        if self.evaluation:
            print("Scores:")
            for i, agent in enumerate(self.agents):
                print(f"Player {agent.index} ({agent.__class__.__name__}, id {agent.id}): {np.mean(scores[agent.index])}")




