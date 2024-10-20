import numpy as np
from ...agent import Agent
from ...util import action_type, onehot_board, get_valid_moves, epsilon, standardized_board
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt

TRANSITION_HISTORY_SIZE = 500  # keep only ... last transitions
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQUENCY = 100 # In rounds
GAMMA = 0.7

base_path = "src/agents/stupid_dqn"

class DQN(nn.Module):

    def __init__(
            self, 
            input_shape: tuple = (9, 9, 5),
            n_actions: int  = 1440,
            hidden_dim: int = 250,
            device: str = "cpu"
        ):
        super(DQN, self).__init__()

        # The first dimension of the input is the batch size
        # Because of thi
        self.conv1 = nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1, device=device) # 9x9x5 -> 9x9x3
        self.conv2 = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=2, device=device) # 9x9x3 -> 9x9x1 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], hidden_dim, device=device)
        self.fc2 = nn.Linear(hidden_dim, n_actions, device=device)

        self.relu = nn.ReLU()
        self.device = device


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent(Agent):
    def __init__(self, training: bool = False):
        super().__init__(training)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.resume_training = False
        self.possible_moves: list[action_type] = pickle.load(open("src/assets/possible_moves.pkl", "rb"))

    def setup(self, player: int) -> None:
        if self.resume_training or not self.training:
            self.online_net = torch.load(f"{base_path}/models/dqn.pth").to(self.device)
            if self.training:
                self.target_net = torch.load(f"{base_path}/models/dqn.pth").to(self.device)
        else:
            self.online_net = DQN(
                input_shape=(9, 9, 5),
                n_actions=len(self.possible_moves),
                device=self.device
            )
            self.target_net = DQN(
                input_shape=(9, 9, 5),
                n_actions=len(self.possible_moves),
                device=self.device
            )
        
        if self.training:
            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
            self.transitions = []
            self.rounds = 0
            self.scores = []
            self.losses = []
            self.exp_dir = None

        super().setup(player)

    def refresh_knowledge(self) -> None:
        if self.training:
            torch.save(self.online_net, f"{base_path}/models/dqn_knowledge_dump.pth")
        else:
            self.online_net = torch.load(f"{base_path}/models/dqn_knowledge_dump.pth").to(self.device)

    def act(self, board: np.ndarray) -> action_type:
        board = standardized_board(board, self.player)
        valid_moves = get_valid_moves(board, 1)
        if self.training and epsilon(1, 0.1, 30, self.rounds):
            return valid_moves[np.random.randint(0, len(valid_moves))]
        board = torch.tensor(onehot_board(board), dtype=torch.float).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(board)
        
        q_values = np.argsort(q_values.cpu().numpy().flatten())[::-1]
        for q in q_values:
            move = self.possible_moves[q]
            if move in valid_moves:
                return move
            
        return valid_moves[0]

    def train(self, old_board: np.ndarray, old_action: action_type, new_board: np.ndarray) -> None:
        old_board = standardized_board(old_board, self.player)
        new_board = standardized_board(new_board, self.player)

        reward = self.calculate_reward(new_board)

        new_board = torch.tensor(onehot_board(new_board), dtype=torch.float).to(self.device)
        old_board = torch.tensor(onehot_board(old_board), dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        action = torch.tensor(self.action_to_index(old_action), dtype=torch.long).to(self.device)
        self.transitions.append((old_board, action, new_board, reward))

        if len(self.transitions) > TRANSITION_HISTORY_SIZE:
            self.transitions.pop(0)

        batch_size = min(BATCH_SIZE, len(self.transitions))
        self.learn(batch_size)

    def learn(self, batch_size: int):

        batch = [self.transitions[i] for i in np.random.choice(len(self.transitions), batch_size, replace=False)]

        old_states, actions, new_states, rewards = zip(*batch)

        old_states = torch.stack(old_states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)

        old_q_values = self.online_net(old_states)
        with torch.no_grad():
            new_q_values = self.target_net(new_states)

        expected_new_q_values = rewards + GAMMA * torch.max(new_q_values, dim=1)[0]

        old_q_values = old_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(old_q_values, expected_new_q_values) / batch_size

        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rounds += 1

        if self.rounds % TARGET_UPDATE_FREQUENCY == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

     
    def action_to_index(self, action: action_type) -> int:
        return self.possible_moves.index(action)

    def calculate_reward(self, board: np.ndarray) -> int:
        
        endangered_pieces = 0
        safe_pieces = 0
        
        player_pieces = np.argwhere(board == self.player)

        total_pieces = len(player_pieces)

        for player_piece in player_pieces:
            row, col = player_piece
            in_danger = False
            for i in range(-1, 1):
                if in_danger:
                    break
                for j in range(-1, 1):
                    if (i, j) == (0, 0):
                        continue
                    if board[row + i, col + j] == 0:
                        endangered_pieces += 1
                        in_danger = True
                        break
            if not in_danger:
                safe_pieces += 1

        return  3 * total_pieces + 2 * safe_pieces + endangered_pieces

    
    def round_end(self, board: np.ndarray, last_action: action_type, n_round: int) -> None:
        plt.figure()
        plt.plot(self.losses)
        plt.yscale("log")
        plt.savefig(f"{base_path}/outputs/losses.png")
        plt.close()

        self.scores.append((board == self.player).sum())
        plt.figure()
        plt.plot(self.scores)
        plt.plot(np.convolve(self.scores, np.ones(10) / 10, mode="valid"))
        plt.savefig(f"{base_path}/outputs/scores.png")
        plt.close()

        if n_round % 50 == 0:
            torch.save(self.online_net, f"{base_path}/models/dqn_{n_round}.pth")


    def dispose(self):
        torch.save(self.online_net, f"{base_path}/models/dqn.pth")