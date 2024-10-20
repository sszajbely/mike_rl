import numpy as np
from ...agent import Agent
from ...util import action_type, onehot_board, get_valid_moves, epsilon, standardized_board, unpadded_board, transform_move, get_valid_moves_tensor, event_dict_type
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import os

TRANSITION_HISTORY_SIZE = 5000  # keep only ... last transitions
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
TARGET_UPDATE_FREQUENCY = 3000 # In rounds
GAMMA = 0.99

base_path = "src/agents/unet_dqn"

class UNet(nn.Module):
    """
    Implements a UNet architecture.
    Going from input 5 channels 9x9 to ouput 3 channels 9x9.
    Using inbetween steps of 5x5 and 3x3.
    """
    def __init__(
            self, 
            device: str = "cpu"
        ):
        super(UNet, self).__init__()
        
        self.relu = nn.ReLU()
        self.device = device


        self.initial = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=1, device=device),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, device=device),
            nn.ReLU(),
            nn.Conv2d(8, 12, kernel_size=3, padding=1, device=device),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=0) # 5x5
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=1, device=device),
            nn.ReLU(),
            nn.Conv2d(12, 20, kernel_size=3, padding=1, device=device),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0) # 3x3
        )
    
        self.up1 = nn.Sequential(
            nn.Upsample(size=(5, 5)),
            nn.Conv2d(20, 20, kernel_size=1, device=device),
            nn.ReLU(),
            nn.Conv2d(20, 12, kernel_size=3, padding=1, device=device),
        )

        # The up2 and final layer need more channels because of the skip connections

        self.up2 = nn.Sequential(
            nn.Upsample(size=(9, 9)),
            nn.Conv2d(24, 24, kernel_size=1, device=device),
            nn.ReLU(),
            nn.Conv2d(24, 8, kernel_size=3, padding=1, device=device),
        )

        self.final = nn.Conv2d(16, 2, kernel_size=1, device=device)


    def forward(self, x):
        out = self.relu(self.initial(x))
        down1 = self.relu(self.down1(out))
        down2 = self.relu(self.down2(down1))
        up1 = torch.cat([down1, self.relu(self.up1(down2))], dim=1)
        up2 = torch.cat([out, self.relu(self.up2(up1))], dim=1)
        out = self.final(up2)
        return out

class UNetAgent(Agent):
    def __init__(self, training: bool, index: int, id: int|None = None):
        super().__init__(training, index, id if id is not None else index)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_collective_experience = False
        self.resume_training = False
        self.possible_moves_mask: torch.Tensor = torch.load("src/assets/possible_moves_mask.pt").to(self.device)
        if not os.path.exists(f"{base_path}/models/agent_{index}"):
            os.makedirs(f"{base_path}/models/agent_{index}")
        if not os.path.exists(f"{base_path}/outputs/agent_{index}"):
            os.makedirs(f"{base_path}/outputs/agent_{index}")

    def setup(self, player: int) -> None:
        if self.resume_training or not self.training:
            try:
                self.online_net = torch.load(f"{base_path}/models/agent_{self.id}/dqn.pth").to(self.device)
                if self.training:
                    self.target_net = torch.load(f"{base_path}/models/agent_{self.id}/dqn.pth").to(self.device)
            except FileNotFoundError:
                self.online_net = UNet(device=self.device)
                self.target_net = UNet(device=self.device)
        else:
            self.online_net = UNet(device=self.device)
            self.target_net = UNet(device=self.device)
        
        if self.training:
            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
            self.transitions = []
            self.rounds = 0
            self.scores = []
            self.losses = []
            self.exp_dir = None

        print("Number of parameters:", sum(p.numel() for p in self.online_net.parameters()))
        
        super().setup(player)

    def refresh_knowledge(self) -> None:
        if self.training:
            torch.save(self.online_net, f"{base_path}/models/agent_{self.id}/dqn_knowledge_dump.pth")
        else:
            self.online_net = torch.load(f"{base_path}/models/agent_{self.id}/dqn_knowledge_dump.pth").to(self.device)

    def plot_inside(self, board: np.ndarray, rotation_number: int):
        oh_board = torch.tensor(onehot_board(board), dtype=torch.float).to(self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.online_net.initial(oh_board)
            down1 = self.online_net.down1(out)
            down2 = self.online_net.down2(down1)
            up1 = self.online_net.up1(down2)
            up2 = self.online_net.up2(torch.cat([down1, up1], dim=1))
            final = self.online_net.final(torch.cat([out, up2], dim=1))
        
        all = oh_board.cpu().numpy()[0], out.cpu().numpy()[0], down1.cpu().numpy()[0], down2.cpu().numpy()[0], up1.cpu().numpy()[0], up2.cpu().numpy()[0], final.cpu().numpy()[0]
        col_num = np.max([len(x) for x in all])
        row_num = len(all)
        plt.figure(figsize=(col_num, row_num))
        plt.tight_layout()
        # 7 rows, 44 cols (maximum number of filters)
        for i in range(len(all)):
            for j in range(len(all[i])):
                plt.subplot(row_num, col_num, i * col_num + j + 1)
                plt.imshow(all[i][j])
                plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"{base_path}/outputs/agent_{self.id}/inside.pdf")
        plt.show()



    def act(self, board: np.ndarray) -> action_type:
        std_board, rotation_number = standardized_board(board, self.player)
        valid_moves = get_valid_moves(std_board, 1)
        if self.training and epsilon(0.8, 0, 250, self.rounds, self.resume_training) > np.random.rand():
            move = valid_moves[np.random.randint(0, len(valid_moves))]
            return transform_move(move, 4 - rotation_number)
        oh_board = torch.tensor(onehot_board(std_board), dtype=torch.float).to(self.device).unsqueeze(0)
        with torch.no_grad():
            output = self.online_net(oh_board)
            q_values = self.action_scores_from_output(output)
            # plt.figure(figsize=(8, 8))
            # plt.tight_layout()
            # plt.subplot(2, 2, 1)
            # up_board = unpadded_board(board)
            # plt.imshow(up_board)
            # plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            # plt.subplot(2, 2, 2)
            # plt.imshow(q_values.cpu().numpy().reshape(81, 82))
            # plt.colorbar()
            # plt.subplot(2, 2, 3)
            # rotated_from = np.rot90(output[0, 0].cpu().numpy(), 4 - rotation_number)
            # plt.imshow(np.exp(np.log(rotated_from * (up_board == self.player))))
            # # plt.imshow(output[0, 0].cpu().numpy())
            # plt.title("From")
            # plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            # plt.subplot(2, 2, 4)
            # rotated_to = np.rot90(output[0, 1].cpu().numpy(), 4 - rotation_number)
            # plt.imshow(np.exp(np.log(rotated_to * (up_board == 0))))
            # # plt.imshow(output[0, 1].cpu().numpy())
            # plt.title("To")
            # plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            # plt.show()
        # self.plot_inside(std_board, rotation_number)

        sorted_moves = np.argsort(q_values.cpu().numpy()[0])[::-1]
        for move in sorted_moves:
            action = self.index_to_action(move)
            if action in valid_moves:
                return transform_move(action, 4 - rotation_number)

        # Choose the move from a softmax distribution, temperature given by the epsilon function
        # if self.training and epsilon(0.8, 0, 35, self.rounds, self.resume_training):
        #     q_values = q_values.cpu().numpy()[0] # Only one batch in the act function
        #     valid_action_indices = q_values > 0
        #     valid_q_values = q_values[valid_action_indices]
        #     # Standardize the q values so near q values are more distinguishable
        #     valid_q_values = (valid_q_values - np.average(valid_q_values)) / np.std(valid_q_values)
        #     probabilities 



    def train(self, old_boards: list[np.ndarray], old_actions: list[action_type], new_boards: list[np.ndarray], events: list[event_dict_type]) -> None:
        for i in range(len(old_boards)) if self.use_collective_experience else [self.player - 1]:
            old_board, rotation_number = standardized_board(old_boards[i], i + 1)
            new_board, _ = standardized_board(new_boards[i], i + 1)

            old_action = transform_move(old_actions[i], rotation_number)

            reward = self.calculate_reward(new_board, True, events)

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

        old_q_values: torch.Tensor = self.action_scores_from_output(self.online_net(old_states))
        with torch.no_grad():
            new_q_values = self.action_scores_from_output(self.target_net(new_states))

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

     
    def action_scores_from_output(self, nn_output: torch.Tensor, valid_moves: torch.Tensor|None = None) -> torch.Tensor:
        """
        Function to convert the output of the network to a 81x82 tensor of scores, where the 81x81 part are the jumps, the 81st column is putting down a new piece.
        This is done by computing the outer product of flattened the from squares and the to squares, concatenating the new piece column.
        The batch dimension is the first dimension of the tensor.
        The output tensor is masked with the possible moves mask.
        The output tensor is flattened (except for the batch dimension) for easier indexing.
        """

        from_scores = nn_output[:, 0].flatten(start_dim=1)
        to_scores = nn_output[:, 1].flatten(start_dim=1)

        all_move_scores = torch.cat([(to_scores.unsqueeze(2) + from_scores.unsqueeze(1)) / 2, to_scores.unsqueeze(2)], dim=2)

        possible_scores = self.possible_moves_mask * all_move_scores

        if valid_moves is not None:
            possible_scores = possible_scores * valid_moves

        return possible_scores.flatten(start_dim=1)
        

    def action_to_index(self, action: action_type) -> int:
        to_index, from_index = action
        flattenedToIndex = (to_index[0] - 2) * 9 + (to_index[1] - 2)
        flattenedFromIndex = (from_index[0] - 2) * 9 + (from_index[1] - 2) if from_index is not None else 81

        return flattenedToIndex * 82 + flattenedFromIndex

    def index_to_action(self, index: int) -> action_type:
        flattenedToIndex, flattenedFromIndex = index // 82, index % 82

        return (
            (flattenedToIndex // 9 + 2, flattenedToIndex % 9 + 2), 
            ((flattenedFromIndex // 9 + 2, flattenedFromIndex % 9 + 2) if flattenedFromIndex < 81 else None)
        )


    def calculate_reward(self, board: np.ndarray, is_standardized: bool, events: event_dict_type) -> int:
        player = self.player if not is_standardized else 1
        player_pieces = np.argwhere(board == player)

        total_pieces = len(player_pieces)

        immediate_weight, waypoint_weight, final_weight = 1, 0, 0

        total_pieces = total_pieces / 81

        for event in events:
            if event == "waypoint":
                waypoint_weight = 2
            elif event == "game_end":
                final_weight = 4

        return (immediate_weight * total_pieces + waypoint_weight * total_pieces + final_weight * total_pieces) / (immediate_weight + waypoint_weight + final_weight)

    
    def round_end(self, board: np.ndarray, last_action: action_type, n_round: int) -> None:
        plt.switch_backend('agg')
        plt.figure()
        plt.plot(self.losses)
        plt.yscale("log")
        plt.savefig(f"{base_path}/outputs/agent_{self.id}/losses.png")
        plt.close()

        self.scores.append((board == self.player).sum())
        plt.figure()
        plt.plot(self.scores)
        plt.plot(np.convolve(self.scores, np.ones(20) / 20, mode="valid"))
        plt.ylim(0, 40)
        plt.yticks(range(0, 40, 5))
        plt.grid(axis="y")
        plt.savefig(f"{base_path}/outputs/agent_{self.id}/scores.png")
        plt.close()

        if n_round % 50 == 0:
            torch.save(self.online_net, f"{base_path}/models/agent_{self.id}/dqn_{n_round}.pth")


    def dispose(self):
        if self.training:
            torch.save(self.online_net, f"{base_path}/models/agent_{self.id}/dqn.pth")