import numpy as np
import torch
from typing import Literal

event_type = Literal["waypoint", "game_end"]
event_dict_type = dict[event_type, bool]
action_type = tuple[tuple[int, int], tuple[int, int] | None]
BOARD_SIZE = 9

def make_move(board: np.ndarray, player: int, to_field: tuple[int, int], from_field: tuple[int, int] | None = None):
# Check if the target field is empty
    if board[to_field] != 0:
        raise ValueError("Field is not empty")
    
    # Slicing adjacent fields around the target
    row, col = to_field
    adjacent_fields = board[row-1:row+2, col-1:col+2]
    
    # If no adjacent piece belongs to the player and it's not a jump, the move is invalid
    if from_field is None and not np.any(adjacent_fields == player):
        raise ValueError("No adjacent field with player's piece")
    
    if from_field is not None:
        # Convert fields into numpy arrays for efficient computation
        from_row, from_col = from_field
        
        # Check if the from_field contains the player's piece
        if board[from_field] != player:
            raise ValueError("Field does not contain player's piece")
        
        # Ensure that the jump is within the valid range (distance <= 2)
        if np.any(np.abs(np.array(from_field) - np.array(to_field)) > 2):
            raise ValueError("Field is too far to jump from")
        
        # Clear the original position after a valid jump
        board[from_row, from_col] = 0
    
    # Set the target field with the player's piece
    board[row, col] = player
    
    # Efficiently capture opponent's pieces by updating adjacent fields
    # We ensure opponent's pieces (not -1 and not empty) are replaced by player's piece
    mask = np.logical_and(adjacent_fields != -1, adjacent_fields != 0)
    adjacent_fields[mask] = player

def board_full(board: np.ndarray) -> bool:
    return np.all(board != 0)

def has_valid_move(board: np.ndarray, player: int) -> bool:
    player_fields = np.argwhere(board == player)
    for player_field in player_fields:
        row, col = player_field
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (i, j) == (0, 0):
                    continue
                if board[row + i, col + j] == 0:
                    return True
    return False
         
def calculate_next_player(player: int) -> int:
    return player % 4 + 1

def get_valid_moves(board: np.ndarray, player: int) -> list[action_type]:
    valid_moves = []

    # Get player's fields (positions where the player's pieces are)
    players_fields = np.argwhere(board == player)

    # Precompute all possible moves for a single position
    all_neighbors = np.array([[i, j] for i in range(-2, 3) for j in range(-2, 3) if (i, j) != (0, 0)])

    # Flatten board for faster access (optional but helps)
    board_flat = board.ravel()
    width = board.shape[1]

    # For each player's field
    for player_field in players_fields:
        row, col = player_field

        # Get all neighboring positions (distance 1 and 2)
        neighbors = player_field + all_neighbors

        # Convert neighbor coordinates to flat indices
        neighbor_indices = (neighbors[:, 0] * width + neighbors[:, 1])

        # Filter valid moves (empty spaces)
        valid_neighbors = neighbor_indices[board_flat[neighbor_indices] == 0]

        # Split valid moves into direct (distance 1) and extended (distance 2)
        for idx in valid_neighbors:
            pos_2d = (idx // width, idx % width)
            if np.abs(pos_2d[0] - row) <= 1 and np.abs(pos_2d[1] - col) <= 1:
                valid_moves.append(((int(pos_2d[0]), int(pos_2d[1])), None))  # Direct move
                valid_moves.append(((int(pos_2d[0]), int(pos_2d[1])), (int(row), int(col))))  # Extended move
            else:
                valid_moves.append(((int(pos_2d[0]), int(pos_2d[1])), (int(row), int(col))))  # Extended move

    return valid_moves
    
def get_valid_moves_tensor(board: np.ndarray, player: int, device = "cpu") -> torch.Tensor:
    board = unpadded_board(board)
    valid_moves = torch.zeros((BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE + 1), device=device)

    # Get player's fields (positions where the player's pieces are)
    players_fields = np.argwhere(board == player)

    # Precompute all possible moves for a single position
    all_neighbors = np.array([[i, j] for i in range(-2, 3) for j in range(-2, 3) if (i, j) != (0, 0)])

    # Flatten board for faster access (optional but helps)
    board_flat = board.ravel()
    width = board.shape[1]

    # For each player's field
    for player_field in players_fields:
        row, col = player_field

        # Get all neighboring positions (distance 1 and 2)
        neighbors = player_field + all_neighbors

        neighbors = neighbors[(neighbors[:, 0] >= 0) & (neighbors[:, 0] < BOARD_SIZE) & (neighbors[:, 1] >= 0) & (neighbors[:, 1] < BOARD_SIZE)]

        # Convert neighbor coordinates to flat indices
        neighbor_indices = (neighbors[:, 0] * width + neighbors[:, 1])

        # Filter valid moves (empty spaces)
        valid_neighbors = neighbor_indices[board_flat[neighbor_indices] == 0]        

        # Split valid moves into direct (distance 1) and extended (distance 2)
        for idx in valid_neighbors:
            pos_2d = (idx // width, idx % width)
            if np.abs(pos_2d[0] - row) <= 1 and np.abs(pos_2d[1] - col) <= 1:
                valid_moves[idx, -1] = 1  # Direct move
                valid_moves[idx, row * width + col] = 1  # Extended move
            else:
                valid_moves[idx, row * width + col] = 1  # Extended move

    return valid_moves


def unpadded_board(board: np.ndarray):
    return board[2:2 + BOARD_SIZE, 2:2 + BOARD_SIZE]

def onehot_board(board: np.ndarray):
    onehot = np.zeros((5, BOARD_SIZE, BOARD_SIZE))
    real_board = unpadded_board(board)
    for i in range(5):
        onehot[i, :, :] = (real_board == i).astype(int)
    return onehot

def unOnehot_board(onehot: np.ndarray):
    return np.argmax(onehot, axis=0)

def padded_board(board: np.ndarray):
    padded = np.ones((BOARD_SIZE + 4, BOARD_SIZE + 4)) * -1
    padded[2:2 + BOARD_SIZE, 2:2 + BOARD_SIZE] = board
    return padded

def standardized_board(board: np.ndarray, player: int) -> tuple[np.ndarray, int]:
    """
        Standardize the board such that the player's pieces are always 1, 
        and the opponents play pieces 2, 3 and 4 one after the other.
        Rotates the board to the orientation where in the starting position 1s are in the top left corner. 
    """
    masks = [
        board == 1,
        board == 2,
        board == 3,
        board == 4,
    ]

    new_board = board.copy()
    for i in range(4):
        new_board[masks[(i + player - 1) % 4]] = i + 1

    # Rotate the board to the orientation where in the starting position 1s are in the top left corner
    if player == 1:
        return new_board, 0
    elif player == 2:
        return np.rot90(new_board, 1), 1
    elif player == 3:
        return np.rot90(new_board, -1), 3
    return np.rot90(new_board, 2), 2

def transform_coords(coord: tuple[int, int], rotation_number: int):
    """
    Transforms the given coordinates according to the given rotation number and transpose flag.
    The rotation number is the number of 90 degree rotations to the left.
    The function assumes padding of 2 around the board.
    """
    rotation_number = rotation_number % 4
    x, y = coord
    x, y = x - 2, y - 2
    for _ in range(rotation_number):
        x, y = 8 - y, x
    return x + 2, y + 2

def transform_move(move: action_type, rotation_number: int) -> action_type:
    """
    Transforms the given move according to the given rotation number.
    The rotation number is the number of 90 degree rotations to the left.
    The function assumes padding of 2 around the board.
    """
    if move[1] is None:
        return transform_coords(move[0], rotation_number), None
    return transform_coords(move[0], rotation_number), transform_coords(move[1], rotation_number)

def epsilon(max_eps: float, min_eps: float, epsilon_decay_steps: int, step: int, upwards_curve: bool = False) -> float:
    """
    If upwards_curve is True, the epsilon will increase until the epsilon_decay_steps and then decrease.
    This is useful for exploitation in the beginning of the training, when resuming training.
    """    
    if upwards_curve:
        max_eps = max_eps * 2
        downwards = min_eps + (max_eps - min_eps) * np.exp(-step / epsilon_decay_steps)
        upwards = max_eps - (max_eps - min_eps) * np.exp(-step / epsilon_decay_steps)
        return min(upwards, downwards)

    return min_eps + (max_eps - min_eps) * np.exp(-step / epsilon_decay_steps)