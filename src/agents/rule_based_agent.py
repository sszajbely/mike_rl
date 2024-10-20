import numpy as np
from ..agent import Agent, action_type
from ..util import make_move, get_valid_moves, calculate_next_player, has_valid_move, board_full

class RuleBasedAgent(Agent):
    def __init__(self, training: bool, evaluation: bool, index: int) -> None:
        super().__init__(training, evaluation, index)

    def act(self, board: np.ndarray) -> action_type:
        valid_moves = get_valid_moves(board, self.player)
        scores = []
        for move in valid_moves:
            new_board = board.copy()
            make_move(new_board, self.player, *move)
            score = self.recursive_score(new_board, calculate_next_player(self.player), 1)
            scores.append((score, move))

        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        best_moves = [move for score, move in scores if score == scores[0][0]]
        best_move = best_moves[np.random.randint(0, len(best_moves))]

        if best_move is None:
            return valid_moves[np.random.randint(0, len(valid_moves))]

        return best_move

    def recursive_score(self, board: np.ndarray, player: int, depth: int) -> int:
        if depth == 0 or board_full(board):
            return np.sum(board == self.player)
        valid_moves = get_valid_moves(board, player)
        best_score = -np.inf
        for move in valid_moves:
            new_board = board.copy()
            make_move(new_board, player, *move)
            score = self.recursive_score(new_board, calculate_next_player(player), depth - 1)
            if score > best_score:
                best_score = score
        return best_score