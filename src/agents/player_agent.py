from ..util import action_type, unpadded_board
from ..agent import Agent
import numpy as np
import matplotlib.pyplot as plt

class PlayerAgent(Agent):
    def __init__(self, training, evaluation, index):
        super().__init__(training, evaluation, index)

    def move_parser(self, move: str, allow_none: bool = False) -> action_type:
        if move == "":
            if not allow_none:
                raise ValueError("Field cannot be empty")
            return None
        return tuple(map(lambda x: int(x) + 1, move.split()))

    def act(self, board: np.ndarray) -> action_type:
        # plt.imshow(unpadded_board(board))
        # plt.show()
        to_field = self.move_parser(input("Enter field to put a new piece/jump to. "))
        from_field = self.move_parser(input("Enter to field to jump from. Leave empty for `None` "), True)
        return (to_field, from_field)
