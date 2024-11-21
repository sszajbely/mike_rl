import numpy as np
from .util import action_type, event_dict_type

class Agent:
    def __init__(self, training: bool, index: int, id: int) -> None:
        self.training = training
        self.id = id
        self.index = index
        pass

    def name(self) -> str:
        return f"Agent {__name__} {self.index}"

    def setup(self, player: int) -> None:
        self.player = player

    def set_player(self, player: int) -> None:
        self.player = player

    def refresh_knowledge(self) -> None:
        pass

    def act(self, board: np.ndarray, training: bool = False) -> action_type:
        raise NotImplementedError()

    def train(self, old_board: np.ndarray, old_action: action_type, new_board: np.ndarray, old_events: event_dict_type, new_events: event_dict_type) -> None:
        pass
    
    def round_end(self, board: np.ndarray, last_action: action_type, n_round: int) -> None:
        pass
    
    def dispose(self):
        pass