import time

from a0_new.protocols.game import A0Game, A0State, A0Action, T_state, T_action, Player
from a0_new.protocols.player import PlayerProtocol

from typing import Optional, Any, Protocol, Sequence, Generic

from a0_new.policy import Policy

import logging
logger = logging.getLogger(__name__)

class PlayStopSignal(Protocol):
    '''
    Protocol for a stop signal to be used during gameplay.
    The signal should be thread-safe and allow checking if a stop has been requested.
    '''
    def is_set(self) -> bool:
        ...
class TurnData(Generic[T_state, T_action]):
    def __init__(self,
                 state: T_state,
                 actions: Sequence[T_action],
                 action: T_action,
                 value: float,
                 policy: Policy[T_action]
                ):
        self.state: T_state = state
        self.actions: Sequence[T_action] = actions
        self.action: T_action = action
        self.value: float = value
        self.policy: Policy[T_action] = policy

class GameData(Generic[T_state, T_action]):
    def __init__(self, game: A0Game[T_state, T_action], turn_limit: Optional[int] = None):
        self.game: A0Game[T_state, T_action] = game
        self.turn_limit: Optional[int] = turn_limit
        self.turn_data: list[TurnData[T_state, T_action]] = []
        self.ended: bool = False
        self.winner: Optional[Player] = None
        self.time: float = 0.0 # seconds
        self.final_state: Optional[T_state] = None

def play(
        game: A0Game[T_state, T_action],
        players: list[PlayerProtocol[T_state, T_action]],
        turn_limit: Optional[int] = None,
        stop_signal: Optional[PlayStopSignal] = None
    ) -> GameData[T_state, T_action]:
    '''
    Plays the given game with the given players.
    Both game and players should conform to the appropriate protocols,
    and use the same state and action types.
    The game will continue until it ends or the turn limit is reached.
    This function returns data about the game and each turn.
    '''
    logger.info("Game started.")
    start = time.perf_counter()

    data = GameData[T_state, T_action](game, turn_limit)

    turn = 0
    ended = False

    game.reset()
    next_state = game.get_current_state()

    while (
            not ended and
            (turn_limit is None or turn < turn_limit) and
            (stop_signal is None or not stop_signal.is_set())
        ):
        player = players[turn % len(players)]

        state = next_state
        actions = game.get_actions()
        
        action, value, policy = player.process_state(state, actions)

        turn_data = TurnData(
            state=state.clone(),
            actions=actions,
            action=action,
            value=value,
            policy=policy
        )
        data.turn_data.append(turn_data)

        next_state, _, ended, _, _ = game.step(action)

        turn += 1
    
    winner = game.get_winner()
    if winner is None:
        logger.info("Draw!")
    else:
        logger.info(f"{winner} wins!")
    
    data.final_state = next_state
    
    data.ended = ended
    data.winner = winner

    end = time.perf_counter()
    data.time = end - start

    return data
