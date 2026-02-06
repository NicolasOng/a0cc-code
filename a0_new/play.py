import time

from a0_new.protocols.game import A0Game, A0State, A0Action, T_state, T_action, Player
from a0_new.protocols.player import PlayerProtocol

from typing import Optional, Any, Protocol, Sequence

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
class TurnData:
    def __init__(self,
                 state: A0State[Any],
                 actions: Sequence[A0Action],
                 action: A0Action,
                 value: float,
                 policy: Policy[Any]
                ):
        self.state: A0State[Any] = state
        self.actions: Sequence[A0Action] = actions
        self.action: A0Action = action
        self.value: float = value
        self.policy: Policy[Any] = policy

class GameData:
    def __init__(self, game: A0Game[Any, Any], turn_limit: Optional[int] = None):
        self.game: A0Game[Any, Any] = game
        self.turn_limit: Optional[int] = turn_limit
        self.turn_data: list[TurnData] = []
        self.ended: bool = False
        self.winner: Optional[Player] = None
        self.time: float = 0.0 # seconds
        self.final_board: Optional[A0State[Any]] = None

def play(
        game: A0Game[T_state, T_action],
        players: list[PlayerProtocol[T_state, T_action]],
        turn_limit: Optional[int] = None,
        stop_signal: Optional[PlayStopSignal] = None
    ) -> GameData:
    '''
    Plays the given game with the given players.
    Both game and players should conform to the appropriate protocols,
    and use the same state and action types.
    The game will continue until it ends or the turn limit is reached.
    This function returns data about the game and each turn.
    '''
    logger.info("Game started.")
    start = time.perf_counter()

    data = GameData(game, turn_limit)

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
        logger.info("It's a draw!")
    else:
        logger.info(f"Player {winner} wins!")
    
    data.final_board = next_state
    
    data.ended = ended
    data.winner = winner

    end = time.perf_counter()
    data.time = end - start

    return data
