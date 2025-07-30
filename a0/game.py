import copy
import time

from cc.core import Game, Board, Move, Player

from typing import Protocol, Optional, Any

import logging
logger = logging.getLogger(__name__)

class PlayerClass(Protocol):
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        ...

class TurnData:
    def __init__(self, board: Board, move: Move, player_data: Any):
        self.board: Board = board
        self.move: Move = move
        self.player_data: Any = player_data

class GameData:
    def __init__(self, game: Game, turn_limit: Optional[int] = None):
        self.game: Game = game
        #self.players = players
        self.turn_limit: Optional[int] = turn_limit
        self.turn_data: list[TurnData] = []
        self.ended: bool = False
        self.winner: Optional[Player] = None
        self.time: float = 0.0 # seconds
        self.final_board: Optional[Board] = None

def play(game: Game, players: list[PlayerClass], turn_limit: Optional[int] = None) -> GameData:
    '''
    Plays a game of chinese checkers with the given players.
    The players should implement the Player protocol, which requires a select_move method.
    The game will continue until it ends or the turn limit is reached.
    This function returns data about the game and each turn.
    '''
    logger.info("Game started.")
    start = time.perf_counter()

    data = GameData(game, turn_limit)

    turn = 0
    while not game.end and (turn_limit is None or turn < turn_limit):
        player = players[turn % len(players)]

        moves = game.start_turn()
        
        move, player_data = player.select_move(game.board, moves)

        turn_data = TurnData(
            board=copy.deepcopy(game.board),
            move=move,
            player_data=player_data
        )
        data.turn_data.append(turn_data)

        game.end_turn(move)

        turn += 1
    
    if game.winner is None:
        logger.info("It's a draw!")
    else:
        logger.info(f"Player {game.winner} wins!")
    
    data.final_board = copy.deepcopy(game.board)
    
    data.ended = game.end
    data.winner = game.winner

    end = time.perf_counter()
    data.time = end - start

    return data
