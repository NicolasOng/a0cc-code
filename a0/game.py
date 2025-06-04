import copy

from cc.core import Game, Board, Move, Player

from typing import Protocol, TypedDict, Optional, Any

import logging
logger = logging.getLogger(__name__)

class PlayerClass(Protocol):
    def select_move(self, state: Board, moves: list[Move]) -> tuple[Move, Any]:
        ...

class TurnData(TypedDict):
    board: Board
    move: Move
    player_data: Any

class GameData(TypedDict):
    game: Game
    players: list[PlayerClass]
    turn_limit: Optional[int]
    turn_data: list[TurnData]
    ended: bool
    winner: Optional[Player]

def play(game: Game, players: list[PlayerClass], turn_limit: Optional[int] = None) -> Any:
    '''
    Plays a game of chinese checkers with the given players.
    The players should implement the Player protocol, which requires a select_move method.
    The game will continue until it ends or the turn limit is reached.
    This function returns data about the game and each turn.
    '''
    data: GameData = {
        'game': game,
        'players': players,
        'turn_limit': turn_limit,
        'turn_data': [],
        'ended': False,
        'winner': None
    }

    turn = 0
    while not game.end and (turn_limit is None or turn < turn_limit):
        player = players[turn % len(players)]

        moves = game.start_turn()
        
        move, player_data = player.select_move(game.board, moves)

        turn_data: TurnData = {
            'board': copy.deepcopy(game.board),
            'move': move,
            'player_data': player_data
        }
        data['turn_data'].append(turn_data)

        game.end_turn(move)

        turn += 1
    
    if game.winner is None:
        logger.info("It's a draw!")
    else:
        logger.info(f"Player {game.winner} wins!")
    
    data['ended'] = game.end
    data['winner'] = game.winner

    return data
