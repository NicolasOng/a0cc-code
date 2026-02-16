from typing import Optional

from a0_new.protocols.ground_truth import GTProtocol, PolicyType

from a0_new.cc.game import CCState, CCAction
from a0_new.protocols.game import Player
from a0_new.policy import Policy

from cc.ground_truth import GroundTruth
from cc.core import Game, Board, Move, Player as CCPlayer

from config import config

class CCGT(GTProtocol[CCState, CCAction]):
    # Implementation of GTProtocol for the specific game
    def __init__(self) -> None:
        self.gt = GroundTruth()
        self.home_size = self.gt.unrank(0).home_size
        self.board_size = config.board_size
    
    def rank(self, state: CCState) -> int:
        return self.gt.rank(state.board)
    
    def unrank(self, rank: int) -> CCState:
        board = self.gt.unrank(rank)
        state = CCState(self.board_size, self.home_size)
        state.init_from_board(board)
        return state

    def get_max_rank(self) -> int:
        return self.gt.get_max_rank()
    
    def get_outcome(self, state: CCState) -> float:
        return self.gt.get_outcome(state.board)

    def get_winner(self, state: CCState) -> Optional[Player]:
        p = self.gt.get_winner(state.board)
        if p is None:
            return None
        return Player.X if p == CCPlayer.PLAYER_X else Player.O
    
    def is_illegal(self, state: CCState) -> bool:
        return self.gt.is_illegal(state.board)
    
    def is_draw(self, state: CCState) -> bool:
        return self.gt.is_draw(state.board)
    
    def get_policy(self, state: CCState, p_type: PolicyType) -> Policy[CCAction]:
        '''
        Returns the ground truth policy for the given state.
        '''
        moves, outcomes = self.gt.get_1ply_policy_moves(state.board)
        p = Policy[CCAction]([CCAction(move.start.x, move.start.y, move.end.x, move.end.y) for move in moves], outcomes)
        if p_type == PolicyType.PROB_DIST:
            p.softmax(1)
            return p
        elif p_type == PolicyType.OUTCOMES:
            return p
        else:
            raise ValueError(f"Unknown policy type: {p_type}")
    
    def is_trivial(self, state: CCState) -> bool:
        return self.gt.is_trivial(state.board)
    
    def is_terminal(self, state: CCState) -> bool:
        return self.gt.cc.get_done(state.board)
