from __future__ import annotations

from a0_new.protocols.player import A0Player

from a0_new.cc.game import CCGame, CCState, CCAction
from a0_new.cc.model import CCModel

class CCPlayer(A0Player[CCModel, CCState, CCAction]):
    def __init__(self, game: CCGame, model: CCModel) -> None:
        self.game = game
        self.model = model

    def process_state(self, state: CCState, legal_actions: list[CCAction]) -> tuple[CCAction, float, Any]:
        pass