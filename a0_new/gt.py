from typing import Any
from a0_new.protocols.ground_truth import GTProtocol
from a0_new.protocols.game import A0State, A0Action

_gt_instance = None

def get_gt() -> GTProtocol[A0State[Any], A0Action]:
    global _gt_instance
    if _gt_instance is None:
        raise RuntimeError("GT not set! Call set_gt() first.")
    return _gt_instance

def set_gt(gt: GTProtocol[A0State[Any], A0Action]) -> None:
    global _gt_instance
    _gt_instance = gt
