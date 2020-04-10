from sc2.helpers.control_group import ControlGroup
import enum
from sc2.position import Point2
from typing import Optional


class ArmyMode(enum.Enum):
    PASSIVE = 0,
    ATTACK = 1,
    RETREAT = 2,
    SCOUT = 3,
    STRIKE = 4


class ArmyGroup(ControlGroup):
    def __init__(self):
        super().__init__(self)
        self.target_expansion: Optional[Point2] = None
        self.mode: ArmyMode = ArmyMode.PASSIVE
        self.waiting_spot: Optional[Point2] = None
        self.leader_unit: Optional[int] = None
