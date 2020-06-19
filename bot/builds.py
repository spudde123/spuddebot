from enum import Enum
from sc2.constants import *
import math


class ResourcePriority(Enum):
    STRUCTURES = 0,
    EXPANSION = 1,
    UNITS = 2


class ArmyPriority(Enum):
    BIO = 0,
    MECH = 1,
    BIOVSZERG = 2


class Tech:
    def __init__(self):
        self._unit_counts = dict()
        self._unit_types = dict()
        self._time = 0

        self.builds = {
            ArmyPriority.MECH: {
                'prerequisites': {
                    UnitTypeId.ENGINEERINGBAY: UnitTypeId.STARPORT
                },
                'building_limits': [
                    {
                        UnitTypeId.BARRACKS: 1,
                        UnitTypeId.FACTORY: 1,
                        UnitTypeId.STARPORT: 1,
                        UnitTypeId.ENGINEERINGBAY: 0,
                        UnitTypeId.ARMORY: 0,
                        UnitTypeId.REFINERY: 2
                    },
                    {
                        UnitTypeId.BARRACKS: 1,
                        UnitTypeId.FACTORY: 2,
                        UnitTypeId.STARPORT: 1,
                        UnitTypeId.ENGINEERINGBAY: 0,
                        UnitTypeId.ARMORY: 1,
                        UnitTypeId.REFINERY: 3
                    },
                    {
                        UnitTypeId.BARRACKS: 1,
                        UnitTypeId.FACTORY: 3,
                        UnitTypeId.STARPORT: 1,
                        UnitTypeId.ENGINEERINGBAY: 1,
                        UnitTypeId.ARMORY: 2,
                        UnitTypeId.REFINERY: 6
                    },
                    {
                        UnitTypeId.BARRACKS: 1,
                        UnitTypeId.FACTORY: 5,
                        UnitTypeId.STARPORT: 2,
                        UnitTypeId.ENGINEERINGBAY: 1,
                        UnitTypeId.ARMORY: 2,
                        UnitTypeId.REFINERY: 8
                    },
                    {
                        UnitTypeId.BARRACKS: 1,
                        UnitTypeId.FACTORY: 7,
                        UnitTypeId.STARPORT: 3,
                        UnitTypeId.ENGINEERINGBAY: 1,
                        UnitTypeId.ARMORY: 2,
                        UnitTypeId.REFINERY: 10
                    }
                ],
                'priority_units': {
                    UnitTypeId.REAPER: 2,
                    UnitTypeId.MARINE: 12,
                    UnitTypeId.HELLION: 4,
                    UnitTypeId.SIEGETANK: 1,
                    UnitTypeId.BANSHEE: 1,
                    UnitTypeId.CYCLONE: 2
                },
                'add_ons': {
                    UnitTypeId.BARRACKS: False,
                    UnitTypeId.FACTORY: True,
                    UnitTypeId.STARPORT: True
                },
                'units': {
                    UnitTypeId.MARINE: 4,
                    UnitTypeId.THOR: 200,
                    UnitTypeId.CYCLONE: 15,
                    UnitTypeId.HELLION: 18,
                    UnitTypeId.SIEGETANK: 2,
                    UnitTypeId.RAVEN: 2,
                    UnitTypeId.BANSHEE: 5,
                    UnitTypeId.VIKINGFIGHTER: 5
                },
                'upgrades': {
                    UnitTypeId.ENGINEERINGBAY: False,
                    UnitTypeId.BARRACKSTECHLAB: False,
                    UnitTypeId.ARMORY: True,
                    UnitTypeId.FACTORYTECHLAB: True,
                    UnitTypeId.STARPORTTECHLAB: True
                },
                'attack_timing': 150

            },
            ArmyPriority.BIO: {
                'prerequisites': {
                    UnitTypeId.ENGINEERINGBAY: UnitTypeId.STARPORT
                },
                'building_limits': [
                    {
                        UnitTypeId.BARRACKS: 2,
                        UnitTypeId.FACTORY: 0,
                        UnitTypeId.STARPORT: 0,
                        UnitTypeId.ENGINEERINGBAY: 0,
                        UnitTypeId.ARMORY: 0,
                        UnitTypeId.REFINERY: 1
                    },
                    {
                        UnitTypeId.BARRACKS: 2,
                        UnitTypeId.FACTORY: 1,
                        UnitTypeId.STARPORT: 1,
                        UnitTypeId.ENGINEERINGBAY: 1,
                        UnitTypeId.ARMORY: 0,
                        UnitTypeId.REFINERY: 3
                    },
                    {
                        UnitTypeId.BARRACKS: 4,
                        UnitTypeId.FACTORY: 1,
                        UnitTypeId.STARPORT: 2,
                        UnitTypeId.ENGINEERINGBAY: 2,
                        UnitTypeId.ARMORY: 1,
                        UnitTypeId.REFINERY: 6
                    },
                    {
                        UnitTypeId.BARRACKS: 6,
                        UnitTypeId.FACTORY: 2,
                        UnitTypeId.STARPORT: 2,
                        UnitTypeId.ENGINEERINGBAY: 2,
                        UnitTypeId.ARMORY: 1,
                        UnitTypeId.REFINERY: 8
                    },
                    {
                        UnitTypeId.BARRACKS: 7,
                        UnitTypeId.FACTORY: 2,
                        UnitTypeId.STARPORT: 3,
                        UnitTypeId.ENGINEERINGBAY: 2,
                        UnitTypeId.ARMORY: 2,
                        UnitTypeId.REFINERY: 10
                    }
                ],
                'priority_units': {
                    UnitTypeId.REAPER: 2,
                    UnitTypeId.MARINE: 200,
                    UnitTypeId.SIEGETANK: 2,
                    UnitTypeId.MEDIVAC: 2
                },
                'add_ons': {
                    UnitTypeId.BARRACKS: True,
                    UnitTypeId.FACTORY: True,
                    UnitTypeId.STARPORT: True
                },
                'units': {
                    UnitTypeId.SIEGETANK: 6,
                    UnitTypeId.MARINE: 200,
                    UnitTypeId.MARAUDER: 200,
                    UnitTypeId.MEDIVAC: 5,
                    UnitTypeId.RAVEN: 2,
                    UnitTypeId.VIKINGFIGHTER: 4
                },
                'upgrades': {
                    UnitTypeId.ENGINEERINGBAY: True,
                    UnitTypeId.BARRACKSTECHLAB: True,
                    UnitTypeId.ARMORY: True,
                    UnitTypeId.FACTORYTECHLAB: False,
                    UnitTypeId.STARPORTTECHLAB: False
                },
                'poke_timing': {
                    UpgradeId.SHIELDWALL: 1,
                    UpgradeId.STIMPACK: 0.8,
                    UnitTypeId.MEDIVAC: 1,
                    UnitTypeId.MARINE: 10
                },
                'attack_timing': 130
            }
        }

    def _get_building_limit(self, army_type: ArmyPriority, building_type: UnitTypeId, base_count: int) -> int:
        if base_count <= 0:
            return self.builds[army_type]['building_limits'][0][building_type]
        elif base_count < len(self.builds[army_type]['building_limits']) + 1:
            return self.builds[army_type]['building_limits'][base_count - 1][building_type]
        else:
            return self.builds[army_type]['building_limits'][-1][building_type]

    def _get_techlab_limit(self, army_type: ArmyPriority, building_type: UnitTypeId, building_count: int) -> int:
        if army_type == ArmyPriority.MECH:
            if building_type == UnitTypeId.BARRACKS:
                return 0
            elif building_type == UnitTypeId.FACTORY:
                return math.floor(0.7*building_count)
            elif building_type == UnitTypeId.STARPORT:
                return math.ceil(building_count / 2)
        elif army_type == ArmyPriority.BIO:
            if building_type == UnitTypeId.BARRACKS:
                return math.ceil(building_count / 3)
            elif building_type == UnitTypeId.FACTORY:
                return building_count
            elif building_type == UnitTypeId.STARPORT:
                return 1
        return math.ceil(building_count / 3)

    def should_build(self, army_type: ArmyPriority, building_type: UnitTypeId, current_amount: int, base_count: int) -> bool:
        if building_type in self.builds[army_type]['prerequisites']:
            req = self.builds[army_type]['prerequisites'][building_type]
            if req not in self._unit_counts or self._unit_counts[req] == 0:
                return False

        return current_amount < self._get_building_limit(army_type, building_type, base_count)

    def should_build_techlab(self, army_type: ArmyPriority, building_type: UnitTypeId, building_count: int,
                             techlab_count: int) -> bool:

        return techlab_count < self._get_techlab_limit(army_type, building_type, building_count)

    def should_train_unit(self, army_type: ArmyPriority, unit_type: UnitTypeId) -> bool:

        if self._time < 400:
            if unit_type in self.builds[army_type]['priority_units']:
                return unit_type not in self._unit_counts or self._unit_counts[unit_type] < self.builds[army_type]['priority_units'][unit_type]
            else:
                return False
        else:
            return (unit_type in self.builds[army_type]['units']
                    and (unit_type not in self._unit_counts
                         or self._unit_counts[unit_type] < self.builds[army_type]['units'][unit_type]))

    def should_build_refinery(self, army_type: ArmyPriority, base_count: int, current_amount: int) -> bool:
        if base_count == 0:
            return False
        elif len(self.builds[army_type]['building_limits']) >= base_count:
            relevant_buildings = {UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT,
                                  UnitTypeId.ENGINEERINGBAY,
                                  UnitTypeId.ARMORY}
            count = 0
            for building_type in relevant_buildings:
                if building_type in self._unit_counts:
                    count += 2

            return current_amount < min(count, self.builds[army_type]['building_limits'][base_count - 1][UnitTypeId.REFINERY])
        else:
            return current_amount < 2*base_count

    def unit_created(self, unit_type: UnitTypeId, tag: int):
        if unit_type not in self._unit_counts:
            self._unit_counts[unit_type] = 0

        self._unit_counts[unit_type] += 1
        self._unit_types[tag] = unit_type

    def unit_destroyed(self, tag: int, army_type: ArmyPriority):
        if tag in self._unit_types:
            unit_type = self._unit_types[tag]

            # if the unit is meant to be made only a certain
            # amount early game we don't count the deaths
            if unit_type in self.builds[army_type]["units"]:
                self._unit_counts[unit_type] -= 1
                del self._unit_types[tag]

    def update_time(self, time: float):
        self._time = time

    def delay_attack_timing(self, army_type: ArmyPriority):
        self.builds[army_type]['attack_timing'] = 180

