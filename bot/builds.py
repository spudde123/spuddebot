from enum import Enum
import sc2
from sc2.constants import *


class ResourcePriority(Enum):
    STRUCTURES = 0,
    EXPANSION = 1,
    UNITS = 2


class ArmyPriority(Enum):
    BIO = 0,
    MECH = 1


class Tech:
    def __init__(self, bot: sc2.BotAI):
        self._unit_counts = dict()
        self._pre_addon_units = dict()
        self._unit_types = dict()
        self._time = 0
        self.bot = bot

        self.builds = {
            ArmyPriority.BIO: {
                'build_after': {
                    (UnitTypeId.ENGINEERINGBAY, 1): (UnitTypeId.STARPORT, 1),
                    (UnitTypeId.BARRACKS, 2): (UnitTypeId.FACTORY, 1),
                    (UnitTypeId.BARRACKS, 2): (UnitTypeId.STARPORT, 1),
                    (UnitTypeId.COMMANDCENTER, 2): (UnitTypeId.ORBITALCOMMAND, 1)
                },
                'pre_addon_units': {
                    UnitTypeId.STARPORT: (UnitTypeId.MEDIVAC, 1),
                    UnitTypeId.BARRACKS: (UnitTypeId.REAPER, 1)
                },
                'building_limits': [
                    {
                        UnitTypeId.BARRACKS: 1,
                        UnitTypeId.BARRACKSTECHLAB: 1,
                        UnitTypeId.FACTORY: 0,
                        UnitTypeId.STARPORT: 0,
                        UnitTypeId.ENGINEERINGBAY: 0,
                        UnitTypeId.ARMORY: 0,
                        UnitTypeId.REFINERY: 1
                    },
                    {
                        UnitTypeId.BARRACKS: 2,
                        UnitTypeId.BARRACKSTECHLAB: 1,
                        UnitTypeId.BARRACKSREACTOR: 1,
                        UnitTypeId.FACTORY: 1,
                        UnitTypeId.FACTORYTECHLAB: 1,
                        UnitTypeId.STARPORT: 1,
                        UnitTypeId.STARPORTTECHLAB: 1,
                        UnitTypeId.ENGINEERINGBAY: 1,
                        UnitTypeId.ARMORY: 0,
                        UnitTypeId.REFINERY: 3
                    },
                    {
                        UnitTypeId.BARRACKS: 4,
                        UnitTypeId.BARRACKSTECHLAB: 2,
                        UnitTypeId.BARRACKSREACTOR: 2,
                        UnitTypeId.FACTORY: 1,
                        UnitTypeId.FACTORYTECHLAB: 1,
                        UnitTypeId.STARPORT: 2,
                        UnitTypeId.STARPORTTECHLAB: 1,
                        UnitTypeId.STARPORTREACTOR: 1,
                        UnitTypeId.ENGINEERINGBAY: 2,
                        UnitTypeId.ARMORY: 1,
                        UnitTypeId.REFINERY: 6
                    },
                    {
                        UnitTypeId.BARRACKS: 6,
                        UnitTypeId.BARRACKSTECHLAB: 2,
                        UnitTypeId.BARRACKSREACTOR: 4,
                        UnitTypeId.FACTORY: 2,
                        UnitTypeId.FACTORYTECHLAB: 2,
                        UnitTypeId.STARPORT: 2,
                        UnitTypeId.STARPORTTECHLAB: 1,
                        UnitTypeId.STARPORTREACTOR: 1,
                        UnitTypeId.ENGINEERINGBAY: 2,
                        UnitTypeId.ARMORY: 1,
                        UnitTypeId.REFINERY: 8
                    },
                    {
                        UnitTypeId.BARRACKS: 7,
                        UnitTypeId.BARRACKSTECHLAB: 3,
                        UnitTypeId.BARRACKSREACTOR: 4,
                        UnitTypeId.FACTORY: 2,
                        UnitTypeId.FACTORYTECHLAB: 2,
                        UnitTypeId.STARPORT: 3,
                        UnitTypeId.STARPORTTECHLAB: 1,
                        UnitTypeId.STARPORTREACTOR: 2,
                        UnitTypeId.ENGINEERINGBAY: 2,
                        UnitTypeId.ARMORY: 2,
                        UnitTypeId.REFINERY: 10
                    }
                ],
                'priority_units': {
                    UnitTypeId.REAPER: 1,
                    UnitTypeId.MARINE: 200,
                    UnitTypeId.SIEGETANK: 2,
                    UnitTypeId.MEDIVAC: 1
                },
                'units': {
                    UnitTypeId.SIEGETANK: 6,
                    UnitTypeId.MARINE: 200,
                    UnitTypeId.MARAUDER: 200,
                    UnitTypeId.MEDIVAC: 4,
                    UnitTypeId.RAVEN: 2,
                    UnitTypeId.VIKINGFIGHTER: 8
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
                    UpgradeId.STIMPACK: 0.7,
                    UnitTypeId.MEDIVAC: 1,
                    UnitTypeId.MARINE: 10
                },
                'attack_timing': 130
            }
        }

    def _count_buildings(self, unit_type: UnitTypeId) -> int:
        ready = self.bot.structures(unit_type).ready.amount
        if unit_type == UnitTypeId.BARRACKS:
            ready += self.bot.structures(UnitTypeId.BARRACKSFLYING).ready.amount
        elif unit_type == UnitTypeId.FACTORY:
            ready += self.bot.structures(UnitTypeId.FACTORYFLYING).ready.amount
        elif unit_type == UnitTypeId.STARPORT:
            ready += self.bot.structures(UnitTypeId.STARPORTFLYING).ready.amount
        elif unit_type == UnitTypeId.ORBITALCOMMAND:
            ready += self.bot.structures(UnitTypeId.ORBITALCOMMANDFLYING).ready.amount
        elif unit_type == UnitTypeId.COMMANDCENTER:
            ready += self.bot.structures(UnitTypeId.COMMANDCENTERFLYING).ready.amount

        return ready + self.bot.already_pending(unit_type)

    def _get_building_limit(self, army_type: ArmyPriority, building_type: UnitTypeId) -> int:
        base_count = self.bot.townhalls.amount

        if base_count <= 0:
            index_to_check = 0
        elif base_count < len(self.builds[army_type]['building_limits']) + 1:
            index_to_check = base_count - 1
        else:
            index_to_check = -1

        if building_type in self.builds[army_type]['building_limits'][index_to_check]:
            return self.builds[army_type]['building_limits'][index_to_check][building_type]
        else:
            return 0

    def should_expand(self, army_type: ArmyPriority):
        base_count = self.bot.townhalls.amount

        for ordering in self.builds[army_type]['build_after']:
            if ordering[0] == UnitTypeId.COMMANDCENTER and base_count + 1 == ordering[1]:
                req_type, req_amount = self.builds[army_type]['build_after'][ordering]
                if self.bot.all_own_units(req_type).ready.amount + self.bot.already_pending(req_type) < req_amount:
                    return False

        return True

    def should_build(self, army_type: ArmyPriority, building_type: UnitTypeId) -> bool:
        current_amount = self._count_buildings(building_type)

        for ordering in self.builds[army_type]['build_after']:
            if ordering[0] == building_type and current_amount + 1 == ordering[1]:
                req_type, req_amount = self.builds[army_type]['build_after'][ordering]
                if req_type not in self._unit_counts or self._unit_counts[req_type] < req_amount:
                    return False

        return current_amount < self._get_building_limit(army_type, building_type)

    def _count_trained_pre_addon_units(self, unit: UnitTypeId):
        if unit not in self._pre_addon_units:
            return self.bot.already_pending(unit)
        else:
            return self._pre_addon_units[unit] + self.bot.already_pending(unit)

    def should_build_addon(self, army_type: ArmyPriority, building_type: UnitTypeId, addon_type: UnitTypeId):
        current_amount = self._count_buildings(addon_type)

        if ('pre_addon_units' in self.builds[army_type]
                and building_type in self.builds[army_type]['pre_addon_units']
                and (self._count_trained_pre_addon_units(self.builds[army_type]['pre_addon_units'][building_type][0])
                     < self.builds[army_type]['pre_addon_units'][building_type][1])):
            return False

        for ordering in self.builds[army_type]['build_after']:
            if ordering[0] == addon_type and current_amount + 1 == ordering[1]:
                req_type, req_amount = self.builds[army_type]['build_after'][ordering]
                if req_type not in self._unit_counts or self._unit_counts[req_type] < req_amount:
                    return False

        return current_amount < self._get_building_limit(army_type, addon_type)

    def should_train_unit(self, army_type: ArmyPriority, unit_type: UnitTypeId) -> bool:

        if self._time < 400:
            if unit_type in self.builds[army_type]['priority_units']:
                return unit_type not in self._unit_counts or self._unit_counts[unit_type] + self.bot.already_pending(unit_type) < self.builds[army_type]['priority_units'][unit_type]
            else:
                return False
        else:
            return (unit_type in self.builds[army_type]['units']
                    and (unit_type not in self._unit_counts
                         or self._unit_counts[unit_type] + self.bot.already_pending(unit_type) < self.builds[army_type]['units'][unit_type]))

    def should_build_refinery(self, army_type: ArmyPriority) -> bool:
        base_count = self.bot.townhalls.amount
        current_amount = self._count_buildings(UnitTypeId.REFINERY)

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

        if unit_type not in self._pre_addon_units:
            self._pre_addon_units[unit_type] = 0

        self._pre_addon_units[unit_type] += 1
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

