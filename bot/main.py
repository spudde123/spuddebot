import sc2
from sc2.constants import *
from sc2.helpers import ControlGroup
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2
from sc2.data import ActionResult, CloakState, Race
from sc2.cache import property_cache_forever, property_cache_once_per_frame
from sc2.game_info import Ramp
import time
import math
from typing import List, Optional, Union
from .builds import ArmyPriority, ResourcePriority, Tech
from .unit_control import ArmyMode, ArmyGroup
from .data import TerranData
from MapAnalyzer import MapData

RETREAT_TIME_THRESHOLD = 60


class MyBot(sc2.BotAI):
    def __init__(self):
        super().__init__()
        self.prev_game_loop = -math.inf
        self.prev_game_seconds = 0
        self.iterations_used = 0

        self.tech = Tech(self)
        self.data = TerranData()
        self.resource_priority = ResourcePriority.STRUCTURES
        self.cheese_defense_scv: Optional[Unit] = None

        self.time_budget_available: float = 1
        self.last_scan_time: float = 0

        self.map_has_inner_expansion = False

        self.save_energy_for_scan = False
        self.poke_scan_done = False
        self.should_be_aggressive = True

        self.building_status = dict()

        self.main_army = ArmyGroup()
        self.army_retreat_start_time = 0

        # reaper group, hellion group, banshee group
        self.strike_teams: List[ArmyGroup] = [ArmyGroup(), ArmyGroup(), ArmyGroup()]
        self.strike_teams[0].mode = ArmyMode.ATTACK

        self.scouting_units = ControlGroup({})
        self.new_units = ControlGroup({})
        self.area_clearing_units = ArmyGroup()

        self.block_depots = ControlGroup({})
        self.block_rax = ControlGroup({})
        self.rax_techlabs = ControlGroup({})
        self.rax_reactors = ControlGroup({})
        self.fac_reactors = ControlGroup({})
        self.fac_techlabs = ControlGroup({})
        self.sp_reactors = ControlGroup({})
        self.sp_techlabs = ControlGroup({})

        # order of likely enemy expansions and when it has been checked
        self.enemy_expansion_order: List[Point2] = []
        self.enemy_expansion_checks = dict()

        self.expansion_order: List[Point2] = []

        self.enemy_unit_sizes = dict()
        self.unit_sizes = dict()
        self.townhall_tags = set()
        self.dead_unit_counter = [0, 0, 0, 0]
        self.expansion_attempts = dict()
        self.failed_addons = set()

    def should_build_scv(self) -> bool:
        room = 0
        for th in self.townhalls.filter(lambda x: not self.base_is_depleted(x)):
            room += th.surplus_harvesters

        for ref in self.structures(UnitTypeId.REFINERY).filter(lambda x: x.vespene_contents > 0):
            room += ref.surplus_harvesters

        # produce a few more harvesters than needed early on
        # so new bases can be saturated faster
        # don't build if orbitals should be prioritized first
        orbitals_needed = self.townhalls.ready.amount - (self.structures(UnitTypeId.ORBITALCOMMAND)
                                                         | self.structures(UnitTypeId.ORBITALCOMMANDFLYING)
                                                         | self.structures(UnitTypeId.PLANETARYFORTRESS)).amount
        if orbitals_needed > 0:
            for th in self.townhalls.filter(lambda x: not x.is_idle):
                if (th.orders[0].ability.id == AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND
                        or th.orders[0].ability.id == AbilityId.UPGRADETOPLANETARYFORTRESS_PLANETARYFORTRESS):
                    orbitals_needed -= 1

        # make sure i build workers if i lose them in the beginning
        return self.workers.amount < 14 or (self.workers.amount < 75 and room < 16
                                            and (not self.building_requirements_satisfied(UnitTypeId.ORBITALCOMMAND) or orbitals_needed <= 0))
    
    def should_build_supply(self) -> bool:
        below_supply_cap = self.supply_cap < 200
        depots_in_production = self.already_pending(UnitTypeId.SUPPLYDEPOT)
        early_game_condition = self.supply_left < 6 and self.supply_used >= 13 and depots_in_production < 1
        later_game_condition = self.supply_left < 12 and self.supply_cap >= 40 and depots_in_production <= 2

        return below_supply_cap and (early_game_condition or later_game_condition)

    def print_debug_info(self):

        print("Army mode: {}".format(self.main_army.mode))
        print("Army sizes: {} {}".format(len(self.main_army), len(self.new_units)))
        if self.time_budget_available:
            print("Budget: {}".format(self.time_budget_available))

    def send_early_scout(self, unit: Unit):
        self.scouting_units.add_unit(unit)
        for i in range(2):
            unit(AbilityId.MOVE, self.enemy_start_locations[0], queue=True)
            unit(AbilityId.MOVE, self.enemy_expansion_order[1], queue=True)

    def building_requirements_satisfied(self, building_type: UnitTypeId) -> bool:
        return (self.data.tech_tree[building_type] is None
                or self.structures.of_type(self.data.tech_tree[building_type]).ready.exists)

    def base_is_depleted(self, th: Unit) -> bool:
        mfs = self.mineral_field.closer_than(15, th).filter(lambda x: x.mineral_contents > 0)
        return mfs.empty

    def count_depleted_and_full_bases(self) -> int:
        count = 0
        for th in self.townhalls.ready:
            if self.base_is_depleted(th) or th.surplus_harvesters >= 0:
                count += 1
            elif th.surplus_harvesters < 0:
                count -= 1
        return count

    def get_next_free_expansion(self) -> Point2:
        next_exp = None
        for exp in self.expansion_order:
            def is_near_to_expansion(t):
                return t.distance_to(exp) < 10

            if (any(map(is_near_to_expansion, self.owned_expansions))
                    or any(map(is_near_to_expansion, self.known_enemy_expansions))):
                continue

            next_exp = exp
            break

        return next_exp

    async def try_to_expand(self):

        exp = self.get_next_free_expansion()

        if exp and self.workers.exists:
            w = self.workers.closest_to(exp)
            if w:
                # keeping track of expansion attempts so burrowed units
                # dont mess with the expanding for long
                if exp not in self.expansion_attempts:
                    self.expansion_attempts[exp] = 0

                self.expansion_attempts[exp] += 1

                if w.distance_to(exp) < 5 and self.expansion_attempts[exp] > 2 and self.all_enemy_units.closer_than(5, exp).empty:
                    self.assign_units_to_clear_base(exp)

                loc = await self.find_placement(UnitTypeId.COMMANDCENTER, exp, max_distance=2,
                                                random_alternative=False,
                                                placement_step=1)
                if loc:
                    w.build(UnitTypeId.COMMANDCENTER, loc)

    # lists the expansions in the order the botai expansion code would suggest them
    # pathing doesn't work to my home base or the enemy home base
    # so need to add them to ends
    async def calculate_expansion_order(self, starting_location: Point2, enemy: bool) -> List[Point2]:

        expansion_list = [starting_location]
        expansion_dist_dict = {starting_location: 0}

        for exp in self.expansion_locations_list:
            d = await self._client.query_pathing(starting_location, exp)
            if d is None:
                continue

            insert_index = 0

            for old_exp in expansion_list:
                if d < expansion_dist_dict[old_exp]:
                    break
                else:
                    insert_index += 1

            expansion_list.insert(insert_index, exp)
            expansion_dist_dict[exp] = d

        if enemy:
            expansion_list.append(self.start_location)
        else:
            expansion_list.append(self.enemy_start_locations[0])

        return expansion_list

    @property_cache_once_per_frame
    def known_enemy_expansions(self):
        townhall_names = {UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND,
                          UnitTypeId.PLANETARYFORTRESS, UnitTypeId.HATCHERY,
                          UnitTypeId.LAIR, UnitTypeId.HIVE,
                          UnitTypeId.NEXUS}
        enemy_townhalls = self.enemy_structures.filter(lambda x: x.type_id in townhall_names)

        expansions = {}
        for th in enemy_townhalls:
            expansions[th.position] = th

        return expansions

    @property_cache_forever
    def common_expansion_entrance(self) -> Point2:
        min_dist_exp = min(self.expansion_locations_list, key=lambda x: x.distance_to(self.main_base_ramp.bottom_center))
        return min_dist_exp.towards(self.game_info.map_center, 3)

    @property_cache_forever
    def early_unit_rally_point(self) -> Point2:
        return self.main_base_ramp.top_center.towards(self.start_location, 5).to2

    @property_cache_once_per_frame
    def current_army_resting_point(self) -> Point2:

        num_of_expansions = self.townhalls.amount

        """
        if num_of_expansions <= 1:
            return self.early_unit_rally_point
        elif num_of_expansions == 2 or (num_of_expansions == 3 and self.map_has_inner_expansion):
            army_units = self.main_army.select_units(self.units)
            marine_count = army_units(UnitTypeId.MARINE).amount
            army_units_count = army_units.amount

            if self.structures(UnitTypeId.BUNKER).ready.exists:
                marine_count += self.structures(UnitTypeId.BUNKER).ready.first.cargo_used
                army_units_count += self.structures(UnitTypeId.BUNKER).ready.first.cargo_used
            if army_units_count > 10 or self.structures(UnitTypeId.BUNKER).ready and marine_count >= 4:
                return self.common_expansion_entrance
            else:
                return self.early_unit_rally_point
        """

        if num_of_expansions <= 2 or (num_of_expansions == 3 and self.map_has_inner_expansion):
            return self.common_expansion_entrance
        elif num_of_expansions == 3:
            return self.expansion_order[2].towards(self.game_info.map_center, 5)
        else:
            return (self.expansion_order[2].towards(self.game_info.map_center, 5)
                    if self.expansion_order[2].distance_to(self.game_info.map_center) <= self.expansion_order[3].distance_to(self.game_info.map_center)
                    else self.expansion_order[3].towards(self.game_info.map_center, 5))

    @property_cache_once_per_frame
    def current_strike_target(self) -> Point2:
        if self.time > 200:
            return self.enemy_expansion_order[2]
        else:
            return self.early_strike_target

    @property_cache_forever
    def early_strike_target(self) -> Point2:
        return self.enemy_start_locations[0].towards(self.game_info.map_center, -5)

    @property_cache_forever
    def enemy_natural_entrance(self) -> Point2:
        min_dist = math.inf
        for ramp in self.game_info.map_ramps:
            cur_dist = self.enemy_start_locations[0]._distance_squared(ramp.top_center)
            if len(ramp.upper2_for_ramp_wall) == 2 and cur_dist < min_dist:
                min_dist = cur_dist
                enemy_block_ramp = ramp

        closest_exp = min(self.enemy_expansion_order, key=lambda x: x._distance_squared(enemy_block_ramp.bottom_center))

        return closest_exp.towards(self.game_info.map_center, 25)

    # place production buildings in nicely spaced out rows so bigger units dont
    # get stuck. using CC size so that there will also be enough space
    # around the edges

    async def find_production_placement(self, near: Point2) -> Optional[Point2]:

        assert isinstance(near, Point2)

        building = self._game_data.units[UnitTypeId.COMMANDCENTER.value].creation_ability

        near = near.offset((1, 0))

        if await self.can_place(building, near):
            return near.offset((-1, 0))

        x_step = 7
        max_distance = 45
        y_steps = [0, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40]

        all_other_level_positions = []

        for x_distance in range(0, max_distance, x_step):
            for y_distance in y_steps:
                possible_positions = [Point2(p).offset(near).to2 for p in (
                    [(x_distance, y_distance), (-x_distance, y_distance), (x_distance, -y_distance), (-x_distance, -y_distance)]
                )]

                res = await self._client.query_building_placement(building, possible_positions)
                possible = [p for r, p in zip(res, possible_positions) if r == ActionResult.Success]
                if not possible:
                    continue

                same_level_positions = list(filter(lambda x: self.get_terrain_height(x) == self.get_terrain_height(near), possible))
                other_level_positions = list(filter(lambda x: self.get_terrain_height(x) != self.get_terrain_height(near), possible))

                if not same_level_positions:
                    all_other_level_positions += other_level_positions
                    continue

                distance_calculations = [[near, same_level_positions[i]] for i in
                                         range(len(same_level_positions))]
                path_lengths = await self._client.query_pathings(distance_calculations)

                viable_same_level_positions = [val for idx, val in enumerate(same_level_positions) if
                                                path_lengths[idx] > 0]
                viable_path_lengths = [val for val in path_lengths if val > 0]
                if len(viable_same_level_positions) > 0 and min(viable_path_lengths) < 120:

                    minimum_length, closest_index = min((val, idx) for (idx, val) in enumerate(viable_path_lengths))
                    closest = viable_same_level_positions[closest_index]
                    if closest.y > near.y:
                        y_offset = -1
                    elif closest.y < near.y:
                        y_offset = 1
                    else:
                        y_offset = 0

                    return closest.offset((-1, y_offset))

        if all_other_level_positions:
            distance_calculations = [[near, all_other_level_positions[i]] for i in range(len(all_other_level_positions))]
            path_lengths = await self._client.query_pathings(distance_calculations)

            viable_other_level_positions = [val for idx, val in enumerate(all_other_level_positions) if path_lengths[idx] > 0]
            viable_path_lengths = [val for val in path_lengths if val > 0]

            if len(viable_other_level_positions) > 0 and min(viable_path_lengths) < 120:

                minimum_length, closest_index = min((val, idx) for (idx, val) in enumerate(viable_path_lengths))
                closest = viable_other_level_positions[closest_index]
                if closest.y > near.y:
                    y_offset = -1
                elif closest.y < near.y:
                    y_offset = 1
                else:
                    y_offset = 0

                return closest.offset((-1, y_offset))

        return None

    async def build_near_base(self, unit_type: UnitTypeId, th: Optional[Point2] = None):
        if th is None:
            th = self.start_location

        if self.workers.gathering.exists:
            w = self.workers.gathering.closest_to(th)
            production_types = {UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT}

            if unit_type in production_types:
                existing_production = self.structures.filter(lambda x: x.type_id in production_types)
                if existing_production.exists:
                    loc = await self.find_production_placement(self.main_base_ramp.barracks_correct_placement)
                else:
                    loc = await self.find_production_placement(th.towards(self.game_info.map_center, 5))
            elif unit_type == UnitTypeId.MISSILETURRET:
                loc = await self.find_placement(unit_type, th.towards(self.enemy_start_locations[0], 2),
                                                placement_step=2, random_alternative=False)
            else:
                closest_minerals = self.mineral_field.closest_to(th).position

                loc = await self.find_placement(unit_type, th.towards(closest_minerals, 8),
                                                placement_step=2, random_alternative=False,
                                                max_distance=24)
            if loc:
                w.build(unit_type, loc)

    async def build_bunker(self):
        th = min(self.expansion_order, key=lambda x: x.distance_to(self.common_expansion_entrance))

        if self.workers.gathering.exists:
            w = self.workers.gathering.closest_to(th)
            base_ramp = self.get_base_ramp(th)

            if base_ramp is not None:
                loc = await self.find_placement(UnitTypeId.BUNKER, base_ramp.top_center, placement_step=1, random_alternative=False)
            else:
                loc = await self.find_placement(UnitTypeId.BUNKER,
                                                th.towards(self.main_base_ramp.bottom_center, 2).towards(self.game_info.map_center, 6),
                                                placement_step=2,
                                                random_alternative=False)
            if loc:
                w.build(UnitTypeId.BUNKER, loc)

    def get_base_ramp(self, exp: Point2) -> Optional[Ramp]:
        for ramp in self.game_info.map_ramps:
            if (ramp.top_center.distance_to(exp) < 15
                    and self.get_terrain_height(ramp.top_center) == self.get_terrain_height(exp.position)):
                return ramp

        return None

    async def build_depot(self):
        ws = self.workers.gathering
        depot_count = self.structures.of_type([UnitTypeId.SUPPLYDEPOT, UnitTypeId.SUPPLYDEPOTLOWERED]).amount
        if ws.exists:
            w = ws.closest_to(self.start_location)

            if depot_count == 0:
                loc = await self.find_placement(UnitTypeId.SUPPLYDEPOT, list(self.main_base_ramp.corner_depots)[0],
                                                placement_step=2, random_alternative=False)
            elif depot_count == 1:
                loc = await self.find_placement(UnitTypeId.SUPPLYDEPOT, list(self.main_base_ramp.corner_depots)[1],
                                                placement_step=2, random_alternative=False)
            else:
                loc = await self.find_placement(UnitTypeId.SUPPLYDEPOT, w.position, placement_step=2,
                                                random_alternative=False)
            if loc:
                w.build(UnitTypeId.SUPPLYDEPOT, loc)

    async def build_addon(self, building: Unit, addon: UnitTypeId):
        # there may be cases where for some reason addon cant be built this way
        # make sure we don't try it all the time if it already failed
        if building.tag in self.failed_addons:
            return

        addon_offset = Point2((2.5, -0.5))
        can_build_to_current_location = await self.can_place(UnitTypeId.SUPPLYDEPOT, building.position.offset(addon_offset))
        if can_build_to_current_location:
            building.build(addon)
        else:
            # if this is the ramp blocking barracks, don't build addon here
            loc = await self.find_production_placement(building.position)
            if loc:
                building.build(addon, loc)
            else:
                self.failed_addons.add(building.tag)

    def control_blocking_depots(self):
        depots = self.block_depots.select_units(self.structures)
        if depots.empty:
            return

        enemy_ground_units = self.get_army_supply(self.enemy_units.filter(lambda x: not x.is_flying).closer_than(10, self.main_base_ramp.top_center))
        army_units = self.units - self.workers
        my_ground_units = self.get_army_supply(army_units.closer_than(10, self.main_base_ramp.top_center))

        if enemy_ground_units == 0 or my_ground_units - 2 > enemy_ground_units:
            for depot in depots.filter(lambda x: x.type_id == UnitTypeId.SUPPLYDEPOT):
                depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)
        else:
            for depot in depots.filter(lambda x: x.type_id == UnitTypeId.SUPPLYDEPOTLOWERED):
                depot(AbilityId.MORPH_SUPPLYDEPOT_RAISE)

    async def on_building_construction_complete(self, unit: Unit):
        if unit.type_id == UnitTypeId.SUPPLYDEPOT:
            if (self.structures(UnitTypeId.SUPPLYDEPOT) | self.structures(UnitTypeId.SUPPLYDEPOTLOWERED)).ready.amount <= 2:
                self.block_depots.add_unit(unit)
            else:
                unit(AbilityId.MORPH_SUPPLYDEPOT_LOWER)
        elif unit.type_id == UnitTypeId.BARRACKSREACTOR:
            self.rax_reactors.add_unit(unit)
        elif unit.type_id == UnitTypeId.BARRACKSTECHLAB:
            self.rax_techlabs.add_unit(unit)
        elif unit.type_id == UnitTypeId.BARRACKS and self.structures(UnitTypeId.BARRACKS).ready.amount <= 1:
            self.block_rax.add_unit(unit)
        elif unit.type_id == UnitTypeId.FACTORYTECHLAB:
            self.fac_techlabs.add_unit(unit)
        elif unit.type_id == UnitTypeId.FACTORYREACTOR:
            self.fac_reactors.add_unit(unit)
        elif unit.type_id == UnitTypeId.STARPORTTECHLAB:
            self.sp_techlabs.add_unit(unit)
        elif unit.type_id == UnitTypeId.STARPORTREACTOR:
            self.sp_reactors.add_unit(unit)

    def set_building_rallypoints(self):
        production_buildings = (self.structures(UnitTypeId.BARRACKS) | self.structures(UnitTypeId.FACTORY) | self.structures(UnitTypeId.STARPORT)).ready
        for building in production_buildings:
            building(AbilityId.RALLY_BUILDING, self.early_unit_rally_point)

    async def on_building_construction_started(self, unit: Unit):
        self.tech.unit_created(unit.type_id, unit.tag)

        # reset expansion attempts counter
        if unit.type_id == UnitTypeId.COMMANDCENTER:
            min_dist = math.inf
            min_dist_exp = None
            for exp in self.expansion_attempts:
                cur_dist = exp.distance_to(unit.position)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    min_dist_exp = exp

            if min_dist_exp:
                self.expansion_attempts[min_dist_exp] = 0

    async def on_unit_created(self, unit: Unit):
        if not unit.is_structure and unit.type_id != UnitTypeId.SCV and unit.type_id != UnitTypeId.MULE:

            self.new_units.add_unit(unit)
            self.unit_sizes[unit.tag] = self.calculate_supply_cost(unit.type_id)

        self.tech.unit_created(unit.type_id, unit.tag)

    async def on_unit_destroyed(self, unit_tag: int):
        if unit_tag in self.enemy_unit_sizes:
            self.dead_unit_counter[-1] += self.enemy_unit_sizes[unit_tag]
            del self.enemy_unit_sizes[unit_tag]
        else:
            self.tech.unit_destroyed(unit_tag, self.army_type)

        if unit_tag in self.main_army:
            self.dead_unit_counter[-1] -= self.unit_sizes[unit_tag]
            del self.unit_sizes[unit_tag]
            self.main_army.remove(unit_tag)

    # need to put scv's to harvest things farther away if current base is saturated
    def assign_mineral_workers(self):
        if self.townhalls.amount - self.count_depleted_and_full_bases() <= 1:
            return

        min_saturation_th = None
        min_saturation = math.inf
        for th in self.townhalls.ready:
            if th.surplus_harvesters < min_saturation:
                min_saturation = th.surplus_harvesters
                min_saturation_th = th

        if min_saturation >= 0:
            return

        mfs = self.mineral_field.closer_than(15, min_saturation_th)
        target_mf = None
        if mfs:
            target_mf = mfs.closest_to(min_saturation_th)

        for th in self.townhalls.ready:
            close_mfs = self.mineral_field.closer_than(15, th)
            if th.surplus_harvesters > 0:
                ws = self.workers.gathering.closer_than(20, th).filter(
                    lambda x: x.order_target in close_mfs.tags)
                if ws:
                    for w in ws.take(th.surplus_harvesters):
                        w.gather(target_mf)
                th(AbilityId.RALLY_BUILDING, target_mf)
            elif close_mfs.exists:
                th(AbilityId.RALLY_BUILDING, close_mfs.closest_to(th))

    def assign_gas_workers(self):
        # quick fix so we dont get stuck taking too much gas if workers die etc
        ref_tags = self.structures(UnitTypeId.REFINERY).ready.tags
        if 2*self.minerals < self.vespene and self.vespene > 300:
            ws = (self.workers.gathering.filter(lambda x: x.order_target in ref_tags)
                  .filter(lambda x: self.townhalls.closest_to(x.position).surplus_harvesters <= 0))
            workers_to_move = ws.take(min(6, ws.amount))
            for w in workers_to_move:
                w.gather(self.mineral_field.closest_to(w.position))

            return

        min_saturation_th = None
        min_saturation = math.inf
        for th in self.townhalls.ready:
            if th.surplus_harvesters < min_saturation:
                min_saturation = th.surplus_harvesters
                min_saturation_th = th

        target_mf = None

        if min_saturation_th is not None:
            mfs = self.mineral_field.closer_than(15, min_saturation_th)
            if mfs:
                target_mf = mfs.closest_to(min_saturation_th)

        for ref in self.structures(UnitTypeId.REFINERY).ready:
            if ref.surplus_harvesters < 0:
                ws = self.workers.gathering.closer_than(20, ref).filter(lambda x: x.order_target not in ref_tags)
                if ws.exists:
                    for w in ws.take(-ref.surplus_harvesters):
                        w.gather(ref)
            elif ref.surplus_harvesters > 0:
                ws = self.workers.gathering.filter(lambda x: x.order_target == ref.tag)
                if ws.exists:
                    if target_mf:
                        for w in ws.take(ref.surplus_harvesters):
                            w.gather(target_mf)

    def assign_idle_workers(self):
        idle_workers = self.workers.idle | self.units(UnitTypeId.MULE).idle - self.scouting_units.select_units(self.workers).idle
        ths_with_resources = self.townhalls - self.townhalls.filter(self.base_is_depleted)
        attacking_workers = self.workers.filter(lambda x: x.is_attacking)
        if ths_with_resources.exists:
            for w in idle_workers:
                th = ths_with_resources.closest_to(w)
                mfs = self.mineral_field.closer_than(15, th)
                if mfs:
                    w.gather(mfs.closest_to(th))

            # bring back attacking workers if they are too far away
            for w in attacking_workers:
                closest_th = ths_with_resources.closest_to(w.position)
                if closest_th.position.distance_to(w) > 30:
                    mfs = self.mineral_field.closer_than(15, closest_th)
                    if mfs:
                        w.gather(mfs.closest_to(closest_th))

        else:
            exp = self.get_next_free_expansion()
            if exp:
                mfs = self.mineral_field.closer_than(15, exp)
                if mfs:
                    for w in (idle_workers | attacking_workers):
                        w.gather(mfs.closest_to(exp))

    # it may happen that the bot builds something right under a flying
    # building that is trying to land
    async def fix_idle_flying_buildings(self):
        for flying_building in (self.structures(UnitTypeId.BARRACKSFLYING) | self.structures(UnitTypeId.FACTORYFLYING)
                                | self.structures(UnitTypeId.STARPORTFLYING)).idle:
            loc = await self.find_production_placement(flying_building.position)
            if loc:
                flying_building(AbilityId.LAND, loc)

    def fix_interrupted_construction(self):
        for building in self.structures.not_ready:
            if (building.tag in self.building_status
                    and self.building_status[building.tag]['health_percentage'] > building.health_percentage
                    and building.health_percentage < 0.1):
                building(AbilityId.CANCEL)
            elif (building.tag in self.building_status
                    and self.building_status[building.tag]['build_progress'] == building.build_progress
                    and self.building_status[building.tag]['health_percentage'] == building.health_percentage
                    and self.workers.filter(lambda x: x.order_target == building.tag).empty):
                if self.workers.collecting.exists:
                    w = self.workers.collecting.closest_to(building.position)
                    w(AbilityId.SMART, building)

            self.building_status[building.tag] = {
                'build_progress': building.build_progress,
                'health_percentage': building.health_percentage
            }

    def lift_buildings_under_attack(self):
        liftable_ths = {UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND}
        for th in self.townhalls.ready.filter(lambda x: x.type_id in liftable_ths):
            enemy_can_attack_air = self.all_enemy_units.filter(lambda x: x.can_attack_air).closer_than(15, th.position).exists
            if (th.tag in self.building_status and not enemy_can_attack_air
                    and self.building_status[th.tag]['health_percentage'] > th.health_percentage
                    and th.health_percentage < 0.6):
                if len(th.orders) > 0:
                    th(AbilityId.CANCEL)
                else:
                    if th.type_id == UnitTypeId.COMMANDCENTER:
                        th(AbilityId.LIFT_COMMANDCENTER)
                    else:
                        th(AbilityId.LIFT_ORBITALCOMMAND)

            self.building_status[th.tag] = {
                'build_progress': th.build_progress,
                'health_percentage': th.health_percentage
            }

        landable_ths = {UnitTypeId.COMMANDCENTERFLYING, UnitTypeId.ORBITALCOMMANDFLYING}
        for th in self.structures.ready.filter(lambda x: x.type_id in landable_ths):
            enemy_can_attack_ground = self.all_enemy_units.filter(lambda x: x.can_attack_ground).closer_than(15, th.position).exists
            if not enemy_can_attack_ground:
                if th.type_id == UnitTypeId.COMMANDCENTERFLYING:
                    th(AbilityId.LAND_COMMANDCENTER, th.position, queue=True)
                else:
                    th(AbilityId.LAND_ORBITALCOMMAND, th.position, queue=True)

            self.building_status[th.tag] = {
                'build_progress': th.build_progress,
                'health_percentage': th.health_percentage
            }

    def control_building_fixing(self):
        for building in self.structures.ready:
            if (building.tag in self.building_status
                    and self.building_status[building.tag]['health_percentage'] > building.health_percentage
                    and self.workers.filter(lambda x: x.order_target == building.tag).amount < 3):

                existing_fixers = self.workers.filter(lambda x: x.order_target == building.tag).amount
                new_fixers = self.workers.filter(lambda x: x.is_collecting or x.is_idle).closest_n_units(building.position, 3 - existing_fixers)
                for w in new_fixers:
                    if w.distance_to(building.position) < 15:
                        w(AbilityId.EFFECT_REPAIR_SCV, building)

                self.building_status[building.tag] = {
                    'build_progress': building.build_progress,
                    'health_percentage': building.health_percentage
                }

    def calldown_mules(self):
        ocs = self.townhalls(UnitTypeId.ORBITALCOMMAND).ready.filter(lambda x: x.energy >= 50)
        if ocs.empty or (self.save_energy_for_scan and len(ocs) < 2 and ocs.filter(lambda x: x.energy >= 100).empty):
            return

        best_mf = None
        best_mf_mineral_contents = 0

        for th in self.townhalls.ready:
            mfs = self.mineral_field.closer_than(15, th)
            if mfs:
                mf = max(mfs, key=lambda x: x.mineral_contents)
                if mf.mineral_contents > best_mf_mineral_contents:
                    best_mf = mf
                    best_mf_mineral_contents = mf.mineral_contents

        if best_mf is not None:
            max_energy_oc = ocs.sorted(lambda x: x.energy).first
            max_energy_oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, best_mf)

    # need to improve the scan timings still
    def use_scan(self, target: Point2) -> bool:
        ocs = self.townhalls(UnitTypeId.ORBITALCOMMAND).ready.filter(lambda x: x.energy >= 50)
        # temporary measure to make sure we don't pointlessly scan twice to the same location
        if ocs.empty or self.time - self.last_scan_time < 10:
            return False

        self.last_scan_time = self.time
        ocs.first(AbilityId.SCANNERSWEEP_SCAN, target)
        return True

    # still need to figure out targeting properly. what target should
    # be chosen and what enemy units are relevant to be considered
    # some of the attack selection is still nonsensical

    async def attack_towards_position(self, units: Units, target: Union[Point2, Unit],
                                      close_enemy_units: Optional[Units] = None,
                                      position_to_hold: Point2 = None):
        if units.empty:
            return

        cloaked_enemies = self.enemy_units.filter(lambda x: not x.can_be_attacked)
        if cloaked_enemies.exists and units.filter(lambda x: x.target_in_range(cloaked_enemies.first, 1)).exists:
            self.use_scan(cloaked_enemies.first.position)

        # don't stim etc because of a random observer flying around
        if close_enemy_units:
            close_enemy_units = close_enemy_units - cloaked_enemies

        # these should be used to determine whether to fight or not
        supply = 0
        for unit in units:
            supply += self.calculate_supply_cost(unit.type_id)

        enemy_supply = 0
        if close_enemy_units:
            for enemy_unit in close_enemy_units:
                enemy_supply += self.calculate_supply_cost(enemy_unit.type_id)

        enemy_units_to_dodge = None
        if close_enemy_units:
            enemy_units_to_dodge = (close_enemy_units(UnitTypeId.BANELING) | close_enemy_units(UnitTypeId.ZEALOT))
            enemy_units_to_dodge = enemy_units_to_dodge.random_group_of(min(enemy_units_to_dodge.amount, 20))

        effects_to_dodge = {}
        for effect in self.state.effects:
            if effect.id == EffectId.PSISTORMPERSISTENT:
                for pos in effect.positions:
                    effects_to_dodge[pos] = effect.radius

        for unit in (units(UnitTypeId.MARINE) | units(UnitTypeId.MARAUDER)):

            if unit.weapon_cooldown > 0 and not unit.is_moving and enemy_units_to_dodge and enemy_units_to_dodge.exists:
                closest_enemy = enemy_units_to_dodge.closest_to(unit)
                if closest_enemy.distance_to(unit) < unit.ground_range + 2:
                    unit.move(unit.position.towards(closest_enemy, -5))
            elif len(effects_to_dodge) > 0 and (unit.weapon_cooldown > 0
                                                or (not close_enemy_units or close_enemy_units.empty or
                                                    close_enemy_units.in_attack_range_of(unit).empty)):
                min_distance = math.inf
                min_position = None
                min_radius = math.inf

                for effect_position, effect_radius in effects_to_dodge.items():
                    dist = unit.distance_to(effect_position)
                    if dist < effect_radius and dist < min_distance:
                        min_position = effect_position
                        min_radius = effect_radius
                        min_distance = dist

                if min_position:
                    unit.move(unit.position.towards(min_position, -5))
                    if unit.type_id == UnitTypeId.MARINE and unit.health_percentage > 0.5 and not unit.has_buff(
                            BuffId.STIMPACK):
                        unit(AbilityId.EFFECT_STIM_MARINE, queue=True)
                    elif unit.type_id == UnitTypeId.MARAUDER and unit.health_percentage > 0.5 and not unit.has_buff(
                            BuffId.STIMPACKMARAUDER):
                        unit(AbilityId.EFFECT_STIM_MARAUDER, queue=True)
            else:
                if position_to_hold and unit.distance_to(target) <= position_to_hold.distance_to(target):
                    unit(AbilityId.MOVE, position_to_hold.towards(target, -2))
                # if there are only air units around, try to run under them.
                elif (close_enemy_units and close_enemy_units.not_flying.empty and close_enemy_units.flying.exists
                        and unit.weapon_cooldown > 0):
                    unit.move(target)
                else:
                    unit.attack(target)
                    if close_enemy_units and close_enemy_units.exists and target.distance_to(unit.position) < unit.ground_range + 2:
                        if unit.type_id == UnitTypeId.MARINE and unit.health_percentage > 0.5 and not unit.has_buff(BuffId.STIMPACK):
                            unit(AbilityId.EFFECT_STIM_MARINE)
                        elif unit.type_id == UnitTypeId.MARAUDER and unit.health_percentage > 0.5 and not unit.has_buff(BuffId.STIMPACKMARAUDER):
                            unit(AbilityId.EFFECT_STIM_MARAUDER)

        for unit in units(UnitTypeId.REAPER):
            closest_enemy = close_enemy_units.closest_to(unit) if close_enemy_units and close_enemy_units.exists else None
            if unit.weapon_cooldown > 0 and not unit.is_moving and closest_enemy:
                if closest_enemy.distance_to(unit) < closest_enemy.ground_range + 4:
                    unit.move(unit.position.towards(closest_enemy, -5))
            elif closest_enemy:
                unit.attack(target)
                if close_enemy_units.amount > 3 and unit.distance_to(closest_enemy) < 10 and await self.can_cast(unit, AbilityId.KD8CHARGE_KD8CHARGE, closest_enemy.position):
                    unit(AbilityId.KD8CHARGE_KD8CHARGE, closest_enemy.position)
            else:
                unit.move(target)

        hellion_threats = None
        if close_enemy_units:
            hellion_threats = (close_enemy_units(UnitTypeId.BANELING) | close_enemy_units(UnitTypeId.ZEALOT)
                               | close_enemy_units(UnitTypeId.ROACH) | close_enemy_units(UnitTypeId.QUEEN)
                               | close_enemy_units(UnitTypeId.ZERGLING) | close_enemy_units(UnitTypeId.HYDRALISK))
            hellion_threats = hellion_threats.random_group_of(min(hellion_threats.amount, 20))

        leader_hellion = units(UnitTypeId.HELLION).furthest_to(self.start_location) if units(UnitTypeId.HELLION).exists else None

        for unit in units(UnitTypeId.HELLION):
            if unit.distance_to(self.start_location) > 40 and self.structures(UnitTypeId.ARMORY).ready:
                unit(AbilityId.MORPH_HELLBAT)
            else:
                if unit.weapon_cooldown > 0 and hellion_threats and hellion_threats.exists:
                    closest_enemy = hellion_threats.closest_to(unit)
                    if closest_enemy.distance_to(unit) < unit.ground_range + 2:
                        unit.move(leader_hellion.position.towards(closest_enemy, -5))
                else:
                    unit.attack(target)

        banelings = None
        if close_enemy_units:
            banelings = close_enemy_units(UnitTypeId.BANELING)

        for unit in units(UnitTypeId.HELLIONTANK):
            if unit.weapon_cooldown > 0 and not unit.is_moving and banelings and banelings.exists:
                closest_enemy = banelings.closest_to(unit)
                if closest_enemy.distance_to(unit) < unit.ground_range + 5:
                    unit.move(unit.position.towards(closest_enemy, -5))
            else:
                unit.attack(target)

        for unit in units(UnitTypeId.CYCLONE):
            if unit.weapon_cooldown > 0 and not unit.is_moving and enemy_units_to_dodge and enemy_units_to_dodge.exists:
                closest_enemy = enemy_units_to_dodge.closest_to(unit)
                if closest_enemy.distance_to(unit) < unit.ground_range + 2:
                    unit.move(unit.position.towards(closest_enemy, -5))
            else:
                unit.attack(target)

        for unit in units(UnitTypeId.SIEGETANK):
            distance_to_target = unit.position.distance_to(target)
            if 5 < distance_to_target < 12:
                unit(AbilityId.SIEGEMODE_SIEGEMODE)
            else:
                unit.attack(target)

        for unit in units(UnitTypeId.SIEGETANKSIEGED):
            distance_to_target = unit.position.distance_to(target)
            if distance_to_target > 16:
                unit(AbilityId.UNSIEGE_UNSIEGE)
                unit.attack(target, queue=True)

        banshee_threats = None
        if close_enemy_units:
            banshee_threats = close_enemy_units.filter(lambda x: x.can_attack_air and x.distance_to(unit) < x.air_range)

        for unit in units(UnitTypeId.BANSHEE):
            detectors_exist = False
            if close_enemy_units:
                detectors_exist = close_enemy_units.filter(lambda x: x.detect_range > 0).filter(lambda x: x.distance_to(unit) < x.detect_range).exists
            if self.already_pending_upgrade(UpgradeId.BANSHEECLOAK) >= 1:
                if (unit.energy > 40 and banshee_threats and banshee_threats.exists
                        and not detectors_exist):
                    unit(AbilityId.BEHAVIOR_CLOAKON_BANSHEE)
            if banshee_threats and banshee_threats.exists and unit.weapon_cooldown > 0 and detectors_exist:
                if unit.weapon_cooldown > 0 and not unit.is_moving:
                    target_position = banshee_threats.closest_to(unit)
                    unit.move(unit.position.towards(target_position, -5))
            else:
                unit.attack(target)

        # find relevant units ravens can use interference matrix on
        # if the units are spellcasters only target those with enough energy
        # for dangerous spells
        interferable_units = None
        armor_missile_targets = None
        if close_enemy_units:
            interferable_units = (close_enemy_units(UnitTypeId.INFESTOR).filter(lambda x: x.energy >= 75)
                                  | close_enemy_units(UnitTypeId.SIEGETANKSIEGED)
                                  | close_enemy_units(UnitTypeId.THOR)
                                  | close_enemy_units(UnitTypeId.BATTLECRUISER)
                                  | close_enemy_units(UnitTypeId.SIEGETANK)
                                  | close_enemy_units(UnitTypeId.BANSHEE)
                                  | close_enemy_units(UnitTypeId.HIGHTEMPLAR).filter(lambda x: x.energy >= 75)
                                  | close_enemy_units(UnitTypeId.COLOSSUS)
                                  | close_enemy_units(UnitTypeId.ARCHON)).filter(lambda x: not x.has_buff(BuffId.RAVENSCRAMBLERMISSILE))

            armor_missile_targets = (close_enemy_units(UnitTypeId.ULTRALISK)
                                     | close_enemy_units(UnitTypeId.ROACH)
                                     | close_enemy_units(UnitTypeId.BROODLORD)
                                     | close_enemy_units(UnitTypeId.HYDRALISK)
                                     | close_enemy_units(UnitTypeId.BANELING)
                                     | close_enemy_units(UnitTypeId.BATTLECRUISER)
                                     | close_enemy_units(UnitTypeId.IMMORTAL)
                                     | close_enemy_units(UnitTypeId.STALKER)).filter(lambda x: not x.has_buff(BuffId.RAVENSHREDDERMISSILEARMORREDUCTION))

        for raven in units(UnitTypeId.RAVEN):
            if interferable_units and interferable_units.exists and raven.energy >= 75:
                # currently this may use the spell twice on consecutive frames to the same target
                raven(AbilityId.EFFECT_INTERFERENCEMATRIX, interferable_units.first)
                interferable_units.remove(interferable_units.first)
            elif armor_missile_targets and close_enemy_units.amount >= 8 and raven.energy >= 75:
                raven(AbilityId.EFFECT_ANTIARMORMISSILE, armor_missile_targets.closest_to(armor_missile_targets.center))
            #elif close_enemy_units and close_enemy_units.exists and raven.energy >= 5:
            #    loc = await self.find_placement(UnitTypeId.AUTOTURRET, target, random_alternative=False)
            #    if loc:
            #        raven(AbilityId.RAVENBUILD_AUTOTURRET, loc))
            elif (units(UnitTypeId.MARINE) | units(UnitTypeId.MARAUDER) | units(UnitTypeId.HELLION) | units(UnitTypeId.HELLIONTANK)).exists:
                if raven.is_idle:
                    raven.attack((units(UnitTypeId.MARINE) | units(UnitTypeId.MARAUDER) | units(UnitTypeId.HELLION) | units(UnitTypeId.HELLIONTANK)).first)

        for med in units(UnitTypeId.MEDIVAC).idle:
            if (units(UnitTypeId.MARINE) | units(UnitTypeId.MARAUDER)).exists:
                med.attack((units(UnitTypeId.MARINE) | units(UnitTypeId.MARAUDER)).first)

        for viking in units(UnitTypeId.VIKINGFIGHTER):
            if (close_enemy_units and (close_enemy_units.flying | close_enemy_units(UnitTypeId.COLOSSUS)).empty
                    and (close_enemy_units.not_flying - close_enemy_units(UnitTypeId.COLOSSUS)).exists):
                viking(AbilityId.MORPH_VIKINGASSAULTMODE)
            viking.attack(target)

        for viking in units(UnitTypeId.VIKINGASSAULT):
            if (close_enemy_units and (close_enemy_units.not_flying - close_enemy_units(UnitTypeId.COLOSSUS)).empty
                    and (close_enemy_units.flying | close_enemy_units(UnitTypeId.COLOSSUS)).exists):
                viking(AbilityId.MORPH_VIKINGFIGHTERMODE)
            viking.attack(target)

        for thor in units(UnitTypeId.THOR):
            thor.attack(target)

    def move_as_group(self, units: Units, leader_tag: int, target: Point2):
        if units.empty:
            return

        leader = units.find_by_tag(leader_tag)
        if leader is None:
            leader = units.closest_to(target)

        for unit in units - units(UnitTypeId.SIEGETANKSIEGED) - units.flying:
            if leader.type_id in {UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED, UnitTypeId.THOR} and unit.tag != leader.tag:
                unit.move(leader)
            elif unit.distance_to(leader) > 8:
                unit.attack(leader.position)
            else:
                unit.attack(target)

        for siege_tank in units(UnitTypeId.SIEGETANKSIEGED):
            siege_tank(AbilityId.UNSIEGE_UNSIEGE)
            siege_tank.attack(target, queue=True)

        for viking in units(UnitTypeId.VIKINGASSAULT):
            viking(AbilityId.MORPH_VIKINGFIGHTERMODE)
            viking.attack(target, queue=True)

        if units.not_flying.exists:
            random_ground_unit = units.not_flying.random
            for flying_unit in units.flying.idle:
                flying_unit.move(random_ground_unit)

    def get_next_strike_target(self, units: Units) -> Point2:
        if self.enemy_expansion_checks[self.enemy_expansion_order[2]] <= self.enemy_expansion_checks[self.enemy_expansion_order[3]]:
            target = self.enemy_expansion_order[2]
        else:
            target = self.enemy_expansion_order[3]

        self.enemy_expansion_checks[target] = self.time

        return target

    def move_strike_teams_to_main_army(self):
        for team in self.strike_teams:
            units = team.select_units(self.units)
            self.new_units.add_units(units)
            team.remove_units(units)

    async def control_units(self):
        if self.units.empty:
            return

        enemy_units = self.enemy_units.filter(lambda x: x.type_id not in {UnitTypeId.SCV,
                                                                          UnitTypeId.DRONE,
                                                                          UnitTypeId.PROBE,
                                                                          UnitTypeId.OVERSEER,
                                                                          UnitTypeId.OVERLORD,
                                                                          UnitTypeId.OBSERVER,
                                                                          UnitTypeId.CHANGELING,
                                                                          UnitTypeId.CHANGELINGMARINE,
                                                                          UnitTypeId.CHANGELINGMARINESHIELD
                                                                          })

        for unit in enemy_units:
            self.enemy_unit_sizes[unit.tag] = self.calculate_supply_cost(unit.type_id)

        if self.structures(UnitTypeId.BUNKER).ready.exists:
            bunkered_tags = self.structures(UnitTypeId.BUNKER).ready.first.passengers_tags
            for passenger in bunkered_tags:
                if passenger in self.new_units:
                    self.new_units.remove(passenger)
                if passenger in self.main_army:
                    self.main_army.remove(passenger)

        resting_point = self.current_army_resting_point

        changelings = (self.enemy_units(UnitTypeId.CHANGELING) | self.enemy_units(UnitTypeId.CHANGELINGMARINE)
                       | self.enemy_units(UnitTypeId.CHANGELINGMARINESHIELD))
        if changelings.exists and self.units.filter(lambda x: x.order_target == changelings.first).empty:
            closest_unit = self.units.closest_to(changelings.first)
            closest_unit.attack(changelings.first)

        new_units = self.new_units.select_units(self.units)
        if self.main_army.mode != ArmyMode.ATTACK:
            self.main_army.add_units(new_units)
            self.new_units.remove_units(new_units)
        else:
            army_units = self.main_army.select_units(self.units)
            if army_units.filter(lambda x: x.can_attack_ground).exists:
                center = army_units.center
                units_to_be_moved = set()
                for unit in new_units:
                    if unit.distance_to(center) < 10:
                        units_to_be_moved.add(unit)
                self.main_army.add_units(units_to_be_moved)
                self.new_units.remove_units(units_to_be_moved)
            else:
                # all army units died but retreat condition wasn't satisfied
                if new_units.exists:
                    await self.attack_towards_position(new_units, self.current_army_resting_point)
                self.main_army.mode = ArmyMode.PASSIVE
                self.tech.delay_attack_timing(self.army_type)
                self.should_be_aggressive = False

        alive_patrollers = self.scouting_units.select_units(self.workers)

        if self.time >= 300 and (self.time - 300) % 120 < 1 and self.time - self.prev_game_seconds >= 1:
            if alive_patrollers.exists:
                for patroller in alive_patrollers.collecting:
                    self.scouting_units.remove_unit(patroller)

            alive_patrollers = self.scouting_units.select_units(self.workers)

            if alive_patrollers.empty and self.workers.collecting.exists:
                new_patroller = self.workers.collecting.random
                self.scouting_units.add_unit(new_patroller)
                if self.time > 500:
                    for exp in self.enemy_expansion_order[3:]:
                        if exp not in self.owned_expansions:
                            new_patroller(AbilityId.MOVE, exp, queue=True)
                else:
                    new_patroller(AbilityId.MOVE, self.enemy_expansion_order[3])
                    new_patroller(AbilityId.MOVE, self.enemy_expansion_order[3].towards(self.enemy_start_locations[0], -7), queue=True)
                    new_patroller(AbilityId.PATROL, self.enemy_expansion_order[3].towards(self.enemy_start_locations[0], -8), queue=True)

        elif alive_patrollers.idle.exists:
            relevant_expansions = self.enemy_expansion_order[2:][::-1]
            for patroller in alive_patrollers.idle:
                for exp in relevant_expansions:
                    if exp not in self.owned_expansions:
                        patroller(AbilityId.MOVE, exp, queue=True)

        # reaper team
        strike_units = self.strike_teams[0].select_units(self.units)
        if strike_units.exists:
            if self.strike_teams[0].mode == ArmyMode.ATTACK:
                close_enemies = (self.enemy_units.closer_than(15, strike_units.furthest_to(self.start_location))
                                 - self.enemy_units(UnitTypeId.OVERLORD)
                                 - self.all_enemy_units(UnitTypeId.EGG)
                                 - self.all_enemy_units(UnitTypeId.LARVA))
                close_structures = self.enemy_structures.closer_than(20, self.current_strike_target)
                reaper_threats = (close_enemies(UnitTypeId.STALKER) | close_enemies(UnitTypeId.ADEPT)
                                  | close_enemies(UnitTypeId.QUEEN) | close_enemies(UnitTypeId.ROACH)
                                  | close_enemies(UnitTypeId.MARINE) | close_enemies(UnitTypeId.MARAUDER)
                                  | close_enemies(UnitTypeId.WIDOWMINE) | close_enemies(UnitTypeId.WIDOWMINEBURROWED))
                reaper_targets = close_enemies - reaper_threats
                if (self.get_army_supply(reaper_threats) >= self.get_army_supply(strike_units)
                        or strike_units.filter(lambda x: x.health_percentage < 0.4)):
                    for unit in strike_units:
                        unit.move(self.game_info.map_center)
                else:
                    if reaper_targets.exists:
                        lowest_hp_target = reaper_targets.sorted(lambda x: x.health + x.shield).first
                        await self.attack_towards_position(strike_units, lowest_hp_target, close_enemies)
                    elif strike_units.center.distance_to(self.current_strike_target) < 5 and close_structures.exists:
                        await self.attack_towards_position(strike_units, close_structures.closest_to(strike_units.center).position, close_enemies)
                    else:
                        await self.attack_towards_position(strike_units, self.current_strike_target, close_enemies)
            elif self.strike_teams[0].mode == ArmyMode.PASSIVE:
                reapers_to_move = set()
                for reaper in strike_units:
                    if reaper.distance_to(self.early_unit_rally_point) > 5:
                        reaper.move(self.early_unit_rally_point)
                    else:
                        reapers_to_move.add(reaper)

                for movable in reapers_to_move:
                    self.main_army.add_unit(movable)
                    self.strike_teams[0].remove_unit(movable)

        # hellion team
        hellions = self.strike_teams[1].select_units(self.units)
        if hellions.exists:
            close_enemies = self.enemy_units.closer_than(15, hellions.furthest_to(self.start_location))
            close_structures = self.enemy_structures.closer_than(20, self.current_strike_target)
            hellion_threats = (close_enemies(UnitTypeId.QUEEN) | close_enemies(UnitTypeId.ROACH))
            if self.get_army_supply(hellion_threats) >= self.get_army_supply(hellions):
                for unit in hellions:
                    unit.move(self.game_info.map_center)
            else:
                if hellions.center.distance_to(self.current_strike_target) < 5 and close_structures.exists:
                    await self.attack_towards_position(hellions, close_structures.closest_to(hellions.center).position, close_enemies)
                else:
                    await self.attack_towards_position(hellions, self.current_strike_target, close_enemies)

        # banshee team
        banshees = self.strike_teams[2].select_units(self.units)
        if banshees.exists:
            close_enemies = (self.enemy_units.closer_than(15, banshees.furthest_to(self.start_location))
                             | self.all_enemy_units(UnitTypeId.SPORECRAWLER))
            close_structures = self.enemy_structures.closer_than(20, self.current_strike_target)
            banshee_threats = (close_enemies(UnitTypeId.QUEEN) | close_enemies(UnitTypeId.SPORECRAWLER)
                               | close_enemies(UnitTypeId.HYDRALISK))
            if self.get_army_supply(banshee_threats) >= self.get_army_supply(banshees):
                for unit in banshees:
                    unit.move(self.game_info.map_center)
            else:
                if banshees.center.distance_to(self.current_strike_target) < 5 and close_structures.exists:
                    await self.attack_towards_position(banshees,
                                                       close_structures.closest_to(banshees.center).position,
                                                       close_enemies)
                else:
                    await self.attack_towards_position(banshees, self.current_strike_target, close_enemies)

        clearing_units = self.area_clearing_units.select_units(self.units)
        if clearing_units.exists:
            if clearing_units.closer_than(10, self.area_clearing_units.target_expansion).amount == clearing_units.amount:
                if clearing_units(UnitTypeId.RAVEN).empty:
                    ocs = self.townhalls(UnitTypeId.ORBITALCOMMAND).filter(lambda x: x.energy >= 50)
                    if ocs.exists:
                        self.use_scan(self.area_clearing_units.target_expansion)
                        self.new_units.add_units(clearing_units.idle)
                        self.area_clearing_units.remove_units(clearing_units.idle)
                else:
                    self.new_units.add_units(clearing_units.idle)
                    self.area_clearing_units.remove_units(clearing_units.idle)
            else:
                await self.attack_towards_position(clearing_units, self.area_clearing_units.target_expansion)

        army_units = self.main_army.select_units(self.units)
        if self.main_army.mode == ArmyMode.PASSIVE:
            if self.should_attack():
                if self.poke_scan_done:

                    if self.structures(UnitTypeId.BUNKER).ready.exists:
                        bunker = self.structures(UnitTypeId.BUNKER).ready.first
                        bunkered_tags = self.structures(UnitTypeId.BUNKER).ready.first.passengers_tags
                        for passenger in bunkered_tags:
                            self.new_units.add(passenger)
                        bunker(AbilityId.UNLOADALL_BUNKER)

                    self.main_army.mode = ArmyMode.ATTACK

                    if self.supply_used > self.tech.builds[self.army_type]['attack_timing']:
                        self.main_army.target_expansion = self.enemy_start_locations[0]
                    else:
                        target_choice1 = self.enemy_expansion_order[2]
                        choice1_revealed = False
                        target_choice2 = self.enemy_expansion_order[3]
                        choice2_revealed = False
                        for known_exp in self.known_enemy_expansions:
                            if known_exp.distance_to(target_choice1) < 10:
                                choice1_revealed = True
                            elif known_exp.distance_to(target_choice2) < 10:
                                choice2_revealed = True

                        self.main_army.target_expansion = target_choice1
                        if not choice1_revealed and choice2_revealed:
                            self.main_army.target_expansion = target_choice2
                else:
                    scan_target_choice1 = self.enemy_expansion_order[2]
                    choice1_revealed = False
                    scan_target_choice2 = self.enemy_expansion_order[3]
                    choice2_revealed = False
                    for known_exp in self.known_enemy_expansions:
                        if known_exp.distance_to(scan_target_choice1) < 10:
                            choice1_revealed = True
                        elif known_exp.distance_to(scan_target_choice2) < 10:
                            choice2_revealed = True

                    if not choice1_revealed:
                        self.use_scan(scan_target_choice1.towards(self.game_info.map_center, 3))
                    elif not choice2_revealed:
                        self.use_scan(scan_target_choice2.towards(self.game_info.map_center, 3))
                    self.poke_scan_done = True
            if army_units.exists:
                if self.structures(UnitTypeId.BUNKER).ready.exists:
                    bunker = self.structures(UnitTypeId.BUNKER).ready.first
                    bunkered_units = bunker.passengers
                    bunkered_marines = len(bunkered_units) > 0

                    enemy_harass_units = self.enemy_units.exclude_type([UnitTypeId.PROBE,
                                                                        UnitTypeId.SCV,
                                                                        UnitTypeId.DRONE,
                                                                        UnitTypeId.OVERLORD,
                                                                        UnitTypeId.OBSERVER])
                    enemy_behind_bunker = False

                    if enemy_harass_units.exists:
                        closest_enemy = enemy_harass_units.closest_to(bunker)
                        out_of_bunker_range = closest_enemy.distance_to(bunker) > 6

                        if out_of_bunker_range:
                            bunker_to_mid = self.game_info.map_center - bunker.position
                            bunker_to_enemy = closest_enemy.position - bunker.position

                            main_base_region = self.map_data.where(self.start_location)
                            natural_region = self.map_data.where(self.expansion_order[1])

                            if (main_base_region.is_inside_point(closest_enemy.position)
                                    or (natural_region.is_inside_point(closest_enemy.position)
                                        and bunker_to_mid.x * bunker_to_enemy.x + bunker_to_mid.y * bunker_to_enemy.y < 0)):
                                enemy_behind_bunker = True
                    # if enemy units are in main base or have went past the bunker to natural mineral line
                    if (bunkered_marines
                            and enemy_behind_bunker):
                        bunkered_tags = bunker.passengers_tags
                        for passenger in bunkered_tags:
                            self.main_army.add(passenger)
                        bunker(AbilityId.UNLOADALL_BUNKER)
                    elif not enemy_behind_bunker:
                        marines_to_bunker = (army_units(UnitTypeId.MARINE).filter(lambda x: x.distance_to(bunker) < 10)
                                             .take(bunker.cargo_max - bunker.cargo_used))
                        for marine in marines_to_bunker:
                            marine(AbilityId.SMART, bunker)

                min_distance_to_enemy = math.inf
                if self.all_enemy_units.exists:
                    closest_enemy = self.all_enemy_units.first
                    for th in self.townhalls:
                        cur_dist = self.all_enemy_units.closest_distance_to(th.position)
                        if cur_dist < min_distance_to_enemy:
                            min_distance_to_enemy = cur_dist
                            closest_enemy = self.all_enemy_units.closest_to(th.position)

                if min_distance_to_enemy < 20:
                    if army_units.exists:
                        marine_count = army_units(UnitTypeId.MARINE).amount
                        if self.structures(UnitTypeId.BUNKER).ready.exists:
                            marine_count += self.structures(UnitTypeId.BUNKER).ready.first.cargo_used
                        close_enemies = self.all_enemy_units.closer_than(20, army_units.center)

                        position_to_hold = None
                        if (self.townhalls.amount <= 2 and self.structures(UnitTypeId.BUNKER).ready.exists
                                and self.structures(UnitTypeId.BUNKER).ready.first.cargo_used > 0):
                            # compare to bunker position and send back if too far

                            position_to_hold = self.structures(UnitTypeId.BUNKER).first.position

                        elif self.townhalls.amount >= 3:
                            # compare to tank position and send back if too far
                            if self.units(UnitTypeId.SIEGETANKSIEGED).exists and close_enemies.exists:
                                closest_tank = self.units(UnitTypeId.SIEGETANKSIEGED).in_closest_distance_to_group(close_enemies)
                                tank_to_center = self.game_info.map_center - closest_tank.position
                                tank_to_enemy = closest_enemy.position - closest_tank.position
                                if tank_to_center.x * tank_to_enemy.x + tank_to_center.y + tank_to_enemy.y > 0:
                                    if close_enemies.closest_distance_to(closest_tank) <= 10:
                                        position_to_hold = None
                                    else:
                                        position_to_hold = closest_tank.position

                        army_to_center = self.game_info.map_center - army_units.center
                        army_to_enemy = closest_enemy.position - army_units.center
                        if army_to_center.x * army_to_enemy.x + army_to_center.y * army_to_enemy.y >= 0:
                            await self.attack_towards_position(army_units, closest_enemy.position, close_enemies,
                                                               position_to_hold)
                        else:
                            await self.attack_towards_position(army_units, closest_enemy.position, close_enemies)
                else:
                    resting_point = self.current_army_resting_point
                    for unit in army_units:
                        if unit.position.distance_to(resting_point) > 8:
                            if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                                unit(AbilityId.UNSIEGE_UNSIEGE)
                                unit.attack(resting_point, queue=True)
                            else:
                                unit.attack(resting_point)
                        else:
                            if unit.type_id == UnitTypeId.SIEGETANK:
                                unit(AbilityId.SIEGEMODE_SIEGEMODE, queue=True)
            else:
                ramp_distance = self.main_base_ramp.top_center.distance_to(self.start_location)
                units_close = self.all_enemy_units.filter(lambda x: not x.is_flying).closer_than(ramp_distance, self.start_location)

                if units_close.amount == 1 and not self.cheese_defense_scv:
                    ws = self.workers.gathering
                    if ws:
                        self.cheese_defense_scv = ws.random
                        self.cheese_defense_scv.attack(self.all_enemy_units.first)
                elif units_close.amount > 1:
                    attacking_workers = self.workers.filter(lambda x: x.is_attacking)
                    if attacking_workers.amount <= 2*units_close.amount:
                        defenders = self.workers.collecting.take(2*units_close.amount - attacking_workers.amount)

                        for unit in defenders:
                            target = units_close.closest_to(unit.position)
                            unit.attack(target.position)

                    for unit in attacking_workers:
                        target = units_close.closest_to(unit.position)
                        unit.attack(target.position)

        elif self.main_army.mode == ArmyMode.ATTACK:
            self.move_strike_teams_to_main_army()
            if army_units.exists:

                if self.main_army.target_expansion:
                    target = self.main_army.target_expansion
                else:
                    target = self.enemy_start_locations[0]

                if self.all_enemy_units.empty and target.distance_to(army_units.center) < 10 and target == self.enemy_start_locations[0]:
                    self.main_army.mode = ArmyMode.SCOUT
                    self.assign_army_scouting_roles()
                else:
                    if (self.enemy_structures.exists and self.enemy_structures.exclude_type([UnitTypeId.SUPPLYDEPOT,
                                                                                             UnitTypeId.SUPPLYDEPOTLOWERED])
                            or self.get_army_supply(self.enemy_units) >= 6
                            or self.supply_used > self.tech.builds[self.army_type]['attack_timing']):
                        target = self.enemy_start_locations[0]
                        self.main_army.target_expansion = self.enemy_start_locations[0]

                    center = army_units.center
                    close_enemies = enemy_units.closer_than(35, center)
                    relevant_enemy_units = self.all_enemy_units.filter(lambda x: x.type_id not in {UnitTypeId.OVERSEER,
                                                                                                   UnitTypeId.OVERLORD,
                                                                                                   UnitTypeId.OBSERVER,
                                                                                                   UnitTypeId.CHANGELING,
                                                                                                   UnitTypeId.CHANGELINGMARINE,
                                                                                                   UnitTypeId.CHANGELINGMARINESHIELD})

                    if relevant_enemy_units.exists:
                        target = relevant_enemy_units.closest_to(center).position

                    if target.distance_to(center) < 25:
                        if self.get_terrain_height(target) > self.get_terrain_height(center):
                            self.use_scan(target)

                        await self.attack_towards_position(army_units, target, close_enemies)
                        self.main_army.leader_unit = None
                    else:
                        if self.main_army.leader_unit is None or army_units.find_by_tag(self.main_army.leader_unit) is None:
                            tanks_and_thors = army_units(UnitTypeId.THOR) | army_units(UnitTypeId.SIEGETANK) | army_units(UnitTypeId.SIEGETANKSIEGED)
                            if tanks_and_thors.exists:
                                self.main_army.leader_unit = tanks_and_thors.closest_to(target).tag
                            else:
                                self.main_army.leader_unit = army_units.furthest_to(target).tag

                        self.move_as_group(army_units, self.main_army.leader_unit, target)

                if new_units.exists:
                    await self.attack_towards_position(new_units, army_units.center)
        elif self.main_army.mode == ArmyMode.RETREAT:
            # temporary fix to make sure the army doesn't get stuck in retreat mode for ages
            if self.time - self.army_retreat_start_time > RETREAT_TIME_THRESHOLD:
                self.main_army.mode = ArmyMode.PASSIVE
            else:
                close_unit_count = 0
                for unit in army_units:
                    if unit.position.distance_to(resting_point) > 20:
                        if unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                            unit(AbilityId.UNSIEGE_UNSIEGE)
                            unit.move(resting_point, queue=True)
                        else:
                            unit.move(resting_point)
                    else:
                        close_unit_count += 1
                if close_unit_count > 0.8 * army_units.amount:
                    self.main_army.mode = ArmyMode.PASSIVE

        elif self.main_army.mode == ArmyMode.SCOUT:
            scouters = self.scouting_units.select_units(self.units - self.workers)
            if self.enemy_structures.exists or scouters.not_flying.empty:
                self.main_army.add_units(scouters)
                self.scouting_units.remove_units(scouters)
                self.main_army.mode = ArmyMode.ATTACK

            if new_units.exists:
                await self.attack_towards_position(new_units, army_units.center)

    def should_attack(self) -> bool:
        if self.supply_used > self.tech.builds[self.army_type]['attack_timing']:
            return True
        elif self.should_be_aggressive and 'poke_timing' in self.tech.builds[self.army_type]:
            for condition in self.tech.builds[self.army_type]['poke_timing']:
                if isinstance(condition, UnitTypeId):
                    if self.all_own_units(condition).amount < self.tech.builds[self.army_type]['poke_timing'][condition]:
                        return False
                elif isinstance(condition, UpgradeId):
                    if self.already_pending(condition) < self.tech.builds[self.army_type]['poke_timing'][condition]:
                        return False

            return True

        return False

    def assign_units_to_clear_base(self, expansion: Point2):
        existing_units = self.area_clearing_units.select_units(self.units)
        if existing_units.exists:
            existing_units.target_expansion = expansion
            return

        units = self.new_units.select_units(self.units)
        if units.empty:
            units = self.main_army.select_units(self.units)

        if units.exists:
            close_ravens = units(UnitTypeId.RAVEN).closer_than(40, expansion)
            if close_ravens.exists:
                closest_raven = close_ravens.closest_to(expansion)

                self.area_clearing_units.add_unit(closest_raven)
                if closest_raven.tag in self.main_army:
                    self.main_army.remove_unit(closest_raven)
                elif closest_raven.tag in self.new_units:
                    self.new_units.remove_unit(closest_raven)

            ground_army_units = (self.units - self.workers).filter(lambda x: x.can_attack_ground
                                                                   and x.type_id not in {UnitTypeId.SIEGETANK,
                                                                                         UnitTypeId.SIEGETANKSIEGED})
            clearing_units = ground_army_units.closest_n_units(expansion, 5)

            self.area_clearing_units.add_units(clearing_units)
            self.area_clearing_units.target_expansion = expansion

            for unit in clearing_units:
                if unit.tag in self.main_army:
                    self.main_army.remove_unit(unit)
                elif unit.tag in self.new_units:
                    self.new_units.remove_unit(unit)

    def assign_army_scouting_roles(self):
        army_units = self.main_army.select_units(self.units)
        if army_units.exists:
            ground_units = (army_units(UnitTypeId.MARINE) | army_units(UnitTypeId.MARAUDER)
                            | army_units(UnitTypeId.HELLION) | army_units(UnitTypeId.HELLIONTANK)
                            | army_units(UnitTypeId.CYCLONE))
            air_units = (army_units(UnitTypeId.VIKINGFIGHTER) | army_units(UnitTypeId.MEDIVAC))

            if ground_units.exists:
                scouting_units = ground_units.take(4)
                closest_base = min(self.enemy_expansion_order, key=lambda x: x.distance_to(scouting_units[0]))
                closest_base_index = self.enemy_expansion_order.index(closest_base)
                direction = 1
                for unit in scouting_units:
                    unit.attack(self.enemy_expansion_order[closest_base_index])
                    i = (closest_base_index + direction) % len(self.enemy_expansion_order)

                    while i != closest_base_index:
                        unit.attack(self.enemy_expansion_order[i], queue=True)
                        i = (i + direction) % len(self.enemy_expansion_order)

                    unit.attack(self.enemy_expansion_order[closest_base_index], queue=True)
                    direction *= -1

                self.main_army.remove_units(scouting_units)
                self.scouting_units.add_units(scouting_units)

            if air_units.exists:
                scouting_units = air_units.take(4)

                area = self.game_info.playable_area
                corners = [
                    Point2((area.x + 1, area.y + 1)),
                    Point2((area.x + 1, area.y + area.height - 1)),
                    Point2((area.x + area.width - 1, area.y + area.height - 1)),
                    Point2((area.x + area.width - 1, area.y + 1)),
                ]
                closest_corner = min(corners, key=lambda x: x.distance_to(scouting_units[0]))
                closest_corner_index = corners.index(closest_corner)
                direction = 1
                for unit in scouting_units:

                    unit.attack(corners[closest_corner_index])
                    i = (closest_corner_index + direction) % len(corners)

                    while i != closest_corner_index:
                        unit.attack(corners[i], queue=True)
                        i = (i + direction) % len(corners)

                    unit.attack(corners[closest_corner_index], queue=True)
                    direction *= -1

                self.main_army.remove_units(scouting_units)
                self.scouting_units.add_units(scouting_units)

    def count_buildings(self, unit_type: UnitTypeId) -> int:
        ready = self.structures(unit_type).ready.amount
        if unit_type == UnitTypeId.BARRACKS:
            ready += self.structures(UnitTypeId.BARRACKSFLYING).ready.amount
        elif unit_type == UnitTypeId.FACTORY:
            ready += self.structures(UnitTypeId.FACTORYFLYING).ready.amount
        elif unit_type == UnitTypeId.STARPORT:
            ready += self.structures(UnitTypeId.STARPORTFLYING).ready.amount
        elif unit_type == UnitTypeId.ORBITALCOMMAND:
            ready += self.structures(UnitTypeId.ORBITALCOMMANDFLYING).ready.amount
        elif unit_type == UnitTypeId.COMMANDCENTER:
            ready += self.structures(UnitTypeId.COMMANDCENTERFLYING).ready.amount

        return ready + self.already_pending(unit_type)

    async def control_addon_production(self):
        if self.tech.builds[self.army_type]['add_ons'][UnitTypeId.BARRACKS] and self.tech.should_build_addon(self.army_type, UnitTypeId.BARRACKS):
            if (self.structures(UnitTypeId.BARRACKS).ready.idle.filter(lambda x: x.add_on_tag == 0).exists
                    and self.tech.should_build_techlab(self.army_type, UnitTypeId.BARRACKS,
                                                       self.structures(UnitTypeId.BARRACKS).ready.amount,
                                                       self.count_buildings(UnitTypeId.BARRACKSTECHLAB))
                    and self.can_afford(UnitTypeId.BARRACKSTECHLAB)):
                rax = self.structures(UnitTypeId.BARRACKS).ready.idle.filter(lambda x: x.add_on_tag == 0).random
                await self.build_addon(rax, UnitTypeId.BARRACKSTECHLAB)
            elif (self.structures(UnitTypeId.BARRACKS).ready.idle.filter(lambda x: x.add_on_tag == 0).exists
                  and self.can_afford(UnitTypeId.BARRACKSREACTOR) and self.structures(UnitTypeId.STARPORT).exists):
                rax = self.structures(UnitTypeId.BARRACKS).ready.idle.filter(lambda x: x.add_on_tag == 0).random
                await self.build_addon(rax, UnitTypeId.BARRACKSREACTOR)

        if self.tech.builds[self.army_type]['add_ons'][UnitTypeId.FACTORY] and self.tech.should_build_addon(self.army_type, UnitTypeId.FACTORY):
            if (self.structures(UnitTypeId.FACTORY).ready.idle.filter(lambda x: x.add_on_tag == 0).exists
                    and self.tech.should_build_techlab(self.army_type, UnitTypeId.FACTORY,
                                                       self.structures(UnitTypeId.FACTORY).ready.amount,
                                                       self.count_buildings(UnitTypeId.FACTORYTECHLAB))
                    and self.can_afford(UnitTypeId.FACTORYTECHLAB)
                    and (self.structures(UnitTypeId.STARPORT).exists or self.enemy_race == Race.Protoss)):
                fac = self.structures(UnitTypeId.FACTORY).ready.idle.filter(lambda x: x.add_on_tag == 0).random
                await self.build_addon(fac, UnitTypeId.FACTORYTECHLAB)


        if self.tech.builds[self.army_type]['add_ons'][UnitTypeId.STARPORT] and self.tech.should_build_addon(self.army_type, UnitTypeId.STARPORT):
            if (self.structures(UnitTypeId.STARPORT).ready.idle.filter(lambda x: x.add_on_tag == 0).exists
                    and self.tech.should_build_techlab(self.army_type, UnitTypeId.STARPORT,
                                                       self.structures(UnitTypeId.STARPORT).ready.amount,
                                                       self.count_buildings(UnitTypeId.STARPORTTECHLAB))
                    and self.can_afford(UnitTypeId.STARPORTTECHLAB)):
                sp = self.structures(UnitTypeId.STARPORT).ready.idle.filter(lambda x: x.add_on_tag == 0).random
                await self.build_addon(sp, UnitTypeId.STARPORTTECHLAB)
            elif (self.structures(UnitTypeId.STARPORT).ready.idle.filter(lambda x: x.add_on_tag == 0).exists
                    and self.can_afford(UnitTypeId.STARPORTREACTOR)):
                sp = self.structures(UnitTypeId.STARPORT).ready.idle.filter(lambda x: x.add_on_tag == 0).random
                await self.build_addon(sp, UnitTypeId.STARPORTREACTOR)

    async def control_upgrade_production(self):
        if self.tech.builds[self.army_type]['upgrades'][UnitTypeId.ENGINEERINGBAY]:
            if self.structures(UnitTypeId.ENGINEERINGBAY).ready.idle.exists:
                bay = self.structures(UnitTypeId.ENGINEERINGBAY).ready.idle.first
                if (self.already_pending(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1) <= 0
                        and self.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)):
                    bay.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)
                elif (self.already_pending(UpgradeId.TERRANINFANTRYARMORSLEVEL1) <= 0
                      and self.can_afford(UpgradeId.TERRANINFANTRYARMORSLEVEL1)):
                    bay.research(UpgradeId.TERRANINFANTRYARMORSLEVEL1)
                elif (self.already_pending(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1) >= 1
                      and self.already_pending(UpgradeId.TERRANINFANTRYWEAPONSLEVEL2) <= 0
                      and self.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL2)):
                    bay.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL2)
                elif (self.already_pending(UpgradeId.TERRANINFANTRYARMORSLEVEL1) >= 1
                      and self.already_pending(UpgradeId.TERRANINFANTRYARMORSLEVEL2) <= 0
                      and self.can_afford(UpgradeId.TERRANINFANTRYARMORSLEVEL2)):
                    bay.research(UpgradeId.TERRANINFANTRYARMORSLEVEL2)
                elif (self.already_pending(UpgradeId.TERRANINFANTRYWEAPONSLEVEL2) >= 1
                      and self.already_pending(UpgradeId.TERRANINFANTRYWEAPONSLEVEL3) <= 0
                      and self.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL3)):
                    bay.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL3)
                elif (self.already_pending(UpgradeId.TERRANINFANTRYARMORSLEVEL2) >= 1
                      and self.already_pending(UpgradeId.TERRANINFANTRYARMORSLEVEL3) <= 0
                      and self.can_afford(UpgradeId.TERRANINFANTRYARMORSLEVEL3)):
                    bay.research(UpgradeId.TERRANINFANTRYARMORSLEVEL3)

        if self.tech.builds[self.army_type]['upgrades'][UnitTypeId.ARMORY]:

            if self.structures(UnitTypeId.ARMORY).ready.idle.exists:
                armory = self.structures(UnitTypeId.ARMORY).ready.idle.first
                if self.already_pending(UpgradeId.TERRANVEHICLEWEAPONSLEVEL1) <= 0 and self.can_afford(
                        UpgradeId.TERRANVEHICLEWEAPONSLEVEL1):
                    armory.research(UpgradeId.TERRANVEHICLEWEAPONSLEVEL1)
                elif (self.already_pending(UpgradeId.TERRANVEHICLEWEAPONSLEVEL1) >= 1
                      and self.already_pending(UpgradeId.TERRANVEHICLEWEAPONSLEVEL2) <= 0
                      and self.can_afford(UpgradeId.TERRANVEHICLEWEAPONSLEVEL2)):
                    armory.research(UpgradeId.TERRANVEHICLEWEAPONSLEVEL2)
                elif (self.already_pending(UpgradeId.TERRANVEHICLEWEAPONSLEVEL2) >= 1
                      and self.already_pending(UpgradeId.TERRANVEHICLEWEAPONSLEVEL3) <= 0
                      and self.can_afford(UpgradeId.TERRANVEHICLEWEAPONSLEVEL3)):
                    armory.research(UpgradeId.TERRANVEHICLEWEAPONSLEVEL3)
                elif (self.already_pending(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1) <= 0
                      and self.can_afford(UpgradeId.TERRANVEHICLEWEAPONSLEVEL1)):
                    armory(AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL1)
                elif (self.already_pending(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1) >= 1
                      and self.already_pending(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL2) <= 0
                      and self.can_afford(UpgradeId.TERRANVEHICLEWEAPONSLEVEL2)):
                    armory(AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL2)
                elif (self.already_pending(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL2) >= 1
                      and self.already_pending(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL3) <= 0
                      and self.can_afford(UpgradeId.TERRANVEHICLEWEAPONSLEVEL3)):
                    armory(AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL3)
                elif (self.already_pending(UpgradeId.TERRANVEHICLEWEAPONSLEVEL3) > 0
                      and self.already_pending(UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL3) > 0
                      and self.can_afford(UpgradeId.TERRANSHIPWEAPONSLEVEL1)):
                    armory.research(UpgradeId.TERRANSHIPWEAPONSLEVEL1)
                elif (self.already_pending(UpgradeId.TERRANSHIPWEAPONSLEVEL1) >= 1
                      and self.already_pending(UpgradeId.TERRANSHIPWEAPONSLEVEL2) <= 0
                      and self.can_afford(UpgradeId.TERRANSHIPWEAPONSLEVEL2)):
                    armory.research(UpgradeId.TERRANSHIPWEAPONSLEVEL2)
                elif (self.already_pending(UpgradeId.TERRANSHIPWEAPONSLEVEL2) >= 1
                      and self.already_pending(UpgradeId.TERRANSHIPWEAPONSLEVEL3) <= 0
                      and self.can_afford(UpgradeId.TERRANSHIPWEAPONSLEVEL3)):
                    armory.research(UpgradeId.TERRANSHIPWEAPONSLEVEL3)

        if self.tech.builds[self.army_type]['upgrades'][UnitTypeId.BARRACKSTECHLAB] and self.structures(UnitTypeId.BARRACKSTECHLAB).ready.idle.exists:
            if (self.already_pending(UpgradeId.SHIELDWALL) <= 0
                    and self.can_afford(UpgradeId.SHIELDWALL)):
                self.structures(UnitTypeId.BARRACKSTECHLAB).idle.first.research(UpgradeId.SHIELDWALL)
            elif (self.already_pending(UpgradeId.STIMPACK) <= 0
                  and self.already_pending(UpgradeId.SHIELDWALL) > 0
                  and self.can_afford(UpgradeId.STIMPACK)):
                self.structures(UnitTypeId.BARRACKSTECHLAB).idle.first.research(UpgradeId.STIMPACK)
            elif (self.already_pending(UpgradeId.PUNISHERGRENADES) <= 0
                    and self.already_pending(UpgradeId.STIMPACK) > 0
                    and self.can_afford(UpgradeId.PUNISHERGRENADES)):
                self.structures(UnitTypeId.BARRACKSTECHLAB).idle.first.research(UpgradeId.PUNISHERGRENADES)

        if self.tech.builds[self.army_type]['upgrades'][UnitTypeId.FACTORYTECHLAB] and self.structures(UnitTypeId.FACTORYTECHLAB).ready.idle.exists:
            tl = self.structures(UnitTypeId.FACTORYTECHLAB).ready.idle.random
            if self.already_pending(UpgradeId.HIGHCAPACITYBARRELS) <= 0 and self.can_afford(UpgradeId.HIGHCAPACITYBARRELS):
                tl.research(UpgradeId.HIGHCAPACITYBARRELS)
            elif (self.already_pending(UpgradeId.HIGHCAPACITYBARRELS) > 0
                  and self.already_pending(UpgradeId.CYCLONELOCKONDAMAGEUPGRADE) <= 0
                  and self.can_afford(UpgradeId.CYCLONELOCKONDAMAGEUPGRADE)):
                tl.research(UpgradeId.CYCLONELOCKONDAMAGEUPGRADE)
            elif (self.already_pending(UpgradeId.CYCLONELOCKONDAMAGEUPGRADE) > 0
                  and self.already_pending(UpgradeId.SMARTSERVOS) <= 0
                  and self.can_afford(UpgradeId.SMARTSERVOS)):
                tl.research(UpgradeId.SMARTSERVOS)

        if self.tech.builds[self.army_type]['upgrades'][UnitTypeId.STARPORTTECHLAB] and self.structures(UnitTypeId.STARPORTTECHLAB).ready.idle.exists:
            tl = self.structures(UnitTypeId.STARPORTTECHLAB).ready.idle.random
            if self.already_pending(UpgradeId.BANSHEECLOAK) <= 0 and self.can_afford(UpgradeId.BANSHEECLOAK):
                tl.research(UpgradeId.BANSHEECLOAK)

    async def control_tech_production(self):

        if (self.building_requirements_satisfied(UnitTypeId.ORBITALCOMMAND) and self.can_afford(UnitTypeId.ORBITALCOMMAND)
                and (self.structures(UnitTypeId.ORBITALCOMMAND) | self.structures(UnitTypeId.ORBITALCOMMANDFLYING)).amount < 3
                and self.structures(UnitTypeId.COMMANDCENTER).idle.exists):
            idle_cc = self.structures(UnitTypeId.COMMANDCENTER).idle.first
            idle_cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)
        elif (self.building_requirements_satisfied(UnitTypeId.PLANETARYFORTRESS) and self.can_afford(UnitTypeId.PLANETARYFORTRESS)
                and self.townhalls.amount >= 4
                and self.structures(UnitTypeId.COMMANDCENTER).idle.exists):
            idle_cc = self.structures(UnitTypeId.COMMANDCENTER).idle.first
            idle_cc(AbilityId.UPGRADETOPLANETARYFORTRESS_PLANETARYFORTRESS)

        if (self.iterations_used % 10 == 0 and self.time >= self.turret_time
                and self.building_requirements_satisfied(UnitTypeId.MISSILETURRET)
                and self.can_afford(UnitTypeId.MISSILETURRET)
                and self.count_buildings(UnitTypeId.MISSILETURRET) < self.townhalls.amount):
            for th in self.townhalls:
                if self.structures(UnitTypeId.MISSILETURRET).empty or self.structures(UnitTypeId.MISSILETURRET).closer_than(10, th).empty:
                    await self.build_near_base(UnitTypeId.MISSILETURRET, th.position)

        if (self.building_requirements_satisfied(UnitTypeId.BUNKER)
                and self.can_afford(UnitTypeId.BUNKER)
                and self.townhalls.amount >= 2
                and self.count_buildings(UnitTypeId.BUNKER) < 1):
            await self.build_bunker()

        await self.control_upgrade_production()

        if (self.building_requirements_satisfied(UnitTypeId.STARPORT)
                and self.tech.should_build(self.army_type, UnitTypeId.STARPORT,
                                           self.count_buildings(UnitTypeId.STARPORT), self.townhalls.amount)
                and self.can_afford(UnitTypeId.STARPORT)):
            await self.build_near_base(UnitTypeId.STARPORT)

        if (self.building_requirements_satisfied(UnitTypeId.FACTORY)
                and self.tech.should_build(self.army_type, UnitTypeId.FACTORY,
                                           self.count_buildings(UnitTypeId.FACTORY), self.townhalls.amount)
                and self.can_afford(UnitTypeId.FACTORY)):
            await self.build_near_base(UnitTypeId.FACTORY)

        if (self.building_requirements_satisfied(UnitTypeId.BARRACKS)
                and self.tech.should_build(self.army_type, UnitTypeId.BARRACKS,
                                           self.count_buildings(UnitTypeId.BARRACKS), self.townhalls.amount)
                and self.can_afford(UnitTypeId.BARRACKS)):
            ws = self.workers.gathering
            if ws and self.townhalls.exists:
                # build the first rax at the main base ramp
                w = ws.furthest_to(ws.center)
                if self.structures(UnitTypeId.BARRACKS).empty and self.already_pending(UnitTypeId.BARRACKS) <= 0:
                    loc = await self.find_placement(UnitTypeId.BARRACKS, self.main_base_ramp.barracks_correct_placement, placement_step=3)
                    if loc:
                        w.build(UnitTypeId.BARRACKS, loc)
                else:
                    await self.build_near_base(UnitTypeId.BARRACKS)

        if (self.building_requirements_satisfied(UnitTypeId.REFINERY)
                and self.tech.should_build_refinery(self.army_type,
                                                    self.townhalls.ready.amount,
                                                    self.count_buildings(UnitTypeId.REFINERY))
                and self.can_afford(UnitTypeId.REFINERY)
                and self.townhalls.exists):
            vgs = (self.vespene_geyser
                   .filter(lambda x: self.townhalls.ready.closer_than(15, x).exists)
                   .filter(lambda x: self.structures(UnitTypeId.REFINERY).closer_than(1.0, x).empty))
            if vgs.exists:
                vg = vgs.first
                w = self.select_build_worker(vg.position)
                if w:
                    w.build(UnitTypeId.REFINERY, vg)

        if (self.building_requirements_satisfied(UnitTypeId.ENGINEERINGBAY)
                and self.tech.should_build(self.army_type, UnitTypeId.ENGINEERINGBAY,
                                           self.count_buildings(UnitTypeId.ENGINEERINGBAY), self.townhalls.amount)
                and self.can_afford(UnitTypeId.ENGINEERINGBAY)):
            await self.build_near_base(UnitTypeId.ENGINEERINGBAY)

        if (self.building_requirements_satisfied(UnitTypeId.ARMORY)
                and self.tech.should_build(self.army_type, UnitTypeId.ARMORY,
                                           self.count_buildings(UnitTypeId.ARMORY),
                                           self.townhalls.amount)
                and self.can_afford(UnitTypeId.ARMORY)):
            await self.build_near_base(UnitTypeId.ARMORY)

    async def control_unit_production(self):
        if self.supply_used == 200:
            return

        # build depots when supply is needed soon
        if self.townhalls.exists and self.should_build_supply() and self.can_afford(UnitTypeId.SUPPLYDEPOT):
            await self.build_depot()
        # scv building is here to prioritize it together with economy
        # and not when army units are needed
        if self.townhalls.exists and self.should_build_scv() and self.can_afford(UnitTypeId.SCV):
            for th in self.townhalls.idle:
                th.train(UnitTypeId.SCV)
            for th in self.townhalls.filter(lambda x: (len(x.orders) == 1
                                                       and x.orders[0].ability.id == AbilityId.COMMANDCENTERTRAIN_SCV
                                                       and x.orders[0].progress >= 0.95)):
                th.train(UnitTypeId.SCV)

        for fac in self.structures(UnitTypeId.FACTORY).ready.idle.filter(lambda x: x.add_on_tag in self.fac_techlabs):
            if (self.can_afford(UnitTypeId.THOR) and self.structures(UnitTypeId.ARMORY).ready.exists
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.THOR)):
                fac.train(UnitTypeId.THOR)
            elif (self.can_afford(UnitTypeId.WIDOWMINE)
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.WIDOWMINE)):
                fac.train(UnitTypeId.WIDOWMINE)
            elif (self.can_afford(UnitTypeId.SIEGETANK)
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.SIEGETANK)):
                fac.train(UnitTypeId.SIEGETANK)
            elif (self.can_afford(UnitTypeId.CYCLONE)
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.CYCLONE)):
                fac.train(UnitTypeId.CYCLONE)

        for fac in self.structures(UnitTypeId.FACTORY).ready.filter(lambda x: x.add_on_tag not in self.fac_techlabs):
            if (self.can_afford(UnitTypeId.HELLION)
                    and (fac.is_idle or (fac.add_on_tag in self.fac_reactors and len(fac.orders) < 2))
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.HELLION)):
                fac.train(UnitTypeId.HELLION)
            elif (self.can_afford(UnitTypeId.WIDOWMINE)
                    and (fac.is_idle or (fac.add_on_tag in self.fac_reactors and len(fac.orders) < 2))
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.WIDOWMINE)):
                fac.train(UnitTypeId.WIDOWMINE)

        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle.filter(lambda x: x.add_on_tag in self.rax_techlabs):
            if (self.can_afford(UnitTypeId.MARAUDER)
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.MARAUDER)):
                rax.train(UnitTypeId.MARAUDER)
            elif (self.can_afford(UnitTypeId.MARINE)
                  and self.tech.should_train_unit(self.army_type, UnitTypeId.MARINE)):
                rax.train(UnitTypeId.MARINE)

        for rax in self.structures(UnitTypeId.BARRACKS).ready.filter(lambda x: x.add_on_tag not in self.rax_techlabs):
            if rax.is_idle or (rax.add_on_tag in self.rax_reactors and len(rax.orders) < 2):
                if (self.can_afford(UnitTypeId.REAPER)
                        and self.tech.should_train_unit(self.army_type, UnitTypeId.REAPER)):
                    rax.train(UnitTypeId.REAPER)
                elif (self.can_afford(UnitTypeId.MARINE)
                        and self.tech.should_train_unit(self.army_type, UnitTypeId.MARINE)):
                    rax.train(UnitTypeId.MARINE)

        for sp in self.structures(UnitTypeId.STARPORT).ready.idle.filter(lambda x: x.add_on_tag in self.sp_techlabs):
            if (self.can_afford(UnitTypeId.RAVEN)
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.RAVEN)):
                sp.train(UnitTypeId.RAVEN)
            elif (self.can_afford(UnitTypeId.BANSHEE)
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.BANSHEE)):
                sp.train(UnitTypeId.BANSHEE)
            elif (self.can_afford(UnitTypeId.MEDIVAC)
                    and self.tech.should_train_unit(self.army_type, UnitTypeId.MEDIVAC)):
                sp.train(UnitTypeId.MEDIVAC)
            elif (self.can_afford(UnitTypeId.VIKINGFIGHTER)
                  and self.tech.should_train_unit(self.army_type, UnitTypeId.VIKINGFIGHTER)):
                sp.train(UnitTypeId.VIKINGFIGHTER)

        for sp in self.structures(UnitTypeId.STARPORT).ready.filter(lambda x: x.add_on_tag not in self.sp_techlabs):
            if sp.is_idle or (sp.add_on_tag in self.sp_reactors and len(sp.orders) < 2):
                if (self.can_afford(UnitTypeId.MEDIVAC)
                        and self.tech.should_train_unit(self.army_type, UnitTypeId.MEDIVAC)):
                    sp.train(UnitTypeId.MEDIVAC)
                elif (self.can_afford(UnitTypeId.VIKINGFIGHTER)
                      and self.tech.should_train_unit(self.army_type, UnitTypeId.VIKINGFIGHTER)):
                    sp.train(UnitTypeId.VIKINGFIGHTER)

    def _prepare_first_step(self):
        super()._prepare_first_step()

        self.army_type = ArmyPriority.BIO
        self.turret_time = 540
        if self.enemy_race == Race.Protoss:
            self.should_be_aggressive = False
        else:
            self.should_be_aggressive = True

        self.map_data = MapData(self)

    def _prepare_step(self, state, proto_game_info):
        super()._prepare_step(state, proto_game_info)
        self.townhalls: Units = self.structures(self.data.townhall_types)

    def get_army_supply(self, units: Units):
        supply = 0
        for unit in units:
            if unit.type_id == UnitTypeId.BUNKER:
                supply += 4
            elif unit.type_id == UnitTypeId.SPINECRAWLER:
                supply += 4
            elif unit.type_id == UnitTypeId.SPORECRAWLER:
                supply += 4
            else:
                supply += self.calculate_supply_cost(unit.type_id)

        return supply

    async def calculate_map_info(self):
        # check if there is a another ramp on the start location level close by
        base_height = self.get_terrain_height(self.start_location)
        for ramp in self.game_info.map_ramps:
            if ramp == self.main_base_ramp:
                continue
            upper_height = self.get_terrain_height(ramp.top_center)
            if base_height == upper_height and self.start_location.distance_to(ramp.top_center) < 25:
                ramp_dir = ramp.bottom_center - ramp.top_center
                test_offset = 2*ramp_dir
                pathing_possible = await self.client.query_pathing(self.start_location,
                                                                   ramp.bottom_center.offset(test_offset))
                # to deal with ramps blocked by rocks or minerals
                if pathing_possible is not None and pathing_possible < 50:
                    self.map_has_inner_expansion = True
                break

        self.enemy_expansion_order = await self.calculate_expansion_order(self.enemy_start_locations[0], True)
        for exp in self.enemy_expansion_order:
            self.enemy_expansion_checks[exp] = 0
        self.expansion_order = await self.calculate_expansion_order(self.start_location, False)

    async def on_step(self, iteration):
        if 4 + self.prev_game_loop > self.state.game_loop or self.time_budget_available < 0.4:
            return

        self.prev_game_loop = self.state.game_loop
        self.iterations_used += 1

        time_counter = time.monotonic()
        self.tech.update_time(self.time)

        if iteration == 0:
            await self.calculate_map_info()

        # temp disabled because it doesn't work
        # self.lift_buildings_under_attack()

        await self.control_units()

        current_base_count = (self.townhalls.ready.amount + self.already_pending(UnitTypeId.COMMANDCENTER)
                              - self.count_depleted_and_full_bases())
        # need to take into account what should be done
        # if all bases have already been taken

        if current_base_count < 1 and self.tech.should_expand(self.army_type, self.townhalls.amount):
            self.resource_priority = ResourcePriority.EXPANSION
        else:
            self.resource_priority = ResourcePriority.STRUCTURES

        # should change this at some point so we can build in different orders if necessary
        if self.resource_priority == ResourcePriority.STRUCTURES or self.resource_priority == ResourcePriority.UNITS:
            await self.control_addon_production()
            await self.control_unit_production()
            await self.control_tech_production()
        elif self.resource_priority == ResourcePriority.EXPANSION:
            # wait until I have money for expansion
            if self.can_afford(UnitTypeId.COMMANDCENTER):
                await self.try_to_expand()
                await self.control_addon_production()
                await self.control_unit_production()
                await self.control_tech_production()

        self.control_blocking_depots()

        if self.time >= 230:
            self.save_energy_for_scan = True

        self.calldown_mules()

        self.fix_interrupted_construction()
        await self.fix_idle_flying_buildings()
        self.control_building_fixing()

        if self.iterations_used % 5 == 0:
            self.assign_gas_workers()
        elif self.iterations_used % 5 == 1:
            self.assign_idle_workers()
        elif self.iterations_used % 5 == 2:
            self.assign_mineral_workers()

        if self.time - self.prev_game_seconds > 1:
            self.set_building_rallypoints()

            amount = sum(self.dead_unit_counter)
            # need to also check where the fight is happening around the map
            # shouldn't start retreating if i'm defending my base
            # need to come up with something more specific to decide when
            # to go on the attack and when to fall back
            army_supply = self.get_army_supply(self.main_army.select_units(self.units))
            if army_supply > 50 and amount > 20:
                self.main_army.mode = ArmyMode.ATTACK
            elif amount < -30:
                self.main_army.mode = ArmyMode.RETREAT
                self.army_retreat_start_time = self.time
                # if we are forced to retreat, wait longer to attack again
                self.tech.delay_attack_timing(self.army_type)
                self.should_be_aggressive = False

            del self.dead_unit_counter[0]
            self.dead_unit_counter.append(0)

        self.prev_game_seconds = math.floor(self.time)
        diff_time = time.monotonic() - time_counter
        if diff_time > 0.1:
            print("Iteration time {}".format(diff_time))
