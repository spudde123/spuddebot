import json
import sys

sys.path.insert(1, "python-sc2")

from sc2 import run_game, maps, Race, Difficulty, AIBuild
from sc2.player import Bot, Computer, Human

from bot import MyBot
import time
import random


def main():
    with open("botinfo.json") as f:
        info = json.load(f)

    race = Race[info["race"]]
    map_list = [
        "EternalEmpireLE",
        "EverDreamLE",
        "GoldenWallLE",
        "NightshadeLE",
        "SimulacrumLE",
        "ThunderbirdLE",
        "ZenLE"
    ]
    map_name = random.choice(map_list)
    run_game(maps.get(map_name), [
        #Human(Race.Terran, fullscreen=True),
        Bot(race, MyBot()),
        Computer(Race.Zerg, Difficulty.VeryHard)
    ], realtime=False, save_replay_as="./replays/{bot1}_vs_{bot2}_{map}_{time}.SC2Replay".format(
        bot1="spudde", bot2="computer", map=map_name.replace(" ", ""), time=time.strftime("%H_%M_%j")
    ))



if __name__ == '__main__':
    main()
