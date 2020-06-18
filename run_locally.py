import json

from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer

from bot import MyBot
import time


def main():
    with open("botinfo.json") as f:
        info = json.load(f)

    race = Race[info["race"]]
    map_name = "Abyssal Reef LE"
    run_game(maps.get(map_name), [
        Bot(race, MyBot()),
        Computer(Race.Random, Difficulty.Medium)
    ], realtime=False, save_replay_as="./replays/{bot1}_vs_{bot2}_{map}_{time}.SC2Replay".format(
        bot1="spudde", bot2="computer", map=map_name.replace(" ", ""), time=time.strftime("%H_%M_%j")
    ))



if __name__ == '__main__':
    main()
