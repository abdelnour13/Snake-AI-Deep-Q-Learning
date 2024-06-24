from game import Game
from human_agent import HumanControlledAgent
from agent import Agent
from argparse import ArgumentParser
from dataclasses import dataclass

@dataclass
class Config:
    max_memory : int = 100_000
    batch_size : int = 1024
    learning_rate : float = 0.001
    max_games : int = 200
    max_epsilon : float = 0.4
    min_epsilon : float = 0.0
    gamma : float = 0.9
    training : bool = True
    human : bool = False

def main(args : Config):

    game : Game = Game()
    agent = None

    if args.human:
        agent = HumanControlledAgent(game=game)
    else:

        agent = Agent(
            game=game,
            max_memory=args.max_memory,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_games=args.max_games,
            max_epsilon=args.max_epsilon,
            min_epsilon=args.min_epsilon,
            gamma=args.gamma,
            training=args.training
        )

    agent.run()

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--max-memory", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--max-epsilon", type=float, default=0.4)
    parser.add_argument("--min-epsilon", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--training", type=lambda t : t.lower() == "true", default=True)
    parser.add_argument("--human", type=lambda t : t.lower() == "true", default=False)

    args = parser.parse_args()

    main(args)
