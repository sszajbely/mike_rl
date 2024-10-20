from src.environment import MikeWorld
from src.agents.player_agent import PlayerAgent
from src.agents.rule_based_agent import RuleBasedAgent
from src.agents.stupid_dqn.agent import DQNAgent
from argparse import ArgumentParser
from src.agent import Agent
from src.agents.unet_dqn.agent import UNetAgent

AGENTS_MAP: dict[str, Agent] = {
    "player": PlayerAgent,
    "rule": RuleBasedAgent,
    "dqn": DQNAgent,
    "unet": UNetAgent
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--agents", type=str, nargs=4, default=["unet", "unet", "unet", "unet"])
    parser.add_argument("--n_rounds", type=int, default=1)
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--spectate", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    agents = []

    for i, agent in enumerate(args.agents):
        if ":" in agent:
            agent, arg = agent.split(":")
            agents.append(AGENTS_MAP[agent](args.train > i, i, int(arg)))
        else:
            agents.append(AGENTS_MAP[agent](args.train > i, i, i))

    game = MikeWorld(agents, args.n_rounds, "player" in args.agents, args.verbose, args.train > 0, args.spectate, args.eval)
    game.run()