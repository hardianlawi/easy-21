import click
import logging
from src.environment import Easy21
from src.monte_carlo import MonteCarloAgent
from src.sarsa import SarsaAgent
from src.utils import train_and_eval

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S:",
)


@click.command()
@click.option(
    "--output_dir",
    default="./outputs",
    show_default=True,
    help="directory to save outputs",
)
@click.option(
    "--agent_method",
    default="monte_carlo",
    show_default=True,
    help="choice of agents (monte_carlo/sarsa)",
)
@click.option("--mc_method", default="every", show_default=True)
@click.option("--no_episodes", default=1000, show_default=True)
@click.option("--val_no_episodes", default=500, show_default=True)
def main(output_dir, agent_method, mc_method, no_episodes, val_no_episodes):

    env = Easy21()
    if agent_method == "monte_carlo":
        agent = MonteCarloAgent(method=mc_method)
    else:
        agent = SarsaAgent()

    # Training and Eval
    train_and_eval(no_episodes, val_no_episodes, agent, env, output_dir)


if __name__ == "__main__":
    main()
