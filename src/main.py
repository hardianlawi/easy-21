import click
import logging
from .environment import Easy21
from .monte_carlo import MonteCarloAgent
from .sarsa import SarsaAgent
from .utils import train_and_eval

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S:",
)


@click.command()
@click.argument("output_dir", default="./outputs")
@click.option("--agent_method", default="monte_carlo")
@click.option("--mc_method", default="first")
@click.option("--no_episodes", default=1000)
@click.option("--val_no_episodes", default=500)
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
