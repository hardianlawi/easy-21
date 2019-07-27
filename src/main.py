import click
from .environment import Easy21
from .monte_carlo import MonteCarloAgent
from .td_learning import TDLearningAgent
from .utils import train_and_plot


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
        agent = TDLearningAgent()

    # Training
    train_and_plot(no_episodes, agent, env, output_dir)

    assert False

    # Evaluation
    # Validation rounds

    total_win = 0

    for i in range(val_no_episodes):

        terminate = False
        cur_state = env.initial_step()
        action = agent.take_action(cur_state, explore=False)
        while not terminate:
            print("State:", cur_state, "action taken:", action)
            cur_state, reward, terminate = env.move(action)
            action = agent.take_action(cur_state, explore=False)

        if reward != -1:
            total_win += reward

        env.clear()

    print("Out of", val_no_episodes, "Player wins", total_win, "games")
    print("Winning percentage:", float(total_win) / val_no_episodes)


if __name__ == "__main__":
    main()
