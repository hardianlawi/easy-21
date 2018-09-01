import click
from src.environment import Easy21
from src.monte_carlo import MonteCarloAgent


@click.command()
@click.option('--no_episodes', default=500000)
@click.option('--method', default='monte-carlo')
def main(no_episodes, method):

    # TODO: Check Correctness

    possible_actions = ['hit', 'stick']
    env = Easy21()
    agent = MonteCarloAgent(possible_actions)

    terminate = False
    for i in range(no_episodes):
        cur_state = env.initial_step()
        action = agent.take_action(cur_state)
        while not terminate:
            cur_state, reward, terminate = env.step(cur_state, action)
            agent.receive_feedback(reward)
            if not terminate:
                action = agent.take_action(cur_state)


if __name__ == '__main__':
    main()
