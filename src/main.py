import click
from src.environment import Easy21
from src.monte_carlo import MonteCarloAgent
from src.td_learning import TDLearningAgent


@click.command()
@click.argument('output_dir', default='/home/hardian/easy-21/outputs')
@click.option('--method', default='monte_carlo')
@click.option('--no_episodes', default=500000)
@click.option('--save_freq', default=25000,
              help='Will create a snapshot of the agent every no_episodes \
              % save_freq == 0')
def main(output_dir, method, no_episodes, save_freq):

    possible_actions = ['hit', 'stick']
    env = Easy21()
    if method == 'monte_carlo':
        agent = MonteCarloAgent(possible_actions)
    else:
        agent = TDLearningAgent(possible_actions)

    for i in range(no_episodes):

        print('###############')
        print('Game', i)

        terminate = False
        cur_state = env.initial_step()
        action = agent.observe_and_act(cur_state)
        while not terminate:
            print('State:', cur_state, 'action taken:', action)
            cur_state, reward, terminate = env.move(action)
            action = agent.observe_and_act(cur_state, reward, terminate)

        print('Last state:', cur_state)
        print_winner(reward)
        print('###############')

        if i % save_freq == 0:
            agent.save(output_dir, i)

    agent.save(output_dir)


def print_winner(reward):
    if reward == -1:
        print('Dealer wins.')
    if reward == 1:
        print('Player wins.')
    if reward == 0:
        print('Draw.')


if __name__ == '__main__':
    main()
