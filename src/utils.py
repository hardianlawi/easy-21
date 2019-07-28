import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .environment import Easy21
from .agent import RLearningAgent


def _generate_filepath(output_dir: str, agent: RLearningAgent):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return os.path.join(output_dir, agent.base_name)


def train_and_eval(
    steps: int,
    val_steps: int,
    agent: RLearningAgent,
    env: Easy21,
    output_dir: str,
    frames=50,
):
    """Util function to help train, evaluate and visualize the progress

    Args:
        steps (int): Total number of episodes to train the agent
        val_steps (int): Total number of episodes to evaluate the agent
        agent (RLearningAgent): Agent to train
        env (Easy21): Environment to train the agent in
        output_dir (str): Directory to store the progress
        frames (int, optional): Number of frames to show in the visualization. Defaults to 50.
    """
    filepath = _generate_filepath(output_dir, agent)

    def _plot_3d_frame(ax, title):

        ax.clear()

        V = agent.get_action_values()

        # min value allowed accordingly with the documentation is 1
        # we're getting the max value from V dimensions
        min_x = 1
        max_x = V.shape[0]
        min_y = 1
        max_y = V.shape[1]

        # creates a sequence from min to max
        x_range = np.arange(min_x, max_x)
        y_range = np.arange(min_y, max_y)

        # creates a grid representation of x_range and y_range
        X, Y = np.meshgrid(x_range, y_range)

        # get value function for X and Y values
        def get_stat_val(x, y):
            return V[x, y].max(axis=-1)

        Z = get_stat_val(X, Y)

        # creates a surface to be ploted
        # check documentation for details: https://goo.gl/etEhPP
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        ax.set_title(title)

        return ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )

    steps = steps // frames

    # you can change this values to change the size of the graph
    fig = plt.figure(figsize=(16, 5))

    # Explanation about this line: https://goo.gl/LH5E7i
    ax_act_val = fig.add_subplot(111, projection="3d")

    def animate(frame):
        if frame != 0:
            agent.train(steps, env)
        i = steps * frame
        logging.info("Frame %s" % frame)
        logging.info("Iteration %s" % i)
        win_pctg = agent.eval(val_steps, env)
        title = "Iteration %s, frame %s, Win: %s" % (i, frame, win_pctg)
        surf = _plot_3d_frame(ax_act_val, title)
        fig.canvas.draw()
        return surf

    ani = animation.FuncAnimation(fig, animate, frames, repeat=False)
    ani.save(filepath + ".gif", writer="imagemagick", fps=3)


def plot_error_vs_episode(
    sqrt_error,
    lambdas,
    train_steps=1000000,
    eval_steps=1000,
    title="SQRT error VS episode number",
    save_as_file=False,
):
    """
        Given the sqrt error between sarsa(lambda) for multiple lambdas and
        an already trained MC control model this function plots a
        graph: sqrt error VS episode number.

        Args:
            sqrt_error (tensor): multiD tensor.
            lambdas (tensor): 1D tensor.
            train_steps (int): number the total steps used to train the models.
            eval_steps (int): train_steps/eval_steps is the number of time the
                              errors were calculated while training.
            save_as_file (boolean).
    """
    # avoid zero division
    assert eval_steps != 0
    x_range = np.arange(0, train_steps, eval_steps)

    # assert that the inputs are correct
    assert len(sqrt_error) == len(lambdas)
    for e in sqrt_error:
        assert len(list(x_range)) == len(e)

    # create plot
    fig = plt.figure(title, figsize=(12, 6))
    plt.title(title)
    ax = fig.add_subplot(111)

    for i in range(len(sqrt_error) - 1, -1, -1):
        ax.plot(x_range, sqrt_error[i], label="lambda %.2f" % lambdas[i])

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    if save_as_file:
        plt.savefig(title)
    plt.show()


def plot_error_vs_lambda(
    sqrt_error, lambdas, title="SQRT error vs lambda", save_as_file=False
):
    """
        Given the sqrt error between sarsa(lambda) for multiple lambdas and
        an already trainedMC Control ths function plots a graph:
        sqrt error VS lambda.

        Args:
            sqrt_error (tensor): multiD tensor.
            lambdas (tensor): 1D tensor.
            title (string): Plot title.
            save_as_file (boolean).

        The srt_error 1D length must be equal to the lambdas length.
    """

    # assert input is correct
    assert len(sqrt_error) == len(lambdas)

    # create plot
    fig = plt.figure(title, figsize=(12, 6))
    plt.title(title)
    ax = fig.add_subplot(111)

    # Y are the last values found at sqrt_error
    y = [s[-1] for s in sqrt_error]
    ax.plot(lambdas, y)

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    if save_as_file:
        plt.savefig(title)
    plt.show()
