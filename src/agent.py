import os
import pickle
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class RLearningAgent(ABC):

    def _save_local(self, output_dir, iteration=None, extension='pkl'):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fname = self._generate_fname(output_dir, iteration, extension)

        if extension == 'pkl':
            with open(fname, 'wb') as f:
                pickle.dump(self, f)

        if extension == 'png' or extension == 'jpg':
            plt.savefig(fname)

    def _generate_fname(self, output_dir, iteration, extension):
        if iteration is None:
            return os.path.join(output_dir, 'agent.{}'.format(extension))
        return os.path.join(
            output_dir, 'agent_iter_{}.{}'.format(iteration, extension))

    @abstractmethod
    def save(self, output_dir, iteration=None):
        pass

    @abstractmethod
    def load(fname):
        pass
