# Easy 21 [WIP]

This repository contains the code for the *Easy 21* assignment of [Reinforcement Learning course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) by *David Silver*.

## Getting Started

The code has only been tested on *Python 3.6.8*.

## Running the tests

There are Some unit tests though there are many cases that are not covered yet.

To run the unit tests, you have to install [`nosetests`](https://nose.readthedocs.io/en/latest/) (will have been installed if you installed from `requirements.txt`). Then run the code below:

```bash
bash test/test_runner.sh
```

## Training

Currently, there are two methods implemented, i.e. **Monte Carlo** and **TD Learning**.

```bash
python -m src.main --help

Usage: main.py [OPTIONS]

Options:
  --output_dir TEXT          directory to save outputs  [default: ./outputs]
  --agent_method TEXT        choice of agents (monte_carlo/sarsa)  [default:
                             monte_carlo]
  --mc_method TEXT           [default: every]
  --no_episodes INTEGER      [default: 1000]
  --val_no_episodes INTEGER  [default: 500]
  --help                     Show this message and exit.
```

### Results

Both methods will ultimately converge to the optimal solution.

Below are the plots for the optimal action value function during the training.

* Monte Carlo (Every-Visit)

    ![Monte Carlo Every Visit](imgs/monte_carlo_every.gif)

* Temporal Difference Learning

    ![Sarsa](imgs/Sarsa.gif)

## Acknowledgments

All the codes here are implemented based on the Author (*Hardian Lawi*) understanding.
