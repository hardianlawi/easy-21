import random


class Easy21(object):
    def __init__(self):

        # card values range from 1 to 10 (Uniformly)
        self._card_values = list(range(1, 11))

        # prob red 1/3, black 2/3
        self._card_colours = ["red", "black", "black"]

        self._terminate = False
        self._cur_state = None

    def clear(self):
        self._terminate = False
        self._cur_state = None

    def initial_step(self):
        self._cur_state = [self._draw_card()[0], self._draw_card()[0]]
        return self._cur_state

    def step(self, action):
        cur_state = self._cur_state
        self._cur_state, reward = self._step(cur_state, action)
        return self._cur_state, reward

    def _step(self, state, action):
        """Process one step of a player

        Args:
            state (list): a state s, list of length 2 (dealer's first card 1-10
            and the player's sum 1-21)
            action (str): a string to indicate which action to take ('hit' or
            'stick')

        Returns:
            list: the next state `s'` and a reward `r`.
        """
        dealer_hand, player_hand = state[0], state[1]

        if action == "hit":
            player_hand = self._play_card(player_hand)
        else:
            dealer_hand = self._dealer_step(dealer_hand)
            self._terminate = True

        reward = self._get_reward(dealer_hand, player_hand)

        if reward != 0:
            self._terminate = True

        return [dealer_hand, player_hand], reward

    def _dealer_step(self, dealer_hand):
        while dealer_hand < 17 and dealer_hand > 0:
            dealer_hand = self._play_card(dealer_hand)
        return dealer_hand

    def _play_card(self, hand):
        value, colour = self._draw_card()
        if colour == "black":
            hand += value
        else:
            hand -= value
        return hand

    def _draw_card(self):
        return [random.choice(self._card_values), random.choice(self._card_colours)]

    def _get_reward(self, dealer_hand, player_hand):
        if player_hand > 21 or player_hand < 1:
            return -1
        if dealer_hand > 21 or dealer_hand < 1:
            return 1
        if self._terminate:
            if player_hand > dealer_hand:
                return 1
            elif player_hand < dealer_hand:
                return -1
            else:
                return 0
        return 0

    def has_terminated(self):
        return self._terminate
