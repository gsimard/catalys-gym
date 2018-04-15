import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import sys
from six import StringIO, b

import random
import pickle


class TicTacToeEnv(gym.Env):
    """
    Classic Tic-Tac-Toe

    Board state is representated by size 9 numpy.ndarray.

    X is 1.0
    O is -1.0
    Empty is 0.0

    Legal actions are [0-8], corresponding to filling this box for this turn's player.

    Internally board state is flipped by multiplying by -1 at every
    turn so the agent always acts in the best interest of this turn's
    player.

    The episode ends when a winning/loosing condition or draw is reached.
    You receive a reward of 1 if you win, -1 if you loose and zero for a draw.

    """

    import os
    
    metadata = {'render.modes': ['human', 'ansi']}
    tictactoe_dict = {}
    with open('tictactoe_dict.pkl', 'rb') as f:
        tictactoe_dict = pickle.load(f)

        
    def __init__(self):
        self.reward_range = (-1, 1)

        self.action_space = spaces.Discrete(9)
        # In our representation we only allow -1, 0 and +1
        self.observation_space = spaces.Box(low=-1, high=1, shape=[9], dtype=np.int8)

        self.ai_strength = 0.0
        self.first_play = 1
        self.self_play = False
        
        self.state = np.zeros(9, dtype=np.int8)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #  1.0: game won
    #  0.0: game ongoing or draw
    # -1.0: game lost
    def reward(self):
        GAME_OVER_STATES = [[1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1],
                            [1, 0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 1, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 1, 0, 1, 0, 1, 0, 0]]
            
        for x in GAME_OVER_STATES:
            d = np.dot(self.state, x)
            if d == 3:
                return 1.0
            elif d == -3:
                return -1.0
        return 0.0

    # Game is over if grid is full or someone won
    def done(self):
        return np.count_nonzero(self.state) == 9 or self.reward() != 0.0

    # Is this state legal ?
    def legal(self):
        s = sum(self.state)
        return s >= -1 and s <= 1

    def whose_turn(self):
        return -sum(self.state) or self.first_play
        #return (1 + -2*(np.count_nonzero(self.state) % 2)) * self.first_play

    def play_perfect(self):
        # Unless the game is over, play
        if not self.done():
            # Perfect player, randomizing its winning strategy
            # The dictionary has been created with player X in
            # mind but here we play O, so inverse state first
            moves = TicTacToeEnv.tictactoe_dict[tuple(np.multiply(self.state, self.whose_turn()))]
            a = random.choice(moves)
            self.state[a] = self.whose_turn()

            return a

    def play_random(self):
        # Unless the game is over, play
        if not self.done():
            # Select a random available box from the grid
            si = set(np.arange(0,9)).symmetric_difference(set(np.flatnonzero(self.state)))
            a = random.sample(si, 1)[0]
            self.state[a] = self.whose_turn()

            return a
    
    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        if action == -1:
            self.play_perfect() # X playing
            self.play_perfect() # O playing
            return np.array(self.state), self.reward(), self.done(), {}
        elif action == -2:
            self.play_random()  # X playing
            self.play_perfect() # O playing
            return np.array(self.state), self.reward(), self.done(), {}
        elif action == -3:
            self.play_random()  # X playing
            self.play_random() # O playing
            return np.array(self.state), self.reward(), self.done(), {}
        else:
            # Accept a move only if box is available
            if self.state[action] == 0.0:
                self.state[action] = 1.0

                # O's turn, if the agent is not playing against itself
                if not self.self_play:
                    if np.random.rand(1) < self.ai_strength:
                        self.play_perfect()
                    else:
                        self.play_random()
                    
                return np.array(self.state), self.reward(), self.done(), {}

            else:
                # Penalize a move on a used box
                return np.array(self.state), -10, True, {}

    def reset(self):
        # Reset the grid
        self.state = np.zeros(9, dtype=np.int8)

        # Half the time, O will have the first move
        if np.random.rand(1) < 0.5:
            self.first_play = -1
            self.state[np.random.randint(0,9)] = -1
        else:
            self.first_play = 1

        return np.array(self.state)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for i in range(0,3):
            # Extract line from state
            l = self.state[(3*i):(3*(i+1))]

            # For each box on this line, replace numbers with characters
            c = [' ', ' ', ' ']
            for j in range(0,3):
                c[j] = 'X' if l[j] == 1.0 else 'O' if l[j] == -1.0 else ' '

            outfile.write('{:s}|{:s}|{:s}\n'.format(c[0], c[1], c[2]))
            if i < 2:
                outfile.write('-----\n')

        outfile.write('\n')
        outfile.flush()
        
        if mode != 'human':
            return outfile

    def available_boxes(self):
        return set(np.arange(0,9)).symmetric_difference(set(np.flatnonzero(self.state)))
        
    def minimax(self, max=True, first=True):
        if self.done():
            return [-1], [self.reward()]
        else:
            indices = []
            rewards = []
            for i in self.available_boxes():
                # Play all available box as present player
                self.state[i] = 1 if max else -1

                _, r = self.minimax(not max, False)
                indices.append(i)
                rewards.append(r[0])

                # Restore state for this box
                self.state[i] = 0

            # Find index of min or max reward
            iminmax = np.argmax(rewards) if max else np.argmin(rewards)
            # Then keep only similar rewards
            indices = [indices[i] for i, x in enumerate(rewards) if x == rewards[iminmax]]
            rewards = [rewards[i] for i, x in enumerate(rewards) if x == rewards[iminmax]]

            if not first:
                # not first means function is called recursively (internaly)
                return [indices[0]], [rewards[0]] # keep one
            else:
                # top level call, interested in getting all immediate possibilities
                return indices, rewards # keep all
