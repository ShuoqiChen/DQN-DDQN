
#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pylab
from collections import deque
import subprocess
import random

EPISODES = 10000


class DQNAgent:
    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, state_size, action_size):
       # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 

        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 1
        self.learning_rate = 0.0005 # 0.0005
        self.epsilon = 0.5
        self.epsilon_decay = 0.999 # 0.999
        self.epsilon_min = 0.05 # 0.025
        self.batch_size = 32
        self.train_start = 10000
        self.test_episode_length = 20
        # create replay memory using deque
        self.memory = deque(maxlen=80000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./mountainCar_dqn.h5")
            print('load weights succeeded!')

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh',
                        kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='tanh',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def epsilon_greedy_policy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def greedy_policy(self, state):
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon -= self.epsilon_decay
            
    def burn_in_memory(self, env):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        counter = 0
        while counter <= self.train_start:
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            while not done:
                counter += 1
                action = np.random.randint(0, env.action_space.n)  # randomly select action  
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                self.append_memory(state, action, reward, next_state, done)
                #self.epsilon += self.epsilon_decay # <<<<<
                state = next_state

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        return random.sample(self.memory, batch_size)


    # pick samples randomly from replay memory (with batch_size)
    def train(self):

        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # When use replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.

        if len(self.memory) < self.train_start:
            train_status = False
            return train_status, 0
          
        train_status = True
        batch_size = min(self.batch_size, len(self.memory))
        samples = np.array(random.sample(self.memory, batch_size))
        input_v = np.concatenate(tuple(samples[:, 0]), axis=0)
        action_ind = samples[:, 1].astype(int)
        next_pred_v = self.target_model.predict(np.concatenate(tuple(samples[:, 3]), axis=0))
        output_v = self.model.predict(input_v)
        target_v = samples[:, 2] + np.where(samples[:, 4], np.zeros(len(samples)),
                                            self.discount_factor * np.max(next_pred_v, axis=1))
        TD_error = np.sum(target_v - output_v[np.arange(len(samples)), action_ind])
        output_v[np.arange(len(samples)), action_ind] = target_v
        self.model.fit(input_v, output_v, batch_size=self.batch_size,
                       epochs=1, verbose=0)

        return train_status, (TD_error/self.batch_size)
      
    def test(self, env):
        test_scores = []

        for e in range(self.test_episode_length):
            done = False
            scores = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            while not done:
                # get action (using greedy method) for the current state and go one step in environment
                q_value = self.model.predict(state)
                action = np.argmax(q_value[0])

                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                state = next_state

                # accumulate reward
                scores += reward

            test_scores.append(scores)

        return (sum(test_scores)/self.test_episode_length)
    

if __name__ == "__main__":
    # In case of mountainCar_dqn-v0, maximum length of episode is 5000
    env = gym.make('MountainCar-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.burn_in_memory(env=env)

    scores, episodes, errors = [], [], []
    test_scores, test_episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        error = 0
        counter = 0
        while not done:
            counter += 1
            # get action for the current state and go one step in environment
            action = agent.epsilon_greedy_policy(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
#             reward = reward if not done or score == 200 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_memory(state, action, reward, next_state, done)
            #if agent.epsilon > agent.epsilon_min:
            #    agent.apsilon -= agent.epsilon_decay 
            # every time step do the training
            status, TD_error = agent.train()
            if status == True:
              error += TD_error
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
#                 score = score if score == 200 else score + 100
                scores.append(score)
                errors.append(error/counter)
                episodes.append(e)
#                pylab.plot(episodes, scores, 'b')
##                 pylab.plot(episodes, errors, 'b')
#                pylab.savefig("./mountainCar_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                # if the mean of scores of last 500 episode is bigger than 190
                # stop training
                if np.mean(scores[-min(500, len(scores)):]) > 190:
                    sys.exit()
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # save the model
        if e % 50 == 0:
            
            np.savez('./mountainCar-train-result', scores=scores,
                         errors=errors,episodes=episodes)
            
             # get test results
            res = agent.test(env=env)
            test_scores.append(res)
            test_episodes.append(e)
            
            if e % 500 == 0:
                np.savez('./mountainCar-test-result', test_scores=test_scores, test_episodes=test_episodes)
                
                # save model weights
                agent.model.save_weights("./mountainCar_dqn.h5")
            
