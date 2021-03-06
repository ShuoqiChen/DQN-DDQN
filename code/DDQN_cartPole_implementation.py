
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


EPISODES = 5000


# Double DQN Agent for the environment
class DoubleDQNAgent:
    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, state_size, action_size):
        # if we want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 1
        self.learning_rate = 0.001
        self.epsilon = 0.5
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
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
            self.model.load_weights("./ddqn_cartpole_dqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.summary()
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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def burn_in_memory(self, env):
        # Initialize the replay memory with a burn_in number of episodes / transitions. 
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
                state = next_state

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # we will feed this to the model to train.
        return random.sample(self.memory, batch_size)


    # pick samples randomly from replay memory (with batch_size)
    def train(self):

        # In this function, we will train our network. 
        # If training without experience replay_memory, then we will interact with the environment 
        # in this function, while also updating the network parameters. 

        # When use replay memory, we should interact with environment here, and store these 
        # transitions to memory, while also updating the model.


        if len(self.memory) < self.train_start:
            train_status = False
            return train_status, 0

        train_status = True
        batch_size = min(self.batch_size, len(self.memory))

        # sample a mini batch
        sample_batch = self.sample_batch()

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []


        for i in range(batch_size):
            update_input[i] = sample_batch[i][0]
            update_target[i] = sample_batch[i][3]
            action.append(sample_batch[i][1])
            reward.append(sample_batch[i][2])
            done.append(sample_batch[i][4])

        # get the target Q values
        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        prediction = target[i][action[i]]  # this is how prediction is happend

        TD_error = 0
        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # [NOTE!!!] the difference here for Double DQN is that ..
                # selection of action is from DQN model but update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        aim = target[i][action[i]]  # this is how prediction is happend

        TD_error += (aim - prediction)

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
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
    # In case of CartPole-v0, we can play until 200 time step
    env = gym.make('CartPole-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DoubleDQNAgent(state_size, action_size)

    # burn in memory
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

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_memory(state, action, reward, next_state, done)
            # every time step do the training
            status, TD_error = agent.train()

            if status == True:
              error += TD_error

            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()
                scores.append(score)
                errors.append(error/counter)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./ddqn_cartpole.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

             # save data
        if e % 50 == 0:
         
            np.savez('./ddqn-carpoleV0-train-result', scores=scores,
                         errors=errors, episodes=episodes)
            
            # get test results
            res = agent.test(env=env)
            test_scores.append(res)
            test_episodes.append(e)
            
            if e % 500 == 0:
                np.savez('./ddqn-cartpoleV0-test-result', test_scores=test_scores, test_episodes=test_episodes)
                
                # save model weights
                agent.model.save_weights("./ddqn_cartpole_dqn.h5")

def test_video(agent, env, epi):
    # Usage: 
    #   we can pass the arguments within agent.train() as:
    #       if episode % int(self.num_episodes/3) == 0:
    #           test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()

