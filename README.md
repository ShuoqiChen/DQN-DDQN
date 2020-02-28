# DQN-DDQN



**What is and why Q learning**


Q-learning is an off policy, tabular reinforcement learning algorithm. It is off-policy RL because the q-learning function learns from actions that are not the most updated policy. Essentially, q-learning seeks to learn a policy that maximizes the total reward.

Q learning is useful when we seeks to find the best action to take given the current state. In Q-learning related algorithms, an agent tries to learn the optimal policy from its history of interaction with the environment. It's an intuitive an elegant solutions to many RL problems.


**What is DQN and DDQN**

Instead of using the tabular reprenestation of the of the state-action, in deep Q-learnin we use a neural network to approximate the Q function. This is because using matrix to keep track of relative importance in simple Q learning is limited both in reprentational power and in the level of dimension the tabular table can scale. Thus, DQN utilzes deep neural network to approximate the q values, long as the relative importance is preserved.


DDQN or Dueling Deep Q Networks is a reinforcement learning algorithms that tries to create a Q value via two function estimators: one that estimates the advantage function, and another that estimates the value function. The value function calculates the value of a given input frame, and the advantage function calculates the benefits of taking a given action. Together, these can provide a good estimation of how good the next frame performs given a certain state action pair.

