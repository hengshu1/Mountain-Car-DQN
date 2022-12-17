from agent import Agent
import gym

env = gym.make("MountainCar-v0")
spec = gym.spec("MountainCar-v0")

#train
train = 1
test = 0
num_episodes = 1500


graph = True

file_type = 'tf'
file = 'saved_networks/dqn_model20'

dqn_agent = Agent(lr=0.001, discount_factor=0.99, num_actions=3, epsilon=1.0, batch_size=64, input_dims=2)

if train and not test:
    dqn_agent.train_model(env, num_episodes, graph)
else:
    dqn_agent.test(env, num_episodes, file_type, file, graph)