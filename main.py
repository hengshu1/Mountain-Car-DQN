import argparse, sys
import gym
from agent import Agent

'''
This code is developed based on https://github.com/DanielPalaio/MountainCar-v0_DeepRL
'''

if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    spec = gym.spec("MountainCar-v0")

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mode', default='testn', type=str, help='train or test1 or testn')

    args = parser.parse_args()

    dqn_agent = Agent(lr=0.001, discount_factor=0.99, num_actions=3, epsilon=1.0, batch_size=64, input_dims=2)

    if args.mode == 'train':
        num_episodes = 1500
        dqn_agent.train_model(env, num_episodes, graph=True)

    #todo: evaluate all the models and target network models at epochs
    elif args.mode == 'test1':
        num_episodes = 100
        file_type = 'tf'
        # file = 'saved_networks/dqn_model_1499'#can fail
        file = 'saved_networks/dqn_model_1498'#more stable: so the variance is really high; even training is near the end. 
        dqn_agent.test(env, num_episodes, file_type, file, graph=True)
    elif args.mode == 'testn':
        for i in range(1500):
            file = 'saved_networks/dqn_model_'+str(i)
            avg_score, scores = dqn_agent.test(env, num_episodes=1, file_type='tf', file=file, graph=True)

    else:
        print('unknown mode. exit')
        sys.exit(1)