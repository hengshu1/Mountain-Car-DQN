import argparse, sys, time
import numpy as np
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
        num_models = 1500
        avg_score, std_score = np.zeros(num_models), np.zeros(num_models)
        for i in reversed(range(num_models-100, num_models)): #just the last ten models
            t0 = time.time()
            file = 'saved_networks/dqn_model_'+str(i)
            avg_score[i], std_score[i], _ = dqn_agent.test(env, num_episodes=30, file_type='tf', file=file, graph=True)
            print('testing one model takes:', time.time() -  t0)
        np.save('avg_score.npy', avg_score)
        np.save('std_score.npy', std_score)
    else:
        print('unknown mode. exit')
        sys.exit(1)