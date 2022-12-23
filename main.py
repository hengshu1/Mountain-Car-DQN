import argparse, sys, time
import numpy as np
import tensorflow as tf
import gym

from agent import Agent

'''
This code is developed based on https://github.com/DanielPalaio/MountainCar-v0_DeepRL
'''

def combine_models(net1, net2, alpha=0.5):
    '''
    combine the two nets into a single one according to alpha
    '''
    w1 = np.array(net1.get_weights())
    w2 = np.array(net2.get_weights())
    print('w1.shape=', w1.shape)
    print('w2.shape=', w2.shape)
    sys.exit(1)
    net1.set_weights(alpha*w1 + (1-alpha)*w2)
    return net1


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    spec = gym.spec("MountainCar-v0")

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mode', default='tn_search', type=str, help='train, test1, testn, or tn_search')#todo: tn_search is for search better models in the target net space

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
    elif args.mode == 'tn_search':

        file1 = 'saved_networks/dqn_model_' + str(1400)
        file2 = 'saved_networks/dqn_model_' + str(1499)

        net1 = tf.keras.models.load_model(file1)
        net2 = tf.keras.models.load_model(file2)
        model = combine_models(net1, net2, alpha=0.5)

        dqn_agent.q_net = model
        avg_score, std_score, _ = dqn_agent.test(env, num_episodes=30, file_type='tf', file=file, graph=True)


    else:
        print('unknown mode. exit')
        sys.exit(1)