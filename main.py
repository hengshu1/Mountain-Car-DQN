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
    w1 = np.array(net1.get_weights())#todo: get a warning; seems Okay
    w2 = np.array(net2.get_weights())
    # print('w1.shape=', w1.shape)#each layer w,b
    # print('w2.shape=', w2.shape)
    # print(w1[0].shape)
    # print(w2[0].shape)
    # print(w1[0][0, :3])
    # print(w2[0][0, :3])
    net1.set_weights(alpha*w1 + (1.-alpha)*w2)
    # w3 = np.array(net1.get_weights())
    # print(w3[0][0, :3])#seems right
    return net1


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    spec = gym.spec("MountainCar-v0")

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # parser.add_argument('--mode', default='testn', type=str, help='train, test1, testn, or tn_search')
    # parser.add_argument('--mode', default='tn_search', type=str, help='train, test1, testn, or tn_search')
    parser.add_argument('--mode', default='collect', type=str, help='train, test1, testn, tn_search, or collect samples')

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
        dqn_agent.load_model(file_type, file)
        dqn_agent.test(env, num_episodes)
    elif args.mode == 'testn':
        num_models = 1500
        avg_score, std_score = np.zeros(num_models), np.zeros(num_models)
        for i in reversed(range(num_models-100, num_models)): #just the last models, starting from the final model
            print('episode=', i)
            t0 = time.time()
            file = 'saved_networks/dqn_model_'+str(i)
            dqn_agent.load_model(file_type='tf', file=file)
            avg_score[i-(num_models-100)], std_score[i-(num_models-100)], _ = dqn_agent.test(env, num_episodes=100)#avg_score[100] is the final model; avg_score[0] is the starting model
            print('testing one model takes:', time.time() -  t0)
        np.save('avg_score.npy', avg_score)
        np.save('std_score.npy', std_score)
    elif args.mode == 'tn_search':
        # file1 = 'saved_networks/dqn_model_' + str(1400)#bad model
        # file1 = 'saved_networks/dqn_targetnet_' + str(1499)#good model
        # file2 = 'saved_networks/dqn_model_' + str(1499)#good model

        file1 = 'saved_networks/dqn_model_' + str(1436)#top 1
        file2 = 'saved_networks/dqn_model_' + str(1402)#

        net1 = tf.keras.models.load_model(file1)
        net2 = tf.keras.models.load_model(file2)

        dqn_agent.q_net = net1
        avg_score1, std_score1, _ = dqn_agent.test(env, num_episodes=100)

        dqn_agent.q_net = net2
        avg_score2, std_score2, _ = dqn_agent.test(env, num_episodes=100)

        scores = []
        max_score = -np.infty
        best_alpha = -1
        for alpha in np.arange(0, 1.1, 0.1):
            dqn_agent.q_net = combine_models(net1, net2, alpha=alpha)
            avg_score, std_score, _ = dqn_agent.test(env, num_episodes=100)
            scores.append(avg_score)
            if avg_score > max_score:
                max_score = avg_score
                best_alpha = alpha
        print('best score is {} at alpha={}'.format(max_score, alpha))
        print('avg_score1={}, std1={}, avg_score2={}, std2={}'.format(avg_score1, std_score1, avg_score2, std_score2))
    elif args.mode == 'collect':
        file1 = 'saved_networks/dqn_model_' + str(1436)  # top 1
        dqn_agent.q_net = tf.keras.models.load_model(file1)
        dqn_agent.test_and_collect(env, num_episodes=10)
        print('total samples:', dqn_agent.buffer.counter)

        #now evaluate the loss of a given model on this "optimal path test" sample set
        dqn_agent.sample_and_test_a_batch()


    else:
        print('unknown mode. exit')
        sys.exit(1)