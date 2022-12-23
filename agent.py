import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from replay_buffer import ReplayBuffer
from dqn_model import DeepQNetwork
import time

class Agent:
    def __init__(self, lr, discount_factor, num_actions, epsilon, batch_size, input_dims):
        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = 0.001
        self.epsilon_final = 0.01
        self.update_frequency = 100
        self.step_counter = 0 #actually the batch counter: the number of batches that are replayed.
        self.buffer = ReplayBuffer(1000000, input_dims)
        self.q_net = DeepQNetwork(lr, num_actions, input_dims, 256, 256)
        self.q_target_net = DeepQNetwork(lr, num_actions, input_dims, 256, 256)


    def policy_epsilon_greedy(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_net(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def sample_and_train_a_batch(self):
        '''
        If buffer has too few samples, this doesn't do anything
        '''
        if self.buffer.counter < self.batch_size:
            return
        if self.step_counter % self.update_frequency == 0:
            self.q_target_net.set_weights(self.q_net.get_weights())

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.buffer.sample_buffer(self.batch_size)

        q_predicted = self.q_net(state_batch)
        q_next = self.q_target_net(new_state_batch)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()

        #todo: the for-loop can be vectorized. Also the q_predicted is useless here, which can be removed. -- No. Actually the other actions are the same so we need q_target to init from q_predicted .
        q_target = np.copy(q_predicted)
        for idx in range(done_batch.shape[0]):
            target_q_val = reward_batch[idx]
            if not done_batch[idx]:
                target_q_val += self.discount_factor * q_max_next[idx]
            q_target[idx, action_batch[idx]] = target_q_val

        self.q_net.train_on_batch(state_batch, q_target)

        #linearly decreasing the epsilon
        # print('epsilon=', self.epsilon)
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        self.step_counter += 1

    def train_model(self, env, num_episodes, graph):
        scores, episodes, avg_scores, obj = [], [], [], []
        txt = open("saved_networks.txt", "w")
        bad_models = []

        for i in range(num_episodes):
            t0 = time.time()

            done = False
            score = 0.0
            state = env.reset()
            while not done:
                action = self.policy_epsilon_greedy(state)
                new_state, reward, done, _ = env.step(action)
                score += reward
                self.buffer.store_tuples(state, action, reward, new_state, done)
                state = new_state
                self.sample_and_train_a_batch()

            # note the score is the performance for this training episode.
            # In the final step, the model still updates one more time so the final score is not necessarily the final model's.
            # to measure plummeting, we need to test the model during training.
            scores.append(score)

            episodes.append(i)

            avg_score = np.mean(scores[-100:])#smoothing scores in the last 100 episodes
            avg_scores.append(avg_score)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}, #Samples: {5}".format(i, num_episodes, score, self.epsilon, avg_score, self.buffer.counter))

            #book keeping: the target net is some historical model; I'm saving it anyway, because sometimes it can be historical average too.
            #todo: evaluate an episode like in test(): actually you can evaluate after training is done
            self.q_net.save(("saved_networks/dqn_model_{0}".format(i)))
            self.q_net.save_weights(("saved_networks/dqn_model_{0}/net_weights{0}.h5".format(i)))

            self.q_target_net.save(("saved_networks/dqn_targetnet_{0}".format(i)))
            self.q_target_net.save_weights(("saved_networks/dqn_targetnet_{0}/targetnet_weights{0}.h5".format(i)))

            txt.write("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}, #Samples: {5}".format(i, num_episodes, score, self.epsilon, avg_score, self.buffer.counter))

            if i == 0:
                print('one episode time is ', time.time()-t0)


        txt.close()
        if graph:
            df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

            plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
            plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='AverageScore')
            plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Solved Requirement')
            plt.legend()
            plt.savefig('MountainCar_Train.png')
        np.save('bad_models.npy', np.array(bad_models))
        print('bad models are:', bad_models)

    def load_model(self, file_type, file):
        print('loading model ', file)
        if file_type == 'tf':
            self.q_net = tf.keras.models.load_model(file)
        elif file_type == 'h5':
            # self.train_model(env, 5, False)#todo: why train model in testing mode?
            # self.q_net.load_weights(file)
            print('unknown mode yet. ')
            sys.exit(1)
        else:
            pass

    def test(self, env, num_episodes,  rendering=False):
        #set the agent in exploitation mode; no exploration in testing
        self.epsilon = 0.0
        scores = np.zeros(num_episodes)
        for i in range(num_episodes):
            state = env.reset()
            done = False
            episode_score = 0.0
            while not done:
                if rendering:
                    env.render()
                action = self.policy_epsilon_greedy(state)
                new_state, reward, done, _ = env.step(action)
                episode_score += reward
                state = new_state
            scores[i] = episode_score
            print("Episode {0}/{1}, Score: {2} (epsilon={3})".format(i, num_episodes, episode_score, self.epsilon))
        # this can be moved out.
        # if graph:
        #     # df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores})
        #     plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
        #     plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
        #              label='AverageScore')
        #     plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
        #              label='Solved Requirement')
        #     plt.legend()
        #     plt.savefig('MountainCar_Test.png')

        env.close()
        avg_score = scores.mean()
        std_score = scores.std()
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Average score:{}; std: {}'.format(avg_score, std_score))

        return avg_score, std_score, scores

    def test_and_collect(self, env, num_episodes):
        self.epsilon = 0.0
        scores = np.zeros(num_episodes)
        for i in range(num_episodes):
            state = env.reset()
            # print('s0=', state)#different: yes
            done = False
            episode_score = 0.0
            while not done:
                action = self.policy_epsilon_greedy(state)
                new_state, reward, done, _ = env.step(action)
                episode_score += reward
                self.buffer.store_tuples(state, action, reward, new_state, done)
                state = new_state
            scores[i] = episode_score
            print("Episode {0}/{1}, Score: {2} (epsilon={3})".format(i, num_episodes, episode_score, self.epsilon))
        env.close()
        avg_score = scores.mean()
        std_score = scores.std()

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Average score:{}; std: {}'.format(avg_score, std_score))
        return avg_score, std_score, scores

    def sample_and_test_a_batch(self):
        '''
        sample and test batch.
        '''
        if self.buffer.counter < self.batch_size:
            return

        print('self.batch_size=', self.batch_size)
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.buffer.sample_buffer(self.batch_size)

        q_predicted = self.q_net(state_batch)
        q_next = self.q_target_net(new_state_batch)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()

        q_target = np.copy(q_predicted)
        for idx in range(done_batch.shape[0]):
            target_q_val = reward_batch[idx]
            if not done_batch[idx]:
                target_q_val += self.discount_factor * q_max_next[idx]
            q_target[idx, action_batch[idx]] = target_q_val

        loss = self.q_net.test_on_batch(state_batch, q_target)
        print('test loss:', loss)#oh. Huge error!


