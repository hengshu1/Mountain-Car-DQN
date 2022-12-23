import numpy as np
import matplotlib.pyplot as plt

#this is for the last 100 episodes
avg_score = np.load('avg_score.npy')
std_score = np.load('std_score.npy')

num_episodes = 1500
# plt.plot(range(num_episodes-100, num_episodes), avg_score[num_episodes-100: num_episodes], '-b')
plt.plot(range(num_episodes-100, num_episodes), -avg_score[0: 100], '-b', label='score mean')
# plt.plot(range(num_episodes-100, num_episodes), std_score[0: 100], '-r', 'score std')#generally the same trend as mean
plt.title('plumeting behavior; even in the end, some models are not stable')
plt.xlabel('Episodes')
plt.ylabel('AVG Score')
plt.legend()

#find best models
index = np.argsort(avg_score)
print(avg_score[index[-10:]])
print(avg_score[index[:10]])
print(index[:10])
print(index[-10:])

plt.show()
