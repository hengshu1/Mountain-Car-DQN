import numpy as np
import matplotlib.pyplot as plt

#this is for the last 100 episodes
avg_score = np.load('avg_score.npy')
std_score = np.load('std_score.npy')

num_episodes = 1500
num_effective_models = 100#note we only evaluated the last 100 models
model_lower_index = num_episodes-num_effective_models
model_scores = avg_score[0: num_effective_models]
plt.plot(range(model_lower_index, num_episodes), -model_scores, '-b', label='score mean')#todo: this needs to be update because I changed in main.py the indexing
plt.title('plumeting behavior; even in the end, some models are not stable')
plt.xlabel('Episodes')
plt.ylabel('AVG Score')
plt.legend()

#find best models
topk = 10
index = np.argsort(model_scores)#increasing
print('top models:', model_lower_index + index[-topk:])
print('their scores:', model_scores[index[-topk:]])
print('their scores(using original score array; should be the same):', avg_score[index[-topk:]])
plt.show()

