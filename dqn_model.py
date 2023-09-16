from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def DeepQNetwork(lr, num_actions, input_dims, fc1, fc2):
    q_net = Sequential()
    q_net.add(Dense(fc1, input_dim=input_dims, activation='relu'))
    q_net.add(Dense(fc2, activation='relu'))
    q_net.add(Dense(num_actions, activation=None))
    q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')#Keras use the sum over batch by Auto: https://github.com/keras-team/keras/blob/v2.11.0/keras/losses.py#L318
    return q_net

def DeepQNetworkElu1(lr, num_actions, input_dims, fc1, fc2):
    q_net = Sequential()
    q_net.add(Dense(fc1, input_dim=input_dims, activation='elu'))
    q_net.add(Dense(fc2, activation='elu'))
    q_net.add(Dense(num_actions, activation=None))
    q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')#Keras use the sum over batch by Auto: https://github.com/keras-team/keras/blob/v2.11.0/keras/losses.py#L318
    return q_net

def DeepQNetworkElu2(lr, num_actions, input_dims, fc1, fc2):
    q_net = Sequential()
    q_net.add(Dense(fc1, input_dim=input_dims, activation='elu'))
    q_net.add(Dense(fc2, activation='relu'))
    q_net.add(Dense(num_actions, activation=None))
    q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')#Keras use the sum over batch by Auto: https://github.com/keras-team/keras/blob/v2.11.0/keras/losses.py#L318
    return q_net

def DeepQNetworkElu3(lr, num_actions, input_dims, fc1, fc2):
    q_net = Sequential()
    q_net.add(Dense(fc1, input_dim=input_dims, activation='relu'))
    q_net.add(Dense(fc2, activation='elu'))
    q_net.add(Dense(num_actions, activation=None))
    q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')#Keras use the sum over batch by Auto: https://github.com/keras-team/keras/blob/v2.11.0/keras/losses.py#L318
    return q_net