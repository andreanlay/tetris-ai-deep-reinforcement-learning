import keras
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
import sys

class Agent:
	def __init__(self, state_size):
		self.state_size = state_size
		self.memory = deque(maxlen=30000)
		self.discount = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.001 
		self.epsilon_end_episode = 2000
		self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_episode

		self.batch_size = 512
		self.replay_start = 3000
		self.epochs = 1

		self.model = self.build_model()

	def build_model(self):
		model = keras.Sequential([
				Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(1, activation='linear')
		])

		model.compile(loss='mse', optimizer='adam')
		return model

	def add_to_memory(self, current_state, next_state, reward, done):
		self.memory.append([current_state, next_state, reward, done])

	def act(self, states):
		max_value = -sys.maxsize - 1
		best = None

		if random.random() <= self.epsilon:
			return random.choice(list(states))
		else:
			for state in states:
				value = self.model.predict(np.reshape(state, [1, self.state_size]))
				if value > max_value:
					max_value = value
					best = state
		
		return best

	def replay(self):
		if len(self.memory) > self.replay_start:
			batch = random.sample(self.memory, self.batch_size)

			next_states = np.array([s[1] for s in batch])
			next_qvalue = np.array([s[0] for s in self.model.predict(next_states)])

			x = []
			y = []


			for i in range(self.batch_size):
				state, _, reward, done = batch[i][0], None, batch[i][2], batch[i][3]
				if not done:
					new_q = reward + self.discount * next_qvalue[i]
				else:
					new_q = reward

				x.append(state)
				y.append(new_q)

			self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, epochs=self.epochs, verbose=0)
