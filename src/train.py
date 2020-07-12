from engine import Tetris
from agent import Agent
import time

# Initialize tetris enviroment
env = Tetris(10, 20)

# Initialize training variable
max_episode = 3000
max_steps = 25000

agent = Agent(env.state_size)

episodes = []
rewards = []

current_max = 0

for episode in range(max_episode):
	current_state = env.reset()
	done = False
	steps = 0
	total_reward = 0
	print("Running episode " + str(episode))

	while not done and steps < max_steps:	
		# Render the board for visualization
		env.render(total_reward)

		# Get all possible tetromino placement in current board
		next_states = env.get_next_states()

		# If the dict is empty, meaning game is over
		if not next_states:
			break

		# Tell agent to choose the best possible state
		best_state = agent.act(next_states.values())

		# Grab best tetromino position and its rotation chosen by the agent
		best_action = None
		for action, state in next_states.items():
			if (best_state==state).all():
				best_action = action
				break

		reward, done = env.step(best_action)
		total_reward += reward

		# Add to memory for replay
		agent.add_to_memory(current_state, next_states[best_action], reward, done)

		# Set current new state 
		current_state = next_states[best_action]

		steps += 1

	print("Total reward: " + str(total_reward))
	episodes.append(episode)
	rewards.append(total_reward)

	
	agent.replay()

	if agent.epsilon > agent.epsilon_min:
		agent.epsilon -= agent.epsilon_decay
