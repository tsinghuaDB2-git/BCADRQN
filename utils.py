import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e7)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.max_action = np.zeros((action_dim, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	# 抽样训练
	def sample(self):
		ind = np.random.randint(1, self.size)
		temp = ind
		list = []
		#0是末尾
		while self.not_done[temp - 1] != 0:
			temp -= 1
		for i in range(temp, ind + 1):
			list.append(i)
		ind = np.array(list)

		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)

	# 遍历测试
	def traversal(self, i):
		ind = i
		temp = ind
		list = []

		while self.not_done[temp - 1] != 0:
			temp -= 1
		for i in range(temp, ind + 1):
			list.append(i)
		ind = np.array(list)

		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)


	def save(self, save_folder, max_action):
		print('Memory saving...\n')
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)
		np.save(f"{save_folder}_max_action.npy", max_action)

	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]
		self.max_action = np.load(f"{save_folder}_max_action.npy")