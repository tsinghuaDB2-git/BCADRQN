import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.max_action * torch.tanh(self.l3(a))
		return (self.phi * a) + ((1 - self.phi) * action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device

	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))

		return self.max_action * torch.tanh(self.d3(a))


class LSTM(nn.Module):
	def __init__(self, state_dim, action_dim, device, hidden_size):
		super(LSTM, self).__init__()
		self.lstm = nn.LSTM(input_size=state_dim+action_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
		self.hidden_size = hidden_size
		self.device = device

	def forward(self, state, action, length):
		h0 = Variable(torch.zeros(2, state.shape[0], self.hidden_size)).to(self.device)
		c0 = Variable(torch.zeros(2, state.shape[0], self.hidden_size)).to(self.device)
		data = torch.cat([state, action], 2)
		data = pack_padded_sequence(data, length, batch_first=True, enforce_sorted=False)
		z, (h, c) = self.lstm(data, (h0, c0))
		z, out_len = pad_packed_sequence(z, batch_first=True)
		output = []
		for i in range(state.shape[0]):
			output.append(z[i][out_len[i]-1].tolist())
		output = torch.Tensor(output).to(self.device)

		return output
		

class BCQ(object):
	def __init__(self, replay_buffer, state_dim, action_dim, max_action, device, discount=0.90, tau=0.005, lmbda=0.75, phi=0.05, hidden_size=30):
		latent_dim = action_dim * 2

		if torch.cuda.is_available():
			self.max_action = torch.from_numpy(max_action).cuda()
		else:
			self.max_action = torch.from_numpy(max_action)
		self.max_action = self.max_action.float()

		self.actor = Actor(hidden_size, action_dim, self.max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(hidden_size, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

		self.vae = VAE(hidden_size, action_dim, latent_dim, self.max_action, device).to(device)
		self.LSTM = LSTM(state_dim, action_dim, device, hidden_size).to(device)
		self.vae_optimizer = torch.optim.Adam([
				{'params': self.vae.parameters(), 'lr': 1e-4, },
				{'params': self.LSTM.parameters(), 'lr': 1e-4, },])

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.hidden_size = hidden_size
		self.device = device
		self.replay_buffer = replay_buffer

	def test(self):
		state, action, next_state, reward, not_done = self.replay_buffer.sample(1)
		return self.select_action(state)

	def traversal(self):
		_states = []
		_actions = []
		new_actions = []
		Qs = []
		new_Qs = []

		for i in tqdm(range(self.replay_buffer.size)):
		# for i in tqdm(range(10)):
			states = []
			before_actions = []
			actions = []
			next_states = []
			rewards = []
			not_dones = []
			final_actions = []

			state, action, next_state, reward, not_done = self.replay_buffer.traversal(i)

			states.append(state)
			next_states.append(next_state)

			b_action = action[:-1]#without last one
			b_action = np.insert(b_action, 0, action[0], axis=0)
			actions.append(action)
			before_actions.append(b_action)

			rewards.append(reward[-1])
			not_dones.append(not_done[-1])
			final_actions.append(action[-1])

			# seq length
			length = []
			for i in range(len(states)):
				length.append(len(states[i]))

			#  transfrom data to the format that LSTM needs
			states = pad_sequence([torch.FloatTensor(state).to(self.device) for state in states], batch_first=True,
								  padding_value=0)
			s_actions = pad_sequence([torch.FloatTensor(action).to(self.device) for action in before_actions],
									 batch_first=True, padding_value=0)

			state = self.LSTM(states, s_actions, length)
			action = torch.FloatTensor(np.array(final_actions)).to(self.device)


			Q = self.get_Q(state, action)
			new_action = self.select_action(state)
			new_Q = self.get_Q(state, torch.from_numpy(new_action))

			_states.append(state.squeeze())
			_actions.append(action.squeeze())
			new_actions.append(new_action.squeeze())
			Qs.append(Q.squeeze())
			new_Qs.append(new_Q.squeeze())
		return _states, _actions, new_actions, Qs, new_Qs

	def select_action(self, state):
		#choose the action with the highest Q value among 100 actions for each state
		with torch.no_grad():
			state = state.reshape(1, -1).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()

	def get_Q(self, state, action):
		with torch.no_grad():
			state = state.reshape(1, -1).to(self.device)
			action = action.reshape(1, -1).to(self.device)
			Q = self.critic_target.q1(state, action)
		return Q

	def train(self, iterations, batch_size=100):
		v_loss = 0
		c_loss = 0
		a_loss = 0

		for it in range(iterations):
			states = []
			before_actions = []
			actions = []
			next_states = []
			rewards = []
			not_dones = []
			final_actions = []
			for i in range(batch_size):
				state, action, next_state, reward, not_done = self.replay_buffer.sample()

				states.append(state)
				next_states.append(next_state)

				b_action = action[:-1]
				# b_action = np.insert(b_action, 0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), axis=0)
				b_action = np.insert(b_action, 0, action[0], axis=0)
				actions.append(action)
				before_actions.append(b_action)

				rewards.append(reward[-1])
				not_dones.append(not_done[-1])
				final_actions.append(action[-1])


			length = []
			for i in range(len(states)):
				length.append(len(states[i]))

			states = pad_sequence([torch.FloatTensor(state).to(self.device) for state in states], batch_first=True, padding_value=0)
			next_states = pad_sequence([torch.FloatTensor(state).to(self.device) for state in next_states], batch_first=True, padding_value=0)
			s_actions = pad_sequence([torch.FloatTensor(action).to(self.device) for action in before_actions], batch_first=True, padding_value=0)
			n_actions = pad_sequence([torch.FloatTensor(action).to(self.device) for action in actions], batch_first=True, padding_value=0)

			state = self.LSTM(states, s_actions, length)
			next_state = self.LSTM(next_states, n_actions, length)
			action = torch.FloatTensor(np.array(final_actions)).to(self.device)
			reward = torch.FloatTensor(np.array(rewards)).to(self.device)
			not_done = torch.FloatTensor(np.array(not_dones)).to(self.device)

			#  VAE Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			v_loss += vae_loss.item()
			vae_loss.backward(retain_graph=True)
			self.vae_optimizer.step()

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			c_loss += critic_loss.item()
			critic_loss.backward(retain_graph=True)
			self.critic_optimizer.step()

			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()

			self.actor_optimizer.zero_grad()
			a_loss += actor_loss.item()
			actor_loss.backward()
			self.actor_optimizer.step()

			#  update target networks
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return v_loss / iterations, c_loss / iterations, a_loss / iterations

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "actor_optimizer")

		torch.save(self.vae.state_dict(), filename + "vae")
		torch.save(self.LSTM.state_dict(), filename + "LSTM")
		torch.save(self.vae_optimizer.state_dict(), filename + "vae_optimizer")

	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

		self.vae.load_state_dict(torch.load(filename + "vae"))
		self.LSTM.load_state_dict(torch.load(filename + "LSTM"))
		self.vae_optimizer.load_state_dict(torch.load(filename + "vae_optimizer"))