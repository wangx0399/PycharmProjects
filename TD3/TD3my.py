import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import copy
from copy import deepcopy
from PIL import Image
import os
import pandas as pd
import itertools

import matplotlib.pyplot as plt
from matplotlib import animation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


# deterministic policy
class Actor(nn.Module):
	def __init__(self, s_dim, a_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(s_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, a_dim)
		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


# Q1 and Q2 are complete independent
class Critic(nn.Module):
	def __init__(self, s_dim, a_dim):
		super(Critic, self).__init__()
		# Q architecture
		self.l1 = nn.Linear(s_dim + a_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

	def forward(self, s, a):
		sa = torch.cat((s, a), 1)        # sa.shape  [batch_size,  s_dim+a_dim]
		q = F.relu(self.l1(sa))
		q = F.relu(self.l2(q))
		q = self.l3(q)                 # q.shape   [batch_size,  1]
		return q.squeeze(-1)     # q.shape   [batch_size]


class ActorCritic(nn.Module):
	def __init__(self, s_space, a_space, max_action, lr_a=3e-4, lr_q=3e-4, discount=0.99, tau=0.005,
				 policy_noise=0.2, noise_clip=0.5, policy_freq=2
				 ):
		super(ActorCritic, self).__init__()
		s_dim = s_space.shape[0]
		a_dim = a_space.shape[0]
		self.pi = Actor(s_dim, a_dim, max_action)
		self.q1 = Critic(s_dim, a_dim)
		self.q2 = Critic(s_dim, a_dim)

	def step(self, s):
		with torch.no_grad():
			a = self.pi(s)
		return a.detach().numpy()


class ReplayBuffer:
	def __init__(self, buf_size, s_space, a_space):
		self.s_buf = np.zeros([buf_size, s_space.shape[0]], dtype=np.float32)
		self.a_buf = np.zeros([buf_size, a_space.shape[0]], dtype=np.float32)
		self.r_buf = np.zeros(buf_size, dtype=np.float32)
		self.d_buf = np.zeros(buf_size, dtype=np.float32)  # if done, = 0; else = 1
		self.sn_buf = np.zeros([buf_size, s_space.shape[0]], dtype=np.float32)
		self.buf_size = buf_size
		self.pointer = 0
		self.size = 0

	def store(self, s, a, r, d, sn):
		self.s_buf[self.pointer] = s
		self.a_buf[self.pointer] = a
		self.r_buf[self.pointer] = r
		self.d_buf[self.pointer] = d
		self.sn_buf[self.pointer] = sn
		self.pointer = (self.pointer + 1) % self.buf_size
		self.size = min(self.size + 1, self.buf_size)

	def sample_minibatch(self, batch_size=256):
		indexs = np.random.randint(0, self.size, size=batch_size)
		minibatch = dict(s=self.s_buf[indexs],
						 a=self.a_buf[indexs],
						 r=self.r_buf[indexs],
						 d=self.d_buf[indexs],
						 sn=self.sn_buf[indexs])
		return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in minibatch.items()}  # to tensor


def TD3(args):

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)

	env = gym.make(args.env)
	s_space = env.observation_space
	a_space = env.action_space
	a_dim = a_space.shape[0]
	max_action = a_space.high[0]
	print('state_dim: ', s_space.shape, ' ------ ', 'action_dim: ', a_space.shape)
	print('{}_max_episode_steps: '.format(args.env), env._max_episode_steps)
	print('  -------------------------------------------------  ')

	ac = ActorCritic(s_space, a_space, max_action)
	optimizer_pi = torch.optim.Adam(ac.pi.parameters(), lr=args.lr_pi)
	optimizer_q1 = torch.optim.Adam(ac.q1.parameters(), lr=args.lr_q)
	optimizer_q2 = torch.optim.Adam(ac.q2.parameters(), lr=args.lr_q)
	q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
	ac_target = deepcopy(ac)
	for p in ac_target.parameters():
		p.requires_grad = False

	replaybuffer = ReplayBuffer(args.buf_size, s_space, a_space)

	s = env.reset()
	returns = 0
	ep_ret_mean = []
	for epoch in range(args.epochs):
		episode_ret = []
		# collect data (s,a,r,d,sn) every epoch
		for step in range(args.epoch_steps):
			# increase exploration
			if replaybuffer.pointer < args.max_stochastic_steps:
				a = env.action_space.sample()
			else:
				a = ac.step(torch.as_tensor(s, dtype=torch.float32))
				# as action is deterministic policy, increase a's randomness
				a += (np.random.normal(0, max_action * args.explor_noise, size=a_dim)).clip(-max_action, max_action)
			sn, r, done, _ = env.step(a)
			returns += r
			d = 0 if done else 1
			replaybuffer.store(s, a, r, d, sn)
			if done:
				# one trajectory is over
				s = env.reset()
				episode_ret.append(returns)
				returns = 0
			else:
				s = sn

		# until an epoch_steps is over, record
		ep_ret_mean.append({'epoch_num': epoch+1,
							'tarjectory_num': len(episode_ret),
							'mean_traj_ret': np.mean(episode_ret)})
		print('epoch:', epoch+1, ' --- traj_num:', len(episode_ret), ' --- mean_traj_ret:', np.mean(episode_ret))

		# stochastic sample minibatch from replaybuffer to train and update
		if replaybuffer.pointer > args.min_update_steps:
			can_policy_update = 0

			for update in range(args.update_times):
				ac.to(device)
				ac_target.to(device)
				minibatch = replaybuffer.sample_minibatch(args.batch_size)
				state, action, reward, notdone, state_next = minibatch['s'].to(device), \
															 minibatch['a'].to(device), \
															 minibatch['r'].to(device), \
															 minibatch['d'].to(device), \
															 minibatch['sn'].to(device)
				with torch.no_grad():
					a_noise = (torch.randn_like(action) * args.pi_noise)      # torch.randn = N(0,1)
					a_noise = a_noise.clamp(-args.noise_clip * max_action, args.noise_clip * max_action)
					action_next = (ac_target.pi(state) + a_noise).clamp(-max_action, max_action)
					target_q_next = torch.min(ac_target.q1(state_next, action_next), ac_target.q2(state_next, action_next))
					target_q = reward + args.gamma * notdone * target_q_next
				# update q1 and q2
				q1 = ac.q1(state, action)
				q2 = ac.q2(state, action)
				loss_q1 = F.mse_loss(q1, target_q)
				loss_q2 = F.mse_loss(q2, target_q)
				optimizer_q1.zero_grad()
				loss_q1.backward()
				optimizer_q1.step()
				optimizer_q2.zero_grad()
				loss_q2.backward()
				optimizer_q2.step()

				can_policy_update += 1

				if can_policy_update == args.policy_update:
					can_policy_update = 0
					# freeze Q ?
					loss_pi = -torch.min(ac.q1(state, ac.pi(state)), ac.q2(state, ac.pi(state))).mean()
					optimizer_pi.zero_grad()
					loss_pi.backward()
					optimizer_pi.step()
					# update ac_target
					for param, target_param in zip(ac.parameters(), ac_target.parameters()):
						target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
				ac.cpu()
				ac_target.cpu()
	return ac, ac_target, ep_ret_mean


class args(object):
	names = dict(a='Ant-v2',
				 b='HalfCheetah-v2',
				 c='Hopper-v2',
				 d='Humanoid-v2',
				 e='HumanoidStandup-v2',
				 f='Swimmer-v2',
				 g='Walker2d-v2',
				 h='BipedalWalker-v3')
	env = names['g']
	seed = 0
	lr_pi = 0.0005
	lr_q = 0.001
	gamma = 0.99
	tau = 0.005
	explor_noise = 0.1
	pi_noise = 0.2
	noise_clip = 0.5
	buf_size = int(1e6)
	batch_size = 256
	epochs = 200
	epoch_steps = 1500
	update_times = 300
	max_stochastic_steps = 25000
	min_update_steps = 5000
	policy_update = 2


def test(name, ac, save_gif=False):
	env = gym.make(name)
	imags = []
	s = env.reset()
	done = False
	i = 0
	env.render()
	while not done:
		if save_gif:
			env.render()
			imags.append(env.render(mode='rgb_array'))
		else:
			env.render()
		a = ac.step(torch.as_tensor(s, dtype=torch.float32))
		s, _, done, _ = env.step(a)
		i += 1
		if done or i == 1000:
			env.close()
			print('Stop at {:.0f} step.'.format(i))
			break
	if save_gif:
		#print(imags[1].shape, type(imags[1]), len(imags))
		#plt.imshow((imags[10]))
		#plt.show()
		display_imags_as_gif(imags, name)


# save interact images into gif
def display_imags_as_gif(frames, env):
	patch = plt.imshow(frames[0])
	plt.axis('off')

	def animate(i):
		patch.set_data(frames[i])

	anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
	anim.save("./results/gif/{}.gif".format(env), writer="imagemagick", fps=30)

'''
def save(ac, path):
	torch.save(ac.state_dict(), path)
def load(ac, path):
	ac.load_state_dict(torch.load(path))
'''

if __name__ == '__main__':

	is_train = True
	save_model = False
	save_record = False
	save_gif = False

	file_name = f"{args.env}_{args.seed}"
	if not os.path.exists("./results/gif"):
		os.makedirs("./results/gif")
	model_path = "./results/{}_{}_ac.pth".format(args.env, args.seed)
	record_path = "./results/{}_{}_ac.csv".format(args.env, args.seed)

	if is_train:
		ac, ac_target, record = TD3(args)
		if save_model:
			torch.save(ac.state_dict(), model_path)
		if save_record:
			record = pd.DataFrame(record)
			record.to_csv(record_path)
	else:
		env = gym.make(args.env)
		s_space = env.observation_space
		a_space = env.action_space
		max_action = a_space.high[0]
		ac = ActorCritic(s_space, a_space, max_action)
		ac.load_state_dict(torch.load(model_path))

	test(args.env, ac, save_gif)
