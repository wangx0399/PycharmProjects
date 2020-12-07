# Soft actor-critic for continuous action space

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gym
from gym.spaces import Box, Discrete
import numpy as np
from copy import deepcopy
import itertools


# for weights orthogonal normalization
def Orth_norm(layer):
    torch.nn.init.orthogonal_(layer.weight, 1.0)
    torch.nn.init.constant_(layer.bias, 0)


# define policy(include continuous and discrete action) and value network
class SquashedGaussianActor(nn.Module):
    def __init__(self, s_dim, a_dim, hidden1=64, hidden2=64, orth_norm=False):
        super(SquashedGaussianActor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.linear1 = nn.Linear(s_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.mu = nn.Linear(hidden2, a_dim)
        self.linear3 = nn.Linear(hidden1, hidden2)
        self.log_sigma = nn.Linear(hidden2, a_dim)
        if orth_norm:
            Orth_norm(self.linear1)
            Orth_norm(self.linear2)
            Orth_norm(self.mu)
            Orth_norm(self.linear3)
            Orth_norm(self.log_sigma)

    def forward(self, s):           # under pi, giving state and computing a rsample action
        x = F.relu(self.linear1(s))
        mu = self.mu(F.relu(self.linear2(x)))
        log_sigma = self.log_sigma(F.relu(self.linear3(x)))
        # log sigma clamp()
        sigma = torch.exp(log_sigma)
        pi = Normal(mu, sigma)
        a = pi.rsample()           # reparameterization
        logp_a = pi.log_prob(a).sum(-1) # sum of row
        logp_a -= (2*(np.log(2) - a - F.softplus(-2*a))).sum(-1)
        #print(logp_a)
        a = torch.tanh(a)
        return a, logp_a           # output action'logp_a and tanh(action)


class Critic(nn.Module):   # Q function
    def __init__(self, s_dim, a_dim, hidden1=64, hidden2=64, orth_norm=False):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(s_dim+a_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.v = nn.Linear(hidden2, 1)
        if orth_norm:
            Orth_norm(self.linear1)
            Orth_norm(self.linear2)
            Orth_norm(self.v)

    def forward(self, s, a):
        pred_v = self.v(F.relu(self.linear2(F.relu(self.linear1(torch.cat((s,a),dim=1))))))
        return pred_v.squeeze(-1)          # torch.Size([]) to torch.Size()


class ActorCritic(nn.Module):
    def __init__(self, s_space, a_space, hid_pi1=128, hid_pi2=64, hid_v1=128, hid_v2=64, orth_norm=False):
        super(ActorCritic, self).__init__()
        s_dim = s_space.shape[0]
        a_dim =  a_space.shape[0]
        self.pi = SquashedGaussianActor(s_dim, a_dim, hid_pi1, hid_pi2, orth_norm)
        self.q1 = Critic(s_dim, a_dim, hid_v1, hid_v2, orth_norm)
        self.q2 = Critic(s_dim, a_dim, hid_v1, hid_v2, orth_norm)

    def step(self, s):           # for interacting with environment
        with torch.no_grad():
            a = self.pi(s)[0]
            return a.detach().numpy()


class ReplayBuffer:
    """
    Soft actor-critic is an off-policy, it can use replaybuffer
    store 'state, action, r(s,a), done, state_next' in the intercation
    one step bootstrapping trains Q function, so needn't compute reward-to-go or GAE
    """
    def __init__(self, buf_size, s_space, a_space):
        self.s_buf = np.zeros([buf_size, s_space.shape[0]], dtype=np.float32)
        self.a_buf = np.zeros([buf_size, a_space.shape[0]], dtype=np.float32)
        self.r_buf = np.zeros(buf_size, dtype=np.float32)
        self.d_buf = np.zeros(buf_size, dtype=np.float32)   # if done, = 0; else = 1
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
        self.pointer = (self.pointer+1) % self.buf_size
        self.size = min(self.size+1, self.buf_size)

    def sample_minibatch(self, batch_size):
        indexs = np.random.randint(0, self.size, size=batch_size)
        minibatch = dict(s = self.s_buf[indexs],
                         a = self.a_buf[indexs],
                         r = self.r_buf[indexs],
                         d = self.d_buf[indexs],
                         sn = self.sn_buf[indexs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in minibatch.items()}  # to tensor


def sac(args):
    """
    with temperature parameter respect to alpha
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # initialization
    env = gym.make(args.game)
    s_space = env.observation_space
    a_space = env.action_space
    print(s_space.shape, '    ', a_space.shape)
    ac = ActorCritic(s_space, a_space, orth_norm=False)
    optimizer_pi = torch.optim.Adam(ac.pi.parameters(), lr=args.lr_pi)
    optimizer_q1 = torch.optim.Adam(ac.q1.parameters(), lr=args.lr_q)
    optimizer_q2 = torch.optim.Adam(ac.q2.parameters(), lr=args.lr_q)
    ac_target = deepcopy(ac)    # initialize target Q1 and Q2
    for p in ac_target.parameters():    # freeze target networks' parameters
        p.requires_grad = False
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())    # q1 and q2 params getheter
    alpha = args.alpha
    entropy_target = -np.prod(a_space.shape)

    def update_q(s, a):
        # compute Q loss, and update
        q1 = ac.q1(s, a)
        q2 = ac.q2(s, a)
        with torch.no_grad():  # notice here!
            an, logp_an = ac.pi(sn)  # target actions come from current policy
            q1_hat = ac_target.q1(sn, an)
            q2_hat = ac_target.q2(sn, an)
            q_target = r + args.gamma * d * (torch.min(q1_hat, q2_hat) - alpha * logp_an)
        for p in q_params:
            p.requires_grad = True
        loss_q1 = ((q1 - q_target) ** 2).mean()
        loss_q2 = ((q2 - q_target) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        optimizer_q1.zero_grad()
        optimizer_q2.zero_grad()
        loss_q.backward()
        optimizer_q1.step()
        optimizer_q2.step()

    def update_pi(s):
        # compute pi loss, and update
        a_new, logp_a_new = ac.pi(s)
        q1_pi = ac.q1(s, a_new)
        q2_pi = ac.q2(s, a_new)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha * logp_a_new - q_pi).mean()
        optimizer_pi.zero_grad()
        loss_pi.backward()
        optimizer_pi.step()

    # Main loop
    ep_r_mean = []
    returns = 0  # compute an episode's total reward
    s = env.reset()

    replay_buffer = ReplayBuffer(args.buf_size, s_space, a_space)
    for epoch in range(args.epochs):    # times of getting replay buffer
        """
        for example:
        every epoch sample step(s,a,r,d,sn) 2000 times
        then get minibatch and train and update 200 times
        """

        episode_ret = []  # collect every episode's total reward

        for step in range(args.pre_epoch_steps):
            if replay_buffer.pointer < 10000:       # increase exploration
                a = env.action_space.sample()
            else:
                a = ac.step(torch.as_tensor(s, dtype=torch.float32))
            sn, r, done, _ = env.step(a)
            returns += r
            d = 0 if done else 1
            replay_buffer.store(s, a, r, d, sn)
            if done:
                s = env.reset()
                episode_ret.append(returns)
                returns = 0
            else:
                s = sn
        # record mean episode's total return in every epoch
        ep_r_mean.append({'epochs': epoch+1, 'traj_nums': len(episode_ret), 'mean_ep_r': np.mean(episode_ret)})
        print('epoch:', epoch+1, '  ---  tarj_nums:', len(episode_ret), '  ---  mean_ep_r:', np.mean(episode_ret))
        # now we have completed to collect a replay_buffer
        # sample mini batch from buffer and update pi\q networks
        if replay_buffer.pointer > 3000:
            for update in range(args.update_time):
                # prepare
                ac.to(device)
                ac_target.to(device)
                mini_batch = replay_buffer.sample_minibatch(args.batch_size)
                s_gpu, a, r, d, sn = mini_batch['s'].to(device),\
                                 mini_batch['a'].to(device),\
                                 mini_batch['r'].to(device),\
                                 mini_batch['d'].to(device),\
                                 mini_batch['sn'].to(device)
                # compute Q loss, and update
                update_q(s_gpu, a)
                    # Freeze Q networks, so don't waste computational effort
                for p in q_params:
                    p.requires_grad = False
                # compute pi loss, and update
                update_pi(s_gpu)
                    # Unfreeze Q networks
                for p in q_params:
                    p.requires_grad = True
                # compute alpha loss, and update
                with torch.no_grad():
                    a_new, logp_a_new = ac.pi(s_gpu)
                alpha = alpha * (1 + args.lr_alpha * ((logp_a_new + entropy_target).mean()))
                if alpha > 1 or alpha < 0:
                    print("Warning: alpha's value is error! ", alpha)
                # Update ac_target.q1\2
                with torch.no_grad():
                    for p, p_target in zip(ac.parameters(), ac_target.parameters()):
                        p_target.data.copy_(args.tau * p.data + (1.0-args.tau) * p_target.data)
                ac.cpu()
                ac_target.cpu()
    return ac, ac_target, ep_r_mean


def test(args):
    ac, _, ep_r_mean = sac(args)
    env = gym.make(args.game)
    s = env.reset()
    done = False
    i = 0
    while (not done):
        env.render()
        a = ac.step(torch.as_tensor(s, dtype=torch.float32))
        s, _, done, _ = env.step(a)
        i += 1
        if done or i==1200:
            print('Stop at {:.0f} steps.'.format(i))
            env.close()
            break


class args(object):
    envs = dict(a = 'Ant-v2',
                b = 'HalfCheetah-v2',
                c = 'Hopper-v2',
                d = 'Humanoid-v2',
                e = 'HumanoidStandup-v2',
                f = 'Swimmer-v2',
                g = 'Walker2d-v2')
    game = envs['g']
    seed = 4321
    lr_pi = 0.001
    lr_q = 0.002
    lr_alpha = 0.002
    gamma = 0.99
    tau = 0.05
    buf_size = int(1e6)
    pre_epoch_steps = 2048
    batch_size = 256
    epochs = 1000
    update_time = 200
    alpha = 0.8


if __name__ == '__main__':
    test(args)




'''
q1 = ac.q1(s, a)
q2 = ac.q2(s, a)
with torch.no_grad():       # notice here!
    an, logp_an = ac.pi(sn)      # target actions come from current policy
    q1_hat = ac_target.q1(sn, an)
    q2_hat = ac_target.q2(sn, an)
    q_target = r + args.gamma * d * (torch.min(q1_hat, q2_hat) - args.alpha * logp_an)
for p in q_params:
    p.requires_grad = True
loss_q1 = ((q1 - q_target)**2).mean()
loss_q2 = ((q2 - q_target)**2).mean()
loss_q = loss_q1 + loss_q2
optimizer_q1.zero_grad()
optimizer_q2.zero_grad()
loss_q.backward()
optimizer_q1.step()
optimizer_q2.step()
'''
'''
a_new, logp_a_new = ac.pi(s)
q1_pi = ac.q1(s, a_new)
q2_pi = ac.q2(s, a_new)
q_pi = torch.min(q1_pi, q2_pi)
loss_pi = (args.alpha * logp_a_new - q_pi).mean()
optimizer_pi.zero_grad()
loss_pi.backward()
optimizer_pi.step()
'''
