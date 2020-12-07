import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gym
from gym.spaces import Box, Discrete
import numpy as np


# for weights orthogonal normalization
def Orth_norm(layer):
    torch.nn.init.orthogonal_(layer.weight, 1.0)
    torch.nn.init.constant_(layer.bias, 0)


# define policy(include continuous and discrete action) and value network
class ActorGaussian(nn.Module):
    def __init__(self, s_dim, a_dim, hidden1=64, hidden2=64, orth_norm=True):
        super(ActorGaussian, self).__init__()
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

    def forward(self, s, a=None):
        x = F.relu(self.linear1(s))
        mu = self.mu(F.relu(self.linear2(x)))
        log_sigma = torch.tanh(self.log_sigma(F.relu(self.linear3(x))))
        sigma = torch.exp(log_sigma)
        pi = Normal(mu, sigma)
        logp_a = None
        if a is not None:
            logp_a = pi.log_prob(a).sum(1) # sum of row
        return pi, logp_a


class ActorCategorical(nn.Module):
    def __init__(self, s_dim, a_dim, hidden1=64, hidden2=64, orth_norm=True):
        super(ActorCategorical, self).__init__()
        self.linear1 = nn.Linear(s_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.probs = nn.Linear(hidden2, a_dim)    # or self.logits =
        if orth_norm:
            Orth_norm(self.linear1)
            Orth_norm(self.linear2)
            Orth_norm(self.probs)

    def forward(self, s, a=None):
        x = torch.sigmoid(self.probs(F.relu(self.linear2(F.relu(self.linear1(s))))))
        pi = Categorical(probs=x)         # also, logits=x is OK, but needn't sigmoid()
        logp_a = None
        if a is not None:
            logp_a = pi.log_prob(a)       # no sum, because output of discrete action is single
        return pi, logp_a


class Critic(nn.Module):
    def __init__(self, s_dim, hidden1=64, hidden2=64, orth_norm=True):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(s_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.v = nn.Linear(hidden2, 1)
        if orth_norm:
            Orth_norm(self.linear1)
            Orth_norm(self.linear2)
            Orth_norm(self.v)

    def forward(self, obs):
        pred_v = self.v(F.relu(self.linear2(F.relu(self.linear1(obs)))))
        return pred_v.squeeze(-1)          # torch.Size([]) to torch.Size()


class ActorCritic(nn.Module):
    def __init__(self, s_space, a_space, hid_pi1=64, hid_pi2=64, hid_v1=64, hid_v2=64, orth_norm=True):
        super(ActorCritic, self).__init__()
        s_dim = s_space.shape[0]
        if isinstance(a_space, Box):
            self.pi = ActorGaussian(s_dim, a_space.shape[0], hid_pi1, hid_pi2, orth_norm)
        elif isinstance(a_space, Discrete):
            self.pi = ActorCategorical(s_dim, a_space.n, hid_pi1, hid_pi2, orth_norm)
        self.v = Critic(s_dim, hid_v1, hid_v2, orth_norm)

    def step(self, s):
        pi = self.pi(s)[0]
        a = pi.sample()
        logp_a = pi.log_prob(a).sum(0)
        pred_v = self.v(s)
        return a.detach().numpy(), logp_a.detach().numpy(), pred_v.detach().numpy()


class Buffer:
    def __init__(self, buf_size, s_space, a_space, gamma=0.999, lam=0.998):
        self.states = np.zeros([buf_size, s_space.shape[0]], dtype=np.float32)
        if isinstance(a_space, Box):
            self.actions = np.zeros([buf_size, a_space.shape[0]], dtype=np.float32)
        elif isinstance(a_space, Discrete):
            self.actions = np.zeros(buf_size, dtype=np.float32)
        self.logp_as = np.zeros(buf_size, dtype=np.float32)
        self.values = np.zeros(buf_size, dtype=np.float32)
        self.rewards = np.zeros(buf_size, dtype=np.float32)
        self.notdones = np.zeros(buf_size, dtype=np.float32)
        self.rtgs = np.zeros(buf_size, dtype=np.float32)
        self.deltas = np.zeros(buf_size, dtype=np.float32)
        self.advs = np.zeros(buf_size, dtype=np.float32)
        self.size = buf_size
        self.gamma = gamma
        self.lam = lam

    def store(self, state, action, logp_a, value, reward, notdone, pointer):
        self.states[pointer] = state
        self.actions[pointer] = action
        self.logp_as[pointer] = logp_a
        self.values[pointer] = value
        self.rewards[pointer] = reward
        self.notdones[pointer] = notdone

    def compute_rtg_adv(self, v_later, adv_norm=True):
        rtg_later = v_later                          # for trajectory end
        adv_later = 0
        for i in reversed(range(self.size)):
            self.rtgs[i] = self.rewards[i] + self.gamma * rtg_later * self.notdones[i]
            self.deltas[i] = self.rewards[i] + self.gamma * v_later * self.notdones[i] - self.values[i]
            self.advs[i] = self.deltas[i] + self.gamma * self.lam * adv_later * self.notdones[i]
            rtg_later = self.rtgs[i]
            v_later = self.values[i]
            adv_later = self.advs[i]
        if adv_norm:
            self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

    def get_data(self):
        s = torch.as_tensor(self.states)
        a = torch.as_tensor(self.actions)
        logp_a= torch.as_tensor(self.logp_as)
        rtg = torch.as_tensor(self.rtgs)
        adv = torch.as_tensor(self.advs)
        data = dict(states=s, actions=a, logp_old=logp_a, rtgs=rtg, advs=adv)
        return data


class State_norm:
    def __init__(self, s_dim, s_clip=5.0):
        self.pointer = 0
        self.mean = np.zeros(s_dim, dtype=np.float32)
        self.nvar = np.zeros(s_dim, dtype=np.float32)         # nvar = n * var
        self.var = np.zeros(s_dim, dtype=np.float32)
        self.std = np.zeros(s_dim, dtype=np.float32)
        self.clip = s_clip

    def mean_std(self, s):
        s = np.asarray(s, dtype=np.float32)
        assert s.shape == self.mean.shape
        self.pointer += 1
        if self.pointer == 1:
            self.mean = s                                     # self.std = [0, 0, 0]
        else:
            old_mean = self.mean
            self.mean = old_mean + (s - old_mean) / self.pointer
            self.nvar = self.nvar + (s - old_mean) * (s - self.mean)
            self.var = self.nvar / (self.pointer -1)
            self.std = np.sqrt(self.var)
        return self.mean, self.std

    def normalize(self, s, isclip=True):
        mean, std = self.mean_std(s)
        if std.all() == 0:
            s = s
        else:
            s = (s - mean) / (std + 1e-6)
        if isclip:
            s = np.clip(s, -self.clip, self.clip)
        return s

def ppo(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env_name)
    s_space = env.observation_space
    a_space = env.action_space
    actic = ActorCritic(s_space, a_space, hid_pi1=64, hid_pi2=64, hid_v1=64, hid_v2=64, orth_norm=args.p_orth_norm)
    #print(actic)
    optimizer_pi = torch.optim.Adam(actic.pi.parameters(), lr=args.lr_pi)
    optimizer_v = torch.optim.Adam(actic.v.parameters(), lr=args.lr_v)
    scheduler_pi = torch.optim.lr_scheduler.StepLR(optimizer_pi, step_size=50, gamma=0.1)
    scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=50, gamma=0.1)

    global_record = []

    for epoch in range(args.epochs):
        buffer = Buffer(args.buf_size, s_space, a_space, args.gamma, args.lam)
        norm_s = State_norm(s_space.shape[0], s_clip=5.0)
        s = env.reset()
        if args.state_norm:
            s = norm_s.normalize(s, args.s_isclip)
        traj_r = 0
        traj_len = 0
        traj_r_list = []
        for step in range(args.buf_size):
            a, logp_a, v = actic.step(torch.as_tensor(s, dtype=torch.float32))
            s_next, r, done, _ = env.step(a)
            traj_r += r
            traj_len += 1
            notdone = 0 if done else 1
            buffer.store(s, a, logp_a, v, r, notdone, step)
            s = s_next
            if args.state_norm:
                s = norm_s.normalize(s, args.s_isclip)
            runout = traj_len == args.traj_len_max
            terminal = done or runout
            buf_end = step == args.buf_size-1
            if buf_end:
                # traj_r_list.append(traj_r)
                if not terminal:
                    print('Notice:last trajectory cut off by buffer_size_end at {} step.'.format(traj_len))
                    _, v_end, _ = actic.step(torch.as_tensor(s, dtype=torch.float32))
                else:
                    v_end = 0
                buffer.compute_rtg_adv(v_later=v_end, adv_norm=True)
            elif terminal:
                traj_r_list.append(traj_r)
                traj_r = 0
                traj_len = 0
                s = env.reset()
                if args.state_norm:
                    s = norm_s.normalize(s, args.s_isclip)

        global_record.append({'episode': epoch, 'mean_trajs_r': np.mean(traj_r_list)})
        print("epoch: {:.0f}--------------mean_trajs_r: {:.4f}".format(epoch, np.mean(traj_r_list)))

        data = buffer.get_data()
        states, actions, logp_olds, rtgs, advs = data['states'].to(device), data['actions'].to(device), \
                                        data['logp_old'].to(device), data['rtgs'].to(device), data['advs'].to(device)
        actic.to(device)
        if args.mode == 1:
            for tarin in range(args.train_iters):
                pi, logp_news = actic.pi(states, actions)
                values = actic.v(states)
                ratio = torch.exp(logp_news - logp_olds)
                ratio_clip = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon) * advs
                loss_pi = -(torch.min(ratio * advs, ratio_clip)).mean()
                loss_v = ((values - rtgs) ** 2).mean()
                optimizer_pi.zero_grad()
                loss_pi.backward()
                optimizer_pi.step()
                optimizer_v.zero_grad()
                loss_v.backward()
                optimizer_v.step()

        else:
            for train in range(args.train_iters):
                pi, logp_news = actic.pi(states, actions)
                values = actic.v(states)
                ratio = torch.exp(logp_news - logp_olds)
                ratio_clip = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon) * advs
                loss_pi = -(torch.min(ratio * advs, ratio_clip)).mean()
                loss_v = ((values - rtgs) ** 2).mean()
                loss_entropy = (torch.exp(logp_news) * logp_news).mean()
                loss_total = loss_pi + args.coeff_v * loss_v + args.coeff_ent * loss_entropy
                optimizer_pi.zero_grad()
                optimizer_v.zero_grad()
                loss_total.backward()
                optimizer_pi.step()
                optimizer_v.step()
        print(optimizer_pi.param_groups[0]['lr'])
        scheduler_pi.step()
        scheduler_v.step()
        actic.cpu()
    return actic, norm_s, global_record

def Run(args):
    network, norm_s, _ = ppo(args)
    env = gym.make(args.env_name)
    s = env.reset()
    done = False
    i = 0
    while (not done):
        if args.state_norm:
            s = norm_s.normalize(s, args.s_isclip)
        env.render()
        a = network.step(torch.as_tensor(s, dtype=torch.float32))[0]
        s, _, done, _ = env.step(a)
        i += 1
        if done or i==400:
            print('Stop at {:.0f} steps.'.format(i))
            env.close()
            break


class args(object):
    env_name = 'Pendulum-v0'
    seed = 1234
    epochs = 100
    buf_size = 10240
    traj_len_max = 1000
    gamma = 0.999
    lam = 0.999
    lr_pi = 0.0004
    lr_v = 0.008
    epsilon = 0.2
    coeff_v = 0.5
    coeff_ent = -0.1                        # larger, nice for mode1
    train_iters = 80
    state_norm = True
    s_isclip = True
    p_orth_norm = False
    mode = 2

if __name__ == '__main__':
    args.env_name = 'Reacher-v2'
    Run(args)

