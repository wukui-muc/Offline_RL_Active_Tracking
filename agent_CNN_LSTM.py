import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor, CNN_LSTM,CNN_simple
import numpy as np
import math
import copy
from torch.autograd import Variable
from utils_basic import weights_init
class CQLSAC_CNN_LSTM(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 tau,
                 hidden_size,
                 learning_rate,
                 temp,
                 with_lagrange,
                 cql_weight,
                 target_action_gap,
                 device,
                 stack_frames,
                 lstm_seq_len,
                 lstm_layer,
                 lstm_out
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC_CNN_LSTM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.stack_frames=stack_frames
        self.device = device
        self.lstm_seq_len = lstm_seq_len
        self.gamma = torch.FloatTensor([0.99]).to(device)
        # self.gamma = torch.FloatTensor([0.9]).to(device)

        self.tau = tau
        hidden_size = hidden_size
        learning_rate = learning_rate
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)
        # self.alpha=torch.tensor([0.5])

        # CQL params
        self.with_lagrange = with_lagrange
        self.temp = temp
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # image processing network
        self.lstm_layer = lstm_layer
        self.CNN_LSTM = CNN_LSTM(state_size=self.state_size,
                                action_size=self.action_size,
                                hidden_size=hidden_size,
                                 stack_frames=self.stack_frames,
                                 lstm_out=lstm_out,
                                 lstm_layer=self.lstm_layer
                                ).to(self.device)  # obs_shape,frame_stack
        self.CNN_LSTM_optimizer = optim.Adam(self.CNN_LSTM.parameters(), lr=learning_rate)

        # Actor Network

        self.actor_local = Actor(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        # self.actor_local = Actor_CNN(self.state_size, action_size, hidden_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Critic Network (w/ Target Network)

        # self.critic1 = Critic_CNN(self.state_size, action_size, hidden_size, 2).to(device)
        # self.critic2 = Critic_CNN(self.state_size, action_size, hidden_size, 1).to(device)
        self.critic1 = Critic(self.CNN_LSTM.outdim, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(self.CNN_LSTM.outdim, action_size, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        # self.critic1_target = Critic_CNN(self.state_size, action_size, hidden_size).to(device)
        self.critic1_target = Critic(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())

        # self.critic2_target = Critic_CNN(self.state_size, action_size, hidden_size).to(device)
        self.critic2_target = Critic(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)

        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)



    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
                self.actor_local.train()
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states, actions_pred.squeeze(0))
        q2 = self.critic2(states, actions_pred.squeeze(0))
        min_Q = torch.min(q1, q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q)).mean()

        # actor_loss = ((alpha * log_pis.cpu() - min_Q)).sum(axis=1).mean()
        return actor_loss.cuda(), log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        # with torch.no_grad():
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)

        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)

        return qs1 - log_pis.detach(), qs2 - log_pis.detach()

    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs

    def learn(self, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences


        batch_size=states.shape[0]
        actions=np.array(actions.cpu())
        #action space 归一化
        # 定义每个维度的最小和最大值,每个环境不一样
        # min_val = np.array([-30, -100]).astype(np.float32)
        # max_val = np.array([30, 100]).astype(np.float32)
        min_val = np.array([-30, -100]).astype(np.float32)
        max_val = np.array([30, 100]).astype(np.float32)

        # 将数据归一化到0到1的范围
        normalized_data = ((actions - min_val) / (max_val - min_val)).astype(np.float32)

        # 将数据归一化到-1到1的范围
        normalized_data = (2 * normalized_data - 1).astype(np.float32)
        actions = torch.from_numpy(normalized_data).to(self.device)

        #--------------------------------Image processing------------------------#

        states = self.CNN_LSTM(states)
        states=states.reshape(batch_size*self.lstm_seq_len,-1)
        with torch.no_grad():
            next_states = self.CNN_LSTM(next_states)
            next_states = next_states.reshape(batch_size*self.lstm_seq_len,-1)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)

        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()


        # Compute alpha loss
        # alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu().sum(axis=1) + self.target_entropy).detach().cpu()).mean().cuda()
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean().cuda()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()



        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
            # Compute Q targets for current states (y_i)

            # Q_targets = rewards.unsqueeze(2) + (self.gamma * (1 - dones) * Q_target_next)
            # Q_targets =rewards.unsqueeze(2) + (self.gamma * (1 - dones.unsqueeze(2)) * Q_target_next)
            Q_targets = rewards.reshape(batch_size*self.lstm_seq_len,1) + (self.gamma * (1 - dones.reshape(batch_size*self.lstm_seq_len,1)) * Q_target_next)

        #
        # # Compute critic loss
        q1 = self.critic1(states, actions.reshape(batch_size*self.lstm_seq_len,-1))
        q2 = self.critic2(states, actions.reshape(batch_size*self.lstm_seq_len,-1))
        q1_train =q1.mean()
        q2_train =q2.mean()

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)

        # # CQL addon
        # random_actions = torch.FloatTensor(q1.shape[0] * 10,q1.shape[1], actions.shape[-1]).uniform_(-1, 1).to(self.device)
        # num_repeat = int(random_actions.shape[0] / states.shape[0])
        #
        # # temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        # # temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat,
        # #                                                                           next_states.shape[1])
        # temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1, 1).view(states.shape[0] * num_repeat,states.shape[1], states.shape[2])
        # temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1, 1).view(next_states.shape[0] * num_repeat,next_states.shape[1], next_states.shape[2])
        #
        # current_pi_values1, current_pi_values2 = self._compute_policy_values(temp_states, temp_states)
        # next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)
        #
        # random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0],num_repeat,states.shape[1], 1)
        # random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0],num_repeat,states.shape[1], 1)
        #
        # current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat,states.shape[1], 1)
        # current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat,states.shape[1], 1)
        #
        # next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat,states.shape[1], 1)
        # next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat,states.shape[1],1)
        #
        # cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        # cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
        #
        # assert cat_q1.shape == (states.shape[0], 3 * num_repeat,states.shape[1], 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        # assert cat_q2.shape == (states.shape[0], 3 * num_repeat,states.shape[1], 1), f"cat_q2 instead has shape: {cat_q2.shape}"
        #
        # cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp,
        #                                      dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
        # cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp,
        #                                      dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight

        random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        num_repeat = int(random_actions.shape[0] / states.shape[0])
        # temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1,1,1).view(states.shape[0] * num_repeat, states.shape[1],states.shape[2],states.shape[3])
        # temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1,1,1).view(next_states.shape[0] * num_repeat, next_states.shape[1],next_states.shape[2],next_states.shape[3])
        temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat,
                                                                                  next_states.shape[1])
        current_pi_values1, current_pi_values2 = self._compute_policy_values(temp_states, temp_states)
        next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)

        random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0],
                                                                                                        num_repeat, 1)
        random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0],
                                                                                                        num_repeat, 1)

        current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)

        next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)

        cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)

        assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"

        cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp,
                                             dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp,
                                             dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight

        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        # critic 2
        self.CNN_LSTM_optimizer.zero_grad()

        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        # CNN_loss.backward()
        self.CNN_LSTM_optimizer.step()




        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)


        return (q1_train.item(),q2_train.item(),actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(),cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item())

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class CQLSAC_CNN(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 tau,
                 hidden_size,
                 learning_rate,
                 temp,
                 with_lagrange,
                 cql_weight,
                 target_action_gap,
                 device
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLSAC_CNN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device

        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.tau = tau
        hidden_size = hidden_size
        learning_rate = learning_rate
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.1], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)
        # self.alpha=torch.tensor([0.5])

        # CQL params
        self.with_lagrange = with_lagrange
        self.temp = temp
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # image processing network
        self.CNN_Simple = CNN_simple(self.state_size, 1).to(device)  # obs_shape,frame_stack
        self.CNN_optimizer = optim.Adam(self.CNN_Simple.parameters(), lr=learning_rate)

        # Actor Network

        self.actor_local = Actor(self.CNN_Simple.outdim, action_size, hidden_size).to(device)
        # self.actor_local = Actor_CNN(self.state_size, action_size, hidden_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Critic Network (w/ Target Network)

        # self.critic1 = Critic_CNN(self.state_size, action_size, hidden_size, 2).to(device)
        # self.critic2 = Critic_CNN(self.state_size, action_size, hidden_size, 1).to(device)
        self.critic1 = Critic(self.CNN_Simple.outdim, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(self.CNN_Simple.outdim, action_size, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        # self.critic1_target = Critic_CNN(self.state_size, action_size, hidden_size).to(device)
        self.critic1_target = Critic(self.CNN_Simple.outdim, action_size, hidden_size).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())

        # self.critic2_target = Critic_CNN(self.state_size, action_size, hidden_size).to(device)
        self.critic2_target = Critic(self.CNN_Simple.outdim, action_size, hidden_size).to(device)

        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)



    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        # state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
                self.actor_local.train()
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        min_Q = torch.min(q1, q2).cpu()
        actor_loss = ((alpha * log_pis.cpu() - min_Q)).mean()
        return actor_loss.cuda(), log_pis

    def _compute_policy_values(self, obs_pi, obs_q):
        # with torch.no_grad():
        actions_pred, log_pis = self.actor_local.evaluate(obs_pi)

        qs1 = self.critic1(obs_q, actions_pred)
        qs2 = self.critic2(obs_q, actions_pred)

        return qs1 - log_pis.detach(), qs2 - log_pis.detach()

    def _compute_random_values(self, obs, actions, critic):
        random_values = critic(obs, actions)
        random_log_probs = math.log(0.5 ** self.action_size)
        return random_values - random_log_probs

    def learn(self, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        batch_size=states.shape[0]
        actions=np.array(actions.cpu())
        #action space 归一化
        # 定义每个维度的最小和最大值
        min_val = np.array([-30, -100]).astype(np.float32)
        max_val = np.array([30, 100]).astype(np.float32)

        # 将数据归一化到0到1的范围
        normalized_data = ((actions - min_val) / (max_val - min_val)).astype(np.float32)

        # 将数据归一化到-1到1的范围
        normalized_data = (2 * normalized_data - 1).astype(np.float32)
        actions = torch.from_numpy(normalized_data).to(self.device)

        #--------------------------------Image processing------------------------#

        states = self.CNN_Simple(states)
        states=states.reshape(batch_size,-1)
        with torch.no_grad():
            next_states = self.CNN_Simple(next_states)
            next_states = next_states.reshape(batch_size,-1)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)

        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        actor_loss.requires_grad_(True)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()


        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean().cuda()
        alpha_loss.requires_grad_(True)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # alpha_loss = - (self.alpha * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean().cuda()


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
            # Compute Q targets for current states (y_i)

            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.squeeze())

        #
        # # Compute critic loss
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q1_train =q1.mean()
        q2_train =q2.mean()

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)

        # # CQL addon
        # random_actions = torch.FloatTensor(q1.shape[0] * 10, actions.shape[-1]).uniform_(-1, 1).to(self.device)
        # num_repeat = int(random_actions.shape[0] / states.shape[0])
        # # temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1,1,1).view(states.shape[0] * num_repeat, states.shape[1],states.shape[2],states.shape[3])
        # # temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1,1,1).view(next_states.shape[0] * num_repeat, next_states.shape[1],next_states.shape[2],next_states.shape[3])
        # temp_states = states.unsqueeze(1).repeat(1, num_repeat, 1).view(states.shape[0] * num_repeat, states.shape[1])
        # temp_next_states = next_states.unsqueeze(1).repeat(1, num_repeat, 1).view(next_states.shape[0] * num_repeat,
        #                                                                           next_states.shape[1])
        # current_pi_values1, current_pi_values2 = self._compute_policy_values(temp_states, temp_states)
        # next_pi_values1, next_pi_values2 = self._compute_policy_values(temp_next_states, temp_states)
        #
        # random_values1 = self._compute_random_values(temp_states, random_actions, self.critic1).reshape(states.shape[0],
        #                                                                                                 num_repeat, 1)
        # random_values2 = self._compute_random_values(temp_states, random_actions, self.critic2).reshape(states.shape[0],
        #                                                                                                 num_repeat, 1)
        #
        # current_pi_values1 = current_pi_values1.reshape(states.shape[0], num_repeat, 1)
        # current_pi_values2 = current_pi_values2.reshape(states.shape[0], num_repeat, 1)
        #
        # next_pi_values1 = next_pi_values1.reshape(states.shape[0], num_repeat, 1)
        # next_pi_values2 = next_pi_values2.reshape(states.shape[0], num_repeat, 1)
        #
        # cat_q1 = torch.cat([random_values1, current_pi_values1, next_pi_values1], 1)
        # cat_q2 = torch.cat([random_values2, current_pi_values2, next_pi_values2], 1)
        #
        # assert cat_q1.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q1 instead has shape: {cat_q1.shape}"
        # assert cat_q2.shape == (states.shape[0], 3 * num_repeat, 1), f"cat_q2 instead has shape: {cat_q2.shape}"
        #
        # cql1_scaled_loss = ((torch.logsumexp(cat_q1 / self.temp,
        #                                      dim=1).mean() * self.cql_weight * self.temp) - q1.mean()) * self.cql_weight
        # cql2_scaled_loss = ((torch.logsumexp(cat_q2 / self.temp,
        #                                      dim=1).mean() * self.cql_weight * self.temp) - q2.mean()) * self.cql_weight
        #
        # cql_alpha_loss = torch.FloatTensor([0.0])
        # cql_alpha = torch.FloatTensor([0.0])
        # if self.with_lagrange:
        #     cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
        #     cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
        #     cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)
        #
        #     self.cql_alpha_optimizer.zero_grad()
        #     cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
        #     cql_alpha_loss.backward(retain_graph=True)
        #     self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss #+ cql1_scaled_loss
        total_c2_loss = critic2_loss #+ cql2_scaled_loss
        total_c1_loss.requires_grad_(True)
        total_c2_loss.requires_grad_(True)
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        # critic 2
        self.CNN_optimizer.zero_grad()

        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        # CNN_loss.backward()
        self.CNN_optimizer.step()




        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)


        # return (actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(),cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item())
        return (actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(),  current_alpha)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
