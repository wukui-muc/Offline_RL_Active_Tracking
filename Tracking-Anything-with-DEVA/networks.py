import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils_basic import weights_init
# import clip
class CNN_simple(nn.Module):
    def __init__(self, obs_shape, stack_frames):
        super(CNN_simple, self).__init__()
        c,w,h = obs_shape
        # self.conv1 = nn.Conv2d(obs_shape[0], 32, 5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(c, 32, 5, stride=1, padding=2)

        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(stack_frames, c, w, h))
        out = self.forward(dummy_state)
        self.outshape = out.shape
        out = out.view(stack_frames, -1)
        cnn_dim = out.size(-1)
        self.outdim = cnn_dim
        self.apply(weights_init)
        self.train()

    def forward(self, x, batch_size=1, fc=False):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        return x

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class CNN_LSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size,stack_frames,lstm_out,lstm_layer):
        super(CNN_LSTM, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.stack_frames = stack_frames
        self.CNN_Simple = CNN_simple(self.input_shape,self.stack_frames)
        self.cnn_dim = self.CNN_Simple.outdim
        self.lstm_layer=lstm_layer
        self.lstm_out = lstm_out
        # self.outdim = layer_size
        self.outdim=self.lstm_out
        self.lstm = nn.LSTM(input_size=self.cnn_dim, hidden_size=self.lstm_out, num_layers=self.lstm_layer,batch_first=True)

        self.ht = None
        self.ct = None

        # self.head_1 = nn.Linear(self.lstm_out, layer_size)
        #
        # self.ff_1 = nn.Linear(layer_size, layer_size)
    def forward(self, input):
        """

        """


        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        x=self.CNN_Simple(input)
        x=x.reshape(x.shape[0],-1)
        x=x.reshape(batch_size,seq_len,-1)
        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(x,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        x = self.CNN_Simple(input)
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(batch_size, seq_len, -1)
        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(x)

        else:
            x, (ht, ct) = self.lstm(x,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct
class LSTM(nn.Module):
    def __init__(self, input_dim, action_size, hidden_size,lstm_out,lstm_layer):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.action_size = action_size



        self.lstm_layer=lstm_layer
        self.lstm_out = lstm_out
        # self.outdim = layer_size
        self.outdim=self.lstm_out
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_out, num_layers=self.lstm_layer,batch_first=True)

        self.ht = None
        self.ct = None

        # self.head_1 = nn.Linear(self.lstm_out, layer_size)
        #
        # self.ff_1 = nn.Linear(layer_size, layer_size)
    def forward(self, input):
        """

        """


        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        # input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        # x=self.CNN_Simple(input)
        # x=x.reshape(x.shape[0],-1)
        x=input.reshape(batch_size,seq_len,-1)

        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(x,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        x = input.reshape(batch_size, seq_len, -1)
        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(x)

        else:
            x, (ht, ct) = self.lstm(x,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct


class MLP_LSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size,stack_frames,lstm_out,lstm_layer):
        super(MLP_LSTM, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.stack_frames = stack_frames
        self.mlp= torch.nn.Linear(20,128)
        self.mlp_out_dim = 128
        self.lstm_layer=lstm_layer
        self.lstm_out = lstm_out
        # self.outdim = layer_size
        self.outdim=self.lstm_out
        self.lstm = nn.LSTM(input_size=self.mlp_out_dim, hidden_size=self.lstm_out, num_layers=self.lstm_layer,batch_first=True)

        self.ht = None
        self.ct = None

        # self.head_1 = nn.Linear(self.lstm_out, layer_size)
        #
        # self.ff_1 = nn.Linear(layer_size, layer_size)
    def forward(self, input):
        """

        """


        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(batch_size,seq_len,-1)
        input = input.reshape(-1,input.shape[-1])

        x=self.mlp(input)
        x=x.reshape(x.shape[0],-1)
        x=x.reshape(batch_size,seq_len,-1)
        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(x,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(-1)
        x = self.mlp(input)
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(batch_size, seq_len, -1)
        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(x)

        else:
            x, (ht, ct) = self.lstm(x,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        # log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(2, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

