import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device,lstm_seq_len,config):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_id=[]
        self.st_id =[]
        self.lstm_seq_len = lstm_seq_len
        self.input_type = config.input_type


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(
            state,
            action,
            reward,
            next_state,
            done)
        self.memory.append(e)
    
    def sample(self):
        # for lstm,seq frame input
        if ('cnn'in self.input_type.lower() and 'lstm' in self.input_type.lower()) or 'mlp' in self.input_type.lower() or 'clip' in self.input_type.lower():
            states_test = []
            actions_test = []
            rewards_test = []
            next_states_test = []
            done_test = []
            self.batch_id = []
            self.st_id = []
            for i in range(self.batch_size):
                self.st_id.append(random.randint(0, 499- self.lstm_seq_len))
                self.batch_id.append(random.randint(0, int(len(self.memory)/499))-1)

            for i in range(self.batch_size):
                experiences_test = []
                for j in range(0, self.lstm_seq_len):
                    experiences_test.append(self.memory[self.batch_id[i]*499+self.st_id[i]])
                    self.st_id[i]+=1


                # states_test.append(np.concatenate([e.state for e in experiences_test if e is not None]))
                states_test.append([e.state for e in experiences_test if e is not None])
                # actions_test.append(np.array([e.action for e in experiences_test if e is not None]))
                actions_test.append([e.action for e in experiences_test if e is not None])
                # rewards_test.append(np.array([e.reward for e in experiences_test if e is not None]))
                rewards_test.append([e.reward for e in experiences_test if e is not None])
                next_states_test.append([e.next_state for e in experiences_test if e is not None])
                # done_test.append(np.array([e.done for e in experiences_test if e is not None]))
                done_test.append([e.done for e in experiences_test if e is not None])

            states=torch.stack([torch.stack(s) for s in states_test])
            actions=torch.stack([torch.stack(a) for a in actions_test])
            rewards=torch.stack([torch.stack(r) for r in rewards_test])
            next_states=torch.stack([torch.stack(n) for n in next_states_test])
            dones=torch.stack([torch.stack(d) for d in done_test])


        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
