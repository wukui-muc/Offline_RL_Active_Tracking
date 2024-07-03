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
                    # experiences_test.append(self.memory[self.st_id])

                    self.st_id[i]+=1
                    # if self.st_id >= len(self.memory):
                    #     self.st_id=0
                        # if self.st_id[i]>=499:
                    #     self.st_id[i]=random.randint(0, 499- self.lstm_seq_len)
                    #     self.batch_id[i]=random.randint(0, len(self.memory)/499-1)

                # states_test.append(np.concatenate([e.state for e in experiences_test if e is not None]))
                states_test.append([e.state for e in experiences_test if e is not None])
                # actions_test.append(np.array([e.action for e in experiences_test if e is not None]))
                actions_test.append([e.action for e in experiences_test if e is not None])
                # rewards_test.append(np.array([e.reward for e in experiences_test if e is not None]))
                rewards_test.append([e.reward for e in experiences_test if e is not None])
                next_states_test.append([e.next_state for e in experiences_test if e is not None])
                # done_test.append(np.array([e.done for e in experiences_test if e is not None]))
                done_test.append([e.done for e in experiences_test if e is not None])

            #
            # states = torch.from_numpy(np.stack([s for s in states_test])).float().to(self.device)
            # actions = torch.from_numpy(np.stack([s for s in actions_test])).float().to(self.device)
            # rewards = torch.from_numpy(np.stack([s for s in rewards_test])).float().to(self.device)
            # next_states = torch.from_numpy(np.stack([s for s in next_states_test])).float().to(self.device)
            # dones = torch.from_numpy(np.stack([s for s in done_test])).float().to(self.device)

            states=torch.stack([torch.stack(s) for s in states_test])
            actions=torch.stack([torch.stack(a) for a in actions_test])
            rewards=torch.stack([torch.stack(r) for r in rewards_test])
            next_states=torch.stack([torch.stack(n) for n in next_states_test])
            dones=torch.stack([torch.stack(d) for d in done_test])
        elif 'cnn' in self.input_type.lower() and 'online' not in self.input_type.lower():
            """Randomly sample a batch of experiences from memory."""
            experiences = random.sample(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

            # states = torch.stack([e.state for e in experiences if e is not None])
            # actions = torch.stack([e.action for e in experiences if e is not None])
            # rewards = torch.stack([e.reward for e in experiences if e is not None])
            # next_states = torch.stack([e.next_state for e in experiences if e is not None])
            # dones = torch.stack([e.done for e in experiences if e is not None])
        elif 'online'  in self.input_type.lower():
            experiences = [self.memory[-1]]
            states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
                self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
                self.device)

        else:
            print('specified network structure')


        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
