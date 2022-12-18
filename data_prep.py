import numpy as np
import torch
import os
from torch.utils.data import Dataset
from dynamics import InvertedPendSinForce
import random
import pandas as pd
import pickle
class SystemIdentDataset(Dataset):
    def __init__(self, train=True):
        
        if not os.path.exists("./sys_ident_data.pkl"):
            self.generate_trajectories()
        with open('./sys_ident_data.pkl', 'rb') as f:
            df = pd.DataFrame(pickle.load(f))
        
        random_seed = 42
        train_set = df.sample(frac=0.8, random_state=random_seed)
        val_set = df.drop(train_set.index).to_numpy()
        train_set = train_set.to_numpy()
        if train:
            self.data = train_set
        else:
            self.data = val_set
        
    def generate_trajectories(self):
        data = np.zeros((500*7*1000,8))
        count = 0
        for freq in [i/100 for i in range(500)]:
            for theta0 in [i*np.pi/6 for i in range(7)]:
                i_conds = [0, 0, theta0, 0]
                solver = InvertedPendSinForce(t_final=100, t_step=0.1, freq=freq, i_conditions=i_conds)
                t, states = solver.solve()
                for i in range(len(states)-1):
                    s = states[i]
                    u = solver.u(t[i],s)
                    s_next = states[i+1]
                    ds = s_next - s
                    s[0] = u
                    s[2] = s[2]%(2*np.pi)
                    np.copyto(data[count][:4],s)
                    np.copyto(data[count][-4:],ds)
                    count += 1
                    
                    
        with open('./sys_ident_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][:4]), torch.tensor(self.data[idx][-4:])

class ControllerDataset(Dataset):
    def __init__(self, num_examples=100000):
        self.data = np.zeros((num_examples,4))
        
        for i in range(num_examples):
            x = (random.random() - 0.5)*50
            v = (random.random() - 0.5)*10
            th = (random.random() - 0.5)*2*np.pi
            om = (random.random() - 0.5)*20
            np.copyto(self.data[i],np.array([x, v, th, om]))
            
        self.label = torch.tensor([0,0,np.pi,0],dtype=torch.float64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float64), self.label
    
        