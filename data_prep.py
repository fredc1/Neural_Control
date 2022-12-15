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
        data = np.zeros((250*7*1000,8))
        count = 0
        for freq in [2*i/100 for i in range(250)]:
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
        