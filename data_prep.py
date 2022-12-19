import numpy as np
import torch
import os
from torch.utils.data import Dataset
from dynamics import InvertedPendSinForce, InvertedPendUnforced
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
        assert num_examples%2 == 0
        self.data = np.zeros((num_examples,4))
        self.label = torch.tensor([0,0,0.5,0],dtype=torch.float64)
        for i in range(int(len(self.data)/2)):
            #x = 2*np.array(torch.rand((1,4),dtype=torch.float64)) - 1
            #x[0] = 0 
            x = np.zeros((1,4))
            x[0][2] = 0.5 + (2*random.random()-1)/10
            np.copyto(self.data[i],x)
            #np.copyto(self.data[i+1],np.zeros_like(x))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx],dtype=torch.float64), self.label

class SystemIdentDatasetNormed(Dataset):
    def __init__(self, num_examples=100000):
        
        self.data = np.zeros((num_examples,8))
        (m, Mm, Ll,   g, b) = (1, 4, 3, -10, 4)
        max_acc = 200 #adjust to constrain the force on the cart
        max_u = max_acc*(m+Mm)
        max_v = max_acc*1
        max_th = 2*np.pi
        max_om = np.sqrt((m+Mm)/(m*Ll**2)*max_v**2)
        self.scale = np.array([max_u,max_v,max_th,max_om])
        

        self.dx = lambda u, y: InvertedPendUnforced.dx(0, y, m, Mm, Ll, g, b, u)
        for i in range(num_examples):
            x = 2*np.array(torch.rand((1,4),dtype=torch.float64)) - 1
            scaled_x = x*self.scale
            u = scaled_x[0][0]
            v = scaled_x[0][1]
            th = scaled_x[0][2]
            om = scaled_x[0][3]
            dx, dv, dth, dom = self.dx(u, (0,v,th,om))
            dx = dx*0.1
            dv = dv*0.1/scaled_x[0][1]
            dth = dth*0.1/scaled_x[0][2]
            dom = dom*0.1/scaled_x[0][3]
            res = np.array([dx, dv, dth, dom])#*self.ouput_scale
            np.copyto(self.data[i][:4],x)
            np.copyto(self.data[i][-4:],res)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][:4], dtype=torch.float64), torch.tensor(self.data[idx][-4:], dtype=torch.float64)
    
        