import numpy as np
from scipy.integrate import solve_ivp
import math
import torch
import pickle

class PendDataStreamer:
    def __init__(self, t_final, t_step) -> None:
        self.t_span = (0,t_final)
        self.t_eval = [i*t_step for i in range(int(t_final/t_step))]
        self.t_final = t_final
        self.t_step = t_step
    def get_data(self) -> tuple:
        print("unimplemented, should return x and theta values for animation")
    
    def get_t_vals(self):
        return self.t_eval

class InvertedPendUnforced(PendDataStreamer):
    
    def __init__(self, t_final, t_step, i_conditions=[0, 0, 5*math.pi/6, 0]) -> None:
        super().__init__(t_final, t_step)
                   #(m, M, L,   g, b, u)
        self.args = (1, 4, 3, -10, 4)
                         #[x, v,   theta, omega]
        self.init_conds = i_conditions
        self.dx = lambda t, y, m, M, L, g, b: InvertedPendUnforced.dx(t, y, m, M, L, g, b, self.u(t, y))
        self.force_scale = 2000
    def solve(self) -> tuple:
         sol = solve_ivp(self.dx, t_span=self.t_span, y0=self.init_conds, method='RK45', args=self.args, t_eval=self.t_eval)

         return (sol.t.T, sol.y.T)

    def get_data(self):
        (t, state) = self.solve()
        x_vals = list(state.T[0])
        theta_vals = list(state.T[2])
        time_vals = list(t)
        return (time_vals, x_vals, theta_vals)

    def u(self, t, y):
        (x,v,theta,omega) = y
        return 0

    @staticmethod
    def dx(t, y, m, M, L, g, b, u):
        x, v, th, om = y
        S = math.sin(th)
        C = math.cos(th)
        D = m*L*L*(M+m*(1-C**2))
        dx = v
        dv = (1/D) * (-1*m**2 * L*2 * g *C *S + m*L**2 * (m*L*om**2*S - b*v) + m*L*L*(1/D)*u)
        dth = om
        dom = (1/D) * ((M+m) * m*g*L *S - m*L*C *(m*L*om**2 * S - b*v) - m*L*C*(1/D)*u)
        return [dx, dv, dth, dom]


class InvertedPendSinForce(InvertedPendUnforced):
    
    def __init__(self, t_final, t_step, freq, i_conditions=None) -> None:
        if i_conditions is not None:
            super(self.__class__, self).__init__(t_final, t_step, i_conditions)
        else:
            super(self.__class__, self).__init__(t_final, t_step)
        self.freq = freq
        self.umax = self.force_scale
    def u(self, t, y):
        wait_period = self.t_final/4
        
        
        if (int(t/wait_period)%2 == 0):
            return self.force_scale*np.sin(self.freq*t)
        else:
            return 0

class InvertedPendNNControl(InvertedPendUnforced):
    
    def __init__(self, t_final, t_step, i_conditions=None) -> None:
        if i_conditions is not None:
            super(self.__class__, self).__init__(t_final, t_step, i_conditions)
        else:
            super(self.__class__, self).__init__(t_final, t_step)
        with open('./controller_model2.pkl', 'rb') as f:
            self.controller = pickle.load(f)
        
        (m, Mm, Ll,   g, b) = (1, 4, 3, -10, 4)
        max_acc = 200 #adjust to constrain the force on the cart
        max_u = max_acc*(m+Mm)
        max_v = max_acc*1
        max_th = 2*np.pi
        max_om = np.sqrt((m+Mm)/(m*Ll**2)*max_v**2)
        self.scale = [max_u,max_v,max_th,max_om]
        
    def u(self, t, y):
        (x,v,th,om) = y
        
        max_u,max_v,max_th,max_om = self.scale
        input = torch.zeros((1,4),dtype=torch.float64)
        input[0][0] = x
        input[0][1] = v/max_v
        input[0][2] = th/max_th
        input[0][3] = om/max_om
        
        return self.controller(input).item()*max_u
        

