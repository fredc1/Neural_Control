import numpy as np
from scipy.integrate import solve_ivp
import math

class PendDataStreamer:
    #statics

    def __init__(self, L) -> None:
        self.L = L

    def get_data(self) -> tuple:
        print("unimplemented")

class InvertedPendUnforced(PendDataStreamer):
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

    def __init__(self, L) -> None:
        super().__init__(L)
        #(m, M, L, g, b, u)
        self.args = (1, 4, L, -10, 4, 0)
    
    def solve(self) -> tuple:
         sol = solve_ivp(InvertedPendUnforced.dx, t_span=(0,50), y0=[0, 0, 5*math.pi/6, 0], method='RK23', args=self.args)
         #print(sol.y.T)
         #print(len(sol.y.T))
         return (sol.t.T, sol.y.T)

    def get_data(self):
        (t, state) = self.solve()
        x_vals = list(state.T[0])
        theta_vals = list(state.T[2])
        time_vals = list(t)
        return (time_vals, x_vals, theta_vals)
