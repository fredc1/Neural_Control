import numpy as np
from scipy.integrate import solve_ivp
import math

class DataStreamer:
    #statics

    def __init__(self) -> None:
        pass

    def get_data() -> list:
        pass



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set the initial position of the ball and rectangle
ball_x = 0
rect_x = 0

# Set the figure and axes for the plot
fig, ax = plt.subplots()
cart = plt.Rectangle((0, 0), 2, 2)
# Create the ball and rectangle objects
ball, = ax.plot(ball_x, 0, 'bo', markersize=20)
rect = ax.bar(rect_x, 5, width=10, color='r')
ax.add_artist(cart)
# Set the limits for the x and y axes
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)

# Function to update the plot at each time step
def update(i):
    # Move the ball and rectangle to the right by 1 unit
    ball.set_xdata(ball_x + i)
    #rect.center = (rect_x + i,0)
    cart.center = (i,i)

# Create the animation using the update function
animation = FuncAnimation(fig, update, frames=range(100),interval=1)
plt.show()
