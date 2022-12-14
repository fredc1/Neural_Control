import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle


n_steps = 1000
theta = [np.pi for i in range(n_steps)]
x = [i/50 for i in range(n_steps)]
theta0 = theta[0]
x0 = x[0]
cart_w = 1
cart_h = 0.5
L = 3

fig, ax = plt.subplots()
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])

# Create a line to represent the pendulum rod
rod, = plt.plot([x[0], x[0] + L*np.sin(np.pi - theta[0])], [cart_h, L*np.cos(np.pi - theta[0])], color="black")

# Create a point to represent the pendulum bob
bob, = plt.plot(x[0] + L*np.sin(np.pi - theta[0]), cart_h + L*np.cos(np.pi - theta[0]), "o", color="red")

rect = Rectangle((x[0] - cart_w/2, 0), cart_w, cart_h, color='r')
ax.add_patch(rect)

# Define the update function for the animation
def update(i):
    # Update the position of the pendulum bob
    bob.set_data(x[i] + L*np.sin(np.pi - theta[i]), cart_h + L*np.cos(np.pi - theta[i]))

    # Update the position of the pendulum rod
    rod.set_data([x[i], x[i] + L*np.sin(np.pi - theta[i])], [cart_h, cart_h + L*np.cos(np.pi - theta[i])])
    rect.set_x(x[i] - cart_w/2)

    return rod, bob, rect

# Create the animation using the update function and the lists of theta and omega values
ani = animation.FuncAnimation(plt.gcf(), update, frames=range(n_steps), interval=10)

# Show the animation
plt.show()
