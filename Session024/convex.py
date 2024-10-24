import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Function for convex optimization
def convex_function(x):
    return (x - 2)**2 + 3  # A simple quadratic function

# Set up the plot for convex optimization
x = np.linspace(-1, 5, 400)
y = convex_function(x)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
line, = ax.plot(x, y, label='Convex Function: $(x-2)^2 + 3$')
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Convex Optimization Example')
ax.legend()
ax.set_ylim(0, 10)

# Add a slider to find the minimum point
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'x', -1, 5, valinit=2)

# Update function for the slider
def update(val):
    x_val = slider.val
    min_point = convex_function(x_val)
    line.set_ydata(y)
    ax.scatter(x_val, min_point, color='red')  # Mark the minimum point
    ax.set_title(f'Convex Optimization Quadratic Function - Min at x={x_val:.2f}')
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
