import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Budget constraint function
def budget_constraint(x):
    return (12 - 2*x) / 3  # Derived from 2x + 3y <= 12

# Create a grid for integer points
x_values = np.arange(0, 7)  # Possible integer values for candies
y_values = np.arange(0, 5)  # Possible integer values for chocolates
X, Y = np.meshgrid(x_values, y_values)

# Objective function: maximizing total packs (x + y)
Z = X + Y

# Set up the plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Plot the budget line
x = np.linspace(0, 6, 100)
y = budget_constraint(x)
ax.plot(x, y, label='Budget Constraint: $2x + 3y \leq 12$', color='blue')

# Plot integer points
ax.scatter(X, Y, c='green', label='Integer Solutions', marker='o')

# Set plot limits and labels
ax.set_xlim(0, 6)
ax.set_ylim(0, 5)
ax.set_xlabel('Packs of Candies (x)')
ax.set_ylabel('Packs of Chocolates (y)')
ax.set_title('Birthday party Budget Explanation')
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)
ax.legend()

# Add a slider to adjust the budget
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Budget', 5, 20, valinit=12)

# Update function for the slider
def update(val):
    budget = slider.val
    new_y = (budget - 2*x) / 3
    ax.clear()
    ax.plot(x, new_y, label=f'Budget Constraint: $2x + 3y \leq {budget}$', color='blue')
    ax.scatter(X, Y, c='green', label='Integer Solutions', marker='o')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5)
    ax.set_xlabel('Packs of Candies (x)')
    ax.set_ylabel('Packs of Chocolates (y)')
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.legend()
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
