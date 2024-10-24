import numpy as np
import matplotlib.pyplot as plt

# Create a grid of x1 (indoor bulbs) and x2 (outdoor bulbs)
x1 = np.linspace(0, 50, 200)  # indoor bulbs
x2_t1 = (150 - 4 * x1) / 2  # from the constraint 4x1 + 2x2 <= 150
x2_t2 = (50 - 2 * x1) / 3  # from the constraint 2x1 + 3x2 <= 50

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot the constraints
plt.plot(x1, x2_t1, label=r'$4x_1 + 2x_2 \leq 150$', color='blue')
plt.plot(x1, x2_t2, label=r'$2x_1 + 3x_2 \leq 50$', color='orange')

# Fill feasible region
plt.fill_between(x1, 0, np.minimum(x2_t1, x2_t2), where=(x2_t1 >= 0) & (x2_t2 >= 0),
                 color='lightgreen', alpha=0.5, label='Feasible Region')

# Axes and labels
plt.xlim(0, 50)
plt.ylim(0, 30)
plt.xlabel('Indoor Bulbs (x1)')
plt.ylabel('Outdoor Bulbs (x2)')
plt.title('Profit Maximization for Bulb Production')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.text(0, 15, 'Feasible Region', fontsize=12, verticalalignment='center')

# Show the plot
plt.show()