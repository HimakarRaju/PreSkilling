import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from sympy import symbols, Eq, solve, N

# Define the variables
x1, x2 = symbols('x1 x2')

# Define the two non-linear equations
eq1 = Eq(x1**2 + x2, 10)  # x1^2 + x2 = 10
eq2 = Eq(x1 + x2**2, 5)   # x1 + x2^2 = 5

# Solve the system of equations symbolically
solutions = solve([eq1, eq2], (x1, x2))
print("Solutions to the system of equations (including complex ones): ", solutions)

# Filter out complex solutions
real_solutions = []
for sol in solutions:
    real_x1 = sol[0].as_real_imag()[0]
    real_x2 = sol[1].as_real_imag()[0]
    if real_x1.is_real and real_x2.is_real:
        real_solutions.append((float(N(real_x1)), float(N(real_x2))))

print("Filtered real solutions: ", real_solutions)

# Plotting the equations
x_vals = np.linspace(-5, 5, 400)
y_vals_eq1 = 10 - x_vals**2  # Rearranged eq1: x2 = 10 - x1^2
y_vals_eq2 = np.sqrt(5 - x_vals)  # Rearranged eq2: x2 = sqrt(5 - x1)
y_vals_eq2_neg = -np.sqrt(5 - x_vals)  # For negative part of sqrt

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals_eq1, label=r'$x_1^2 + x_2 = 10$', color='blue')
plt.plot(x_vals, y_vals_eq2, label=r'$x_1 + x_2^2 = 5$', color='green')
plt.plot(x_vals, y_vals_eq2_neg, color='green')  # For negative square root part

# Set plot limits
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Mark the intersection points (real solutions)
for sol in real_solutions:
    plt.plot(sol[0], sol[1], 'ro', label=f'Solution: {sol}')

# Identify the most optimal solution (here, the one closest to the origin)
if real_solutions:
    optimal_solution = min(real_solutions, key=lambda s: np.sqrt(s[0]**2 + s[1]**2))
    plt.plot(optimal_solution[0], optimal_solution[1], 'yo', label=f'Optimal Solution: {optimal_solution}', markersize=10)

# Adding labels and title
plt.title('Non-linear Equations: Solutions and Plot')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend(loc='best')

# Add interactivity with mplcursors
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"x1={sel.target[0]:.2f}, x2={sel.target[1]:.2f}"))

# Show the plot
plt.grid(True)
plt.show()
