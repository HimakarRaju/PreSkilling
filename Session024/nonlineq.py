import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

x1,x2 = symbols('x1 x2')

#Define two equations that are non -linear
eq1 = Eq(x1**2 + x2,10)
eq2 = Eq(x1 +x2**2,5)

solutions= solve([eq1,eq2], (x1,x2))
real_solutions = [(sol[0].evalf(),sol[1].evalf()) for sol in solutions if sol[0].is_real and sol[1].is_real]
print ("Real Solutions to the system of euqations:",real_solutions)

print("Possible solutions to the equations given ",solutions)

x_vals =np.linspace(-5,5,400)
y_vals_eq1 = 10-x_vals**2
y_vals_eq2_pos = np.sqrt(np.maximum(5-x_vals,0))

y_vals_eq2_neg = np.sqrt(np.maximum(5-x_vals,0))



plt.figure(figsize=(10,6))
plt.plot(x_vals,y_vals_eq1,label=r'$x_1^2 + x_2 = 10$',color='blue')
plt.plot(x_vals,-y_vals_eq2_neg,label=r'$x_1 + x_2^2 = 5$',color='green')
plt.plot(x_vals,y_vals_eq2_pos,label=r'$x_1 + x_2^2 = 5$',color='red')

plt.xlim(-5,5)
plt.ylim(-5,5)

for sol in real_solutions:
    plt.plot(sol[0],sol[1],'ro',markersize=8)
    #Show the equation here
    
    plt.text(sol[0],sol[1] + 0.5 ,'Click for equation', fontsize=10,bbox=dict(boxstyle='round,pad=0.5',edgecolor='black',facecolor='lightyellow'), ha='center',va='center')

plt.title = ('Non Linear Python Code : Solutions and Plot')
plt.xlabel=(r'$x_1$')
plt.ylabel=(r'$x_2$')

plt.axhline(0,color='black',linewidth=0.5)
plt.axvline(0,color='black',linewidth=0.5)
plt.legend(loc='best')


def onclick(event):
    for sol in real_solutions:
        if (sol[0] - 0.2 < event.xdata < sol[0] + 0.2 and sol[1] - 0.2 < event.ydata < sol[1] + 0.2 ):
           print(f"Equation at {sol[0]},{sol[1]}: ")
           print(f"{eq1},{eqn2}")
plt.gcf().canvas.mpl_connect('button_press_event',onclick)



plt.grid(True)
plt.show()
