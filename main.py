import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from math import log, sin

def f(t, y):
    return (2*y/(1+t**2))

def func(t, y):
    return np.log(t) / (np.sin(y/3))**2

t0 = 1
tk = 1.7
y0 = 1

analytic = solve_ivp(f, (t0, tk), [y0])

t = analytic.t
y = analytic.y[0]

print("Розв'язок")
for i in range(len(t)):
    print("t =", t[i], "; y =", y[i])

def method_milna(f, t0, tk, y0, h):
    n = int((tk-t0) / h)
    t = np.linspace(t0, tk, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(3):
        nset1 = h * f(t[i], y[i])
        nset2 = h * f(t[i] + h/2, y[i] + nset1/2)
        nset3 = h * f(t[i] + h/2, y[i] + nset2/2)
        nset4 = h * f(t[i+1], y[i] + nset3)
        y[i+1] = y[i] + (nset1 + 2*nset2 + 2*nset3 + nset4)/6
    for i in range(3, n):
        PFRAy = y[i-3] + 4*h/3 * (2*f(t[i], y[i]) - f(t[i-1], y[i-1]) + 2*f(t[i-2], y[i-2]))
        UFNFy = y[i-1] + h/3 * (f(t[i+1], PFRAy) + 4*f(t[i], y[i]) + f(t[i-1], y[i-1]))
        y[i+1] = UFNFy
    return t, y

h = 0.1
t, y = method_milna(f, t0, tk, y0, h)
print("Розв'язок методом Мілна")
for i in range(len(t)):
    print("t=",t[i],"; y=",y[i])

print("Метод Адамса:")
def adams_method(h):
    t = np.arange(t0, tk+h, h)
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(0, 3):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h/2, y[i] + k1/2)
        k3 = h * f(t[i] + h/2, y[i] + k2/2)
        k4 = h * f(t[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    for i in range(3, len(t)-1):
        y[i+1] = y[i] + h * (55*f(t[i], y[i]) - 59*f(t[i-1], y[i-1]) + 37*f(t[i-2], y[i-2]) -9*f(t[i-3], y[i-3])) / 24
    return t, y

h1 = 0.05
t1, y1 = adams_method(h1)
h2 = 0.1
t2, y2 = adams_method(h2)
print("h1 =", h1)
for i in range(len(t1)):
    print("t=", t1[i], "; y=", y1[i])
print("h2 =",h2)
for i in range(len(t2)):
    print("t=", t2[i], "; y=", y2[i])
er1 = np.linalg.norm(np.abs(y1 - np.interp(t1, t2, y2)))
er2 = np.linalg.norm(np.abs(y2 - np.interp(t2, t1, y1)))
print("Похибка між ", h1, "та", h2, "(обчисл. як норма):")
print("y1:", er1,"y2:",er2)

num_points = 20
h = (tk - t0) / (num_points - 1)
t = np.linspace(t0, tk, num_points)
y = np.zeros(num_points)
y[0] = y0
for i in range(1, num_points):
    y[i] = y[i-1] + h * func(t[i-1], y[i-1])
print("Явний однокроковий метод:")
for i in range(len(t)):
    print("t=",t[i],"; y=",y[i])

def rk23(y, y_prev, t, h):
    nset1 = h * func(t, y)
    nset2 = h * func(t + h, y + nset1)
    return y - y_prev - (nset1 + nset2) / 2
for i in range(1, num_points):
    y[i] = fsolve(rk23, y[i-1], args=(y[i-1], t[i-1], h))
print("Метод Рунге - Кутти 2(3):")
for i in range(len(t)):
    print("t=",t[i],"; y=",y[i])

def rk45(y, y_prev, t, h):
    nset1 = h * func(t, y)
    nset2 = h * func(t + h/2, y + nset1/2)
    nset3 = h * func(t + h/2, y + nset2/2)
    nset4 = h * func(t + h, y + nset3)
    return y - y_prev - (nset1 + 2*nset2 + 2*nset3 + nset4) / 6
for i in range(1, num_points):
    y[i] = fsolve(rk45, y[i-1], args=(y[i-1], t[i-1], h))
print("Метод Рунге - Кутти 4(5):")
for i in range(len(t)):
    print("t=",t[i],"; y=",y[i])

def equation_euler(y, y_prev, t , h):
    return y - y_prev - h * func(t, y)

def equation_rk23(y, y_prev, t , h):
    nset1 = h * func(t, y)
    nset2 = h * func(t + h, y + nset1)
    nset3 = h * func(t + h/2, y + nset1/4 + nset2/4)
    return y - y_prev - (nset1 + 3*nset3) / 4

def equation_rk45(y, y_prev, t, h):
    nset1 = h * func(t, y)
    nset2 = h * func(t + h/2, y + nset1/2)
    nset3 = h * func(t + h/2, y + nset2/2)
    nset4 = h * func(t + h, y + nset3)
    nset5 = h * func(t + h, y + nset1/6 + nset2/3 + nset3/3 + nset4/6)
    return y - y_prev - (nset1 + 4*nset4 + nset5) / 6

y_euler = y_rk23 = y_rk45 = np.zeros(num_points)
y_euler[0] = y_rk23[0] = y_rk45[0] = y0

for i in range(1, num_points):
    y_euler[i] = fsolve(equation_euler, y_euler[i-1], args=(y_euler[i-1], t[i-1], h))
    y_rk23[i] = fsolve(equation_rk23, y_rk23[i-1], args=(y_rk23[i-1], t[i-1], h))
    y_rk45[i] = fsolve(equation_rk45, y_rk45[i-1], args=(y_rk45[i-1], t[i-1], h))

error_euler = np.abs(y - y_euler)
error_rk23 = np.abs(y - y_rk23)
error_rk45 = np.abs(y - y_rk45)

for i in range(1, num_points):
    y_euler[i] = fsolve(equation_euler, y_euler[i-1], args=(y_euler[i-1], t[i-1], h))
    y_rk23[i] = fsolve(equation_rk23, y_rk23[i-1], args=(y_rk23[i-1], t[i-1], h))
    y_rk45[i] = fsolve(equation_rk45, y_rk45[i-1], args=(y_rk45[i-1], t[i-1], h))

error_euler = np.abs(y - y_euler)
error_rk23 = np.abs(y - y_rk23)
error_rk45 = np.abs(y - y_rk45)
plt.figure(figsize=(10, 6))
plt.plot(t, error_euler, label="Метод Ейлера")
plt.plot(t, error_rk23, label="Рунге - Кутти2(3)")
plt.plot(t, error_rk45, label="Рунге - Кутти4(5)")
plt.xlabel("t")
plt.ylabel("Похибка")
plt.show()
print("Максимальні похибки (обчисл. як норма)\nЕйлера:", np.linalg.norm(error_euler), "Рунге - Кутти2(3):",
      np.linalg.norm(error_rk23), "Рунге - Кутти4(5):", np.linalg.norm(error_rk45))
plt.show()
