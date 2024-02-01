"""
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.ticker import FixedFormatter,FixedLocator
pi = math.pi

t = 0
for t in range(9):
    l = 3  # длина закрепления
    n = 2  # период косинусов
    nshtrix = 2  # период синусов
    #t = 1  # момент времени
    v0 = 10  # постоянная скорость движения полотна
    c = 100  # скорость распространения колебаний в покоящемся полотне
    m = 200  # удельная масса полотна на единицу площади
    D = 300  # величина жесткости на изгиб

    pinl = (pi * n) / l
    pinshtrixl = (pi * nshtrix) / l
    tetaA = (1 + (((D) / (m * (c ** 2))) * ((pinl ** 2)))) ** (1 / 2)
    tetaB = (1 + (((D) / (m * (c ** 2))) * ((pinshtrixl ** 2)))) ** (1 / 2)

    chastiargumentcos1 = pinl * t * ((c * tetaA) - v0)
    chastiargumentcos2 = pinl * t * ((c * tetaA) + v0)
    chastiargumentsin1 = pinshtrixl * t * ((c * tetaB) - v0)
    chastiargumentsin2 = pinshtrixl * t * ((c * tetaB) + v0)
    mnogitelcos1 = (1 + (v0) / (c * tetaA))
    mnogitelcos2 = (1 - (v0) / (c * tetaA))
    mnogitelsinusov = ((1) / ((pinshtrixl) * (c) * (tetaB)))

    x = np.arange(-2 * pi, 2 * pi, 0.1)


    def function(x):
        return l ** 8 / 315 + 480 * ((l ** 8) / (pi ** 8)) * (
                    (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                    * ((mnogitelcos1 * np.cos(chastiargumentcos1 + pinl * x)
                        + mnogitelcos2 * np.cos(chastiargumentcos2 - pinl * x))
                       +
                       ((mnogitelsinusov) * (np.sin(chastiargumentsin1 + pinshtrixl * x)
                                             + np.sin(chastiargumentsin2 - pinshtrixl * x))))
                    )


    z = function(x)

    fig, ax = plt.subplots(1, 1)
    plt.plot(x, z)

    ax.xaxis.set_major_locator(FixedLocator([-4*pi,-2 * pi, -7 * pi / 4, -3 * pi / 2, -5 * pi / 4, -pi, -3 * pi / 4,
                                             -pi / 2, -pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4,
                                             3 * pi / 2, 7 * pi / 4, 2 * pi, 4*pi]))
    ax.xaxis.set_major_formatter(FixedFormatter([r'-$4\pi$',r'-$2\pi$', r'-$\frac{7}{4}\pi$',
                                                 r'-$\frac{3}{2}\pi$', r'-$\frac{5}{4}\pi$', r'-$\pi$',
                                                 r'-$\frac{3}{4}\pi$',
                                                 r'-$\frac{1}{2}\pi$', r'-$\frac{1}{4}\pi$', '0', r'$\frac{1}{4}\pi$',
                                                 r'$\frac{1}{2}\pi$',
                                                 r'$\frac{3}{4}\pi$', r'$\pi$', r'$\frac{5}{4}\pi$',
                                                 r'$\frac{3}{2}\pi$', r'$\frac{7}{4}\pi$', r'$2\pi$', r'$4\pi$']))

    ax.grid()
    ax.text(1, 1, 't = ', size=15)
    ax.text(2, 1, t, size=15)
    ax.vlines(0, 0, 40, color='black')
    ax.hlines(0, -2*pi, 2*pi, color='black')


plt.show()

"""












"""

import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.ticker import FixedFormatter,FixedLocator
pi = math.pi

for n in range(10):
    n = n + 1
    l = 3  # длина закрепления
    # n = 2 #период косинусов
    nshtrix = 2  # период синусов
    t = 1  # момент времени
    v0 = 10  # постоянная скорость движения полотна
    c = 100  # скорость распространения колебаний в покоящемся полотне
    m = 200  # удельная масса полотна на единицу площади
    D = 300  # величина жесткости на изгиб

    pinl = (pi * n) / l
    pinshtrixl = (pi * nshtrix) / l
    tetaA = (1 + (((D) / (m * (c ** 2))) * ((pinl ** 2)))) ** (1 / 2)
    tetaB = (1 + (((D) / (m * (c ** 2))) * ((pinshtrixl ** 2)))) ** (1 / 2)

    chastiargumentcos1 = pinl * t * ((c * tetaA) - v0)
    chastiargumentcos2 = pinl * t * ((c * tetaA) + v0)
    chastiargumentsin1 = pinshtrixl * t * ((c * tetaB) - v0)
    chastiargumentsin2 = pinshtrixl * t * ((c * tetaB) + v0)
    mnogitelcos1 = (1 + (v0) / (c * tetaA))
    mnogitelcos2 = (1 - (v0) / (c * tetaA))
    mnogitelsinusov = ((1) / ((pinshtrixl) * (c) * (tetaB)))

    x = np.arange(-2 * pi, 2 * pi, 0.1)


    def function(x):
        return l ** 8 / 315 + 480 * ((l ** 8) / (pi ** 8)) * (
                    (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                    * ((mnogitelcos1 * np.cos(chastiargumentcos1 + pinl * x)
                        + mnogitelcos2 * np.cos(chastiargumentcos2 - pinl * x))
                       +
                       ((mnogitelsinusov) * (np.sin(chastiargumentsin1 + pinshtrixl * x)
                                             + np.sin(chastiargumentsin2 - pinshtrixl * x))))
                    )


    z = function(x)

    fig, ax = plt.subplots(1, 1)
    plt.plot(x, z)

    ax.xaxis.set_major_locator(FixedLocator([-2 * pi, -7 * pi / 4, -3 * pi / 2, -5 * pi / 4, -pi, -3 * pi / 4,
                                             -pi / 2, -pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4,
                                             3 * pi / 2, 7 * pi / 4, 2 * pi]))
    ax.xaxis.set_major_formatter(FixedFormatter([r'-$2\pi$', r'-$\frac{7}{4}\pi$',
                                                 r'-$\frac{3}{2}\pi$', r'-$\frac{5}{4}\pi$', r'-$\pi$',
                                                 r'-$\frac{3}{4}\pi$',
                                                 r'-$\frac{1}{2}\pi$', r'-$\frac{1}{4}\pi$', '0', r'$\frac{1}{4}\pi$',
                                                 r'$\frac{1}{2}\pi$',
                                                 r'$\frac{3}{4}\pi$', r'$\pi$', r'$\frac{5}{4}\pi$',
                                                 r'$\frac{3}{2}\pi$', r'$\frac{7}{4}\pi$', r'$2\pi$']))

    ax.grid()
    ax.text(1, 1, 'n = ', size=15)
    ax.text(2, 1, n, size=15)
    ax.vlines(0, 0, 40, color='black')
    ax.hlines(0, -2 * pi, 2 * pi, color='black')



plt.show()
"""

"""
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.ticker import FixedFormatter,FixedLocator

pi = math.pi
sum = 0
i = 200 # количество слагаемых в сумме
l = 2  # длина закрепления
t = 10  # момент времен
v0 = 6  # постоянная скорость движения полотна
c = 5  # скорость распространения колебаний в покоящемся полотне
m = 9  # удельная масса полотна на единицу площади
D = 500  # величина жесткости на изгиб

while t >= 0:
    for n in range(i):
        n = n + 1

        pi_n_l = (pi * n) / l
        teta = (1 + (((D) / (m * (c ** 2))) * ((pi_n_l ** 2)))) ** (1 / 2)

        chasti_argument_cos_i_sin_1 = pi_n_l * t * ((c * teta) - v0)
        chasti_argument_cos_i_sin_2 = pi_n_l * t * ((c * teta) + v0)

        mnogitel_cos_1 = 1 + (v0 / (c * teta))
        mnogitel_cos_2 = 1 - (v0 / (c * teta))
        mnogitel_sinusov = 1 / (pi_n_l * c * teta)

        x = np.arange(0, l, 0.0001)

        U_x_t = 480 * ((l ** 8) / (pi ** 8)) * (
                (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                * ((mnogitel_cos_1 * np.cos(chasti_argument_cos_i_sin_1 + pi_n_l * x)
                    + mnogitel_cos_2 * np.cos(chasti_argument_cos_i_sin_2 - pi_n_l * x))
                   +
                   ((mnogitel_sinusov) * (np.sin(chasti_argument_cos_i_sin_1 + pi_n_l * x)
                                          + np.sin(chasti_argument_cos_i_sin_2 - pi_n_l * x))))
        )
        sum = sum + U_x_t

    sum = sum + l ** 8 / 315


    def function(x):
        return sum


    z = function(x)

    fig, ax = plt.subplots(1, 1)
    plt.plot(x, z)

    ax.xaxis.set_major_locator(FixedLocator([0, l / 2, l]))
    ax.xaxis.set_major_formatter(FixedFormatter(['0', r'$\frac{1}{2}\l$', r'l', ]))
    ax.grid()
    ax.text(1, 10, 't = ', size=10)
    ax.text(1.2, 10, t, size=10)
    ax.text(1, 7, 'n = ', size=10)
    ax.text(1.2, 7, n, size=10)

    t = t - 0.1

plt.show()
"""








"""
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.ticker import FixedFormatter,FixedLocator
pi = math.pi

n = 1
l = 3  # длина закрепления
# n = 2 #период косинусов
nshtrix = 2  # период синусов
t = 1  # момент времени
v0 = 10  # постоянная скорость движения полотна
c = 100  # скорость распространения колебаний в покоящемся полотне
m = 200  # удельная масса полотна на единицу площади
D = 300  # величина жесткости на изгиб

pinl = (pi * n) / l
pinshtrixl = (pi * nshtrix) / l
tetaA = (1 + (((D) / (m * (c ** 2))) * ((pinl ** 2)))) ** (1 / 2)
tetaB = (1 + (((D) / (m * (c ** 2))) * ((pinshtrixl ** 2)))) ** (1 / 2)

chastiargumentcos1 = pinl * t * ((c * tetaA) - v0)
chastiargumentcos2 = pinl * t * ((c * tetaA) + v0)
chastiargumentsin1 = pinshtrixl * t * ((c * tetaB) - v0)
chastiargumentsin2 = pinshtrixl * t * ((c * tetaB) + v0)
mnogitelcos1 = (1 + (v0) / (c * tetaA))
mnogitelcos2 = (1 - (v0) / (c * tetaA))
mnogitelsinusov = ((1) / ((pinshtrixl) * (c) * (tetaB)))

x = np.arange(-2 * pi, 2 * pi, 0.1)

reh1 =  480 * ((l ** 8) / (pi ** 8)) * (
                    (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                    * ((mnogitelcos1 * np.cos(chastiargumentcos1 + pinl * x)
                        + mnogitelcos2 * np.cos(chastiargumentcos2 - pinl * x))
                       +
                       ((mnogitelsinusov) * (np.sin(chastiargumentsin1 + pinshtrixl * x)
                                             + np.sin(chastiargumentsin2 - pinshtrixl * x))))
                    )




n = 2
l = 3  # длина закрепления
# n = 2 #период косинусов
nshtrix = 2  # период синусов
t = 1  # момент времени
v0 = 10  # постоянная скорость движения полотна
c = 100  # скорость распространения колебаний в покоящемся полотне
m = 200  # удельная масса полотна на единицу площади
D = 300  # величина жесткости на изгиб

pinl = (pi * n) / l
pinshtrixl = (pi * nshtrix) / l
tetaA = (1 + (((D) / (m * (c ** 2))) * ((pinl ** 2)))) ** (1 / 2)
tetaB = (1 + (((D) / (m * (c ** 2))) * ((pinshtrixl ** 2)))) ** (1 / 2)

chastiargumentcos1 = pinl * t * ((c * tetaA) - v0)
chastiargumentcos2 = pinl * t * ((c * tetaA) + v0)
chastiargumentsin1 = pinshtrixl * t * ((c * tetaB) - v0)
chastiargumentsin2 = pinshtrixl * t * ((c * tetaB) + v0)
mnogitelcos1 = (1 + (v0) / (c * tetaA))
mnogitelcos2 = (1 - (v0) / (c * tetaA))
mnogitelsinusov = ((1) / ((pinshtrixl) * (c) * (tetaB)))

x = np.arange(-2 * pi, 2 * pi, 0.1)

reh2 = 480 * ((l ** 8) / (pi ** 8)) * (
                    (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                    * ((mnogitelcos1 * np.cos(chastiargumentcos1 + pinl * x)
                        + mnogitelcos2 * np.cos(chastiargumentcos2 - pinl * x))
                       +
                       ((mnogitelsinusov) * (np.sin(chastiargumentsin1 + pinshtrixl * x)
                                             + np.sin(chastiargumentsin2 - pinshtrixl * x))))
                    )


n = 3
l = 3  # длина закрепления
# n = 2 #период косинусов
nshtrix = 2  # период синусов
t = 1  # момент времени
v0 = 10  # постоянная скорость движения полотна
c = 100  # скорость распространения колебаний в покоящемся полотне
m = 200  # удельная масса полотна на единицу площади
D = 300  # величина жесткости на изгиб

pinl = (pi * n) / l
pinshtrixl = (pi * nshtrix) / l
tetaA = (1 + (((D) / (m * (c ** 2))) * ((pinl ** 2)))) ** (1 / 2)
tetaB = (1 + (((D) / (m * (c ** 2))) * ((pinshtrixl ** 2)))) ** (1 / 2)

chastiargumentcos1 = pinl * t * ((c * tetaA) - v0)
chastiargumentcos2 = pinl * t * ((c * tetaA) + v0)
chastiargumentsin1 = pinshtrixl * t * ((c * tetaB) - v0)
chastiargumentsin2 = pinshtrixl * t * ((c * tetaB) + v0)
mnogitelcos1 = (1 + (v0) / (c * tetaA))
mnogitelcos2 = (1 - (v0) / (c * tetaA))
mnogitelsinusov = ((1) / ((pinshtrixl) * (c) * (tetaB)))

x = np.arange(-2 * pi, 2 * pi, 0.1)

reh3 = 480 * ((l ** 8) / (pi ** 8)) * (
                    (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                    * ((mnogitelcos1 * np.cos(chastiargumentcos1 + pinl * x)
                        + mnogitelcos2 * np.cos(chastiargumentcos2 - pinl * x))
                       +
                       ((mnogitelsinusov) * (np.sin(chastiargumentsin1 + pinshtrixl * x)
                                             + np.sin(chastiargumentsin2 - pinshtrixl * x))))
                    )
n = 4
l = 3  # длина закрепления
# n = 2 #период косинусов
nshtrix = 2  # период синусов
t = 1  # момент времени
v0 = 10  # постоянная скорость движения полотна
c = 100  # скорость распространения колебаний в покоящемся полотне
m = 200  # удельная масса полотна на единицу площади
D = 300  # величина жесткости на изгиб

pinl = (pi * n) / l
pinshtrixl = (pi * nshtrix) / l
tetaA = (1 + (((D) / (m * (c ** 2))) * ((pinl ** 2)))) ** (1 / 2)
tetaB = (1 + (((D) / (m * (c ** 2))) * ((pinshtrixl ** 2)))) ** (1 / 2)

chastiargumentcos1 = pinl * t * ((c * tetaA) - v0)
chastiargumentcos2 = pinl * t * ((c * tetaA) + v0)
chastiargumentsin1 = pinshtrixl * t * ((c * tetaB) - v0)
chastiargumentsin2 = pinshtrixl * t * ((c * tetaB) + v0)
mnogitelcos1 = (1 + (v0) / (c * tetaA))
mnogitelcos2 = (1 - (v0) / (c * tetaA))
mnogitelsinusov = ((1) / ((pinshtrixl) * (c) * (tetaB)))

x = np.arange(-2 * pi, 2 * pi, 0.1)

reh4 = 480 * ((l ** 8) / (pi ** 8)) * (
                    (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                    * ((mnogitelcos1 * np.cos(chastiargumentcos1 + pinl * x)
                        + mnogitelcos2 * np.cos(chastiargumentcos2 - pinl * x))
                       +
                       ((mnogitelsinusov) * (np.sin(chastiargumentsin1 + pinshtrixl * x)
                                             + np.sin(chastiargumentsin2 - pinshtrixl * x))))
                    )
print(reh1)
print(reh2)
print(reh3)
print(reh4)

def function(x):
    return l ** 8/315  +reh2

print(l ** 8/315 + reh2 )
z = function(x)

fig, ax = plt.subplots(1, 1)
plt.plot(x, z)

ax.xaxis.set_major_locator(FixedLocator([-2 * pi, -7 * pi / 4, -3 * pi / 2, -5 * pi / 4, -pi, -3 * pi / 4,
                                             -pi / 2, -pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4,
                                             3 * pi / 2, 7 * pi / 4, 2 * pi]))
ax.xaxis.set_major_formatter(FixedFormatter([r'-$2\pi$', r'-$\frac{7}{4}\pi$',
                                                 r'-$\frac{3}{2}\pi$', r'-$\frac{5}{4}\pi$', r'-$\pi$',
                                                 r'-$\frac{3}{4}\pi$',
                                                 r'-$\frac{1}{2}\pi$', r'-$\frac{1}{4}\pi$', '0', r'$\frac{1}{4}\pi$',
                                                 r'$\frac{1}{2}\pi$',
                                                 r'$\frac{3}{4}\pi$', r'$\pi$', r'$\frac{5}{4}\pi$',
                                                 r'$\frac{3}{2}\pi$', r'$\frac{7}{4}\pi$', r'$2\pi$']))

ax.grid()
ax.text(1, 1, 'n = ', size=15)
ax.text(2, 1, n, size=15)
#ax.vlines(0, 0, 40, color='black')
#ax.hlines(0, -2 * pi, 2 * pi, color='black')



plt.show()
"""


"""
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.ticker import FixedFormatter,FixedLocator
pi = math.pi
sum = 0
l = 3
x = np.arange(0, l, 0.1)

def function(x):
    return (x**4)*(l-x)**4

z = function(x)

fig, ax = plt.subplots(1, 1)
plt.plot(x, z)

ax.xaxis.set_major_locator(FixedLocator([-2 * pi, -7 * pi / 4, -3 * pi / 2, -5 * pi / 4, -pi, -3 * pi / 4,
                                             -pi / 2, -pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4,
                                             3 * pi / 2, 7 * pi / 4, 2 * pi]))
ax.xaxis.set_major_formatter(FixedFormatter([r'-$2\pi$', r'-$\frac{7}{4}\pi$',
                                                 r'-$\frac{3}{2}\pi$', r'-$\frac{5}{4}\pi$', r'-$\pi$',
                                                 r'-$\frac{3}{4}\pi$',
                                                 r'-$\frac{1}{2}\pi$', r'-$\frac{1}{4}\pi$', '0', r'$\frac{1}{4}\pi$',
                                                 r'$\frac{1}{2}\pi$',
                                                 r'$\frac{3}{4}\pi$', r'$\pi$', r'$\frac{5}{4}\pi$',
                                                 r'$\frac{3}{2}\pi$', r'$\frac{7}{4}\pi$', r'$2\pi$']))

ax.grid()

#ax.vlines(0, 0, 40, color='black')
#ax.hlines(0, -2 * pi, 2 * pi, color='black')



plt.show()
"""

"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math

pi = math.pi
sum = 0
i = 300  # количество слагаемых в сумме
l = 3  # длина закрепления
t = 0  # момент времен
s = t
v0 = 95  # постоянная скорость движения полотна
c = 95  # скорость распространения колебаний в покоящемся полотне
m = 10  # удельная масса полотна на единицу площади
D = 900  # величина жесткости на изгиб

while t <= 10:
    for n in range(i):
        n = n + 1

        pi_n_l = (pi * n) / l
        teta = (1 + (((D) / (m * (c ** 2))) * ((pi_n_l ** 2)))) ** (1 / 2)

        chasti_argument_cos_i_sin_1 = pi_n_l * t * ((c * teta) - v0)
        chasti_argument_cos_i_sin_2 = pi_n_l * t * ((c * teta) + v0)

        mnogitel_cos_1 = 1 + (v0 / (c * teta))
        mnogitel_cos_2 = 1 - (v0 / (c * teta))
        mnogitel_sinusov = 1 / (pi_n_l * c * teta)

        x = np.arange(0-0.25, l+0.25, 0.0001)

        U_x_t = 480 * ((l ** 8) / (pi ** 8)) * (
                (((((-1) ** n) + 1) / (n ** 8)) * (((pi ** 2) * (n ** 2)) - 42))
                * ((mnogitel_cos_1 * np.cos(chasti_argument_cos_i_sin_1 + pi_n_l * x)
                    + mnogitel_cos_2 * np.cos(chasti_argument_cos_i_sin_2 - pi_n_l * x))
                   +
                   ((mnogitel_sinusov) * (np.sin(chasti_argument_cos_i_sin_1 + pi_n_l * x)
                                          + np.sin(chasti_argument_cos_i_sin_2 - pi_n_l * x))))
        )
        sum = sum + U_x_t

    sum = sum + l ** 8 / 315


    def function(x):
        return sum


    z = function(x)



    t = t + 0.1

    plt.ion()
    plt.clf()
    plt.plot(x, z, color='black')
    #c = plt.Circle((0, 10), 0.5, color='black')
    #plt.gca().add_artist(c)
    plt.text(1, 605, 't = ', size=10)
    plt.text(1.2, 605, t, size=10)
    plt.grid(True)
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.001)


plt.ioff()
plt.show()
"""




"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math

pi = math.pi
l = 2  # длина закрепления
t = 0  # момент времени
v0 = 100  # постоянная скорость движения полотна
c = 100  # скорость распространения колебаний в покоящемся полотне
m = 10  # удельная масса полотна на единицу площади
D = 90  # величина жесткости на изгиб

n = 3
n = n + 1
# ny = (pi/2)*((2*n)-1)
ny = 0.5
x = np.arange(0-10.25, l+0.25, 0.0001)

def function(x, t):
    ny =  t
    print(ny)
    tay = ((v0 ** 2) - ((D / m) * ((ny ** 2) / (l ** 2)))) ** (1 / 2)
    U_x_t = (1/2) * ( ((1+(v0/tay))*(np.cosh((ny/l)*((t*(tay-v0))+x)))) + ((1-(v0/tay))*(np.cosh((ny/l)*((t*(tay+v0))-x)))) )
    return U_x_t

plt.ion()
for _ in range(25):  # Подстройте диапазон по необходимости
    z = function(x, t)
    plt.clf()
    plt.plot(x, z, color='black')
    plt.text(1, 605, 't = ', size=10)
    plt.text(1.2, 605, t, size=10)
    plt.grid(True)
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(2.1)
    t += 0.1

plt.ioff()
plt.show()
"""



"""
import numpy as np
import matplotlib.pyplot as plt
import time
import math

pi = math.pi
l = 2  # длина закрепления
t = 0  # момент времени
v0 = 30  # постоянная скорость движения полотна
c = 30  # скорость распространения колебаний в покоящемся полотне
m = 500  # удельная масса полотна на единицу площади
D = 100  # величина жесткости на изгиб

ny = 0.5
x = np.arange(0-0.25, l+0.25, 0.0001)

def function(x, t):
    n = t*10
    print(t)
    if (n == 0):
        ny = 1.875
    elif (n == 1):
        ny = 4.694
    else:
        ny = (pi / 2) * ((2 * n) - 1)
    print(ny)
    gam = ((v0 ** 2) + ((D / m) * ((ny ** 2) / (l ** 2)))) ** (1 / 2)
    U_x_t = (1/2) * ( ((1+(v0/gam))*(np.sin((ny/l)*((t*(gam-v0))+x)))) - ((1-(v0/gam))*(np.sin((ny/l)*((t*(gam+v0))-x)))) )
    return U_x_t

plt.ion()
for _ in range(6):  # Подстройте диапазон по необходимости
    z = function(x, t)
    plt.clf()
    plt.plot(x, z, color='black')
    plt.text(1.05, 1.005, 't = ', size=10)
    plt.text(1.25, 1.005, round(t,1), size=10)
    plt.grid(True)
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.5)
    t += 0.1

plt.ioff()
plt.show()
"""

#
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import math
#
# pi = math.pi
# l = 2  # длина закрепления
# t = 0  # момент времени
# v0 = 100  # постоянная скорость движения полотна
# c = 100  # скорость распространения колебаний в покоящемся полотне
# m = 10  # удельная масса полотна на единицу площади
# D = 900  # величина жесткости на изгиб
#
# n = 3
# n = n + 1
# # ny = (pi/2)*((2*n)-1)
# ny = 0.5
# step = 0.5
# x = np.arange(0-0.25, l+0.25, 0.0001)
#
# def function(x, t):
#     ny =  t
#     print(ny)
#     tay = ((v0 ** 2) - ((D / m) * ((ny ** 2) / (l ** 2)))) ** ((step))
#     gam = ((v0 ** 2) + ((D / m) * ((ny ** 2) / (l ** 2)))) ** (1 / 2)
#     U_x_t = (step) * ( ((1+(v0/gam))*(np.sin((ny/l)*((t*(gam-v0))+x)))) - ((1-(v0/gam))*(np.sin((ny/l)*((t*(gam+v0))-x)))) )
#     U_x_t_1 = (1 / 2) * (((1 + (v0 / tay)) * (np.cosh((ny / l) * ((t * (tay - v0)) + x)))) + (
#                 (1 - (v0 / tay)) * (np.cosh((ny / l) * ((t * (tay + v0)) - x)))))
#     return U_x_t + U_x_t_1
#
# plt.ion()
# for _ in range(250):  # Подстройте диапазон по необходимости
#     z = function(x, t)
#     plt.clf()
#     plt.plot(x, z, color='black')
#     plt.text(1, 605, 't = ', size=10)
#     plt.text(1.2, 605, t, size=10)
#     plt.grid(True)
#     plt.draw()
#     plt.gcf().canvas.flush_events()
#     time.sleep(0.1)
#     t += 0.1
#
# plt.ioff()
# plt.show()




# ДЛЯ 3 ЗАДАНИЯ, СТАТЬИ (РУДАКОВ)
import numpy as np
import matplotlib.pyplot as plt
import time
import math

pi = math.pi
sum = 0
i = 300  # количество слагаемых в сумме
l = pi  # длина закрепления
t = 0  # момент времен
v0 = 70  # постоянная скорость движения полотна
c = 90  # скорость распространения колебаний в покоящемся полотне
m = 10  # удельная масса полотна на единицу площади
D = 100  # величина жесткости на изгиб
n = 0

while t <= 2:
    for n in range(i):
        n = n + 1
        a = (D/m)**(1/2)
        alfa = round(((c ** 2)-(v0 ** 2))**(1/2),2)
        tetan = 1/4 # (0; 1/4)
        n_for_lymbda = 2
        lymbdan = round((((n_for_lymbda+(1/4)-tetan))**4)+(((alfa/a)**2)*((n_for_lymbda+(1/4)-tetan)**2)),2)

        q = round(((((-2)*(alfa**2))+2*(((alfa**4)+(4*(a**2)*(lymbdan)))**(1/2))) ** (1/2))/(2*a),2)
        s = round((1 + (((D) / (m * (c ** 2))) * (q ** 2))) ** (1 / 2),2)

        arg_sin_1 = round(t * (v0 - (c * s) ),2)
        arg_sin_2 = round(t * (v0 + (c * s) ),2)

        mnogitel_sin_1 = round(1 + (v0 / (c * s)),2)
        mnogitel_sin_2 = round(1 - (v0 / (c * s)),2)


        x = np.arange(0-0.25, l+0.25, 0.0001)
        print(a, alfa, tetan,lymbdan,q,s,arg_sin_1,arg_sin_2, mnogitel_sin_2,mnogitel_sin_1)

        U_x_t = 72/(pi) * (
                (((((-1) ** n) - 1) / (n ** 7)) * (((pi ** 2) * (n ** 2)) - 10))
                * ((mnogitel_sin_1 * np.sin(q*n*(x-arg_sin_1))) + (mnogitel_sin_2 * np.sin(q*n*(x-arg_sin_2))))
        )
        sum = sum + U_x_t

    sum = sum +1

    def function(x):
        return sum

    z = function(x)

    plt.ion()
    plt.clf()
    plt.plot(x, z, color='black')
    #c = plt.Circle((0, 10), 0.5, color='black')
    #plt.gca().add_artist(c)

    plt.axvline(x=pi, color='gray')
    plt.axvline(x=0, color='gray')
    plt.text(1.1, 20, 't = ', size=10)
    plt.text(1.3, 20, round(t,2), size=10)
    t = t + 0.1

    plt.grid(True)
    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.001)


plt.ioff()
plt.show()