import taichi as ti
import matplotlib.cm as cm
from func import *

ti.init(arch=ti.gpu)

w = 640
h = 320

pixels = ti.Vector.field(3, dtype=float, shape=(w, h))
scaleX = 6.0
scaleY = 3.0

max_iteration = 50

@ti.kernel
def paint(power: int, c0: float, c1:float, t: int):
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([c0, c1])
        x = cartesianX(i, w, scaleX)
        y = cartesianX(j, h, scaleY)
        z = ti.Vector([x, y])
        iterations = 0
        while z.norm() < 30 and iterations < max_iteration:
            z = complex_power(z, power) + c
            iterations += 1
        pixels[i, j] = hsv2rgb(angle(x, y) + t, 0.8, 1 - iterations / max_iteration)

gui = ti.GUI("Julia Set", res=(w, h))
power = gui.slider('Power', 0, 10, step=1)
c0 = gui.slider('C0', -scaleX, scaleX, step=0.01)
c1 = gui.slider('C1', -scaleY, scaleY, step=0.01)

# set default value
power.value = 2
c0.value = -0.899
c1.value = 0.0

t = 0
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 's':
            power.value -= 1
        elif e.key == 'w':
            power.value += 1
        elif e.key == 'Left':
            c0.value -= 0.01
        elif e.key == 'Right':
            c0.value += 0.01
        elif e.key == 'Up':
            c1.value -= 0.01
        elif e.key == 'Down':
            c1.value += 0.01
    paint(int(power.value), float(c0.value), float(c1.value), t)
    gui.set_image(pixels)
    gui.show()
    t += 1