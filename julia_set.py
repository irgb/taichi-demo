import taichi as ti
import matplotlib.cm as cm

ti.init(arch=ti.gpu)

w = 640
h = 320

pixels = ti.field(dtype=float, shape=(w, h))
# set origin coordinate
orig_x = w / 2
orig_y = h / 2
unit_x = w / 8.0 # split x axis to 8 units, each unit contains unit_x pixels
unit_y = h / 4.0

@ti.func
def complex_power(z, power=2):
    res = ti.Vector([1.0, 0.0]) # z^0 = 1.0 + 0.0j
    while power > 0:
        res = ti.Vector([res[0] * z[0] - res[1] * z[1], res[0] * z[1] + res[1] * z[0]])
        power -= 1
    return res

@ti.kernel
def paint(power: int, c0: float, c1:float):
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([c0, c1])
        x = (i - orig_x) / unit_x
        y = (orig_y - j) / unit_y
        z = ti.Vector([x, y])
        iterations = 0
        while z.norm() < 300 and iterations < 50:
            z = complex_power(z, power) + c
            iterations += 1
        pixels[i, j] = (1 - iterations * 0.02)


gui = ti.GUI("Julia Set", res=(w, h))
power = gui.slider('Power', 0, 10, step=1)
c0 = gui.slider('C0', -w / unit_x, w / unit_x, step=0.01)
c1 = gui.slider('C1', -h / unit_y, h / unit_y, step=0.01)

# set default value
power.value = 2
c0.value = -0.899
c1.value = 0.0

cmap = cm.get_cmap('jet')

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
    paint(int(power.value), float(c0.value), float(c1.value))
    # gui.set_image(pixels)
    gui.set_image(cmap(pixels.to_numpy()))
    gui.show()