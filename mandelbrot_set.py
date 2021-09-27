import taichi as ti
import matplotlib.cm as cm

ti.init(arch=ti.gpu)

w = 640
h = 320

pixels = ti.field(dtype=float, shape=(w, h))
# set origin coordinate
orig_x = w / 5 * 3
orig_y = h / 2
unit_x = w / 5 # split x axis to 5 units, each unit contains unit_x pixels
unit_y = h / 2.5

@ti.func
def complex_power(z, power=2):
    res = ti.Vector([1.0, 0.0]) # z^0 = 1.0 + 0.0j
    while power > 0:
        res = ti.Vector([res[0] * z[0] - res[1] * z[1], res[0] * z[1] + res[1] * z[0]])
        power -= 1
    return res

@ti.kernel
def paint(power: int):
    for i, j in pixels:  # Parallelized over all pixels
        x = (i - orig_x) / unit_x
        y = (orig_y - j) / unit_y
        c = ti.Vector([x, y])
        z = ti.Vector([0.0, 0.0])
        iterations = 0
        while z.norm() <= 300 and iterations < 50:
            z = complex_power(z, power) + c
            iterations += 1
        pixels[i, j] = (1 - iterations * 0.02)


gui = ti.GUI("Mandelbrot Set", res=(w, h))
power = gui.slider('Power', 0, 10, step=1)

# set default value
power.value = 2

cmap = cm.get_cmap('jet')

while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'Up':
            power.value -= 1
        elif e.key == 'Down':
            power.value += 1
    paint(int(power.value))
    gui.set_image(cmap(pixels.to_numpy()))
    gui.show()