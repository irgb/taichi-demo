import taichi as ti

#常用函数
@ti.func
def angle(x: float, y: float) -> float:
	"""Convert radian [-π/2, π/2) to angle [0, 360)"""
	return ti.atan2(x, y) / 3.1415926 * 180.0 + 180.0

@ti.func
def cartesian(x: float, y: float, w: int, h: int, scale: float=1.0) -> ti.Vector:
	"""
	Convert screen coordinate to cartesian coordinate. 
	Example: x=20, y=20, w=40, h=40, return [0, 0]

	:param float x: screen x
	:param float y: screen y
	:param int w: screen width
	:param int h: screen height
	:param float scale: range of cartesian coordinate is [-scale/2, scale/2]
	"""
	return ti.Vector([x / w - 0.5, 0.5 - y / h]) * scale

@ti.func
def cartesianX(x: float, w: int, scale: float=2.0) -> float:
	return (x / w - 0.5) * scale

@ti.func
def cartesianY(y: float, h: int, scale: float=2.0) -> float:
	return (0.5 - y / h) * scale

@ti.func
def hsv2rgb(h: float, s: float, v: float) -> ti.Vector:
	"""
	Convert HSV to RGB, algorithm: https://stackoverflow.com/a/6930407/5432806

	:param float h: hue ∈ [0, 360)
	:param float s: saturation ∈ [0, 1)
	:param float v: value  ∈ [0, 1)
	:return: ti.Vector([R, G, B])
	"""
	h = h % 360
	hh = h / 60
	i = ti.floor(hh)
	ff = hh - i
	p = v * (1.0 - s)
	q = v * (1.0 - (s * ff))
	t = v * (1.0 - (s * (1.0 - ff)))

	r, g, b = 0.0, 0.0, 0.0
	if i == 0: r, g, b = v, t, p
	elif i == 1: r, g, b = q, v, p
	elif i == 2: r, g, b = p, v, t
	elif i == 3: r, g, b = p, q, v
	elif i == 4: r, g, b = t, p, v
	elif i == 5: r, g, b = v, p, q
	return ti.Vector([r, g, b])

@ti.func
def complex_power(z, power=2) -> ti.Vector:
	"""
	Power of complex number.
	Example: z = 1 + 1j, z^2 = 2j
	"""
	res = ti.Vector([1.0, 0.0]) # z^0 = 1.0 + 0.0j
	while power > 0:
		res = ti.Vector([res[0] * z[0] - res[1] * z[1], res[0] * z[1] + res[1] * z[0]])
		power -= 1
	return res
