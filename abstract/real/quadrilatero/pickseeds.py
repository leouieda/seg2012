"""
Pick the seed locations from the gzz map
"""
import numpy
from matplotlib import pyplot
from fatiando import vis, logger, ui

x, y, gzz = numpy.loadtxt('data.xyz', usecols=[0, 1, -1], unpack=True)
depth = -950.
density = 1000.
pyplot.figure()
pyplot.axis('scaled')
ax = pyplot.gca()
vis.map.contourf(y, x, gzz, (100, 100), 15, interpolate=True)
points = ui.picker.points((y.min(), y.max(), x.min(), x.max()), ax)
yseed, xseed = numpy.transpose(points)
data = [xseed, yseed, depth*numpy.ones_like(xseed),
        density*numpy.ones_like(xseed)]
with open('seeds.xyz', 'w') as f:
    f.write(logger.header(comment='#'))
    f.write('\n# x y z density\n')
    numpy.savetxt(f, numpy.transpose(data), fmt='%.4f')
