"""
Plot the results
"""
import cPickle as pickle
import numpy
from matplotlib import pyplot
from fatiando import vis, mesher

pyplot.rc('font', size=6)

# Plot the observed and predicted data
x, y, gzz = numpy.loadtxt('data.txt', usecols=[0, 1, -1]).T
gzzshape = numpy.loadtxt('predicted-shape.xyz', usecols=[-1])
gzzstd = numpy.loadtxt('predicted-std.xyz', usecols=[-1])
x, y = x*0.001, y*0.001
xseed, yseed = [2.5], [2.5]
shape = (20, 20)

pyplot.figure(figsize=(3.6, 2))
pyplot.subplots_adjust(bottom=0.26, right=0.97, left=0.05, wspace=0.01)

pyplot.subplot(1, 2, 1)
pyplot.axis('scaled')
levels = vis.map.contourf(y, x, gzz, shape, 4)
vis.map.contour(y, x, gzzstd, shape, levels, color='k')
pyplot.plot(yseed, xseed, 'ow')
pyplot.xlabel('Horizontal coordinate y (km)')
pyplot.ylabel('Horizontal coordinate x (km)')

ax = pyplot.subplot(1, 2, 2)
pyplot.axis('scaled')
levels = vis.map.contourf(y, x, gzz, shape, 4)
cb = pyplot.colorbar(shrink=1)
vis.map.contour(y, x, gzzshape, shape, levels, color='k')
pyplot.plot(yseed, xseed, 'ow')
pyplot.xlabel('Horizontal coordinate y (km)')

pyplot.show()

# Plot the estimate and the seeds
with open('estimate-shape.pickle') as f:
    shape = pickle.load(f)
with open('estimate-std.pickle') as f:
    std = pickle.load(f)
with open('model.pickle') as f:
    model = pickle.load(f)
bounds = [0, 5000, 0, 5000, 0, 2000]
vis.vtk.figure()
vis.vtk.prisms(model, mesher.ddd.extract('density', model), style='wireframe')
vis.vtk.prisms(shape, mesher.ddd.extract('density', shape), vmin=0)
vis.vtk.prisms(std, mesher.ddd.extract('density', std), vmin=0)
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
    ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=3)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
