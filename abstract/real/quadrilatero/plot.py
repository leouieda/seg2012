"""
Plot the results
"""
import cPickle as pickle
import numpy
from matplotlib import pyplot
from fatiando import vis, mesher

pyplot.rc('font', size=6)

# Plot the observed and predicted data
x, y, gyz, gzz = numpy.loadtxt('data.xyz', usecols=[0, 1, -2, -1]).T
gyz_pred, gzz_pred = numpy.loadtxt('predicted.xyz', usecols=[-2, -1]).T
xseed, yseed = numpy.loadtxt('seeds.xyz', usecols=[0, 1]).T
x, y = x*0.001, y*0.001
xseed, yseed = xseed*0.001, yseed*0.001
shape = (100, 100)

pyplot.figure(figsize=(3.6, 4))
pyplot.subplots_adjust(wspace=0, right=0.98, left=0.11)

pyplot.subplot(1, 2, 1)
pyplot.title('$g_{yz}$', fontsize=12)
pyplot.axis('scaled')
levels = vis.map.contourf(y, x, gyz, shape, 10, interpolate=True)
cb = pyplot.colorbar(orientation='horizontal', shrink=0.9)
cb.set_ticks([l for i, l in enumerate(levels) if i%2 != 0])
vis.map.contour(y, x, gyz_pred, shape, levels, interpolate=True, color='k')
pyplot.plot(yseed, xseed, 'ow')
pyplot.xlabel('Horizontal coordinate y (km)')
pyplot.ylabel('Horizontal coordinate x (km)')

ax = pyplot.subplot(1, 2, 2)
pyplot.title('$g_{zz}$', fontsize=12)
pyplot.axis('scaled')
levels = vis.map.contourf(y, x, gzz, shape, 10, interpolate=True)
cb = pyplot.colorbar(orientation='horizontal', shrink=0.9)
cb.set_ticks([l for i, l in enumerate(levels) if i%2 != 0])
vis.map.contour(y, x, gzz_pred, shape, levels, interpolate=True, color='k')
pyplot.plot(yseed, xseed, 'ow')
pyplot.xlabel('Horizontal coordinate y (km)')
ax.set_yticklabels([])

pyplot.show()

# Plot the estimate and the seeds
with open('estimate.pickle') as f:
    estimate = pickle.load(f)
with open('seeds.pickle') as f:
    seeds = pickle.load(f)
with open('mesh.pickle') as f:
    mesh = pickle.load(f)
bounds = mesh.bounds
vis.vtk.figure()
vis.vtk.prisms(estimate, mesher.ddd.extract('density', estimate), vmin=0)
vis.vtk.prisms(seeds, mesher.ddd.extract('density', seeds), vmin=0)
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
    ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=5)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
