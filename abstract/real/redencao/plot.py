"""
Plot the results
"""
import cPickle as pickle
import numpy
from matplotlib import pyplot
from fatiando import vis, mesher

pyplot.rc('font', size=7)

# Plot the observed and predicted data
x, y, gz = numpy.loadtxt('data.xyz').T
gz_pred = numpy.loadtxt('predicted.xyz', usecols=[-1])
xseed, yseed = numpy.loadtxt('seeds.xyz', usecols=[0, 1]).T
xc, yc = numpy.loadtxt('outcrop.xyz').T
x, y = x*0.001, y*0.001
xseed, yseed = xseed*0.001, yseed*0.001
shape = (20, 20)
pyplot.figure(figsize=(3.5, 3.7))
pyplot.subplots_adjust(left=0.17, right=0.9, bottom=0.15, top=0.93)
pyplot.axis('scaled')
levels = vis.map.contourf(y, x, gz, shape, 8)
cb = pyplot.colorbar(orientation='vertical', shrink=0.68)
cb.set_ticks([l for i, l in enumerate(levels) if i%2 != 0])
cb.set_label('mGal')
vis.map.contour(y, x, gz_pred, shape, levels, color='k')
pyplot.plot(yc*0.001, xc*0.001, '-r', linewidth=2)
pyplot.plot(yseed, xseed, 'ow')
pyplot.xlabel('Horizontal coordinate y (km)')
pyplot.ylabel('Horizontal coordinate x (km)')
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
vis.vtk.prisms(estimate, mesher.ddd.extract('density', estimate))
vis.vtk.prisms(seeds, mesher.ddd.extract('density', seeds))
vis.vtk.mlab.plot3d(xc, yc, numpy.zeros_like(xc), color=(1,0,0),
                    tube_radius=300)
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
                 ranges=[i*0.001 for i in bounds], fmt='%.0f', nlabels=3)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
