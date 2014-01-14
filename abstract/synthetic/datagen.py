"""
Generate synthetic data from interfering sources.
"""
import cPickle as pickle
import numpy
from matplotlib import pyplot
from fatiando.mesher.ddd import Prism, extract
from fatiando import potential, logger, gridder, utils, vis

log = logger.tofile(logger.get(), 'datagen.log')
log.info(logger.header())

# Generate a synthetic model
bounds = [0, 5000, 0, 5000, 0, 1500]
model = [Prism(500, 4500, 2200, 2800, 200, 800, {'density':1000})]
# show it
vis.vtk.figure()
vis.vtk.prisms(model, extract('density', model))
vis.vtk.add_axes(vis.vtk.add_outline(bounds), ranges=[i*0.001 for i in bounds],
    fmt='%.1f', nlabels=6)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
# and use it to generate some tensor data
shape = (20, 20)
area = bounds[0:4]
noise = 2
x, y, z = gridder.regular(area, shape, z=-150)
gxx = utils.contaminate(potential.prism.gxx(x, y, z, model), noise)
gxy = utils.contaminate(potential.prism.gxy(x, y, z, model), noise)
gxz = utils.contaminate(potential.prism.gxz(x, y, z, model), noise)
gyy = utils.contaminate(potential.prism.gyy(x, y, z, model), noise)
gyz = utils.contaminate(potential.prism.gyz(x, y, z, model), noise)
gzz = utils.contaminate(potential.prism.gzz(x, y, z, model), noise)
# show the data
pyplot.figure()
for i, v in enumerate([gxx, gxy, gxz, gyy, gyz, gzz]):
    pyplot.subplot(2, 3, i + 1)
    pyplot.axis('scaled')
    vis.map.contourf(y, x, v, shape, 12)
    pyplot.colorbar()
pyplot.show()
# Dump data to a file and pickle the model
numpy.savetxt('data.txt',
              numpy.transpose([x, y, z, gxx, gxy, gxz, gyy, gyz, gzz]),
              fmt='%.4f')
with open('model.pickle', 'w') as f:
    pickle.dump(model, f)

