"""
Run the inversion using the shape-of-anomaly misfit function
"""
import cPickle as pickle
import numpy
from matplotlib import pyplot
from fatiando.mesher.ddd import PrismMesh, extract, vfilter
from fatiando.potential import harvester
from fatiando import logger, vis

log = logger.tofile(logger.get(), 'run-shape.log')
log.info(logger.header())

# Load the data and the model
with open('model.pickle') as f:
    model = pickle.load(f)
x, y, z, gxx, gxy, gxz, gyy, gyz, gzz = numpy.loadtxt('data.txt', unpack=True)
shape = (20, 20)
# Create a prism mesh
bounds = [0, 5000, 0, 5000, 0, 2000]
mesh = PrismMesh(bounds, (20, 50, 50))
# Make the data modules
datamods = harvester.wrapdata(mesh, x, y, z, gzz=gzz, norm=2)
# and the seeds
points =[(2500, 2500, 300)]
seeds = harvester.sow_prisms(points, {'density':[1000]*len(points)}, mesh,
    mu=0.2, delta=0.0005, useshape=True)
# Show the seeds
vis.vtk.figure()
seedprisms = [s.get_prism() for s in seeds]
vis.vtk.prisms(model, extract('density', model), style='wireframe')
vis.vtk.prisms(seedprisms, extract('density', seedprisms), vmin=0)
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
    ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=6)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
# Run the inversion and collect the results
estimate, goals, misfits = harvester.harvest(datamods, seeds)
# Insert the estimated density values into the mesh
mesh.addprop('density', estimate['density'])
# and get only the prisms corresponding to our estimate
density_model = vfilter(900, 1300, 'density', mesh)
# Save the results
with open('estimate-shape.pickle', 'w') as f:
    pickle.dump(density_model, f)
with open('seeds.pickle', 'w') as f:
    pickle.dump(seedprisms, f)
with open('predicted-shape.xyz', 'w') as f:
    output = [x, y, z]
    output.extend(dm.get_predicted() for dm in datamods)
    numpy.savetxt(f, numpy.transpose(output), fmt='%.4f')
# Plot the results
pyplot.figure()
for i, dm in enumerate(datamods):
    pyplot.subplot(1, 1, i + 1)
    pyplot.axis('scaled')
    levels = vis.map.contourf(y*0.001, x*0.001, dm.data, shape, 10)
    pyplot.colorbar()
    vis.map.contour(y*0.001, x*0.001, dm.get_predicted(), shape, levels,
        color='k')
    pyplot.xlabel('Horizontal coordinate y (km)')
    pyplot.ylabel('Horizontal coordinate x (km)')
pyplot.show()
vis.vtk.figure()
vis.vtk.prisms(model, extract('density', model), style='wireframe')
vis.vtk.prisms(density_model, extract('density', density_model), vmin=-1000)
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
    ranges=[i*0.001 for i in bounds], fmt='%.0f', nlabels=6)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
