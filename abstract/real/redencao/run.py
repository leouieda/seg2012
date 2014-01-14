"""
Run a shape-of-anomaly inversion on the data set
"""
import cPickle as pickle
import numpy
from matplotlib import pyplot
from fatiando import vis, logger, mesher
from fatiando.potential import harvester

log = logger.tofile(logger.get(), 'run.log')
log.info(logger.header())

# Load up the data
x, y, gz = numpy.loadtxt('data.xyz').T
# Create the mesh
pad = 5000
bounds = [x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad, 0, 10000]
mesh = mesher.ddd.PrismMesh(bounds, (60, 56, 64))
# Wrap the data into data modules
dms = harvester.wrapdata(mesh, x, y, -0.1 + numpy.zeros_like(x), gz=gz, norm=2)
# Load the seeds
points, densities = harvester.loadseeds('seeds.xyz')
seeds = harvester.sow_prisms(points, {'density':densities}, mesh,
    mu=0.5, delta=0.00005, useshape=True, reldist=True)
# show the seeds
sprisms = [s.get_prism() for s in seeds]
vis.vtk.figure()
vis.vtk.prisms(sprisms, mesher.ddd.extract('density', sprisms))
vis.vtk.add_axes(vis.vtk.add_outline(bounds), nlabels=5,
    ranges=[i*0.001 for i in bounds], fmt='%.1f')
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
# Run the inversion
estimate, goals, misfits = harvester.harvest(dms, seeds)
# fill the mesh with the density values
mesh.addprop('density', estimate['density'])
# Pickle the seeds
log.info("Pickling seeds")
with open('seeds.pickle', 'w') as f:
    pickle.dump(sprisms, f)
# Pickle the mesh
log.info("Pickling mesh")
with open('mesh.pickle', 'w') as f:
    pickle.dump(mesh, f)
# Pickle the estimate
log.info("Pickling estimate")
density_model = mesher.ddd.vfilter(-1000, -1, 'density', mesh)
with open('estimate.pickle', 'w') as f:
    pickle.dump(density_model, f)
# Dump the predicted data to a file
log.info("Dumping predicted data")
with open('predicted.xyz', 'w') as f:
    output = [x, y, dms[0].get_predicted()]
    f.write(logger.header(comment='#'))
    f.write("\n")
    numpy.savetxt(f, numpy.transpose(output), fmt='%10.4f')
# Plot the results
log.info("Plotting results")
shape = (20, 20)
pyplot.figure()
pyplot.title('Observed = color | Predicted = contour')
pyplot.axis('scaled')
levels = vis.map.contourf(y*0.001, x*0.001, gz, shape, 15)
pyplot.colorbar()
vis.map.contour(y*0.001, x*0.001, dms[0].get_predicted(), shape, levels,
    color='k')
pyplot.xlabel('Horizontal coordinate y (km)')
pyplot.ylabel('Horizontal coordinate x (km)')
pyplot.show()
# Get only the prisms corresponding to our estimate
vis.vtk.figure()
vis.vtk.prisms(density_model, mesher.ddd.extract('density', density_model))
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
    ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=5)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
