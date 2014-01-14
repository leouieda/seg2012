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
x, y, height, z, gxx, gxy, gxz, gyy, gyz, gzz = numpy.loadtxt('data.xyz').T
# Create the mesh
bounds = [x.min(), x.max(), y.min(), y.max(), -height.max(), -200]
#mesh = mesher.ddd.PrismMesh(bounds, (46, 200, 270))
mesh = mesher.ddd.PrismMesh(bounds, (23, 100, 135))
#mesh = mesher.ddd.PrismMesh(bounds, (10, 50, 70))
mesh.carvetopo(x, y, height)
# Wrap the data into data modules
dms = harvester.wrapdata(mesh, x, y, z, gyz=gyz, gzz=gzz, norm=2)
# Load the seeds
points, densities = harvester.loadseeds('seeds.xyz')
seeds = harvester.sow_prisms(points, {'density':densities}, mesh,
    mu=0.1, delta=0.0001, useshape=True)
# show the seeds
sprisms = [s.get_prism() for s in seeds]
vis.vtk.figure()
vis.vtk.prisms(sprisms, mesher.ddd.extract('density', sprisms), vmin=0)
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
density_model = mesher.ddd.vfilter(1, 2000, 'density', mesh)
with open('estimate.pickle', 'w') as f:
    pickle.dump(density_model, f)
# Dump the predicted data to a file
log.info("Dumping predicted data")
with open('predicted.xyz', 'w') as f:
    output = [x, y, z]
    output.extend(dm.get_predicted() for dm in dms)
    f.write(logger.header(comment='#'))
    f.write("\n")
    numpy.savetxt(f, numpy.transpose(output), fmt='%.4f')
# Save the estimate to .msh and .den files for ploting in MeshTools3d
#log.info("Saving results to 'estimate.den' and 'estimate.msh'")
#with open('estimate.msh', 'w') as f:
    #nz, ny, nx = mesh.shape
    #x1, x2, y1, y2, z1, z2 = mesh.bounds
    #dx, dy, dz = mesh.dims
    #f.write("%d %d %d\n" % (ny, nx, nz))
    #f.write("%g %g %g\n" % (y1, x1, -z1))
    #f.write("%d*%g\n" % (ny, dy))
    #f.write("%d*%g\n" % (nx, dx))
    #f.write("%d*%g\n" % (nz, dz))
#dens = (v if i not in mesh.mask else -100
        #for i, v in enumerate(estimate['density']))
#with open('estimate.den', 'w') as f:
    #numpy.savetxt(
        #f,
        #numpy.ravel(numpy.reshape(numpy.fromiter(dens, 'f'), mesh.shape),
            #order='F'),
        #fmt='%.4f')
# Plot the results
log.info("Plotting results")
shape = (100, 100)
for i, dm in enumerate(dms):
    pyplot.figure()
    pyplot.title('Observed = color | Predicted = contour')
    pyplot.axis('scaled')
    levels = vis.map.contourf(y*0.001, x*0.001, dm.data, shape, 15,
        interpolate=True)
    pyplot.colorbar()
    vis.map.contour(y*0.001, x*0.001, dm.get_predicted(), shape, levels,
        color='k', interpolate=True)
    pyplot.xlabel('Horizontal coordinate y (km)')
    pyplot.ylabel('Horizontal coordinate x (km)')
pyplot.show()
# Get only the prisms corresponding to our estimate
log.info("Filter estimate for plotting")
vis.vtk.figure()
vis.vtk.prisms(density_model, mesher.ddd.extract('density', density_model),
    vmin=0)
vis.vtk.add_axes(vis.vtk.add_outline(bounds),
    ranges=[i*0.001 for i in bounds], fmt='%.1f', nlabels=5)
vis.vtk.wall_bottom(bounds)
vis.vtk.wall_north(bounds)
vis.vtk.mlab.show()
