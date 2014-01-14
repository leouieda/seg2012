import numpy
from fatiando import vis, gridder, logger

pyplot = vis.map.pyplot

log = logger.tofile(logger.get(), 'fetchdata.log')
log.info(logger.header())

fname = '/home/leo/dat/redencao_grav/data/residual_bouguer.xyz'
log.info("Loading data from file '%s'" % (fname))
x, y, gz = numpy.loadtxt(fname, unpack=True)
shape = (20, 20)
log.info("Gridding data to %s grid" % (str(shape)))
x, y, gz = gridder.interpolate(x, y, gz, shape)
gz = gz.filled(0.).ravel()
x, y = x.ravel(), y.ravel()
with open('data.xyz', 'w') as f:
    f.write(logger.header(comment='#'))
    f.write("\n# Residual bouguer data of the Redencao pluton")
    f.write("\n# Column structure\n")
    f.write('# x (m)   y (m)   gz (mGal)')
    f.write('\n')
    numpy.savetxt(f, numpy.transpose([x, y, gz]), fmt="%10.4f")
fname = '/home/leo/dat/redencao_grav/data/outcrop.xyz'
log.info("Loading outcrop contour from '%s'" % (fname))
xc, yc = numpy.loadtxt(fname, unpack=True)
with open('outcrop.xyz', 'w') as f:
    f.write(logger.header(comment='#'))
    f.write("\n# Contour of the outcropping portion of the Redencao pluton")
    f.write("\n# Column structure\n")
    f.write('# x (m)   y (m)')
    f.write('\n')
    numpy.savetxt(f, numpy.transpose([xc, yc]), fmt="%10.4f")

pyplot.figure()
pyplot.title("Residual gravity anomaly")
pyplot.axis('scaled')
vis.map.contourf(y*0.001, x*0.001, gz, shape, 15)
pyplot.colorbar(shrink=0.9)
pyplot.plot(yc*0.001, xc*0.001, '-k', linewidth=3)
pyplot.ylabel('North')
pyplot.xlabel('East')
pyplot.show()

