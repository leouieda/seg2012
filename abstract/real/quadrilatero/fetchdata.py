import numpy
from matplotlib import pyplot
from fatiando import vis, gridder, logger

log = logger.tofile(logger.get(), 'fetchdata.log')
log.info(logger.header())

data = numpy.loadtxt('/home/leo/dat/boa6/ftg/rawdata/BOA6_FTG.XYZ', unpack=True)
# Remove the coordinates from the raw data
data[0] -= data[0].min()
data[1] -= data[1].min()
area1 = [7970, 12877, 10650, 17270]
y, x, scalars = gridder.cut(data[0], data[1], data[2:], area1)
# The x and y components are switched because the coordinates are mixed up
# (my x is their y)
height, z, gyy, gxy, gyz, gxx, gxz, gzz = scalars
# Remove the coordinates from the cut data
x -= x.min()
y -= y.min()
# Convert altitude into z coordinates
z *= -1
# Save things to a file
fields =  ['x', 'y', 'height', 'alt', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']
data = [x, y, height, z, gxx, gxy, gxz, gyy, gyz, gzz]
with open('data.xyz', 'w') as f:
    f.write(logger.header(comment='#'))
    f.write("\n# Column structure\n# ")
    f.write('  '.join(fields))
    f.write('\n')
    numpy.savetxt(f, numpy.transpose(data), fmt="%.4f")
# Plot
pyplot.figure(figsize=(16, 8))
pyplot.suptitle("Area 1 data set")
pyplot.subplots_adjust(wspace=0.3)
for i, s in enumerate(data[2:]):
    pyplot.subplot(2, 4, i + 1)
    pyplot.axis('scaled')
    pyplot.title(fields[i + 2])
    vis.map.contourf(y*0.001, x*0.001, s, (100, 100), 15, interpolate=True)
    pyplot.colorbar(shrink=0.9)
    pyplot.ylabel('North')
    pyplot.xlabel('East')
pyplot.show()

