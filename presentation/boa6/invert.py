import sys
import cPickle as pickle
import fatiando as ft
import numpy as np

def setview1(scene):
    scene.scene.camera.position = [-5223.8621677191895, 2525.7340618425533, -9868.4188588613615]
    scene.scene.camera.focal_point = [3383.1675754283192, 2308.595674490809, 235.42915513228513]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.76122989162048482, -0.0010492132895237637, -0.64848126515338744]
    scene.scene.camera.clipping_range = [7118.9267881922133, 19208.19334880414]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

def setview2(scene):
    scene.scene.camera.position = [-2083.5891203179367, 2196.562816405461, -698.77837411337339]
    scene.scene.camera.focal_point = [467.08916568702983, 471.73128900160287, 610.85856263017729]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.33814580712856857, -0.1984415469462483, -0.91993389195471487]
    scene.scene.camera.clipping_range = [1619.2366959256301, 5453.5227454505075]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

seedfile = sys.argv[1]
mu = float(sys.argv[2])
delta = float(sys.argv[3])
if sys.argv[4] == 'classic':
    useshape = False
    reldist = False
elif sys.argv[4] == 'shape':
    useshape = True
    reldist = True
else:
    print "invalid argument"
    sys.exit()

log = ft.log.get()
ft.log.tofile(log, 'invert-%s-%s.log' % (seedfile, sys.argv[4]))
log.info(ft.log.header())

data = np.loadtxt('/home/leo/dat/boa6/ftg/rawdata/BOA6_FTG.XYZ', unpack=True)
# Remove the coordinates from the raw data
data[0] -= data[0].min()
data[1] -= data[1].min()
area1 = [7970, 12877, 10650, 17270]
y, x, scalars = ft.grd.cut(data[0], data[1], data[2:], area1)
# The x and y components are switched because the coordinates are mixed up
# (my x is their y)
height, z, gyy, gxy, gyz, gxx, gxz, gzz = scalars
# Remove the coordinates from the cut data
x -= x.min()
y -= y.min()
# Convert altitude into z coordinates
z *= -1
shape = [100, 100]

bounds = [x.min(), x.max(), y.min(), y.max(), -height.max(), -200]
mesh = ft.msh.ddd.PrismMesh(bounds, (23, 100, 135))
mesh.carvetopo(x, y, height)

dms = ft.pot.harvester.wrapdata(mesh, x, y, z, gyz=gyz, gzz=gzz)
rawseeds = ft.pot.harvester.loadseeds(seedfile)
seeds = ft.pot.harvester.sow(rawseeds, mesh, mu=mu, delta=delta,
    useshape=useshape, reldist=reldist)

ft.vis.figure(figsize=(3.5, 5))
ft.vis.axis('scaled')
levels = ft.vis.contourf(y, x, gzz, shape, 10, interp=True)
ft.vis.colorbar(orientation='horizontal', shrink=0.7)
sx, sy = np.transpose(rawseeds)[:2]
ft.vis.plot(sy, sx, 'ow')
ft.vis.xlabel('y (km)')
ft.vis.ylabel('x (km)')
ft.vis.m2km()
ft.vis.savefig('seeds-map-%s-%s.png' % (seedfile, sys.argv[4]), dpi=300)
#ft.vis.show()

scene = ft.vis.figure3d(size=(1000, 1000))
ft.vis.prisms([s.get_prism() for s in seeds], 'density', vmin=0, vmax=1000)
ax = ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[b*0.001 for b in bounds],
    fmt='%0.1f', nlabels=3)
#ax.axes.x_label, ax.axes.y_label, ax.axes.z_label = 'x (km)', 'y (km)', 'h (km)'
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
setview1(scene)
ft.vis.savefig3d('seeds-%s-%s1.png' % (seedfile, sys.argv[4]))
#setview2(scene)
#ft.vis.savefig3d('seeds-%s-%s2.png' % (seedfile, sys.argv[4]))
#ft.vis.show3d()
ft.vis.vtk.mlab.close()

estimate, goals, misfits = ft.pot.harvester.harvest(dms, seeds)
mesh.addprop('density', estimate['density'])
result = ft.msh.ddd.vremove(0, 'density', mesh)

with open('results-%s-%s.pickle' % (seedfile, sys.argv[4]), 'w') as f:
    pickle.dump({'estimate':result, 'predicted':[d.predicted for d in dms],
        'seeds':[s.get_prism() for s in seeds]}, f)

ft.vis.figure(figsize=(3.5, 5))
ft.vis.axis('scaled')
levels = ft.vis.contourf(y, x, gzz, shape, 10, interp=True)
ft.vis.colorbar(orientation='horizontal', shrink=0.7)
ft.vis.contour(y, x, dms[-1].predicted, shape, levels, color='k', linewidth=1,
    interp=True)
ft.vis.xlabel('y (km)')
ft.vis.ylabel('x (km)')
ft.vis.m2km()
ft.vis.savefig('fit-%s-%s.png' % (seedfile, sys.argv[4]), dpi=300)
#ft.vis.show()

scene = ft.vis.figure3d(size=(1000, 1000))
ft.vis.prisms([s.get_prism() for s in seeds], 'density', vmin=0, vmax=1000)
ft.vis.prisms(result, 'density', vmin=0, vmax=1000)
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[b*0.001 for b in bounds],
    fmt='%0.1f', nlabels=3)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
setview1(scene)
ft.vis.savefig3d('result-%s-%s1.png' % (seedfile, sys.argv[4]))
#setview2(scene)
#ft.vis.savefig3d('result-%s-%s2.png' % (seedfile, sys.argv[4]))
ft.vis.show3d()
