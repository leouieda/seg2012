import sys
import cPickle as pickle
import fatiando as ft
import numpy as np

def setview1(scene):
    scene.scene.camera.position = [9051932.3466249667, 520373.26722526352, -23123.157096860283]
    scene.scene.camera.focal_point = [9104478.7864473909, 597360.37386603083, 12748.208589182856]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.22373378623257814, 0.28199845385219274, -0.93296305656923417]
    scene.scene.camera.clipping_range = [12347.437305082298, 211153.9941788023]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

def setview2(scene):
    scene.scene.camera.position = [9028853.7330028508, 535186.74788464373, -696.88677720871965]
    scene.scene.camera.focal_point = [9102757.4690016806, 601731.10779025545, 8521.777620524299]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.071750864386748395, 0.058255137056722008, -0.9957199166765005]
    scene.scene.camera.clipping_range = [8517.6277181604746, 216402.58060204517]
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
ft.log.tofile(log, 'invert-%s.log' % (sys.argv[4]))
log.info(ft.log.header())

outcropx, outcropy = np.loadtxt('outcrop.xyz', unpack=True)
x, y, gz = np.loadtxt('data.xyz', unpack=True)
z = -0.1*np.ones_like(x)
shape = [200, 200]

pad = 5000
bounds = [x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad, 0, 10000]
#mesh = ft.msh.ddd.PrismMesh(bounds, (60, 56, 64))
mesh = ft.msh.ddd.PrismMesh(bounds, (30, 28, 32))

dms = ft.pot.harvester.wrapdata(mesh, x, y, z, gz=gz)

rawseeds = ft.pot.harvester.loadseeds(seedfile)
seeds = ft.pot.harvester.sow(rawseeds, mesh, mu=mu, delta=delta,
    useshape=useshape, reldist=reldist)

ft.vis.figure(figsize=(4, 5))
ft.vis.subplots_adjust(right=0.99)
ft.vis.axis('scaled')
levels = ft.vis.contourf(y, x, gz, shape, 6, interp=True)
ft.vis.colorbar(orientation='horizontal', shrink=0.7)
sx, sy = np.transpose(rawseeds)[:2]
ft.vis.plot(sy, sx, 'ow', markersize=8)
ft.vis.plot(outcropy, outcropx, '-k', linewidth=5)
ft.vis.xlabel('y (km)')
ft.vis.ylabel('x (km)')
ft.vis.m2km()
ft.vis.savefig('seeds-map-%s.png' % (sys.argv[4]), dpi=300)
#ft.vis.show()

scene = ft.vis.figure3d(size=(1400, 800))
ft.vis.prisms([s.get_prism() for s in seeds], 'density')
ft.vis.vtk.mlab.plot3d(outcropx, outcropy, np.zeros_like(outcropx),
    color=(1,0,0), tube_radius=300)
ax = ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[b*0.001 for b in bounds],
    fmt='%0.0f', nlabels=3)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
setview1(scene)
ft.vis.savefig3d('seeds-%s1.png' % (sys.argv[4]))
setview2(scene)
ft.vis.savefig3d('seeds-%s2.png' % (sys.argv[4]))
#ft.vis.show3d()
#ft.vis.vtk.mlab.close()

estimate, goals, misfits = ft.pot.harvester.harvest(dms, seeds)
mesh.addprop('density', estimate['density'])
result = ft.msh.ddd.vremove(0, 'density', mesh)

with open('results-%s.pickle' % (sys.argv[4]), 'w') as f:
    pickle.dump({'estimate':result, 'predicted':[d.predicted for d in dms],
        'seeds':[s.get_prism() for s in seeds]}, f)

ft.vis.figure(figsize=(4, 5))
ft.vis.subplots_adjust(right=0.99)
ft.vis.axis('scaled')
levels = ft.vis.contourf(y, x, gz, shape, 6, interp=True)
ft.vis.colorbar(orientation='horizontal', shrink=0.7)
ft.vis.contour(y, x, dms[-1].predicted, shape, levels, color='k', linewidth=2,
    interp=True)
#ft.vis.plot(outcropy, outcropx, '-r', linewidth=3)
ft.vis.xlabel('y (km)')
ft.vis.ylabel('x (km)')
ft.vis.m2km()
ft.vis.savefig('fit-%s.png' % (sys.argv[4]), dpi=300)
#ft.vis.show()

scene = ft.vis.figure3d(size=(1400, 800))
ft.vis.prisms([s.get_prism() for s in seeds], 'density')
ft.vis.prisms(result, 'density')
ft.vis.vtk.mlab.plot3d(outcropx, outcropy, np.zeros_like(outcropx),
    color=(1,0,0), tube_radius=300)
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[b*0.001 for b in bounds],
    fmt='%0.0f', nlabels=3)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
setview1(scene)
ft.vis.savefig3d('result-%s1.png' % (sys.argv[4]))
setview2(scene)
ft.vis.savefig3d('result-%s2.png' % (sys.argv[4]))
ft.vis.show3d()
