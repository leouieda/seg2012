import sys
import cPickle as pickle
import fatiando as ft
import numpy as np

log = ft.log.get()
log.info(ft.log.header())

def setview1(scene):
    scene.scene.camera.position = [-2267.5718325185544, 516.89047192363171, 325.41328402454576]
    scene.scene.camera.focal_point = [486.1565293791673, 491.11737744276104, 577.28350756789393]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.091053683141375921, -0.0033155163338156627, -0.99584046620823252]
    scene.scene.camera.clipping_range = [1653.5666619996091, 4186.5783724965167]
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
elif sys.argv[4] == 'shape':
    useshape = True
else:
    print "invalid argument"
    sys.exit()

xp, yp, zp, gxx, gxy, gxz, gyy, gyz, gzz = np.loadtxt('data.txt', unpack=True)
with open('model.pickle') as f:
    model = pickle.load(f)

bounds = [0, 1000, 0, 1000, 0, 1000]
mesh = ft.msh.ddd.PrismMesh(bounds, (30, 30, 30))
dms = ft.pot.harvester.wrapdata(mesh, xp, yp, zp, gxx=gxx, gxy=gxy, gxz=gxz,
    gyy=gyy, gyz=gyz, gzz=gzz)
seeds = ft.pot.harvester.sow(ft.pot.harvester.loadseeds(seedfile), mesh,
    mu=mu, delta=delta, useshape=useshape)

scene = ft.vis.figure3d(size=(1000, 1000))
ft.vis.prisms(model, 'density', style='wireframe', linewidth=5)
ft.vis.prisms([s.get_prism() for s in seeds], 'density', vmin=0, vmax=1000)
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[b*0.001 for b in bounds],
    fmt='%0.1f', nlabels=3)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
setview1(scene)
ft.vis.savefig3d('seeds-%s-%s1.png' % (seedfile, sys.argv[4]))
setview2(scene)
ft.vis.savefig3d('seeds-%s-%s2.png' % (seedfile, sys.argv[4]))
#ft.vis.show3d()

estimate, goals, misfits = ft.pot.harvester.harvest(dms, seeds)
mesh.addprop('density', estimate['density'])
result = ft.msh.ddd.vremove(0, 'density', mesh)

with open('results-%s-%s.pickle' % (seedfile, sys.argv[4]), 'w') as f:
    pickle.dump({'estimate':result, 'predicted':dms[-1].predicted,
        'seeds':[s.get_prism() for s in seeds]}, f)

shape = [51, 51]
ft.vis.figure(figsize=(3.33,4))
ft.vis.axis('scaled')
levels = ft.vis.contourf(yp, xp, gzz, shape, 6)
ft.vis.colorbar(orientation='horizontal', shrink=0.8)
ft.vis.contour(yp, xp, dms[-1].predicted, shape, levels, color='k',
    linewidth=1.5)
ft.vis.xlabel('y (km)')
ft.vis.ylabel('x (km)')
ft.vis.m2km()
ft.vis.savefig('fit-%s-%s.png' % (seedfile, sys.argv[4]), dpi=300)
#ft.vis.show()

scene = ft.vis.figure3d(size=(1000, 1000))
ft.vis.prisms(model, 'density', style='wireframe', linewidth=8)
ft.vis.prisms(result, 'density', vmin=0, vmax=1000)
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[b*0.001 for b in bounds],
    fmt='%0.1f', nlabels=3)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
setview1(scene)
ft.vis.savefig3d('result-%s-%s1.png' % (seedfile, sys.argv[4]))
setview2(scene)
ft.vis.savefig3d('result-%s-%s2.png' % (seedfile, sys.argv[4]))
ft.vis.show3d()
