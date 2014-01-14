import cPickle as pickle
import fatiando as ft
import numpy as np

log = ft.log.get()
ft.log.tofile(log, 'datagen.log')
log.info(ft.log.header())

props = {'density':1000}
model = [ft.msh.ddd.Prism(400, 600, 300, 500, 200, 400, props),
         ft.msh.ddd.Prism(400, 600, 400, 600, 400, 600, props),
         ft.msh.ddd.Prism(400, 600, 500, 700, 600, 800, props)]
with open('model.pickle', 'w') as f:
    pickle.dump(model, f)

shape = (51, 51)
bounds = [0, 1000, 0, 1000, 0, 1000]
area = bounds[0:4]
noise = 0.5
x, y, z = ft.grd.regular(area, shape, z=-150)
tensor = [ft.pot.prism.gxx(x, y, z, model),
          ft.pot.prism.gxy(x, y, z, model),
          ft.pot.prism.gxz(x, y, z, model),
          ft.pot.prism.gyy(x, y, z, model),
          ft.pot.prism.gyz(x, y, z, model),
          ft.pot.prism.gzz(x, y, z, model)]
tensor_noisy = [ft.utils.contaminate(d, noise) for d in tensor]
data = [x, y, z]
data.extend(tensor_noisy)

with open('data.txt', 'w') as f:
    f.write("# Noise corrupted tensor components:\n")
    f.write("#   noise = %g Eotvos\n" % (noise))
    f.write("# x   y   z   gxx   gxy   gxz   gyy   gyz   gzz\n")
    np.savetxt(f, np.transpose(data))

titles = "gxx   gxy   gxz   gyy   gyz   gzz".split()
for i in xrange(6):
    ft.vis.figure(figsize=(3.33,4))
    ft.vis.axis('scaled')
    ft.vis.title(titles[i])
    levels = ft.vis.contourf(y, x, tensor_noisy[i], shape, 6)
    ft.vis.colorbar(orientation='horizontal', shrink=0.8)
    ft.vis.xlabel('y (km)')
    ft.vis.ylabel('x (km)')
    ft.vis.m2km()
    ft.vis.savefig('%s.png' % (titles[i]), dpi=300)
#ft.vis.show()

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

scene = ft.vis.figure3d(size=(1000, 1000))
ft.vis.prisms(model, 'density', linewidth=2)
ft.vis.axes3d(ft.vis.outline3d(bounds), ranges=[b*0.001 for b in bounds],
    fmt='%0.1f', nlabels=3)
ft.vis.wall_bottom(bounds)
ft.vis.wall_north(bounds)
setview1(scene)
ft.vis.savefig3d('model1.png')
setview2(scene)
ft.vis.savefig3d('model2.png')
#ft.vis.show3d()
