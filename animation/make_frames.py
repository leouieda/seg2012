"""
Run the inversion with and without shape of anomaly one iteration at a time and
save snapshots.
"""
import sys
import numpy as np
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())

def setview(scene):
    scene.scene.camera.position = [-3786.8876444408993, -38969.725047229782, -8784.4024255316726]
    scene.scene.camera.focal_point = [5409.8557350004485, 5000.9649126339673, 688.33202574906738]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.034959172145713077, 0.20347947158105278, -0.97845488446222295]
    scene.scene.camera.clipping_range = [27320.983736766659, 68695.411956405063]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

def animate(pseeds, mu, delta, useshape, outdir):
    # Create a synthetic model
    model = [
        ft.msh.ddd.Prism(-2000, 12000, 4000, 6000, 2000, 4000, {'density':1000})]
    # and generate synthetic data from it
    shape = (50, 50)
    bounds = [-5000, 15000, 0, 10000, 0, 10000]
    area = bounds[0:4]
    xp, yp, zp = ft.grd.regular(area, shape, z=-1)
    noise = 0.1 # mGal noise
    gz = ft.utils.contaminate(ft.pot.prism.gz(xp, yp, zp, model), noise)

    # Create a mesh
    mesh = ft.msh.ddd.PrismMesh(bounds, (15, 15, 30))
    # Make the data modules
    dms = ft.pot.harvester.wrapdata(mesh, xp, yp, zp, gz=gz)
    # Make the seed and set the compactness regularizing parameter mu
    seeds = ft.pot.harvester.sow(pseeds, mesh, mu=mu, delta=delta,
        useshape=useshape)

    # Setup the 3D plot
    vmin, vmax = 0, 1000
    scene = ft.vis.figure3d(size=(700, 700))
    #ft.vis.prisms(model, 'density', style='wireframe', linewidth=4, vmin=vmin,
        #vmax=vmax, cmap='binary')
    ft.vis.prisms(model, 'density', linewidth=2, vmin=vmin, vmax=vmax)
    ft.vis.outline3d(bounds)
    ft.vis.wall_bottom(bounds)
    ft.vis.wall_east(bounds)
    X = np.reshape(xp, shape, order='F')
    Y = np.reshape(yp, shape, order='F')
    GZ = np.reshape(gz, shape, order='F')
    p = ft.vis.vtk.mlab.contour_surf(X, Y, -GZ, contours=10,
        colormap='gist_rainbow')
    p.contour.filled_contours = True
    p.actor.actor.position = (0, 0, -7000)
    scale = 200
    p.actor.actor.scale = (1, 1, scale)
    pred_pos = -2000
    setview(scene)
    ft.vis.savefig3d('%s/model.png' % (outdir))

    # Plot the mesh first and then remove it
    ft.vis.prisms(mesh, cmap='binary')
    setview(scene)
    ft.vis.savefig3d('%s/mesh.png' % (outdir))
    scene.children[-1].remove()

    # Remove the solid model and plot only the wireframe
    scene.children[0].remove()
    ft.vis.prisms(model, 'density', style='wireframe', linewidth=4, vmin=vmin,
        vmax=vmax)
    ft.vis.outline3d(bounds)

    # Run the inversion step by step
    for i, chset in enumerate(ft.pot.harvester.harvest(dms, seeds, iterate=True)):
        index, props, dms, seeds = chset
        if index is None:
            new = [s.get_prism() for s in seeds]
        else:
            new = [mesh[index]]
            new[0].addprop('density', props['density'])
            scene.children[-1].remove() # Remove the old predicted data
            scene.children[-1].remove() # Remove the old neighbors
        neighbors = []
        for seed in seeds:
            for n in seed.neighbors:
                neighbors.append(mesh[n])
        ft.vis.prisms(new, 'density', vmin=vmin, vmax=vmax)
        ft.vis.prisms(neighbors, style="wireframe", linewidth=1, cmap='black-white')
        data = np.reshape(dms[0].get_predicted(), shape, order='F')
        p = ft.vis.vtk.mlab.contour_surf(X, Y, -data, contours=10,
                                         colormap='gist_rainbow')
        p.contour.filled_contours = True
        p.actor.actor.position = (0, 0, pred_pos)
        p.actor.actor.scale = (1, 1, scale)
        setview(scene)
        ft.vis.savefig3d('%s/frame-%04d.png' % (outdir, i))
    scene.children[-2].remove() # Remove the neighbors
    setview(scene)
    ft.vis.savefig3d('%s/result.png' % (outdir))
    ft.vis.vtk.mlab.close(scene)

    #estimate, goals, misfits = ft.pot.harvester.harvest(dms, seeds)
    #mesh.addprop('density', estimate['density'])
    #ft.vis.prisms(ft.msh.ddd.vremove(0, 'density', mesh), 'density', vmin=vmin,
        #vmax=vmax)
    #p = ft.vis.vtk.mlab.contour_surf(X, Y,
        #-np.reshape(dms[0].get_predicted(), shape, order='F'),
        #contours=10, colormap='gist_rainbow')
    #p.contour.filled_contours = True
    #p.actor.actor.position = (0, 0, pred_pos)
    #p.actor.actor.scale = (1, 1, scale)
    #setview(scene)
    #ft.vis.show3d()

if __name__ == '__main__':
    if sys.argv[1] == 'classic':
        animate([[5000, 5000, 2500, {'density':1000}]], 0.05, 0.001, False,
            'classic')
    elif sys.argv[1] == 'shape':
        animate([[5000, 5000, 2500, {'density':1000}]], 3, 0.001, True,
            'shape')
    else:
        print "Unknown command %s" % (sys.argv[1])
