.PHONY: optirun normal transparent clean

optirun-classic:
	optirun python make_frames.py classic

optirun-shape:
	optirun python make_frames.py shape

transparent:
	cp -r classic classic-trans
	mogrify -transparent white classic-trans/*.png
	cp -r shape shape-trans
	mogrify -transparent white shape-trans/*.png

gif-classic:
	convert -loop 0 \
	-delay 100 classic/model.png classic/mesh.png classic/frame-0000.png \
	-delay 10 classic/frame-*.png \
	-delay 200 classic/result.png \
	-resize 50% classic-lowres.gif

gif-shape:
	convert -loop 0 \
	-delay 100 shape/model.png shape/mesh.png shape/frame-0000.png \
	-delay 10 shape/frame-*.png \
	-delay 200 shape/result.png \
	-resize 50% shape-lowres.gif

clean:
	rm -f classic/*.png shape/*.png
