from pyGcode import makeGcode

g = makeGcode()
xoffset = 0 #8.5 mm
yoffset = 44.07 #94.069  mm
zoffset = 81.31 #81.307 mm

g.xstart = g.xstart + xoffset
g.ystart = g.ystart + yoffset 
g.zstart = g.zstart + zoffset 
delay = 20000 # msec
g.generateGcodefile('B', delay)