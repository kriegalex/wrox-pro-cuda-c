set terminal png size 640,480 enhanced font 'Verdana,10'
set output 'iso.png'

set dgrid3d 30,30
set xrange[0:NX]
set yrange[0:NY]


# splot 'heat.dat' using 1:2:3 with lines
# set hidden3d
# plot 'heat.dat' with lines

set logscale cb
set key off
plot INPUT using 2:1:3 with image
