#!/bin/bash

rm -f out.mp4
cd sph_copy
convert '*.png[640x]' resized_%04d.jpg
cd ..
ffmpeg -i sph_copy/resized_%4d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
