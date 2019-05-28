AWI ALS toolbox
===============

This python package is a software tool to parse and processs airborne laserscanner (ALS) data
from polar research aircraft of the Alfred Wegener Institute, Helmholtz Centre
for Polar and Marine Research.

The airborne laserscanner data is stored as geolocated point clouds in a custom binary file
format. A binary header contains the timing and indices information in the data block
of each file, allowing to parse subsets of its content.

In its current state, the toolbox contains the functionality to parse ALS file and create
grids from the point clouds.