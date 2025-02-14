
from siphon.catalog import TDSCatalog
from datetime import datetime
from xarray.backends import NetCDF4DataStore
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.units import units
from metpy.plots import ctables
from flask import Flask, jsonify, render_template

# Function to find the time variable in a dataset
def find_time_var(var, time_basename='time'):
    for coord_name in var.coords:
        if coord_name.startswith(time_basename):
            return var.coords[coord_name]
    raise ValueError('No time variable found for ' + var.name)



best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                          'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')

best_ds = list(best_gfs.datasets.values())[0]
ncss = best_ds.subset()

# Query data from the server
query = ncss.query()
query.lonlat_box(north=43, south=35, east=260, west=249).time(datetime.utcnow())
query.accept('netcdf4')
query.variables('Temperature_surface')

# Get the data
data = ncss.get_data(query)
data = xr.open_dataset(NetCDF4DataStore(data))

# Extract necessary variables
temp_3d = data['Temperature_surface']
#print (temp_3d)
time_1d = find_time_var(temp_3d)
lat_1d = data['latitude']
lon_1d = data['longitude']

# Reduce the dimensions of the data and get as an array with units
temp_2d = temp_3d.metpy.unit_array.squeeze()
avg_temp = float(np.mean(np.array(temp_2d)))
print (temp_2d)
print (avg_temp -273)

