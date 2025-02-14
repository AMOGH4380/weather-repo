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


app = Flask(__name__)


# Function to find the time variable in a dataset
def find_time_var(var, time_basename='time'):
    for coord_name in var.coords:
        if coord_name.startswith(time_basename):
            return var.coords[coord_name]
    raise ValueError('No time variable found for ' + var.name)

def fetch_gfs_data():
    # Connect to the THREDDS data server
    best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                          'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')
    return best_gfs


# Function to fetch data and plot temperature
def fetch_and_plot_temperature(best_gfs):
       
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
    time_1d = find_time_var(temp_3d)
    lat_1d = data['latitude']
    lon_1d = data['longitude']

    # Reduce the dimensions of the data and get as an array with units
    temp_2d = temp_3d.metpy.unit_array.squeeze()
    avg_temp = float(np.mean(np.array(temp_2d)))

    # Combine latitude and longitudes
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Create a new figure
    fig = plt.figure(figsize=(15, 12))

    # Add the map and set the extent
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-100.03, -111.03, 35, 43])

    # Retrieve the state boundaries using cFeature and add to plot
    ax.add_feature(cfeature.STATES, edgecolor='gray')

    # Contour temperature at each lat/long
    contours = ax.contourf(lon_2d, lat_2d, temp_2d.to('degF'), 200, transform=ccrs.PlateCarree(),
                           cmap='RdBu_r')
    # Plot a colorbar to show temperature
    fig.colorbar(contours)

    # Make a title with the time value
    ax.set_title(f'Temperature forecast (\u00b0F) for {time_1d[0].values}Z ', fontsize=20)

    # Plot markers for each lat/long to show grid points for 0.25 deg GFS
    ax.plot(lon_2d.flatten(), lat_2d.flatten(), linestyle='none', marker='o',
            color='black', markersize=2, alpha=0.3, transform=ccrs.PlateCarree())

    #plt.show()
    plt.savefig('static/temp.png')

# Function to fetch humidity data and plot
def fetch_and_plot_humidity(best_gfs):

    best_ds = list(best_gfs.datasets.values())[0]
    ncss = best_ds.subset()

    # Query data from the server
    query = ncss.query()
    query.lonlat_box(north=43, south=35, east=260, west=249).time(datetime.utcnow())
    query.accept('netcdf4')
    query.variables('Relative_humidity_height_above_ground')

    # Get the data
    data = ncss.get_data(query)
    data = xr.open_dataset(NetCDF4DataStore(data))

    # Extract necessary variables
    humidity_3d = data['Relative_humidity_height_above_ground']
    time_1d = find_time_var(humidity_3d)
    lat_1d = data['latitude']
    lon_1d = data['longitude']

    # Reduce the dimensions of the data and get as an array with units
    humidity_2d = humidity_3d.metpy.unit_array.squeeze()

    # Combine latitude and longitudes
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Create a new figure
    fig = plt.figure(figsize=(15, 12))

    # Add the map and set the extent
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-100.03, -111.03, 35, 43])

    # Retrieve the state boundaries using cFeature and add to plot
    ax.add_feature(cfeature.STATES, edgecolor='gray')

    # Contour humidity at each lat/long
    contours = ax.contourf(lon_2d, lat_2d, humidity_2d, 200, transform=ccrs.PlateCarree(),
                           cmap='Greens')
    # Plot a colorbar to show humidity
    fig.colorbar(contours)

    # Make a title with the time value
    ax.set_title(f'Relative Humidity (%) for {time_1d[0].values}Z', fontsize=20)

    # Plot markers for each lat/long to show grid points for 0.25 deg GFS
    ax.plot(lon_2d.flatten(), lat_2d.flatten(), linestyle='none', marker='o',
            color='black', markersize=2, alpha=0.3, transform=ccrs.PlateCarree())

    #plt.show()
    plt.savefig('static/humidity.png')

# Function to fetch wind speed data and plot
def fetch_and_plot_wind_speed(best_gfs):

    best_ds = list(best_gfs.datasets.values())[0]
    ncss = best_ds.subset()

    # Query data from the server
    query = ncss.query()
    query.lonlat_box(north=43, south=35, east=260, west=249).time(datetime.utcnow())
    query.accept('netcdf4')
    query.variables('Wind_speed_gust_surface')

    # Get the data
    data = ncss.get_data(query)
    data = xr.open_dataset(NetCDF4DataStore(data))

    # Extract necessary variables
    wind_3d = data['Wind_speed_gust_surface']
    time_1d = find_time_var(wind_3d)
    lat_1d = data['latitude']
    lon_1d = data['longitude']

    # Reduce the dimensions of the data and get as an array with units
    wind_2d = wind_3d.metpy.unit_array.squeeze()

    # Combine latitude and longitudes
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # Create a new figure
    fig = plt.figure(figsize=(15, 12))

    # Add the map and set the extent
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-100.03, -111.03, 35, 43])

    # Retrieve the state boundaries using cFeature and add to plot
    ax.add_feature(cfeature.STATES, edgecolor='gray')

    # Contour wind speed at each lat/long
    contours = ax.contourf(lon_2d, lat_2d, wind_2d.to('knots'), 200, transform=ccrs.PlateCarree(),
                           cmap='BuPu')
    # Plot a colorbar to show wind speed
    fig.colorbar(contours)

    # Make a title with the time value
    ax.set_title(f'Wind Speed (knots) for {time_1d[0].values}Z', fontsize=20)

    # Plot markers for each lat/long to show grid points for 0.25 deg GFS
    ax.plot(lon_2d.flatten(), lat_2d.flatten(), linestyle='none', marker='o',
            color='black', markersize=2, alpha=0.3, transform=ccrs.PlateCarree())

    #plt.show()
    plt.savefig('static/wind.png')

def fetch_data(variable_name):
    best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                          'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')
    best_ds = list(best_gfs.datasets.values())[0]
    ncss = best_ds.subset()

    query = ncss.query()
    query.lonlat_box(north=43, south=35, east=260, west=249).time(datetime.utcnow())
    query.accept('netcdf4')
    query.variables(variable_name)

    data = ncss.get_data(query)
    return xr.open_dataset(NetCDF4DataStore(data))

# Route for fetching temperature data
@app.route('/temperature')
def temperature():
    data = fetch_data('Temperature_surface')
    temp_3d = data['Temperature_surface']
    temp_2d = temp_3d.metpy.unit_array.squeeze()
    return temp_2d

# Route for fetching humidity data
@app.route('/humidity')
def humidity():
    data = fetch_data('Relative_humidity_height_above_ground')
    humidity_3d = data['Relative_humidity_height_above_ground']
    humidity_2d = humidity_3d.metpy.unit_array.squeeze()
    return humidity_2d

# Route for fetching wind speed data
@app.route('/wind_speed')
def wind_speed():
    data = fetch_data('Wind_speed_gust_surface')
    wind_3d = data['Wind_speed_gust_surface']
    wind_2d = wind_3d.metpy.unit_array.squeeze()
    return wind_2d

# Route for calculating averages
@app.route('/averages')
def averages():
    temp = temperature()
    hum = humidity()
    wind = wind_speed()
    avg_temp = float(np.mean(np.array(temp)))
    avg_humidity = float(np.mean(np.array(hum)))
    avg_wind_speed = float(np.mean(np.array(wind)))
    return jsonify({'avg_temp': avg_temp, 'avg_humidity': avg_humidity, 'avg_wind_speed': avg_wind_speed})

# Main route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Main program execution
if __name__ == '__main__':
    gfs_data = fetch_gfs_data()
    fetch_and_plot_temperature(gfs_data)
    fetch_and_plot_humidity(gfs_data)
    fetch_and_plot_wind_speed(gfs_data)
    app.run(debug=True)
