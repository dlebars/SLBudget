#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Read monthly temperature and salinity from EN4 data. 
    Select the European coast area.
    Compute the density with GSW package.
    Output results in a netcdf file. """

import numpy as np
import xarray as xr
import gsw

Dir = '/Volumes/Elements/Data/EN4/netcdf_EN.4.2.2.analyses.g10/'

# Choose the area and time of interest
lat_min, lat_max, lon_min, lon_max = 30, 70, -20, 20
year_min, year_max = 1900, 2024

for year in range(year_min, year_max+1):
    EN4_file = 'EN.4.2.2.f.analysis.g10.'+str(year)+'*.nc'
    print('Working on file:'+EN4_file)
    EN4_d = xr.open_mfdataset(Dir+EN4_file)
    EN4_d = EN4_d.assign_coords({'lon':(((EN4_d['lon'] + 180 ) % 360) - 180)})
    EN4_d = EN4_d.sortby(EN4_d.lon)
    EN4_d = EN4_d.sel(lat=slice(lat_min,lat_max),lon=slice(lon_min,lon_max))
    temp = EN4_d.temperature.mean(dim='time') - 272.15 # Convert from Kelvin to Celsius
    sal = EN4_d.salinity.mean(dim='time')
    
    # Calculate pressure from depth, depth should be positive upward
    depth_a = np.array(temp.depth).copy()
    depth_a = depth_a.reshape( len(temp.depth), 1)
    pres = gsw.p_from_z(-depth_a, temp.lat.data)
    
    # Reshape arrays for broadcasting
    pres = np.array(pres)
    pres = pres.reshape( pres.shape[0], pres.shape[1], 1)
    lon_a = np.array(temp.lon)
    lat_a = np.array(temp.lat)
    lon_a = lon_a.reshape(1, 1, len(temp.lon))
    lat_a = lat_a.reshape(1, len(temp.lat), 1)

    # Calculate absolute salinity from practical salinity
    sa = gsw.SA_from_SP(sal, pres, lon_a, lat_a)

    # Calculate conservative temperature from potential temperature
    ct = gsw.CT_from_pt(sa, temp)

    # Calculate density, thermal expansion and haline contraction coefficients
    sa.load()
    ct.load()
    rho, alpha, beta = gsw.rho_alpha_beta(sa, ct, pres)

    # Add metadata and plot with xarray
    rho_at = {'long_name' : 'in-situ density', 'units' : 'kg/m3'}
    rho   = xr.DataArray(rho, coords=[temp.depth, temp.lat, temp.lon], \
                         dims=['depth', 'lat', 'lon'], name='density', attrs=rho_at)
    alpha_at = {'long_name' : 'thermal expansion coefficient with respect to Conservative Temperature' \
                , 'units' : '1/K' }
    alpha = xr.DataArray(alpha, coords=[temp.depth, temp.lat, temp.lon], \
                         dims=['depth', 'lat', 'lon'], name='alpha', attrs=alpha_at)
    beta_at = {'long_name' : 'saline (i.e. haline) contraction coefficient at constant'+ \
               'Conservative Temperature', 'units' : 'kg/g'}
    beta  = xr.DataArray(beta, coords=[temp.depth, temp.lat, temp.lon], \
                         dims=['depth', 'lat', 'lon'], name='beta', attrs=beta_at)
    
    if year == year_min:
        RHO = rho
        ALPHA = alpha
        BETA = beta
    else:
        RHO = xr.concat((RHO, rho),dim='time')
        ALPHA = xr.concat((ALPHA, alpha),dim='time')
        BETA = xr.concat((BETA, beta),dim='time')

# Add all variables into one dataset and export as NetCDF
DENS_d = xr.merge((RHO, ALPHA, BETA))
DENS_d['time'] = range(year_min, year_max+1)

DENS_d.to_netcdf('density_teos10_en422_g10_'+ str(year_min) + '_' + str(year_max) + '.nc')


