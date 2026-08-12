import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import xesmf as xe                     # package for regridding 

#import gravity toolkit 
import gravity_toolkit as gravtk

import sys
sys.path.append('../code')   

def sealevelfunctions(meltrates, lon_target, lat_target, landsea):
    #Uses as input the regridded melt values and the landseamaks with the same dimensions
    
    th = (90 - lat_target)*np.pi/180.0
    LMAX = 360
    LOVE = gravtk.load_love_numbers(LMAX)
    PLM, dPLM = gravtk.plm_holmes(LMAX, np.cos(th))
    #LOVE0 = (LOVE[0]-LOVE[0], LOVE[1]-LOVE[1], LOVE[2]-LOVE[2]) # not needed? 

    Ylms = gravtk.gen_stokes(meltrates.data.T, lon_target, lat_target, UNITS=3, LMIN=0, LMAX=LMAX, LOVE=LOVE, PLM=PLM)

    rsl = gravtk.sea_level_equation(Ylms.clm, Ylms.slm, lon_target, lat_target, landsea.data.T,
    LMAX=LMAX, PLM=PLM, LOVE=LOVE, ITERATIONS=6, POLAR=True)

    rsl_da = xr.DataArray(
        data=rsl.T,              # transpose if needed
        dims=["lat", "lon"],     # match array shape
        coords={"lat": lat_target, "lon": lon_target}
    )

    #rsl_da = rsl_da.where(landsea != 1, np.nan)

    rsl_da = rsl_da.where(landsea == 0) #Only keep the values at sea, otherwise values at land are selected

    return rsl_da


def makefrederiksefile(data, meanrise):
    '''
    Creates a file which has the same format as the data from Frederikse. In this way, the data from frederikse can be updated using this Grace data in the budget. 
    '''
    ant_costg = xr.Dataset(
        coords = {
            'lon': data['lon'].values, 
            'lat': data['lat'].values,
            'time': data['time'].values
        },
        data_vars = {
            'IS_Gravis': (['time', 'lat', 'lon'], data.values),
            'Meanslr': (['time'], meanrise.values)
        },
        attrs = {
            'Info': 'Based on Gravis data and created using the gravity toolkit',
        }
    )
    
    ant_costg['lon'].attrs = {'long_name': 'Longitude', 'units': 'Degrees East'}
    ant_costg['lat'].attrs = {'long_name': 'Latitude', 'units': 'Degrees North'}
    ant_costg['time'].attrs = {'long_name': 'time (yearly average)', 'standard_name': 'time',  'units': 'years', 'axis': 'T'}
    
    
    ant_costg['Meanslr'].attrs = {
        'long_name': 'Mean IS_Gravis over ocean',
        'description': 'Mean over all ocean grid cells (union_mask==0)',
        'units': 'cm'
    }
    return ant_costg