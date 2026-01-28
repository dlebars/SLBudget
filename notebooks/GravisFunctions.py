import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import xesmf as xe                     # package for regridding 

#import gravity toolkit 
import gravity_toolkit as gravtk

import sys
sys.path.append('../code')


def yearlygrace(gracedata):
    '''
    Uses the gridded data from Gravis (greenland or antarctica) as an input value and then creates a similar dataset using the same layout. 
    '''
    
    # Drop dm, time_dec, and the old time coordinate: this is everything that dempends on time
    ant_y = gracedata.drop_vars(['dm', 'time_dec', 'time'], errors='ignore')

    # Resample dm to yearly
    dm_y = gracedata['dm'].resample(time='1Y').mean(keep_attrs=True)
    # Ensure correct order
    dm_y = dm_y.transpose('time', 'y', 'x')

    # Assign dm back
    ant_y = ant_y.assign({'dm': dm_y})

    # Assign the time coordinate from dm_y (now no conflicts)
    ant_y = ant_y.assign_coords(time=dm_y['time'])

    # Compute decimal year as a NumPy array
    time_dec = ant_y["time"].dt.year 

    # Add it as a new variable (not just a coordinate)
    ant_y["time_dec"] = ("time", time_dec.data)

    # Add metadata (optional but good practice)
    ant_y["time_dec"].attrs = {
        "long_name": "Decimal year",
        "units": "year"
    }

    return ant_y 

    

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
            'time': data['time'].dt.year.values
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
        'units': 'm'
    }
    return ant_costg