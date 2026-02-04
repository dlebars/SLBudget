# List of functions to be used in sea level budget

import datetime
import netCDF4
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
import gzip
import xesmf as xe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import regionmask

import statsmodels.api as sm
lowess = sm.nonparametric.lowess

PATH_SLBudgets_data = '../data/'
PATH_Data = '/Users/dewilebars/Data/'
tg_data_dir = f'{PATH_SLBudgets_data}rlr_annual'

# Define a few constants
er = 6.371e6 # Earth's radius in meters
oa = 3.6704e14 # Total ocean area m**2
rho_o = 1030 # Density of ocean water
g = 9.81 # Acceleration of gravity

def find_closest(lat, lon, lat_i, lon_i):
    """lookup the index of the closest lat/lon"""
    Lon, Lat = np.meshgrid(lon, lat)
    idx = np.argmin(((Lat - lat_i)**2 + (Lon - lon_i)**2))
    Lat.ravel()[idx], Lon.ravel()[idx]
    [i, j] = np.unravel_index(idx, Lat.shape)
    return i, j

def make_wind_df(lat_i, lon_i, product):
    """create a dataset for NCEP1 wind (1948-now) at 1 latitude/longitude point 
    or ERA5 (1950-now) or 20CR reanalysis"""
    
    if product == 'NCEP1':
        # Use for OpenDAP:
        #NCEP1 = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/'
        #For local:
        NCEP1_dir = PATH_SLBudgets_data + 'WindPressure/NCEP1/'
        u_file = NCEP1_dir + 'uwnd.10m.mon.mean.nc'
        v_file = NCEP1_dir + 'vwnd.10m.mon.mean.nc'
        p_file = NCEP1_dir + 'pres.sfc.mon.mean.nc'
        latn = 'lat'
        lonn = 'lon'
        timen = 'time'
        un = 'uwnd'
        vn = 'vwnd'
        pn = 'pres'
        
    elif product == 'ERA5':
        ERA5_dir = PATH_SLBudgets_data + 'WindPressure/ERA5/'      
        u_file = [ERA5_dir + 'ERA5_u10.nc']
        v_file = [ERA5_dir + 'ERA5_v10.nc']
        p_file = [ERA5_dir + 'ERA5_msl.nc']
        latn = 'latitude'
        lonn = 'longitude'
        timen = 'valid_time'
        un = 'u10'
        vn = 'v10'
        pn = 'msl'
    
    elif product == '20CR':
        TCR_dir = PATH_SLBudgets_data + 'WindPressure/20CR/'
        u_file = TCR_dir + 'uwnd.10m.mon.mean.nc'
        v_file = TCR_dir + 'vwnd.10m.mon.mean.nc'
        p_file = TCR_dir + 'prmsl.mon.mean.nc'
        latn = 'lat'
        lonn = 'lon'
        timen = 'time'
        un = 'uwnd'
        vn = 'vwnd'
        pn = 'prmsl'
    
    if lon_i < 0:
        lon_i = lon_i + 360
    
    # open the 3 files
    ds_u = xr.open_mfdataset(u_file)
    ds_v = xr.open_mfdataset(v_file)
    ds_p = xr.open_mfdataset(p_file)
    
    # read lat, lon, time from 1 dataset
    lat, lon = ds_u[latn][:], ds_u[lonn][:]
    
    # this is the index where we want our data
    i, j = find_closest(lat, lon, lat_i, lon_i)
    
    # get the u, v, p variables
    print('found point', float(lat[i]), float(lon[j]))    
    u = ds_u[un][:, i, j]
    v = ds_v[vn][:, i, j]
    pres = ds_p[pn][:, i, j]
    pres = pres - pres.mean()
    
    # compute derived quantities
    speed = np.sqrt(u**2 + v**2)
    
    # Inverse barometer effect in cm
    ibe = - pres/(rho_o*g)*100
    ibe = ibe - ibe.mean()
    
    # compute direction in 0-2pi domain
    direction = np.mod(np.angle(u + v * 1j), 2*np.pi)
    
    # Compute the wind squared while retaining sign, as an approximation of stress
    #u2 = u**2 * np.sign(u)
    #v2 = v**2 * np.sign(v)
    wind_magnitude = np.sqrt(u**2 + v**2)
    u2 = wind_magnitude*u
    v2 = wind_magnitude*v
    
    # put everything in a dataframe
    wind_df = pd.DataFrame(data=dict(u=u, v=v, t=u[timen], speed=speed, 
                                     direction=direction, u2=u2, v2=v2, 
                                     pres=pres, ibe=ibe))
    wind_df = wind_df.set_index('t')
    
    annual_wind_df = wind_df.groupby(wind_df.index.year).mean()
    
    return annual_wind_df

def linear_model_zsm(df, with_trend=True, with_nodal=True, with_wind=True, with_pres=True, with_ar=False):
    ''' Define the statistical model, similar to zeespiegelmonitor'''
    
    t = np.array(df.index)
    y = df['height']
    X = np.ones(len(t))
    names = ['Constant']
    if with_wind:
        X = np.c_[ X, df['u2'], df['v2']]
        names.extend(['Wind $u^2$', 'Wind $v^2$'])
    if with_pres:
        X = np.c_[X, df['pres']]
        names.extend(['Pressure'])
    if with_nodal:
        X = np.c_[ X, np.cos(2*np.pi*(t - 1970)/18.613), np.sin(2*np.pi*(t - 1970)/18.613)]
        names.extend(['Nodal U', 'Nodal V'])
    if with_trend:
        X = np.c_[X, t - 1970 ]
        names.extend(['Trend'])
    if with_ar:
        model = sm.GLSAR(y, X, missing='drop', rho=1)
    else:
        model = sm.OLS(y, X, missing='drop')
    fit = model.fit(cov_type='HC0')
    
    return fit, names


def make_wpn_ef(coord, tgm_df, with_nodal, with_trend, product):
    '''Prepare a dataframe of wind, pressure and nodal cycle influences on sea level'''
    
    annual_wind_df = make_wind_df(coord[0], coord[1], product)
    df_c = tgm_df.join(annual_wind_df, how='inner')
    df_c.index.names = ['year']
    linear_fit, names = linear_model_zsm(df_c, with_trend, with_nodal, 
                                         with_wind=True, with_pres=True, with_ar=False)
    
    mod = np.array(linear_fit.params[:]) * np.array(linear_fit.model.exog[:,:])
    
    wpn_ef_df = pd.DataFrame(index=df_c.index)
    
    if with_nodal:
        wpn_ef_df['Nodal'] = mod[:,[4,5]].sum(axis=1)
    
    
    wpn_ef_df['Wind'] = mod[:,[1,2]].sum(axis=1)
    wpn_ef_df['Pressure'] = mod[:,3]
    
    return wpn_ef_df

def make_waqua_df(tg_id):
    '''Read time series of annually averaged sea level from the WAQUA model forced by ERA-interim.'''
    dir_waqua = PATH_SLBudgets_data + 'DataWAQUANinaERAI'
    ds_wa = netCDF4.Dataset(dir_waqua+'/ERAintWAQUA_waterlevels_speed_1979_2015.nc')

    # Get WAQUA tide gauge names that are not editted in the same way as PSMSL names
    names_col = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality')
    filelist_df = pd.read_csv(tg_data_dir + '/filelist.txt', sep=';', header=None, names=names_col)
    filelist_df = filelist_df.set_index('id')
    tgn = filelist_df.name[tg_id].replace('-', '').replace(' ', '')
    tgn = tgn.lower()[:8]
    
    dh = ds_wa[tgn + '/WAQUA_surge'][:]*100
    time_wa = ds_wa['time'][:]
    t_wa = netCDF4.num2date(time_wa, ds_wa.variables['time'].units)

    t_wa_y = np.empty_like(t_wa)
    
    for i in range(len(t_wa)):
        t_wa_y[i] = t_wa[i].year
        
    waqua_df = pd.DataFrame( data = dict( time=t_wa_y, WindPressure=dh.data) )
    waqua_df = waqua_df.set_index('time')
    
    return waqua_df

def make_gtsm_df(tg_id, var):
    ''' 
    Read GTSM yearly averaged data and return a dataframe.
    var: waterlevel, surge 
    '''
    gtsm_df = pd.read_csv('../data/GTSM_yearly/reanalysis_mean_nl.csv', index_col=0, parse_dates=['time'])
    # Change column names to the tide gauge IDs
    stations = [25, 32, 20, 24, 23, 22]
    gtsm_df.columns = stations + list(gtsm_df.columns[-3:])
    gtsm_df['year'] = gtsm_df.time.dt.year
    
    df = gtsm_df.query(f'variable == "{var}"').sort_values('time')
    del(df['time'])
    df.rename(columns={'year':'time'}, inplace=True)
    df = df.groupby('time').agg('mean').reset_index()
    df.set_index('time', inplace=True)

    df = df*100 # Convert from m to cm
    df = df[[tg_id]]
    df.rename(columns={tg_id:'WindPressure'}, inplace=True)
    
    return df

def read_tg_info():
    '''Read csv file containing the information about all tide gauges and export 
    as pandas dataframe'''
    
    names_col = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality')
    filelist_df = pd.read_csv(f'{tg_data_dir}/filelist.txt', sep=';', 
                              header=None, names=names_col)
    filelist_df = filelist_df.set_index('id')
    
    filelist_df['name'] = [n.strip() for n in filelist_df['name']]
    
    return filelist_df

def tide_gauge_obs(tg_id=[20, 22, 23, 24, 25, 32], interp=False):
    '''Read a list of tide gauge data and compute the average. 
    Set interp to True for a linear interpollation of missing values.
    By default use the 6 tide gauges from the Zeespiegelmonitor''' 

    names_col2 = ('time', 'height', 'interpolated', 'flags')

    for i in range(len(tg_id)):
        print(tg_id[i])
        tg_data = pd.read_csv(f'{tg_data_dir}/data/{tg_id[i]}.rlrdata', 
                              sep=';', header=None, names=names_col2)
        tg_data = tg_data.set_index('time')
        tg_data.height = tg_data.height.where(~np.isclose(tg_data.height,-99999))

        #Option to select data from 1960:
        #tg_data = tg_data.loc[1960:]

        tg_data.height = tg_data.height - tg_data.height.mean()

        if i==0:
            tg_data_df = pd.DataFrame(data=dict(time=tg_data.index, 
                                                col_name=tg_data.height))
            tg_data_df = tg_data_df.set_index('time')
            tg_data_df.columns  = [tg_id[i]] 
        else:
            tg_data_df[tg_id[i]] = tg_data.height

    if interp:
        tg_data_df = tg_data_df.interpolate(method='slinear')
        
    tg_data_df['Average'] = tg_data_df.mean(axis=1)
    tg_data_df = tg_data_df * 0.1 # Convert from mm to cm
    
    return tg_data_df

def rotate_longitude(ds, name_lon):

    ds = ds.assign_coords({name_lon:(((ds[name_lon] + 180 ) % 360) - 180)})
    ds = ds.sortby(ds[name_lon])

    return ds

def altimetry_obs(location, box):
    '''Read satellite altimetry data at a list of tide gauge locations
    if location is a list of within a region it location is a polygone.
    Compute the yearly average sea level and the average of the list of 
    tide gauge locations or region.
    
    Give a box value of 0 to select the closest altimetry point to the tide 
    gauge and larger than 0 to include the average altimetry field over 
    an area. The value of area used is:
    lat-box: lat+box
    lon-box: lon+box
    with box in degrees.''' 

    duacs_dir = '~/Data/duacs_cmems/'
    file_name = f'{duacs_dir}cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1M-m_*.nc'
    duacs_ds = xr.open_mfdataset(file_name, chunks={'time':6})  #.load()
    duacs_ds = rotate_longitude(duacs_ds, 'longitude')
    duacs_ds['sla'] = duacs_ds.sla*100 # Convert from meter to cm
    duacs_y_ds = duacs_ds.groupby('time.year').mean()

    df = pd.DataFrame(index=pd.Series(duacs_y_ds.year.values, name="time"))
    
    if type(location) == list:
        # Working with tide gauge IDs
        
        for i in range(len(location)):
            geo_coord = tg_lat_lon(location[i])
        
            if box==0:
                ts = duacs_y_ds.sel(latitude=geo_coord[0], longitude=geo_coord[1], 
                                    method="nearest")
            else:
                ts = duacs_y_ds.sel(latitude=slice(geo_coord[0]-box, geo_coord[0]+box), 
                                    longitude=slice(geo_coord[1]-box, geo_coord[1]+box)
                                   ).mean(dim=["latitude","longitude"])
        
            df[location[i]] = ts.sla
        
    elif type(location) == np.ndarray:
        # Working with a region defined by a polygone
        
        region = regionmask.Regions([location], names=['reg'], abbrevs=['reg'])
        
        # Define the mask and change its value from 0 to 1
        mask_alti = region.mask_3D(duacs_y_ds.longitude, duacs_y_ds.latitude)
        
        duacs_y_ds_m = duacs_y_ds.where(mask_alti)
        
        # Calculate the weighted regional average
        # !!! Only works with regular grids
        weights = np.cos(np.deg2rad(duacs_y_ds.latitude))
        region_average = duacs_y_ds.weighted(mask_alti * weights).mean(dim=('latitude', 'longitude'))
        df['region_average'] = region_average.sla
    
    df['Average'] = df.mean(axis=1)
    
    return df

def steric_masks_north_sea(da, mask_name):
    '''Define a few masks to use to compute the steric expansion that is felt 
    in the North Sea.
    The input data array needs to have a latitude/longitude coordinates with
    longitudes from -180 to 180.'''
    
    if mask_name == 'ENS':
        # Extended North Sea mask
        lat = np.array(da.lat)
        lon = np.array(da.lon)
        LatAr = np.repeat(lat[:,np.newaxis], len(lon), 1)
        LatAr = xr.DataArray(LatAr, dims=['lat', 'lon'], 
                             coords={'lat' : lat, 'lon' : lon})
        LonAr = np.repeat(lon[np.newaxis,:], len(lat), 0)
        LonAr = xr.DataArray(LonAr, dims=['lat', 'lon'], 
                             coords={'lat' : lat, 'lon' : lon})

        mask_med = xr.where(np.isnan(da[0,0,:,:]), np.nan, 1)
        mask_med1 = mask_med.where((LonAr >= -8) & (LatAr <= 42) )
        mask_med1 = xr.where(np.isnan(mask_med1), 1, np.nan)
        mask_med2 = mask_med.where((LonAr >= 1) & (LatAr <= 48) )
        mask_med2 = xr.where(np.isnan(mask_med2), 1, np.nan)
        mask_med = mask_med * mask_med1 * mask_med2

        mask = xr.where(np.isnan(da[0,0,:,:]), np.nan, 1)
        mask = mask.where(mask.lon <= 7)
        mask = mask.where(mask.lon >= -16)
        mask = mask.where(mask.lat <= 69) #Normal value: 60 or 69
        mask = mask.where(mask.lat >= 33)
        mask = mask * mask_med

    elif mask_name == 'EBB':
        # Extended bay of Biscay
        mask = xr.where(np.isnan(da[0,:,:,:]
                                 .sel(depth=2000, method='nearest')), np.nan, 1)
        mask = mask.where(mask.lon <= -2)
        mask = mask.where(mask.lon >= -12)
        mask = mask.where(mask.lat <= 52)
        mask = mask.where(mask.lat >= 35)

    elif mask_name == 'BB':
        # Bay of Biscay
        mask = xr.where(np.isnan(da[0,:,:,:]
                                 .sel(depth=500, method='nearest')), np.nan, 1)
        mask = mask.where(mask.lon <= 0)
        mask = mask.where(mask.lon >= -10)
        mask = mask.where(mask.lat <= 50)
        mask = mask.where(mask.lat >= 44)    
        
    elif mask_name == 'NWS':
        # Norwegian Sea
        mask = xr.where(np.isnan(da[0,:,:,:]
                                 .sel(depth=2000, method='nearest')), np.nan, 1)
        mask = mask.where(mask.lon <= 8)
        mask = mask.where(mask.lon >= -10)
        mask = mask.where(mask.lat <= 69)
        mask = mask.where(mask.lat >= 60)
        
    else:
        print('ERROR: mask_name argument is not available')

    del mask['depth']
    del mask['time']
    
    return mask

def thickness_from_depth(depth):
    '''Define a thickness Data Array from depth coordinate'''
    
    midp = (np.array(depth[1:])+np.array(depth[:-1]))/2
    midp = np.insert(midp, 0, np.array([0]))
    midp = np.insert(midp, len(midp), 
                     np.array(depth[-1]) 
                     + (np.array(depth[-1]) - np.array(depth[-2])))
    thick = midp[1:] - midp[:-1]
    thick_da = xr.DataArray(thick, coords={'depth': depth[:]}, dims='depth')
    
    return thick_da

def read_density(data_source):
    '''Read the density dataset from IAP, EN4_21 or EN4_22'''
    
    if data_source == 'IAP':
        density_ds = xr.open_mfdataset(PATH_SLBudgets_data+
                        'DataSteric/density_teos10_IAP/density_teos10_iap_*.nc')
    elif data_source == 'EN4_21':
        density_ds = xr.open_dataset(PATH_SLBudgets_data + 
                       'DataSteric/density_teos10_EN421f_analysis_g10/' + 
                       'density_teos10_en4_1900_2019.nc')
    elif data_source == 'EN4_22':
        density_ds = xr.open_dataset(PATH_SLBudgets_data + 
                       'DataSteric/density_teos10_en422_g10_1900_2024.nc')
    else:
        print('ERROR: data_source not defined')
        
    return density_ds
    
def StericSL(data_source, mask_name, min_depth, max_depth, window):
    '''Compute the steric sea level in cm integrated between two depth levels 
    given in meters. '''
    
    density_ds = read_density(data_source)
    thick_da = thickness_from_depth(density_ds.depth)
    SumDens = density_ds.density * thick_da

    mask = steric_masks_north_sea(density_ds.density, mask_name)
    
    SumDens = (SumDens * mask).mean(dim=['lat', 'lon'])
    rho_0 = (density_ds.density[0 ,0 ,: ,:] * mask).mean(dim=['lat', 'lon'])
    StericSL = (- SumDens.sel(depth=slice(min_depth,max_depth)).sum(dim='depth') 
                / rho_0) * 100
    
    StericSL = StericSL - StericSL.sel(time=slice(1940,1960)).mean(dim='time')
    StericSL.name = 'Steric'
    StericSL_df = StericSL.to_dataframe()
    del StericSL_df['depth']
    
    if window > 1:
        frac = window/StericSL_df.shape[0]
        StericSL_df['Steric'] = lowess(StericSL_df['Steric'], StericSL_df.index, 
                                       frac, return_sorted=False)
    
    return StericSL_df

def speed2height_ts(variable_name, speed):
    '''Convert a speed float to a height time series dataframe'''
    
    time = np.arange(1900, 2030)
    ts = speed * (time - time[0])
    ts_list = [("time", time),(variable_name, ts)]
    ts_df = pd.DataFrame.from_dict(dict(ts_list))
    ts_df = ts_df.set_index("time")
    
    return ts_df

def GIA_ICE6G(tg_id):
    '''Read the current GIA 250kaBP-250kaAP from the ICE6G model and output a
    time series in a pandas dataframe format'''
    
    dir_ICE6G = PATH_SLBudgets_data + "GIA/ICE6G/"
    locat = []
    gia = []
    
    # This file is difficult to read with Pandas because the number of rows 
    # varies. The name of location have spaces and delimiter is space... 
    with open (dir_ICE6G + "drsl.PSMSL.ICE6G_C_VM5a_O512.txt", "r") as myfile:
        data = myfile.readlines()
        
    for i in range(7,len(data)):
        line = data[i].split()
        locat.append(line[2])
        gia.append(line[-1])
        
    # Now build a pandas dataframe from these lists
    gia_list = [("Location", locat),
                ("GIA", gia)]
    gia_df = pd.DataFrame.from_dict(dict(gia_list))
    gia_df.Location = gia_df.Location.astype(int)
    gia_df.GIA = gia_df.GIA.astype(float)
    gia_df = gia_df.set_index("Location")
    gia_df = gia_df.sort_index()
    gia_avg = (gia_df.loc[tg_id]).GIA.mean() /10 # Convert from mm/y to cm/y
    
    gia_ts_df = speed2height_ts("GIA", gia_avg)
    
    return gia_ts_df

def GIA_ICE6G_region(location):
    '''Input region, output weigthed averaged influence of GIA on absolute sea 
    level.
    GIA has a small influence on absolute sea level. This is the sum of radial
    velocities and relative sea level which are given for ICE6G.'''
    
    dir_ICE6G = f'{PATH_SLBudgets_data}GIA/ICE6G/'
    
    drad_ds = xr.open_dataset(f'{dir_ICE6G}drad.1grid_O512.nc')
    dsea_ds = xr.open_dataset(f'{dir_ICE6G}dsea.1grid_O512.nc')
    
    abs_da = dsea_ds.Dsea_250+drad_ds.Drad_250
    abs_da = abs_da.sortby(abs_da['Lat'])

    region = regionmask.Regions([location], names=['reg'], abbrevs=['reg'])
    
    mask = region.mask_3D(abs_da.Lon, abs_da.Lat)

    masked_abs_da = abs_da.where(mask)
    
    # Compute the weithed average
    weights = np.cos(np.deg2rad(masked_abs_da.Lat))
    region_mean = masked_abs_da.weighted(weights).mean(dim=("Lat", "Lon"))
    region_mean = region_mean/10 # Convert from mm/yr to cm/yr
    
    region_mean_ts_df = speed2height_ts('GIA', region_mean)
    
    return region_mean_ts_df

def tg_lat_lon(tg_id):
    '''Return tide gauge latitude, longitude location given the id as input'''
    
    tg_data_dir = PATH_SLBudgets_data + 'rlr_annual'
    names_col = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality')
    filelist_df = pd.read_csv(tg_data_dir + '/filelist.txt', sep=';', header=None, names=names_col)
    filelist_df = filelist_df.set_index('id')
    
    return filelist_df.loc[tg_id].lat.item(), filelist_df.loc[tg_id].lon.item()

def glaciers_m15_glo():
    '''Provides glacier contributions to local sea level between 1900 and 2013
    from Marzeion et al. 2015.'''
    
    M15_dir = PATH_SLBudgets_data + 'Glaciers/Marzeion2015/tc-9-2399-2015-supplement/'
    M15_glo_df = pd.read_csv(M15_dir + 'data_marzeion_etal_update_2015.txt', 
                             header=None, 
                             names=['time', 'Glaciers', 'CI' ], delim_whitespace=True)
    M15_glo_df = M15_glo_df.set_index('time')
    M15_glo_df['Glaciers'] = - M15_glo_df.Glaciers + M15_glo_df.Glaciers.iloc[0]
    del M15_glo_df['CI']
    M15_glo_df = M15_glo_df/10 # Convert from mm to cm
    M15_glo_df.index = np.array(M15_glo_df.index).astype(int)
    
    return M15_glo_df 

def glaciers_m15(tg_id, extrap=False, del_green=False):
    '''Provides glacier contributions to local sea level between 1900 and 2013. 
    Glacier mass loss is from Marzeion et al. 2015. Fingerprint for Randolph 
    Glacier Inventory regions are from Frederikse et al. 2016.
    If tg_id is None then the global sea level contribution is computed. 
    Gives the possibility to extrapollate values a few years in the future based 
    on trend of the last 10 years: extrap=True
    Give the possibility to exclude Greenland peripheral glaciers, del_green = True,
    this is not possible with the glaciers_m15_glo function.
    This option is handy if the peripheral glaciers are already included in the
    Greenland ice sheet contribution.'''
    
    M15_dir = PATH_SLBudgets_data + 'Glaciers/Marzeion2015/tc-9-2399-2015-supplement/'
    fp_dir = PATH_SLBudgets_data + 'fp_uniform/'
    RGl = []
    for i in range(1,19):
        RGl.append('RG'+str(i))
    M15_reg_df = pd.read_csv(M15_dir + 'data_marzeion_etal_update_2015_regional.txt', 
                             header=None, names=['time'] + RGl, delim_whitespace=True)
    M15_reg_df = M15_reg_df.set_index('time')
    M15_reg_df = - M15_reg_df.cumsum() # Data is in mm/y so needs to be cumulated
    M15_regloc_df = M15_reg_df.copy()
    if tg_id is not None:
        RGI_loc = np.ones(len(tg_id))
        for i in range(1,19):
            filename = 'RGI_'+ str(i) +'.nc'
            RGI = xr.open_dataset(fp_dir + filename)
            for j in range(len(tg_id)):
                tg_lat, tg_lon =  tg_lat_lon(tg_id[j])
                RGI_loc[j] = RGI.rsl.sel(x = tg_lon, y = tg_lat, 
                                         method='nearest').values    
            M15_regloc_df['RG' + str(i)] = M15_regloc_df['RG' + str(i)] * RGI_loc.mean()
    if del_green:
        del M15_regloc_df['RG5']
    M15_regloc_df['Total'] = M15_regloc_df.sum(axis=1)
    M15_regloc_tot_df = pd.DataFrame(data=dict( Glaciers=M15_regloc_df.Total))
    if extrap:
        nby = 10
        trend = np.polyfit(M15_regloc_tot_df.index[-nby:], 
                           M15_regloc_tot_df.Glaciers.iloc[-nby:], 1)[0]
        for i in range(6):
            M15_regloc_tot_df.loc[M15_regloc_tot_df.index.max() + 1] = \
            M15_regloc_tot_df.Glaciers.iloc[-1] + trend
            
    return M15_regloc_tot_df/10 # Convert to cm

def glaciers_zemp19_glo():
    '''Provides glacier contributions to local sea level between 1962 and 2016
    from Zemp et al. 2019'''
    data_dir = (PATH_SLBudgets_data + 
                'Glaciers/Zemp2019/Zemp_etal_results_regions_global_v11/')
    zemp_df = pd.read_csv(data_dir + 'Zemp_etal_results_global.csv', 
                          skiprows=19)
    zemp_df = zemp_df.set_index('Year')
    zemp_df.columns = [i.strip() for i in zemp_df.columns]
    zemp_df = zemp_df['INT_SLE'].cumsum()/10 # Convert from mm to cm
    zemp_df = pd.DataFrame(data={'Glaciers': zemp_df})
    return zemp_df

def glaciers_zemp19(tg_id, extrap=False, del_green=False):
    '''Provides glacier contributions to local sea level between 1962 and 2016. 
    Glacier mass loss is from Zemp et al. 2019. Fingerprint for Randolph 
    Glacier Inventory regions are from Frederikse et al. 2016.
    If tg_id is None then the global sea level contribution is computed. 
    Gives the possibility to extrapollate values a few years in the future based 
    on trend of the last 10 years: extrap=True
    Give the possibility to exclude Greenland peripheral glaciers, del_green = True,
    this is not possible with the glaciers_m15_glo function.
    This option is handy if the peripheral glaciers are already included in the
    Greenland ice sheet contribution.'''
    
    data_dir = (PATH_SLBudgets_data + 
                'Glaciers/Zemp2019/Zemp_etal_results_regions_global_v11/')
    fp_dir = PATH_SLBudgets_data + 'fp_uniform/'
    RegNames = ('ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 
                'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SAN', 'NZL', 
                'ANT')
    zemp_all_df = pd.DataFrame()
    for i in range(1,20):
        zemp_df = pd.read_csv(data_dir + 'Zemp_etal_results_region_'+
                              str(i)+'_'+RegNames[i-1]+'.csv', skiprows=27)
        zemp_df = zemp_df.set_index('Year')
        zemp_df.columns = [i.strip() for i in zemp_df.columns]
        # Convert from Gt to cm slr
        zemp_all_df[RegNames[i-1]] = -zemp_df['INT_Gt'].cumsum()/3600
        
    zemp_loc_df = zemp_all_df.dropna().copy()

    if tg_id is not None:
        RGI_loc = np.ones(len(tg_id))
        for i in range(1,19):
            filename = 'RGI_'+ str(i) +'.nc'
            RGI = xr.open_dataset(fp_dir + filename)
            for j in range(len(tg_id)):
                tg_lat, tg_lon =  tg_lat_lon(tg_id[j])
                RGI_loc[j] = RGI.rsl.sel(x = tg_lon, y = tg_lat, 
                                         method='nearest').values    
            zemp_loc_df[RegNames[i-1]] = zemp_loc_df[RegNames[i-1]] * RGI_loc.mean()
    if del_green:
        del zemp_loc_df['GRL']
    zemp_loc_df['Total'] = zemp_loc_df.sum(axis=1)
    zemp_loc_tot_df = pd.DataFrame(data=dict( Glaciers=zemp_loc_df.Total))
    if extrap:
        nby = 10
        trend = np.polyfit(zemp_loc_tot_df.index[-nby:], 
                           zemp_loc_tot_df.Glaciers.iloc[-nby:], 1)[0]
        for i in range(4):
            zemp_loc_tot_df.loc[zemp_loc_tot_df.index.max() + 1] = \
            zemp_loc_tot_df.Glaciers.iloc[-1] + trend
    return zemp_loc_tot_df

def ant_imbie18(extrap=False):
    '''Read IMBIE 2018 excel data, compute yearly averages and return a data 
    frame of sea level rise in cm'''
    imbie_dir = PATH_SLBudgets_data + 'Antarctica/IMBIE2018/'
    im_df = pd.read_excel(imbie_dir  + 'imbie_dataset-2018_07_23.xlsx', sheet_name='Antarctica')
    im_df = im_df.set_index('Year')
    im_df = pd.DataFrame(data=dict( Antarctica=im_df[im_df.columns[2]]))
    im_df['Year_int'] = im_df.index.astype(int)
    grouped = im_df.groupby('Year_int', axis=0)
    im_full_years = grouped.size() == 12
    im_df = grouped.mean()
    im_df = im_df[im_full_years] # The last year doesn't have 12 month of data available so exclude it

    # Extend the data to 1950 with zeros
    im_df = im_df.reindex(np.arange(1950,2017))
    im_df = im_df.fillna(0)

    # Extrapolate data using the trend in the  last 10 years
    if extrap:
        nby = 10
        trend = np.polyfit(im_df.loc[2007:2016].index , im_df.loc[2007:2016].Antarctica, 1)[0]
        for i in range(3):
            im_df.loc[im_df.index.max() + 1] = im_df.Antarctica.iloc[-1] + trend
    return im_df / 10 # convert from mm to cm

def ant_rignot19():
    '''Use data of mass balance from table 2 of Rignot et al. 2019. 
    Fit a second order polynomial through these data that covers 1979 to 2017. 
    Extend to 1950 assuming that Antarctica did not loose mass before 1979.'''
    ye = 2019 # Last year plus 1
    dM_79_89 = 40    # Gt/y
    dM_89_99 = 49.6
    dM_99_09 = 165.8 
    dM_09_17 = 251.9
    #Fit a second order polynomial to the data
    xy = np.array([1984, 1994, 2004, 2013])
    dM = [dM_79_89, dM_89_99, dM_99_09, dM_09_17]
    dM2f = np.polyfit(xy - xy[0], dM, 2)
    xy2 = np.arange(1979,ye)
    dM2 = dM2f[0] * (xy2 - xy[0])**2 + dM2f[1] * (xy2 - xy[0]) + dM2f[2]
    slr_rig = dM2.cumsum() / 3600 # Convert from Gt to cm
    slr_rig_df = pd.DataFrame(data = dict(time= xy2, Antarctica = slr_rig))
    slr_rig_df = slr_rig_df.set_index('time')
    slr_rig_df = slr_rig_df.reindex(np.arange(1950,ye)).fillna(0)
    return slr_rig_df

def psmsl2mit(tg_id):
    '''Function that translates the tide gauge number from the PSMSL data base to 
    the numbers used by the kernels of Mitrovica et al. 2018'''
    tg_data_dir = PATH_SLBudgets_data + 'rlr_annual'
    kern_dir = PATH_SLBudgets_data + 'Mitrovica2018Kernels/'
    kern_df = pd.read_fwf(kern_dir + 'sites.txt', header=None)
    names_col = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality')
    filelist_df = pd.read_csv(tg_data_dir + '/filelist.txt', sep=';', header=None, names=names_col)
    filelist_df = filelist_df.set_index('id')
    filelist_df['name'] = filelist_df['name'].str.strip() #Remove white spaces in the column
    tg_id_mit = []
    for i in tg_id:
        tg_n = filelist_df['name'][i].upper()
        tg_id_mit_i = kern_df[kern_df.iloc[:,1].str.contains(tg_n)][0].values
        if tg_id_mit_i.size != 1:
            print('ERROR: Tide gauge number '+ tg_n +'is not available or multiple tide gauges have the same name')
        else:
            tg_id_mit.append(int(tg_id_mit_i))
    return tg_id_mit

def ices_fp(tg_id, fp, ices):
    '''Provide the relative sea level rise fingerprint of ice sheet melt for different 
    sea level models and mass losses assumptions. Three assumptions are possible:
    - mit_unif: Use kernels of Mitrovica et al. 2018 and assumes a uniform melt pattern
    - mit_grace: Use the kernels from Mitrovica et al. 2018 and assumes a melting pattern
    similar to that observed by grace (data from Adhikari et al. 2019).
    - fre_unif: Use a normalised fingerprint computed by Thomas Frederikse assuming 
    uniform mass loss
    The three options are available for both Antarctica (ant) and Greenland (green) '''
    fp_val = []
    for i in range(len(tg_id)):
        i_mit = psmsl2mit([ tg_id[i] ])
        if fp == 'mit_unif' or fp == 'mit_grace':
            kern_dir = PATH_SLBudgets_data + 'Mitrovica2018Kernels/'
            kern_t = gzip.open(kern_dir + 'kernels/grid_'+ str(i_mit[0]) +'_' + ices +'.txt.gz','rb')
            kern = np.loadtxt(kern_t)
            kern = kern[::-1,:]
            # The latitude is provided on a Gaussian grid
            gl = np.polynomial.legendre.leggauss(kern.shape[0])
            lat1D = (np.arcsin(gl[0]) / np.pi) * 180
            lon1D = np.linspace( 0, 360. - 360. / kern.shape[1], kern.shape[1])
        if fp == 'mit_unif':
            lat1D_edges = np.zeros(len(lat1D) + 1)
            lat1D_edges[1:-1] = (lat1D[1:] + lat1D[:-1]) /2
            lat1D_edges[0] = -90.
            lat1D_edges[-1] = 90.
            area = np.zeros(lat1D.shape[0])
            area = (np.sin(np.radians(lat1D_edges[1:])) - np.sin(np.radians(lat1D_edges[:-1])))
            area = area[:, np.newaxis]
            kern1 = np.where(kern == 0, kern, 1)
            fp_val.append((kern * area).sum() / ( (kern1 * area).sum() * er**2 / oa ) )
        if fp == 'mit_grace':
            #Read GRACE data from Adhikari et al. 2019.
            # The grid is uniform with 0.5º steps
            Adh_dir = PATH_SLBudgets_data + 'Adhikari2019/'
            slf_ds = xr.open_dataset(Adh_dir + 'SLFgrids_GFZOP_CM_WITHrotation.nc')
            lat = slf_ds.variables['lat'][:]
            lon = slf_ds.variables['lon'][:]
            area = np.zeros(lat.shape[0])
            area =  np.sin(np.radians(lat + 0.25)) - np.sin(np.radians(lat - 0.25))
            area = xr.DataArray(area, dims=('lat')) 

            # Regrid kernels onto the Adhikari grid. The regridder command only needs to be done once. 
            #The weights are then stored locally for further use. 
            #Since the kernels do not have metadata, the coordinates need to be given separately.
            grid_in = {'lon': lon1D, 'lat': lat1D}
            if i == 0:
                regridder = xe.Regridder(grid_in, slf_ds, 'bilinear')
            kern[kern == 0] = np.nan
            kern_rg = regridder(kern)  # regrid a basic numpy array
            kern_rg = xr.DataArray(kern_rg, dims=('lat','lon'))
            weh = slf_ds['weh']
            # Height difference between the last and the first three years of the time series
            weh_diff = weh[-12*3:, :, :].mean(axis=0) -  weh[:12*3, :, :].mean(axis=0)
 
            slr_im = - (kern_rg * weh_diff * area).sum()
            kern_rg_1 = xr.where(kern_rg == 0, np.nan, kern_rg)
            kern_rg_1 = xr.where(np.isnan(kern_rg_1), kern_rg_1, 1)
            slr_glo = - (kern_rg_1 * weh_diff * area).sum() * er**2 / oa
            fp_val.append((slr_im / slr_glo).values.tolist())
        if fp == 'fred_unif':
            fp_dir = PATH_SLBudgets_data + 'fp_uniform/'
            tg_lat, tg_lon =  tg_lat_lon(tg_id[i])
            if ices == 'ant':
                filename = 'AIS.nc' #WAIS and EAIS are also available
            elif ices == 'green':
                filename = 'GrIS.nc'
            fp_fre_ds = xr.open_dataset(fp_dir + filename)
            fp_val.append(fp_fre_ds.rsl.sel(x = tg_lon, y = tg_lat, method='nearest').values.tolist())
    return np.mean(fp_val)

def green_mouginot19_glo():
    '''Read the Greenland contribution to sea level from Mouginot et al. 2019 and 
    export to a dataframe.
    Date available from 1972 to 2018.'''
    green_dir = PATH_SLBudgets_data + 'Greenland/'
    mo_df = pd.read_csv(green_dir + 'Mouginot2019_MB.txt')
    del mo_df['Unnamed: 0']
    mo_df = mo_df.T
    mo_df.columns = ['Greenland']
    mo_df.Greenland = pd.to_numeric(
        mo_df.Greenland.astype(str).str.replace(',','.'), errors='coerce')
    mo_df['Years'] = np.arange(1972,2019)
    mo_df = mo_df.set_index('Years')
    mo_df = - mo_df / 3600 #Convert from Gt to cm
    mo_df = mo_df.reindex(np.arange(1950,2019)).fillna(0)
    return mo_df

def TWS_loc(tg_id):
    '''Read TWS effect on relative sea level derived from GRACE from a file 
    given by Thomas Frederikse.'''
    
    dir_fpg = PATH_SLBudgets_data + 'fp_grace/'
    fpg_ds1 = xr.open_dataset(dir_fpg + 'sle_results.nc')
    fpg_ds = xr.open_dataset(dir_fpg + 'sle_results.nc', group='TWS/rsl/')
    ts_mean = fpg_ds['ts_mean']
    ts_mean = xr.DataArray(ts_mean, coords={'time': fpg_ds1.time[:], 'lat': fpg_ds1.lat[:], 'lon': fpg_ds1.lon[:]})
    for i in range(len(tg_id)):
        tg_lat, tg_lon =  tg_lat_lon([tg_id[i]])
        TWS = np.array(ts_mean.sel(lon = tg_lon.values , lat = tg_lat.values, method='nearest'))
        if i == 0:
            TWS_tot = TWS.copy()
        else:
            TWS_tot = TWS_tot + TWS
    TWS = TWS_tot[:,0,0] / len(tg_id)
    TWS = xr.DataArray(TWS, dims=['time'], coords={'time': fpg_ds1.time[:]})
    TWS.name = 'TWS'
    TWS_df = TWS.to_dataframe()
    TWS_df['Year_int'] = TWS_df.index.astype(int)
    grouped = TWS_df.groupby('Year_int', axis=0)
    TWS_df = grouped.mean()
    #TWS_df = TWS_df.loc[TWS_df.index <= 2016 && TWS_df.index >= 2003]
    TWS_df = TWS_df.loc[2003:2016] # Exclude first and last year
    
    return TWS_df / 10 # Convert from mm to cm

def tws_glo_humphrey19(extrap=False):
    '''Build a pandas data frame from the global terrestrial water storage 
    reconstructions of Humphrey and Gudmundson 2019. Data available from 1901-01
    to 2014-12. Option avialable to '''
    dir_tws = PATH_SLBudgets_data + \
    'TWS/Humphrey2019/04_global_averages_allmodels/monthly/ensemble_means/'
    #Choice of files:
    # 'GRACE_REC_v03_GSFC_ERA5_monthly_ensemblemean_withoutGreenlandAntarctica.txt'
    # 'GRACE_REC_v03_GSFC_GSWP3_monthly_ensemblemean_withoutGreenlandAntarctica.txt'
    
    file_name = \
    'GRACE_REC_v03_GSFC_GSWP3_monthly_ensemblemean_withoutGreenlandAntarctica.txt'
    TWS_glo_df = pd.read_csv(dir_tws + file_name)
    TWS_glo_df['Year'] = pd.to_datetime(TWS_glo_df['Time'], format='%Y%m').dt.year
    TWS_glo_df = TWS_glo_df.set_index('Year')
    del TWS_glo_df['Time']
    del TWS_glo_df['TWS_seasonal_cycle_in_Gt']
    grouped = TWS_glo_df.groupby('Year', axis=0)
    TWS_glo_df = grouped.mean()
    TWS_glo_df = - TWS_glo_df / 3600 # Convert Gt TWS to cm sea level
    TWS_glo_df.columns = ['TWS']
    last5avg = TWS_glo_df['TWS'].iloc[-5:].mean()
    if extrap:
        for i in range(5):
            TWS_glo_df.loc[TWS_glo_df.index.max()+1] = last5avg
    return TWS_glo_df

def LevitusSL(reg = 'Global', extrap_back = False, extrap=False):
    ''' Steric sea level anomaly (NOAA, Levitus) computed in the top 2000m of the ocean. 
    Options for different bassins are available but for now only North Atlantic and
    Global is implemented.
    Possibility to extrapolate the time series to 1950 using the trend of the first 
    20 years with extrap_back.
    Possibility to extrapolate the time series forward up to 2020 using the trend of 
    the last 5 years'''
    Dir_LEV = PATH_Data + 'NOAA/'
    Lev_ds = xr.open_dataset(Dir_LEV + \
                             'mean_total_steric_sea_level_anomaly_0-2000_pentad.nc', \
                             decode_times=False)
    if reg == 'Global':
        LevitusSL = Lev_ds.pent_s_mm_WO.copy() / 10
    elif reg == 'NA':
        LevitusSL = Lev_ds.pent_s_mm_NA.copy() / 10
    LevitusSL['time'] = LevitusSL.time / 12 + 1955 - .5 # Convert from months since 
                                                        #1955 to years
    LevitusSL['time'] = LevitusSL.time.astype(int)
    LevitusSL_df = LevitusSL.to_dataframe()
    LevitusSL_df.rename(columns={'pent_s_mm_WO': 'GloSteric'}, inplace=True)
    if extrap_back:
        nby = 20
        trend = np.polyfit(LevitusSL_df.index[:nby], \
                           LevitusSL_df.GloSteric.iloc[:nby], 1)[0]
        for i in range(7):
            LevitusSL_df.loc[LevitusSL_df.index.min() - 1] = \
            ( LevitusSL_df.GloSteric.loc[LevitusSL_df.index.min()] - trend )
        LevitusSL_df.sort_index(inplace=True)
    if extrap:
        nby = 5
        trend = np.polyfit(LevitusSL_df.index[-nby:], \
                           LevitusSL_df.GloSteric.iloc[-nby:], 1)[0]
        for i in range(3):
            LevitusSL_df.loc[LevitusSL_df.index.max() + 1] = \
            ( LevitusSL_df.GloSteric.loc[LevitusSL_df.index.max()] + trend )
    return LevitusSL_df

def GloSLDang19():
    ''' Global sea level reconstruction from Dangendorf et al. 2019. 
    Looks like read_csv cannot read the first line of data. Why?'''
    
    Dir_GloSL = PATH_Data + 'SeaLevelReconstructions/'
    GloSLDang19_df = pd.read_csv(Dir_GloSL + 'DataDangendorf2019.txt', 
                      names=['time', 'GMSL', 'Error'], header=1, delim_whitespace=True)

    GloSLDang19_df['Year_int'] = GloSLDang19_df.time.astype(int)
    grouped = GloSLDang19_df.groupby('Year_int', axis=0)
    GloSLDang19_df = grouped.mean()
    del GloSLDang19_df['time']
    del GloSLDang19_df['Error'] # Remove error columns because the yearly error is not the average of monthly errors
    GloSLDang19_df.index.names = ['time']
    
    return GloSLDang19_df / 10 # Convert from mm to cm

def nodal_tides_potential(lat, time_years):
    '''Compute the nodal tide potential based on Woodworth et al. 2012,
    https://doi.org/10.2112/JCOASTRES-D-11A-00023.1'''
    
    h2 = 0.6032
    k2 = 0.298

    #nodal cycle correction
    A = 0.44*(1+k2-h2)*20*(3*np.sin(lat*np.pi/180.)**2-1)/10  # mm to cm
    nodcyc = A*np.cos((2*np.pi*(time_years-1922.7))/18.61 + np.pi)
    
    nodcyc_df = pd.DataFrame(data={'time': time_years, 'Nodal': nodcyc})
    nodcyc_df = nodcyc_df.set_index('time')
    
    return nodcyc_df

def contrib_frederikse2020(coord, var, output_type='rsl', extrap=False):
    '''
    Read values from Frederikse et al. 2020 budget.
    Data from 1900 to 2018.
    
    Inputs: 
    coord: [latitude, longitude]
    variable: tws, AIS, glac, GrIS, steric
    output_type: rsl, rad, abs
    These represent the type of fingerprint to use. Influence of mass loss on
    relative sea level (rsl), radial velocities (rad) or absolute sea level
    which is the the sum rsl+rad.
    extrap: Possibility to extrapolate a few years.
    
    Outputs:
    Dataframe giving the average contribution at the tide gauges
    '''    
    
    data_dir = f'{PATH_SLBudgets_data}Frederikse2020/'
    ds = xr.open_dataset(f'{data_dir}{var}.nc')
    
    # Fill coastal points to avoid selecting NaN
    if output_type=='abs':
        sel_da = (ds[f'{var}_rsl_mean'].ffill('lon', 3).bfill('lon', 3)+
                  ds[f'{var}_rad_mean'].ffill('lon', 3).bfill('lon', 3))
    else:
        sel_da = ds[f'{var}_{output_type}_mean'].ffill('lon', 3).bfill('lon', 3)
        
    sel_da = sel_da/10 # Convert from mm to cm
    
    loc_da = sel_da.sel(lat = coord[0], 
                        lon = coord[1], 
                        method = 'nearest')
    
    fr_name = {'tws' : 'TWS', 
           'AIS' : 'Antarctica', 
           'GrIS' : 'Greenland', 
           'glac' : 'Glaciers'}
    
    df = loc_da.squeeze().reset_coords(drop=True).to_dataframe(name=fr_name[var])    
    
    if extrap:
        nby = 10
        trend = np.polyfit(df.index[-nby:], df.iloc[-nby:], 1)
        
        for i in range(8):
            df.loc[df.index.max() + 1] = (trend[1]+trend[0]*(df.index.max()+1))
    
    return df

def contrib_frederikse2020_glob(var, extrap=False, quant='mean'):
    '''
    Read values from Frederikse et al. 2020 budget.
    
    Inputs:
    var (available: GloSteric, glac, for other variables see excel sheet)
    extrap: Extrapollate the trend of the last 10 years forward
    quant: 'mean', 'upper', 'lower'
    
    Outputs:
    Pandas dataframe of this variable with time in years as index
    '''
    
    fr_name = {'tws' : 'Terrestrial Water Storage', 
               'AIS' : 'Antarctic Ice Sheet', 
               'GrIS' : 'Greenland Ice Sheet', 
               'glac' : 'Glaciers',
               'GloSteric' : 'Steric'}
    
    out_names = {'tws' : 'TWS', 
                 'AIS' : 'Antarctica', 
                 'GrIS' : 'Greenland', 
                 'glac' : 'Glaciers',
                 'GloSteric' : 'GloSteric'}
    
    data_dir = f'{PATH_SLBudgets_data}Frederikse2020/'
    fts = pd.read_excel(f'{data_dir}/global_basin_timeseries.xlsx', sheet_name='Global')
    fts = fts.rename(columns = {fts.columns[0]:'time'})
    fts = fts.set_index('time')
    fts = fts/10 # Convert from mm to cm
    out_df = pd.DataFrame(fts[f'{fr_name[var]} [{quant}]'])
    
    out_df = out_df.rename(columns = {f'{fr_name[var]} [{quant}]':out_names[var]})
        
    if extrap:
        nby = 10
        trend = np.polyfit(out_df.index[-nby:], out_df.iloc[-nby:], 1)
        
        for i in range(6):
            out_df.loc[out_df.index.max() + 1] = (
                trend[1]+trend[0]*(out_df.index.max()+1))
    
    return out_df

def local_budget(location, opt_sl, opt_steric, opt_glaciers, opt_antarctica, 
                 opt_greenland, opt_tws, opt_wind_ibe, opt_nodal, 
                 global_steric, avg):
    '''Compute the sea level budget at tide gauge locations. 
    avg (boolean): Compute the average budget over the list of tide gauges'''
    
    if type(location) == list:
        location_type = 'tg_id'
        cond_loop = len(location)
    elif type(location) == np.ndarray:
        location_type = 'region'
        cond_loop = 1
    
    if opt_sl == 'tide_gauge':
        sl_df = tide_gauge_obs(location, interp=True)
        output_type = 'rsl' # Relative sea level
    elif opt_sl == 'altimetry':
        sl_df = altimetry_obs(location, 0)
        output_type = 'abs' # Absolute sea level
    
    if opt_steric[0] in ['EN4_21', 'EN4_22', 'IAP']:
        
        if len(opt_steric[1])==1:
            # Using a one-layer steric representation
            loc_steric_df = StericSL(opt_steric[0], opt_steric[1], 0, opt_steric[2], 
                                     opt_steric[3])
        elif len(opt_steric[1])==2:
            # Using a two-layer steric representation
            loc_steric_up = StericSL(opt_steric[0], opt_steric[1][0], 0, 
                                     opt_steric[2][0], opt_steric[3][0])
            loc_steric_down = StericSL(opt_steric[0], opt_steric[1][1], opt_steric[2][0] , 
                                       opt_steric[2][1], opt_steric[3][1])
            
            loc_steric_df = loc_steric_up+loc_steric_down
        
    else:
        print('ERROR: option for opt_steric[0] undefined')

    steric_df = loc_steric_df
    
    if global_steric:  # split local and global steric effects
        if global_steric == 'levitus':
            glo_steric_df =  LevitusSL(extrap=True, extrap_back=True)
        elif global_steric == 'fred20':
            glo_steric_df = contrib_frederikse2020_glob('GloSteric', extrap=True)
        else:
            print('ERROR: option for global_steric undefined')
        
        steric_df = steric_df.join(glo_steric_df)
        steric_df = steric_df.rename(columns={'Steric': 'LocSteric'})
        steric_df['LocSteric'] = steric_df['LocSteric'] - steric_df['GloSteric']
    
    for i in range(cond_loop):
        
        if location_type == 'tg_id':
            print('Working on tide gauge id: '+ str(location[i]))
            coord = tg_lat_lon(location[i])
            print(f'with lat/lon {coord}')
            sl_loc = sl_df[location[i]]
        elif location_type == 'region':
            coord = [np.mean(location[:,1]), np.mean(location[:,0])]
            print(f'Working on a region with lat/lon: {coord}')
            sl_loc = sl_df['Average']
        
        if opt_sl == 'altimetry':
            gia_ts_df = GIA_ICE6G_region(location)
        else:
            gia_ts_df = GIA_ICE6G(location[i])

        if opt_glaciers == 'marzeion15':
            glac_ts_df = glaciers_m15([location[i]], extrap=True, 
                                      del_green=True)
        elif opt_glaciers == 'zemp19':
            glac_ts_df = glaciers_zemp19([location[i]], extrap=True, 
                                         del_green=True)
        elif opt_glaciers == 'fred20':
            glac_ts_df = contrib_frederikse2020(coord, 'glac', output_type, 
                                                extrap=True)
        else:
            print('ERROR: option for opt_glaciers undefined')

        if opt_antarctica == 'imbie18':
            ant_df = ant_imbie18(extrap=True) * ices_fp([location[i]], 'mit_unif', 
                                                                  'ant')
        elif opt_antarctica == 'rignot19':
            ant_df = ant_rignot19() * ices_fp([location[i]] , 'mit_unif', 'ant')
        elif opt_antarctica == 'fred20':
            ant_df = contrib_frederikse2020(coord, 'AIS', output_type, 
                                            extrap=True)
            # option to add GravIS data 
        elif opt_antarctica == 'Gravis':
            # 1. Use Frederikse for the years without Gravis data: 
            fredant_df = contrib_frederikse2020(coord, 'AIS', output_type, extrap=True)
            #2. Use Gravis data: 
            Gravis_dir = '../outputs/Gravis_Results/'
            Gravis_ant_ds = xr.open_dataset(f'{Gravis_dir}Gravis_Antarctica.nc')
            sel_da = Gravis_ant_ds['IS_Gravis'].ffill('lon', 3).bfill('lon', 3)
            loc_da = sel_da.sel(lat = coord[0], lon = coord[1], method = 'nearest')
            gravis_df = loc_da.squeeze().reset_coords(drop=True).to_dataframe(name='Ant_gravis') 
            gravis_df = gravis_df*1.1837950465122913 #for now corrected... 
            #3. Make Antcombi consisting of the frederikse data 
            Antcombi = fredant_df['Antarctica'].copy()
            # Reference values to lign up the data
            antfredreference = fredant_df['Antarctica'].loc[2014:2018].mean()
            antgracereference = gravis_df['Ant_gravis'].loc[2014:2018].mean()
            # Align GravIS data with frederikse data with the period 2014-2018 as a reference period. 
            Antcombi.loc[2019:] = gravis_df['Ant_gravis'] - antgracereference + antfredreference 
            ant_df = Antcombi.to_frame()
        else:
            print('ERROR: option for opt_antarctica undefined')

        if opt_greenland == 'mouginot19':
            green_df = green_mouginot19_glo() * ices_fp([location[i]] , 
                                                        'mit_unif', 'green')
        elif opt_greenland == 'fred20':
            green_df = contrib_frederikse2020(coord, 'GrIS', output_type, 
                                              extrap=True)
        elif opt_greenland == 'Gravis':
            # Combine Frederikse data and Gravis: Frederikse for the years without GravIS
            fredgre_df = contrib_frederikse2020(coord, 'GrIS', output_type, extrap=True)
            # Only works if land tiles form the landsea mask are nan (currently this is the case in the file). 
            Gravis_dir = '../outputs/Gravis_Results/'
            Gravis_gre_ds = xr.open_dataset(f'{Gravis_dir}Gravis_Greenland.nc')
            sel_da = Gravis_gre_ds['IS_Gravis'].ffill('lon', 3).bfill('lon', 3)
            loc_da = sel_da.sel(lat = coord[0], lon = coord[1], method = 'nearest')
            gravis_df = loc_da.squeeze().reset_coords(drop=True).to_dataframe(name='Gre_gravis') 
            # Make Greencombi consisting of Frederikse data for data before 2018 and use Gravis after 2018. 
            Greencombi = fredgre_df['Greenland'].copy()
            # Create a reference data to allign the data:
            grefredreference = fredgre_df['Greenland'].loc[2014:2018].mean()
            gregracereference = gravis_df['Gre_gravis'].loc[2014:2018].mean()
            # Allign GravIs data with Frederikse data based on the reference period based on the mean betweeen 2014 and 2018. 
            Greencombi.loc[2019:] = gravis_df['Gre_gravis'] - gregracereference + grefredreference 
            green_df = Greencombi.to_frame()

        else:
            print('ERROR: option for opt_greenland undefined')
        
        if opt_tws == 'humphrey19':
            tws_df = tws_glo_humphrey19(extrap=True)
        elif opt_tws == 'fred20':
            tws_df = contrib_frederikse2020(coord, 'tws', output_type, 
                                            extrap=True)
        
        sealevel_df = steric_df.copy()
        sealevel_df = sealevel_df.join([ gia_ts_df, glac_ts_df, 
                                        ant_df, green_df, tws_df], how='inner')
        
        if opt_nodal == 'potential':
            nodal_df = nodal_tides_potential(coord[0], sealevel_df.index)
            sealevel_df = sealevel_df.join(nodal_df)
            with_nodal = False
        elif opt_nodal == 'regression':
            with_nodal = True
        else:
            print('ERROR: option opt_nodal undefined:'+str(opt_nodal))
        
        sealevel_df['Total'] = sealevel_df.sum(axis=1)

        if opt_wind_ibe[0] == 'regression':
            diff_sl_df = sl_loc - sealevel_df.Total
            diff_sl_df = diff_sl_df.to_frame(name='height').dropna()
            wpn_ef_df = make_wpn_ef(coord, diff_sl_df, with_nodal, 
                                    with_trend=False, product=opt_wind_ibe[1])
        elif opt_wind_ibe[0] == 'dynamical_model':
            if opt_wind_ibe[1] == 'WAQUA':
                wpn_ef_df = make_waqua_df(location[i])
            elif opt_wind_ibe[1] == 'GTSM':
                wpn_ef_df = make_gtsm_df(location[i], 'surge')
            else:
                print('ERROR: option for opt_wind_ibe[1] undefined')
        else:
            print('ERROR: option for opt_wind_ibe[0] undefined')

        sealevel_df = sealevel_df.join(wpn_ef_df, how='inner')
        del sealevel_df['Total']
        sealevel_df.insert(0, 'Total', sealevel_df.sum(axis=1))
        sealevel_df['Total'] = sealevel_df['Total'] - sealevel_df['Total'].mean()
        sealevel_df.index.name = 'time'
        sealevel_df = sealevel_df - sealevel_df.iloc[0,:]
        sealevel_df = pd.concat([sealevel_df], axis=1, keys=[str(location[i])])
        
        if i==0:
            slall_df = sealevel_df.copy()
        else:
            slall_df = pd.concat([slall_df, sealevel_df], axis=1)

    if avg:        # Compute the average contributors at all tide gauges
        #slmean_df = slall_df.groupby(level=1, axis=1, sort=False).mean() # Only works for old version of python
        slmean_df = slall_df.T.groupby(level=1, sort=False).mean().T  #the new version of python does not work with axis is 0. in stead of that, t makes the transpose. I compared both outputs and they should be equal. 
        slmean_df = slmean_df.join(sl_df.Average, how='inner')
        slmean_df = slmean_df.rename(columns={'Average': 'Obs'})
        slall_df = slmean_df

    return slall_df

def plot_budget(location_name, slmean_df):
    '''Summary plot of the sea level budget. Should be split in smaller functions.'''
    ### Plot compaison between tide gauge observations and budget
    fig, ax = plt.subplots(2, 2, figsize=(9,9), gridspec_kw={'height_ratios': [1, 1]})
    fig.tight_layout(pad=1.9)

    ax[0,0].plot(slmean_df.Obs - slmean_df.Obs.mean(), 'o-', label='Sea level observations')
    ax[0,0].plot(slmean_df.Total - slmean_df.Total.mean() , 'r-', label='Sum of contributors')

    #ax[0,0].set_xlabel('time')
    ax[0,0].set_ylabel('sea level (cm)')
    ax[0,0].set_title('Yearly average sea level at '+location_name)
    ax[0,0].grid(True)
    ax[0,0].legend(loc='upper left')

    ### Plot the difference between observations and budget
    diff_df = slmean_df.Obs - slmean_df.Total
    diff_df = diff_df - diff_df.mean()

    t = ('Normalised RMSE (cm): '+
         str( round(np.sqrt( (diff_df**2).sum() ) / len(diff_df), 2 ))+ '\n' +
         'Normalised AE (cm): '+
         str( round( np.abs(diff_df).sum() / len(diff_df),2)))
    ax[0,1].text(0.98, 0.98, t, ha='right', va='top', transform=ax[0,1].transAxes)
    ax[0,1].set_title('Difference observations - budget')
    ax[0,1].grid(True)
    # Optionnaly add a running average (keep after diagnostics)
    #diff_df = diff_df.rolling( 3, center=True).mean()
    ax[0,1].plot(diff_df)

    ### Plot the trend and acceleration budget
    lin_trend = np.polyfit(slmean_df.index, 
                           slmean_df * 10, 1)[0,:]  # Convert from cm to mm

    acceleration = 2 * np.polyfit(slmean_df.index, slmean_df * 10, 2)[0,:]
    # Convert from mm^2 / year to 10^-2 mm^2 / year
    acceleration = acceleration * 100 
    
    if 'Steric' in slmean_df.columns:
        colors = ['red', 'blue', 'green', 'brown', 'magenta', 'grey', 'orange', 
                  'black', 'cyan', 'yellow']
            
    else:  
        colors = ['red', 'blue', 'purple', 'green', 'brown', 'magenta', 'grey', 
                  'orange', 'black', 'cyan', 'olive']

    ind = np.arange(len(slmean_df.columns) - 1 )

    legend_elements = []
    for i in ind:
        legend_elements.append(Line2D([0], [0], color = colors[i], lw = 4, 
                                      label = slmean_df.columns[i]))

    legend_elements.append(Line2D([0], [0], color = 'black', lw = 2, 
                                  label = 'tg obs'))
    ax[1,0].set_title('Linear trend budget')
    ax[1,0].bar(ind, lin_trend[:-1], color=colors)
    ax[1,0].hlines(y=lin_trend[-1], xmin=-0.5, xmax=0.5, color='black')
    ax[1,0].set_ylabel('Linear trend (mm/year)')
    ax[1,0].legend(handles=legend_elements, loc='upper right')
    ax[1,0].text(0.02, 0.01, 
                 f'Observed trend: {round(lin_trend[-1],2)}\n'+ 
                 f'Budget trend: {round(lin_trend[0],2)}', 
                 va='bottom', ha='left', 
                 transform=ax[1,0].transAxes)

    ax[1,1].set_title('Acceleration budget')
    ax[1,1].bar(ind, acceleration[:-1], color=colors)
    ax[1,1].hlines(y=acceleration[-1], xmin=-0.5, xmax=0.5, color='black')
    ax[1,1].set_ylabel('Acceleration ($10^{-2} mm/year^2$)')
#     ax[1,1].text(5, 6, f'Observed acceleration: {round(acceleration[-1],2)}\n'+
#            f'Budget acceleration: {round(acceleration[0],2)}')   
    ax[1,1].text(0.98, 0.98, 
                 f'Observed acceleration: {round(acceleration[-1],2)}\n'+ 
                 f'Budget acceleration: {round(acceleration[0],2)}', 
                 va='top', ha='right', 
                 transform=ax[1,1].transAxes)
    
    return fig, ax

def plot_budget_ts(slmean_df):
    '''Plot time series of contributors'''

    df = slmean_df.copy()
    
    # Combined contributors plot:
    df['Steric'] = df['LocSteric'] + df['GloSteric']
    df['IceSheets'] = df['Glaciers'] + df['Antarctica'] + df['Greenland']
    df['WindPressure'] = df['Wind'] + df['Pressure']
    # Columns to plot (aggregated)
    contributors_agg = ['Steric', 'GIA', 'IceSheets', 'TWS', 'Nodal', 'WindPressure']
    
    #legend
    legend_labels = {
        'Steric': 'Regional steric',
        'IceSheets': 'Antarctica, Greenland and glaciers',
        'WindPressure': 'Wind + Pressure',
        'GIA': 'GIA',
        'TWS': 'TWS',
        'Nodal': 'Nodal',
        'Total': 'Total (sum of contributors)',
        'Obs': 'Observations'
    }
    
    # Color scheme
    colors_agg = ['blue', 'green', 'magenta', 'orange', 'black', 'cyan']
    
    # Figure to save: 
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for col, color in zip(contributors_agg, colors_agg):
        ax2.plot(df[col] - df[col].iloc[0], color=color, label=legend_labels[col])
    ax2.plot(df['Total'] - df['Total'].iloc[0], color='red', label='Total')
    ax2.plot(df['Obs'] - df['Obs'].iloc[0], 'o-', label='Observations')
    ax2.set_xlabel('Time (year)', fontsize=16)
    ax2.set_ylabel(f'Sea level contribution relative to {df.index[0].item()} (cm)', fontsize=16)
    ax2.set_title('Local sea level budget', fontsize=18)
    ax2.grid(True)
    ax2.legend(loc='upper left', fontsize=13)
    
    # Increase tick label size
    ax2.tick_params(axis='both', which='major', labelsize=14)

    fig2.tight_layout()

    return fig2, ax2