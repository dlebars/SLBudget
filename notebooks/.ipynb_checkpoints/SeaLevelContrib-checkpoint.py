# List of functions to be used in sea level budget

import datetime
import netCDF4
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
import gzip
import xesmf as xe

PATH_SeaLevelBudgets = '/Users/dewilebars/Projects/Project_SeaLevelBudgets/'
PATH_Data = '/Users/dewilebars/Data/'

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
    or ERA5 (1979-now) """
    if product == 'NCEP1':
        # Use for OpenDAP:
        #NCEP1 = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface_gauss/'
        #For local:
        NCEP1_dir = PATH_SeaLevelBudgets + 'WindPressure/NCEP1/'
        u_file = NCEP1_dir + 'uwnd.10m.mon.mean.nc'
        v_file = NCEP1_dir + 'vwnd.10m.mon.mean.nc'
        p_file = NCEP1_dir + 'pres.sfc.mon.mean.nc'
        latn = 'lat'
        lonn = 'lon'
        timen = 'time'
        un = 'uwnd'
        vn = 'vwnd'
        pn = 'pres'
        if lon_i < 0:
            lon_i = lon_i + 360
        
    elif product == 'ERA5':
        ERA5_dir = PATH_SeaLevelBudgets + 'WindPressure/ERA5/'
        u_file = ERA5_dir + 'ERA5_u10.nc'
        v_file = ERA5_dir + 'ERA5_v10.nc'
        p_file = ERA5_dir + 'ERA5_msl.nc'
        latn = 'latitude'
        lonn = 'longitude'
        timen = 'time'
        un = 'u10'
        vn = 'v10'
        pn = 'msl'
        if lon_i < 0:
            lon_i = lon_i + 360
    
    # open the 3 files
    ds_u = xr.open_dataset(u_file)
    ds_v = xr.open_dataset(v_file)
    ds_p = xr.open_dataset(p_file)
    
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
    
    # Compute the wind squared while retaining sign, as a approximation of stress
    u2 = u**2 * np.sign(u) #!!!! u**2
    v2 = v**2 * np.sign(v)
    
    # put everything in a dataframe
    wind_df = pd.DataFrame(data=dict(u=u, v=v, t=u[timen], speed=speed, direction=direction, u2=u2, v2=v2, 
                                    pres=pres, ibe=ibe))
    wind_df = wind_df.set_index('t')

    annual_wind_df = wind_df.resample('A', label='left', loffset=datetime.timedelta(days=1)).mean()
    annual_wind_df.index = annual_wind_df.index.year
    
    # return it
    return annual_wind_df

def linear_model_zsm(df, with_trend=True, with_nodal=True, with_wind=True, with_pres=True, with_ar=False):
    ''' Define the statistical model, similar to zeespiegelmonitor'''
    t = np.array(df.index)
    y = df['height']
    X = np.ones(len(t))
    names = ['Constant']
    if with_nodal:
        X = np.c_[ X, np.cos(2*np.pi*(t - 1970)/18.613), np.sin(2*np.pi*(t - 1970)/18.613)]
        names.extend(['Nodal U', 'Nodal V'])
    if with_wind:
        X = np.c_[ X, df['u2'], df['v2']]
        names.extend(['Wind $u^2$', 'Wind $v^2$'])
    if with_pres:
        X = np.c_[X, df['pres']]
        names.extend(['Pressure'])
    if with_trend:
        X = np.c_[X, t - 1970 ]
        names.extend(['Trend'])
    if with_ar:
        model = sm.GLSAR(y, X, missing='drop', rho=1)
    else:
        model = sm.OLS(y, X, missing='drop')
    fit = model.fit(cov_type='HC0')
    return fit, names

def make_wpn_ef(tg_id, tgm_df, with_trend, product):
    for i in range( len(tg_id)):
        tg_lat, tg_lon = tg_lat_lon(tg_id[i])
        annual_wind_df = make_wind_df(tg_lat, tg_lon, product)
        df_c = tgm_df.join(annual_wind_df, how='inner')
        df_c.index.names = ['year']
        linear_fit, names = linear_model_zsm(df_c, with_trend, with_nodal=True, 
                                             with_wind=True, with_pres=True, with_ar=False)
        time_y = df_c.index
        mod = np.array(linear_fit.params[:]) * np.array(linear_fit.model.exog[:,:])
        n_ef = mod[:,[1,2]].sum(axis=1)
        w_ef = mod[:,[3,4]].sum(axis=1)
        p_ef = mod[:,5]
        if i==0:
            n_ef_df = pd.DataFrame(data=dict(time=time_y, col_name=n_ef))
            n_ef_df = n_ef_df.set_index('time')
            n_ef_df.columns  = [str(tg_id[i])] 
            w_ef_df = pd.DataFrame(data=dict(time=time_y, col_name=w_ef))
            w_ef_df = w_ef_df.set_index('time')
            w_ef_df.columns  = [str(tg_id[i])]
            p_ef_df = pd.DataFrame(data=dict(time=time_y, col_name=p_ef))
            p_ef_df = p_ef_df.set_index('time')
            p_ef_df.columns  = [str(tg_id[i])]
        else:
            n_ef_df[str(tg_id[i])] = n_ef
            w_ef_df[str(tg_id[i])] = w_ef
            p_ef_df[str(tg_id[i])] = p_ef
            
    wpn_ef_df = pd.DataFrame(data=dict(time=time_y, Nodal=n_ef_df.mean(axis=1), 
                                       Wind=w_ef_df.mean(axis=1), Pressure=p_ef_df.mean(axis=1)))
    wpn_ef_df = wpn_ef_df.set_index('time')
    return wpn_ef_df

def make_waqua_df(tg_id):
    '''Read time series of annually averaged sea level from the WAQUA model forced by ERA-interim.'''
    dir_waqua = PATH_SeaLevelBudgets + 'DataWAQUANinaERAI'
    ds_wa = netCDF4.Dataset(dir_waqua+'/ERAintWAQUA_waterlevels_speed_1979_2015.nc')

    # Get WAQUA tide gauge names that are not editted in the same way as PSMSL names
    tg_data_dir = PATH_SeaLevelBudgets + 'rlr_annual'
    names_col = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality')
    filelist_df = pd.read_csv(tg_data_dir + '/filelist.txt', sep=';', header=None, names=names_col)
    filelist_df = filelist_df.set_index('id')
    tgn = filelist_df.name[tg_id].replace(' ', '').lower()[:8]
    
    dh = ds_wa[tgn + '/WAQUA_surge'][:]*100
    time_wa = ds_wa['time'][:]
    t_wa = netCDF4.num2date(time_wa, ds_wa.variables['time'].units)

    t_wa_y = np.empty_like(t_wa)
    for i in range(len(t_wa)):
        t_wa_y[i] = t_wa[i].year
    waqua_df = pd.DataFrame( data = dict( time=t_wa_y, sealevel=dh.data) )
    waqua_df = waqua_df.set_index('time')
    return waqua_df

def tide_gauge_obs(tg_id=[20, 22, 23, 24, 25, 32], interp=False):
    '''Read a list of tide gauge data and compute the average. 
    Set interp to True for a linear interpollation of missing values.
    By default use the 6 tide gauges from the zeespiegelmonitor''' 
    
    tg_data_dir = PATH_SeaLevelBudgets + 'rlr_annual'
    names_col = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality')
    filelist_df = pd.read_csv(tg_data_dir + '/filelist.txt', sep=';', header=None, names=names_col)
    filelist_df = filelist_df.set_index('id')

    names_col2 = ('time', 'height', 'interpolated', 'flags')

    for i in range(len(tg_id)):
        tg_data = pd.read_csv(tg_data_dir + '/data/' + str(tg_id[i]) + '.rlrdata', sep=';', 
                            header=None, names=names_col2)
        tg_data = tg_data.set_index('time')
        tg_data.height = tg_data.height.where(~np.isclose(tg_data.height,-99999))
        tg_data.height = tg_data.height - tg_data.height.mean()

        if i==0:
            tg_data_df = pd.DataFrame(data=dict(time=tg_data.index, col_name=tg_data.height))
            tg_data_df = tg_data_df.set_index('time')
            tg_data_df.columns  = [str(tg_id[i])] 
        else:
            tg_data_df[str(tg_id[i])] = tg_data.height

    #tg_data_df = tg_data_df[tg_data_df.index >= 1890].copy()
    # 1890 is to follow the choice of the zeespiegelmonitor
    # Alternatively use 1948 to fit with NCEP1 starting date
    if interp:
        tg_data_df = tg_data_df.interpolate(method='slinear')
    tg_data_df['Average'] = tg_data_df.mean(axis=1)
    return tg_data_df * 0.1 # Convert from mm to cm

def StericSL(max_depth, mask_name):
    '''Compute the steric effect in the North Sea in cm integrated from the 
    surface up to a given depth given in meters. '''
    DENS = xr.open_dataset('density_teos10_en4_1900_2019.nc')
    midp = (np.array(DENS.depth[1:])+np.array(DENS.depth[:-1]))/2
    midp = np.insert(midp, 0, np.array([0]))
    midp = np.insert(midp, len(midp), np.array(DENS.depth[-1]) + 
                     (np.array(DENS.depth[-1]) - np.array(DENS.depth[-2])))
    thick = midp[1:] - midp[:-1]
    thick = xr.DataArray(thick, coords={'depth': DENS.depth[:]}, dims='depth')
    SumDens = DENS.density * thick
    if mask_name == 'ENS':
        # Extended North Sea mask
        lat = np.array(DENS.lat)
        lon = np.array(DENS.lon)
        LatAr = np.repeat(lat[:,np.newaxis], len(lon), 1)
        LatAr = xr.DataArray(LatAr, dims=['lat', 'lon'], 
                             coords={'lat' : lat, 'lon' : lon})
        LonAr = np.repeat(lon[np.newaxis,:], len(lat), 0)
        LonAr = xr.DataArray(LonAr, dims=['lat', 'lon'], 
                             coords={'lat' : lat, 'lon' : lon})

        mask_med = xr.where(np.isnan(DENS.density[0,0,:,:]), np.nan, 1)
        mask_med1 = mask_med.where((LonAr >= -8) & (LatAr <= 42) )
        mask_med1 = xr.where(np.isnan(mask_med1), 1, np.nan)
        mask_med2 = mask_med.where((LonAr >= 1) & (LatAr <= 48) )
        mask_med2 = xr.where(np.isnan(mask_med2), 1, np.nan)
        mask_med = mask_med * mask_med1 * mask_med2

        mask = xr.where(np.isnan(DENS.density[0,0,:,:]), np.nan, 1)
        mask = mask.where(mask.lon <= 7)
        mask = mask.where(mask.lon >= -16)
        mask = mask.where(mask.lat <= 69) #Normal value: 60 or 69
        mask = mask.where(mask.lat >= 33)
        mask = mask * mask_med

    elif mask_name == 'EBB':
        # Extended bay of Biscay
        mask = xr.where(np.isnan(DENS.density[0,:,:,:].sel(depth=2000, method='nearest')), np.NaN, 1)
        mask = mask.where(mask.lon <= -2)
        mask = mask.where(mask.lon >= -12)
        mask = mask.where(mask.lat <= 52)
        mask = mask.where(mask.lat >= 35)
        
    elif mask_name == 'NWS':
        # Norwegian Sea
        mask = xr.where(np.isnan(DENS.density[0,:,:,:].sel(depth=2000, method='nearest')), np.NaN, 1)
        mask = mask.where(mask.lon <= 8)
        mask = mask.where(mask.lon >= -10)
        mask = mask.where(mask.lat <= 69)
        mask = mask.where(mask.lat >= 60)
        
    else:
        print('ERROR: mask_name argument is not available')

    del mask['depth']
    del mask['time']
    SumDens_NS = (SumDens * mask).mean(dim=['lat', 'lon'])
    StericSL_NS = (- SumDens_NS.sel(depth=slice(0,max_depth)).sum(dim='depth') 
                   / (DENS.density[0 ,0 ,: ,:] * mask).mean(dim=['lat', 'lon'])) * 100
    StericSL_NS = StericSL_NS - StericSL_NS.sel(time=slice(1940,1960)).mean(dim='time')
    StericSL_NS.name = 'Steric'
    StericSL_NS_df = StericSL_NS.to_dataframe()
    del StericSL_NS_df['depth']
    return StericSL_NS_df

def GIA_ICE6G(tg_id=[20, 22, 23, 24, 25, 32]):
    '''Read the current GIA 250kaBP-250kaAP from the ICE6G model and output a
    time series in a pandas dataframe format'''
    dir_ICE6G = PATH_SeaLevelBudgets + "GIA/ICE6G/"
    locat = []
    gia = []
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
    time = np.arange(1900, 2020)
    gia_ts = gia_avg * (time - time[0])
    gia_ts_list = [("time", time),
                  ("GIA", gia_ts)]
    gia_ts_df = pd.DataFrame.from_dict(dict(gia_ts_list))
    gia_ts_df = gia_ts_df.set_index("time")
    return gia_ts_df

def tg_lat_lon(tg_id):
    '''Give tide gauge latitude, longitude location given the id as input'''
    tg_data_dir = PATH_SeaLevelBudgets + 'rlr_annual'
    names_col = ('id', 'lat', 'lon', 'name', 'coastline_code', 'station_code', 'quality')
    filelist_df = pd.read_csv(tg_data_dir + '/filelist.txt', sep=';', header=None, names=names_col)
    filelist_df = filelist_df.set_index('id')
    return filelist_df.loc[tg_id].lat, filelist_df.loc[tg_id].lon

def glaciers_m15_glo():
    '''Provides glacier contributions to local sea level between 1900 and 2013
    from Marzeion et al. 2015.'''
    M15_dir = PATH_SeaLevelBudgets + 'Glaciers/Marzeion2015/tc-9-2399-2015-supplement/'
    M15_glo_df = pd.read_csv(M15_dir + 'data_marzeion_etal_update_2015.txt', 
                             header=None, 
                             names=['time', 'Glaciers', 'CI' ], delim_whitespace=True)
    M15_glo_df = M15_glo_df.set_index('time')
    M15_glo_df['Glaciers'] = - M15_glo_df.Glaciers + M15_glo_df.Glaciers.iloc[0]
    del M15_glo_df['CI']
    return M15_glo_df / 10 # Convert from mm to cm

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
    
    M15_dir = PATH_SeaLevelBudgets + 'Glaciers/Marzeion2015/tc-9-2399-2015-supplement/'
    fp_dir = PATH_SeaLevelBudgets + 'fp_uniform/'
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
    data_dir = (PATH_SeaLevelBudgets + 
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
    
    data_dir = (PATH_SeaLevelBudgets + 
                'Glaciers/Zemp2019/Zemp_etal_results_regions_global_v11/')
    fp_dir = PATH_SeaLevelBudgets + 'fp_uniform/'
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

def ant_imbie_glo(extrap=False):
    '''Read IMBIE 2018 excel data, compute yearly averages and return a data 
    frame of sea level rise in cm'''
    imbie_dir = PATH_SeaLevelBudgets + 'Antarctica/IMBIE2018/'
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

def ant_rignot19_glo():
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
    tg_data_dir = PATH_SeaLevelBudgets + 'rlr_annual'
    kern_dir = PATH_SeaLevelBudgets + 'Mitrovica2018Kernels/'
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
            kern_dir = PATH_SeaLevelBudgets + 'Mitrovica2018Kernels/'
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
            Adh_dir = PATH_SeaLevelBudgets + 'Adhikari2019/'
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
            fp_dir = PATH_SeaLevelBudgets + 'fp_uniform/'
            tg_lat, tg_lon =  tg_lat_lon(tg_id[i])
            if ices == 'ant':
                filename = 'AIS.nc' #WAIS and EAIS are also available
            elif ices == 'green':
                filename = 'GrIS.nc'
            fp_fre_ds = xr.open_dataset(fp_dir + filename)
            fp_val.append(fp_fre_ds.rsl.sel(x = tg_lon, y = tg_lat, method='nearest').values.tolist())
    return np.mean(fp_val)

def green_mouginot19_glo():
    '''Read the Greenland contribution to sea level from Mouginot et al. 2019 and export in a dataframe.
    Date available from 1972 to 2018.'''
    green_dir = PATH_SeaLevelBudgets + 'Greenland/'
    mo_df = pd.read_csv(green_dir + 'Mouginot2019_MB.txt')
    del mo_df['Unnamed: 0']
    mo_df = mo_df.T
    mo_df.columns = ['Greenland']
    mo_df.Greenland = pd.to_numeric(mo_df.Greenland.astype(str).str.replace(',','.'), errors='coerce')
    mo_df['Years'] = np.arange(1972,2019)
    mo_df = mo_df.set_index('Years')
    mo_df = - mo_df / 3600 #Convert from Gt to cm
    mo_df = mo_df.reindex(np.arange(1950,2019)).fillna(0)
    return mo_df

def TWS_loc(tg_id):
    '''Read TWS effect on relative sea level derived from GRACE from a file given by Thomas Frederikse.'''
    dir_fpg = PATH_SeaLevelBudgets + 'fp_grace/'
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

def TWS_glo(extrap=False):
    '''Build a pandas data frame from the global terrestrial water storage 
    reconstructions of Humphrey and Gudmundson 2019. Data available from 1901-01
    to 2014-12. Option avialable to '''
    dir_tws = PATH_SeaLevelBudgets + \
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
    Possibility to extrapolate the time series forward up to 2019 using the trend of 
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
    LevitusSL_df.rename(columns={'pent_s_mm_WO': 'StericLevitus'}, inplace=True)
    if extrap_back:
        nby = 20
        trend = np.polyfit(LevitusSL_df.index[:nby], \
                           LevitusSL_df.StericLevitus.iloc[:nby], 1)[0]
        for i in range(7):
            LevitusSL_df.loc[LevitusSL_df.index.min() - 1] = \
            ( LevitusSL_df.StericLevitus.loc[LevitusSL_df.index.min()] - trend )
        LevitusSL_df.sort_index(inplace=True)
    if extrap:
        nby = 5
        trend = np.polyfit(LevitusSL_df.index[-nby:], \
                           LevitusSL_df.StericLevitus.iloc[-nby:], 1)[0]
        for i in range(3):
            LevitusSL_df.loc[LevitusSL_df.index.max() + 1] = \
            ( LevitusSL_df.StericLevitus.loc[LevitusSL_df.index.max()] + trend )
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