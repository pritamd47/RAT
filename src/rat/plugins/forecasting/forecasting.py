import xarray as xr
import numpy as np
import requests
import dask
import pandas as pd
import geopandas as gpd
from dask.distributed import LocalCluster, Client
from pathlib import Path
import tempfile
import os
import shutil
import numpy as np
from datetime import date
from rat.utils.run_command import run_command
from rat.toolbox.MiscUtil import get_quantile_from_area

# Obtain GEFS precip data
def download_gefs_chirps(basedate, lead_time, save_dir, low_latency=False):
    """Download forecast precipitation from GEFS-CHIRPS.

    Args:
        basedate (pd.Timestamp): Data starting from next day will be downloaded.
        lead_time (int): Lead time in days.
        save_dir (Path): Directory to save data.
    """
    assert lead_time <= 15, "Maximum lead time is 15 days for GEFS-CHIRPS."

    forecast_dates = pd.date_range(basedate, basedate + pd.Timedelta(days=lead_time))
    if low_latency:
        start_index=0
    else:
        start_index=1

    for forecast_date in forecast_dates[start_index:]: # Skip the first day, download forecast from next day but if latency mode is being used, download will start first day
        by = basedate.year
        bm = basedate.month
        bd = basedate.day
        fy = forecast_date.year
        fm = forecast_date.month
        fd = forecast_date.day

        save_fp = save_dir / f"{basedate:%Y%m%d}" / f"{forecast_date:%Y%m%d}.tif"
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day/{by}/{bm:02}/{bd:02}/"
        url = prefix + f"data.{fy}.{fm:02}{fd:02}.tif"

        r = requests.get(url)
        if r.status_code == 200:
            with open(save_fp, 'wb') as f:
                f.write(r.content)
        else:
            print(f"Could not download {url}")


def process_gefs_chirps(
        basin_bounds, 
        srcpath, 
        dstpath, 
        temp_datadir=None,
    ):
    """For any IMERG Precipitation file located at `srcpath` is clipped, scaled and converted to
    ASCII grid file and saved at `dstpath`. All of this is done in a temporarily created directory
    which can be controlled by the `datadir` path
    """
    src_fn = Path(srcpath)
    dst_fn = Path(dstpath)
    dst_fn.parent.mkdir(parents=True, exist_ok=True) # Create parent directory if it doesn't exist

    date = pd.to_datetime(src_fn.stem.split('_')[0])
    if temp_datadir is not None and not os.path.isdir(temp_datadir):
        STATUS='FAILED'
        return date, 'Precipitaion', STATUS
    
    if not(os.path.exists(dstpath)):
        # log.debug("Processing Precipitation file: %s", srcpath)
        print(f"Processing Precipitation file: {srcpath}")
        STATUS = 'STARTED'
        with tempfile.TemporaryDirectory(dir=temp_datadir) as tempdir:
            clipped_temp_file = os.path.join(tempdir, 'clipped.tif')
            cmd = [
                "gdalwarp",
                "-dstnodata", 
                "-9999.0",
                "-tr",
                "0.0625",
                "0.0625",
                "-te",
                str(basin_bounds[0]),
                str(basin_bounds[1]),
                str(basin_bounds[2]),
                str(basin_bounds[3]),
                '-of',
                'GTiff',
                '-overwrite', 
                f'{srcpath}',
                clipped_temp_file
            ]
            run_command(cmd)

            # Change format, and save as processed file
            aai_temp_file = os.path.join(tempdir, 'processed.tif')
            cmd = [
                'gdal_translate',
                '-of', 
                'aaigrid',
                clipped_temp_file,
                aai_temp_file
            ]
            run_command(cmd)

            # Move to destination
            shutil.move(aai_temp_file, dstpath)
            STATUS = 'SUCCESS'
    else:
        STATUS = 'SKIPPED'
    return date, 'Precipitaion', STATUS


def get_gefs_precip(basin_bounds, forecast_raw_dir, forecast_processed_dir, begin, lead_time, low_latency=False, temp_datadir=None):
    """Download and process forecast data for a day."""
    # download data
    download_gefs_chirps(begin, lead_time, forecast_raw_dir, low_latency)

    # process data
    futures = []
    forecast_raw_date_dir = forecast_raw_dir / f"{begin:%Y%m%d}"
    raw_files = sorted(list(forecast_raw_date_dir.glob("*.tif")))
    for raw_file in raw_files:
        date = pd.to_datetime(raw_file.stem.split('_')[0])
        dstpath = forecast_processed_dir / f"{begin:%Y%m%d}" / f"{date:%Y%m%d}.asc"
        dstpath.parent.mkdir(parents=True, exist_ok=True)

        print(raw_file, dstpath)

        future = dask.delayed(process_gefs_chirps)(basin_bounds, raw_file, dstpath, temp_datadir=temp_datadir)
        futures.append(future)

    processed_statuses = dask.compute(*futures)
    processed_statuses = pd.DataFrame(processed_statuses, columns=['Date','Variable','Status'])
    processed_statuses.sort_values(by=["Date","Variable"], inplace=True)
    return processed_statuses


def download_GFS_files(
        basedate, lead_time, save_dir, hour_of_day=0
    ):
    """Download GFS files for a day and saves it with a consistent naming format for further processing."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    hours = range(bool(lead_time)*24+hour_of_day, lead_time*24+hour_of_day+1, 24) # Added 1 in the end of range because it is exclusive. So just increase by 1

    # determine where to download from. nomads has data for the last 10 days
    if basedate > pd.Timestamp(date.today()) - pd.Timedelta(days=10):
        links = [f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs.{basedate:%Y%m%d}%2F00%2Fatmos&file=gfs.t00z.pgrb2.0p25.f{h:03}&var_TMAX=on&var_TMIN=on&var_UGRD=on&var_VGRD=on&lev_2_m_above_ground=on&lev_10_m_above_ground=on" for h in hours]
    else:
        links = [f"https://data.rda.ucar.edu/ds084.1/{basedate:%Y}/{basedate:%Y%m%d}/gfs.0p25.{basedate:%Y%m%d}00.f{h:03}.grib2" for h in hours]
    
    raw_savefns = [save_dir / f"{basedate:%Y%m%d}/{basedate+pd.Timedelta(h, 'hours'):%Y%m%d}.grib2" for h in hours]

    for link, savefn in zip(links, raw_savefns):
        savefn.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(link)
        with open(savefn, 'wb') as f:
            f.write(r.content)


def aggregate_ncs(ncs, fn, var=None):
    """Will return a single dataset same as ncs[0] with the underlying values being equal to the result of the `fn` aggregation function

    Args:
        ncs (list[DataArray/DataSet]): List of datarrays or datasets to aggregate. If a dataset is passed, `var` must be passed
        fn (function): Aggregation function
        var (str): Variable name 
    """
    if isinstance(ncs[0], xr.Dataset):
        ncs_dataarrays = [nc[var] for nc in ncs]
    else:
        ncs_dataarrays = ncs
    
    res = ncs_dataarrays[0].copy()

    res.values = fn([nc.values for nc in ncs_dataarrays], axis=0)

    return res


def extract_gribs(gfs_dir, date):
    """Extracts GRIB files to NetCDF files and returns the paths to the extracted files."""
    gfs_dir = Path(gfs_dir)
    date = pd.to_datetime(date)
    raw_dir = gfs_dir / "raw" / f"{date:%Y%m%d}"
    save_dir = gfs_dir / "extracted" / f"{date:%Y%m%d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(list(raw_dir.glob("*.grib2")))

    extracted_dir = {
        "tmax": save_dir / f"tmax",
        "tmin": save_dir / f"tmin",
        "uwnd": save_dir / f"uwnd",
        "vwnd": save_dir / f"vwnd"
    }

    extraction = {
        "tmax": lambda ds: (ds['tmax']-273.15).sel(heightAboveGround=2).rename('tmax').drop_vars('heightAboveGround'),
        "tmin": lambda ds: (ds['tmin']-273.15).sel(heightAboveGround=2).rename('tmin').drop_vars('heightAboveGround'),
        "uwnd": lambda ds: ds['u10'].sel(heightAboveGround=10).rename('uwnd').drop_vars('heightAboveGround'),
        "vwnd": lambda ds: ds['v10'].sel(heightAboveGround=10).rename('vwnd').drop_vars('heightAboveGround'),
    }

    extracted_fps = {
        "tmax": [],
        "tmin": [],
        "uwnd": [],
        "vwnd": []
    }

    for fn in filepaths:
        basedate = pd.to_datetime(fn.parent.name)
        forecasted_date = pd.to_datetime(fn.stem)

        # cfgrib_datasets = cfgrib.open_datasets(str(fn))
        tempds = xr.open_dataset(
            str(fn), engine="cfgrib",
            backend_kwargs={
                'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2},
                "indexpath": ""
            }
        )
        windds = xr.open_dataset(
            str(fn), engine="cfgrib",
            backend_kwargs={
                'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 10},
                "indexpath": ""
            }
        )

        ds = xr.merge([
            d.expand_dims("heightAboveGround") 
            for d in [tempds, windds]
        ])
        ds = ds.assign_coords(longitude=((360 + (ds.longitude % 360)) % 360))
        ds = ds.roll(longitude=int(len(ds['longitude']) / 2), roll_coords=True)

        for var in ["tmax", "tmin", "uwnd", "vwnd"]:
            savedir = extracted_dir[var]
            savedir.mkdir(parents=True, exist_ok=True)

            # (1) Read in with xarray and only save the bare minimum
            modified = extraction[var](ds)

            savep = savedir / f"{forecasted_date:%Y%m%d}.nc"
            modified.to_netcdf(savep)
            print(f"{basedate} ({forecasted_date} hours) - {var} - (1) Subsetted nc file")
            extracted_fps[var].append(savep)


def process_GFS_file(fn, basin_bounds, gfs_dir):
    # date = pd.to_datetime(fn.split(os.sep)[-1].split('.')[0], format='%Y%m%d')
    fn = Path(fn)
    gfs_dir = Path(gfs_dir)

    basedate = pd.to_datetime(fn.parent.parent.name)
    forecasted_date = pd.to_datetime(fn.stem)
    var = fn.parent.name
    
    processed_savedirs = {
        "tmax": gfs_dir / f"processed/{basedate:%Y%m%d}/tmax",
        "tmin": gfs_dir / f"processed/{basedate:%Y%m%d}/tmin",
        "uwnd": gfs_dir / f"processed/{basedate:%Y%m%d}/uwnd",
        "vwnd": gfs_dir / f"processed/{basedate:%Y%m%d}/vwnd"
    }
    for v in ["tmax", "tmin", "uwnd", "vwnd"]:
        processed_savedirs[v].mkdir(parents=True, exist_ok=True)

    # (1) Convert to GTiff
    in_fn = fn
    out_fn = fn.with_name(f'{forecasted_date:%Y%m%d}_converted_1.tif')
    cmd = ['gdal_translate', '-of', 'Gtiff', '-a_ullr', "-180", "90", "180", "-90", str(in_fn), str(out_fn)]
    res = run_command(cmd)  #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(f"{basedate:%Y-%m-%d} - {var} - (2) Converted to Geotiff")

    # (2)Change scale and clip
    in_fn = out_fn
    out_fn = in_fn.with_name(f'{forecasted_date:%Y%m%d}_clipped_2.tif')
    cmd = ['gdalwarp', '-dstnodata', '-9999.0', '-tr', '0.0625', '0.0625', '-te', str(basin_bounds[0]), str(basin_bounds[1]), str(basin_bounds[2]), str(basin_bounds[3]), "-of", "GTiff", "-overwrite", str(in_fn), str(out_fn)]
    res = run_command(cmd)  #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(f"{basedate:%Y-%m-%d} - {var} - (3) Scaled and clipped")

    # (3) Change format to AAIgrid
    in_fn = out_fn
    outdir = processed_savedirs[var]
    outdir.mkdir(parents=True, exist_ok=True)

    out_fn = outdir / f"{forecasted_date:%Y%m%d}.asc"
    cmd = ['gdal_translate', '-of', 'aaigrid', str(in_fn), str(out_fn)]
    res = run_command(cmd)#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(f"{basedate:%Y-%m-%d} - {var} - (4) Converted to .asc")

def process_GFS_files(basedate, lead_time, basin_bounds, gfs_dir, hour_of_day=0):
    """Extracts only the required meteorological variables and converts from GRIB2 format to netcdf

    Args:
        fns (string):,
        hour_of_day=hour_of_day Path of GRIB2 GFS file
    """
    gfs_dir = Path(gfs_dir)
    hours = range(bool(lead_time)*24+hour_of_day, lead_time*24+hour_of_day+1, 24)
    forecasted_dates = [basedate + pd.Timedelta(h, 'hours') for h in hours]
    
    in_fns = [
        gfs_dir / f"extracted/{basedate:%Y%m%d}/{var}/{forecasted_date:%Y%m%d}.nc" 
        for forecasted_date in forecasted_dates 
        for var in ["tmax", "tmin", "uwnd", "vwnd"]
    ]
    for i, in_fn in enumerate(in_fns):
        process_GFS_file(in_fn, basin_bounds, gfs_dir)

def get_GFS_data(basedate, lead_time, basin_bounds, gfs_dir, hour_of_day=0):
    """Extracts only the required meteorological variables and converts from GRIB2 format to netcdf

    Args:
        fns (string): Path of GRIB2 GFS file
    """
    download_GFS_files(
        basedate,
        lead_time=lead_time,
        save_dir=gfs_dir / "raw",
        hour_of_day=hour_of_day
    )

    extract_gribs(gfs_dir, basedate)

    process_GFS_files(
        basedate,
        lead_time,
        basin_bounds,
        gfs_dir,
        hour_of_day=hour_of_day
    )

def forecast_scenario_custom_wl(forecast_outflow, initial_sa, aec_data, cust_wl):
    delS = []
    delH = []
    sarea = []
    elevation = []
    outflow = []
    curr_sa = initial_sa
    curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
    cust_wl_area = np.interp(cust_wl, aec_data['elevation'],aec_data['area'])
    init_deficit_elevation = cust_wl - curr_elevation
    init_deficit_storage = 0 if init_deficit_elevation > 0 else init_deficit_elevation*curr_sa*1E6 + forecast_outflow['evaporation'].sum().values
    deficit_elevation = init_deficit_elevation
    for iter,inflow in enumerate(forecast_outflow['inflow'].values):
        
        if(curr_elevation > cust_wl):
            # how is this working?
            deficit_release = init_deficit_storage/(16)
            if(init_deficit_storage==0):
                outflow.append(inflow - forecast_outflow['evaporation'].values[iter])
                delS.append(0.0)
            else:
                outflow.append(inflow - deficit_release)
                delS.append(deficit_release - inflow - forecast_outflow['evaporation'].values[iter])
        else:
            delS.append(inflow - forecast_outflow['evaporation'].values[iter])
            outflow.append(0.0)
        delH.append(delS[-1]/(curr_sa*1E6))        # S2 - S1 = (H2-H1) * (A1 + A2)/2; Is A2 -> A1?
        curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
        new_elevation = curr_elevation + delH[-1]
        elevation.append(new_elevation)
        curr_sa = np.interp(new_elevation, aec_data['elevation'],aec_data['area'])
        sarea.append(curr_sa)

        
    sarea_da = xr.DataArray(sarea, dims='date', coords={'date': forecast_outflow['date']})
    delH_da = xr.DataArray(delH, dims='date', coords={'date': forecast_outflow['date']})
    delS_da = xr.DataArray(delS, dims='date', coords={'date': forecast_outflow['date']})
    elevation_da = xr.DataArray(elevation, dims='date', coords={'date': forecast_outflow['date']})
    outflow_da = xr.DataArray(outflow, dims='date', coords={'date': forecast_outflow['date']})

    forecast_outflow['CL_sarea'] = sarea_da
    forecast_outflow['CL_delH'] = delH_da
    forecast_outflow['CL_delS'] = delS_da
    forecast_outflow['CL_elevation'] = elevation_da
    forecast_outflow['CL_outflow'] = outflow_da

    return forecast_outflow

def rc_SbySmax(date, rule_curve):
    ''' 
    Returns the linearly interpolated S/Smax value for a given date using monthly rule curve values as input.
    
    Parameters
    ----------
    date: str or datetime object
        Date for which the S/Smax value is to be extracted from the rule curve. (str format: 'yyyy-mm-dd' or 'mm-dd-yyyy')
    rule_curve: str
        Path to the rule curve file for the reservoir in .csv format containing headers - (Month, S/Smax)
    '''
    #Extracting day and month from date
    date = pd.to_datetime(date)
    day = date.day
    month = date.month
    
    ## reading rule curve data and returning interpolated S/Smax for given day and month
    rule_curve_data = pd.read_csv(rule_curve)
    rc_SbySmax = rule_curve_data['S/Smax']
    day_inMonths = day/30
    #If month is December, interpolate between Dec and Jan. For all other months, interpolate between the given month and next.
    if(month == 12):
        SbySmax_day = rc_SbySmax.loc[month-1] + (rc_SbySmax.loc[0]-rc_SbySmax.loc[month-1])*day_inMonths
    else:
        SbySmax_day = rc_SbySmax.loc[month-1] + (rc_SbySmax.loc[month]-rc_SbySmax.loc[month-1])*day_inMonths

    return(SbySmax_day)

def forecast_scenario_rule_curve(forecast_outflow, initial_sa, base_date, end_date, rule_curve, s_max, aec_data):
    delSbySmax = rc_SbySmax(end_date, rule_curve) - rc_SbySmax(base_date, rule_curve)
    rc_delS= delSbySmax*(s_max*1E6) #m3
    delS = []
    delH = []
    sarea = []
    elevation = []
    outflow = []
    curr_sa = initial_sa
    curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
    cum_netInflow = forecast_outflow['inflow'].sum() - forecast_outflow['evaporation'].sum()
    for iter,inflow in enumerate(forecast_outflow['inflow'].values):
        if(cum_netInflow < rc_delS):
            delS.append(inflow - forecast_outflow['evaporation'].values[iter])
            delH.append(delS[-1]/(curr_sa*1E6))
            curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
            new_elevation = curr_elevation + delH[-1]
            elevation.append(new_elevation)
            curr_sa = np.interp(new_elevation, aec_data['elevation'], aec_data['area'])
            sarea.append(curr_sa)
            outflow.append(0)
        else:
            net_delS = cum_netInflow - rc_delS
            outflow.append(net_delS/15)
            delS.append(inflow - outflow[-1])
            delH.append(delS[-1]/(curr_sa*1E6))
            curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
            new_elevation = curr_elevation + delH[-1]
            elevation.append(new_elevation)
            curr_sa = np.interp(new_elevation, aec_data['elevation'],aec_data['area'])
            sarea.append(curr_sa)

    sarea_da = xr.DataArray(sarea, dims='date', coords={'date': forecast_outflow['date']})
    delH_da = xr.DataArray(delH, dims='date', coords={'date': forecast_outflow['date']})
    delS_da = xr.DataArray(delS, dims='date', coords={'date': forecast_outflow['date']})
    elevation_da = xr.DataArray(elevation, dims='date', coords={'date': forecast_outflow['date']})
    outflow_da = xr.DataArray(outflow, dims='date', coords={'date': forecast_outflow['date']})

    forecast_outflow['RC_sarea'] = sarea_da
    forecast_outflow['RC_delH'] = delH_da
    forecast_outflow['RC_delS'] = delS_da
    forecast_outflow['RC_elevation'] = elevation_da
    forecast_outflow['RC_outflow'] = outflow_da

    return forecast_outflow

def forecast_scenario_gates_closed(forecast_outflow, initial_sa, aec_data):
    #Initialising delS, delH, sarea, elevation variables. outflow = 0
    delS = []
    delH = []
    sarea = []
    elevation = []
    forecast_outflow['GC_outflow'] = xr.zeros_like(forecast_outflow['inflow'])
    curr_sa = initial_sa
    #Iterating over daily inflow and computing new delS, delH, elevation and sarea values.
    for iter,inflow in enumerate(forecast_outflow['inflow'].values):
        delS.append(inflow - forecast_outflow['evaporation'].values[iter])
        delH.append(delS[-1]/(curr_sa*1E6))
        curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
        new_elevation = curr_elevation + delH[-1]
        elevation.append(new_elevation)
        curr_sa = np.interp(new_elevation, aec_data['elevation'], aec_data['area'])
        sarea.append(curr_sa)
    # Creating new data arrays and merging with the xarray dataset
    sarea_da = xr.DataArray(sarea, dims='date', coords={'date': forecast_outflow['date']})
    delH_da = xr.DataArray(delH, dims='date', coords={'date': forecast_outflow['date']})
    elevation_da = xr.DataArray(elevation, dims='date', coords={'date': forecast_outflow['date']})
    forecast_outflow['GC_sarea'] = sarea_da
    forecast_outflow['GC_delH'] = delH_da
    forecast_outflow['GC_elevation'] = elevation_da

    return forecast_outflow

def forecast_scenario_st(forecast_outflow, initial_sa, cust_st, s_max, aec_data, st_percSmax):
    #Creating positive and negative values for permissible storage cases
    st_percSmax = np.array(st_percSmax)
    # st_percSmax = np.append(st_percSmax, -st_percSmax)

    #Converting storage cases from %Smax to absolute volumes
    st_volumes = np.array(st_percSmax)/100*s_max*1E6
    if(cust_st is not None):
        st_percSmax = np.append(st_percSmax, cust_st)
        st_volumes = np.append(st_volumes, cust_st*1E6)
    # Iterating over all storage cases and creating outflow, sarea, delH, delS, elevation change results
    for iter_vol,vol in enumerate(st_volumes):
        delS = []
        delH = []
        sarea = []
        elevation = []
        outflow = []
        curr_sa = initial_sa
        curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
        cum_netInflow = forecast_outflow['inflow'].sum() - forecast_outflow['evaporation'].sum()
        dels_remaining = vol
        dels_stored = 0
        for iter,inflow in enumerate(forecast_outflow['inflow'].values):
            if(cum_netInflow < vol):
                delS.append(inflow - forecast_outflow['evaporation'].values[iter])
                delH.append(delS[-1]/(curr_sa*1E6))
                curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
                new_elevation = curr_elevation + delH[-1]
                elevation.append(new_elevation)
                curr_sa = np.interp(new_elevation, aec_data['elevation'], aec_data['area'])
                sarea.append(curr_sa)
                outflow.append(0)
            else:
                I_E = inflow - forecast_outflow['evaporation'].values[iter]
                if dels_remaining > 0:
                    if I_E <= dels_remaining:
                        O = 0
                        dels_stored = I_E
                        dels_remaining = dels_remaining - dels_stored
                    else:
                        O = I_E - dels_remaining
                        dels_stored = dels_remaining
                        dels_remaining = 0
                else:
                    # If no remaining storage capacity, release all incoming water minus evaporation
                    O = I_E
                    dels_stored = 0  # No water is stored
                outflow.append(O)
                delS.append(dels_stored)
                delH.append(delS[-1]/(curr_sa*1E6))
                curr_elevation = np.interp(curr_sa, aec_data['area'], aec_data['elevation'])
                new_elevation = curr_elevation + delH[-1]
                elevation.append(new_elevation)
                curr_sa = np.interp(new_elevation, aec_data['elevation'], aec_data['area'])
                sarea.append(curr_sa)
        sarea_da = xr.DataArray(sarea, dims='date', coords={'date': forecast_outflow['date']})
        delH_da = xr.DataArray(delH, dims='date', coords={'date': forecast_outflow['date']})
        delS_da = xr.DataArray(delS, dims='date', coords={'date': forecast_outflow['date']})
        elevation_da = xr.DataArray(elevation, dims='date', coords={'date': forecast_outflow['date']})
        outflow_da = xr.DataArray(outflow, dims='date', coords={'date': forecast_outflow['date']})
        if(iter_vol==len(st_percSmax)-1 and cust_st is not None):
            forecast_outflow[f'ST_sarea_{st_percSmax[iter_vol]}MCM'] = sarea_da
            forecast_outflow[f'ST_delH_{st_percSmax[iter_vol]}MCM'] = delH_da
            forecast_outflow[f'ST_delS_{st_percSmax[iter_vol]}MCM'] = delS_da
            forecast_outflow[f'ST_elevation_{st_percSmax[iter_vol]}MCM'] = elevation_da
            forecast_outflow[f'ST_outflow_{st_percSmax[iter_vol]}MCM'] = outflow_da
        else:
            forecast_outflow[f'ST_sarea_{st_percSmax[iter_vol]}%'] = sarea_da
            forecast_outflow[f'ST_delH_{st_percSmax[iter_vol]}%'] = delH_da
            forecast_outflow[f'ST_delS_{st_percSmax[iter_vol]}%'] = delS_da
            forecast_outflow[f'ST_elevation_{st_percSmax[iter_vol]}%'] = elevation_da
            forecast_outflow[f'ST_outflow_{st_percSmax[iter_vol]}%'] = outflow_da    
    return forecast_outflow

def forecast_outflow_for_res(
        base_date, 
        forecast_lead_time,
        forecast_inflow_fp,
        evap_fp,
        sarea_fp,
        aec_fp,
        dels_scenario = None, 
        cust_wl = None, 
        s_max = None, 
        rule_curve = None, 
        st_percSmax = [0.5, 1, 2.5], 
        cust_st = None, 
        output_path = None,
        actual_st_left = False
    ):
    ''' 
    Generates forecasted outflow for RAT given forecasted inflow time series data based on specific delS scenarios.
    Returns an xarray dataset containing forecasted results.
    
    Parameters
    ----------
    base_date: str or datetime object
        Start date of forecasted outflow estimation. This is typically the end date of hindcast RAT run + 1.
    forecast_lead_time: int
        Lead time of forecast in days.
    forecast_inflow_fp: str
        Path to forecasted inflow file in `rat_outputs` format (in .csv format with streamflow and date as header).
    evap_fp: str
        Path to evaporation data file in `rat_outputs` format (in .csv format with OUT_EVAP and date as header).
    sarea_fp: str
        Path to surface area data file in `rat_outputs` format (in .csv format with area and date as header).
    aec_fp: str
        Path to area-elevation curve data file (in .csv format with area and elevation).
    dels_scenario: str
        Specifies the delS scenario to use. If None, returns results for all scenarios
        values: 
            GC - Gates Closed scenario: All inflow will be accumulated as change in storage. Final surface area, storage state, water level is returned
            GO - Gates Open scenario: All inflow will be released as outflow. Outflows returned 
            CL - Custom Water Level scenario: Inflow will be accumulated till desired water level, then released as outflow
            RC - Rule Curve scenario: Outflow will be calculated depending on the permissible rule curve based change in storage
            ST - storage change scenario: Outflows will be calculated based on permissible storage values provided 
    cust_wl: float
        Specifies the value for custom water level in meters above sea level. Only used if dels_scenario = CL or None.
    s_max: float
        Maximum storage capacity of the reservoir in Million cubic meters (MCM)
    rule_curve: str
        Path to rule curve file in .csv with headers (Month, S/Smax)   
    st_percSmax: list
        A list of values of storages as percentage of Smax
    cust_st: float
        User defined custom storage value in Million cubic meters (MCM). Used if dels_scenario = ST or None  
    output_path: str
        If provided, generates a netcdf file at the specified path containing the forecasted outflow results.
    '''
    # If actual_st_left is true
    if actual_st_left and 'ST' in dels_scenario:
        actual_st_left_percent = np.round(100*(1-get_quantile_from_area(sarea_fp)),1)
        st_percSmax.append(actual_st_left_percent)

    # Reading forecasted inflow data, computing forecasting period as base date to base date + 15 days
    forecast_inflow = pd.read_csv(forecast_inflow_fp, parse_dates=['date']).set_index('date').rename({'streamflow': 'inflow'}, axis=1).to_xarray()
    forecast_inflow = forecast_inflow*86400 # convert m3/s to m3/day
    base_date = pd.to_datetime(base_date)
    base_date_15lead = base_date + pd.Timedelta(days=forecast_lead_time)

    # Reading evaporation, sarea, aec data
    evaporation_data = pd.read_csv(evap_fp)
    sarea_data = pd.read_csv(sarea_fp)
    aec_data = pd.read_csv(aec_fp)
    
    ## Initialising forecasted_outflow dataset. 
    # Inititial surface area is taken as the final surface area state of hindcasted data
    # Evaporation is taken to be constant across the forecasted period and equals the final evaporation value of hindcasted data
    # forecast_outflow = forecast_inflow.copy()
    forecast_outflow = forecast_inflow.sel(date = slice(base_date, base_date_15lead))
    initial_sa = sarea_data['area'].values[-1]
    evaporation_mm = evaporation_data['OUT_EVAP'].values[-1]
    forecast_outflow['evaporation'] = xr.ones_like(forecast_outflow['inflow'])*(evaporation_mm*1E-3*initial_sa*1E6)
    
    ################################################ delS scenario based forecasted outflow estimation #######################################################
       
    ######### delS_scenario = GC (Gates Closed) ############ 
    # outflow = 0, dels = I - E
    # surface area and delH is iteratively estimated
    if(dels_scenario is None or dels_scenario == 'GC'):
        forecast_outflow_gates_closed = forecast_scenario_gates_closed(forecast_outflow, initial_sa, aec_data)
        forecast_outflow = forecast_outflow.merge(forecast_outflow_gates_closed)

    ######### delS_scenario = GO (Gates Open to match inflow) ############ 
    # outflow = I, dels = 0
    if(dels_scenario is None or dels_scenario == 'GO'):
        forecast_outflow['GO_outflow'] = forecast_outflow['inflow'] - forecast_outflow['evaporation']
        forecast_outflow['GO_sarea'] = xr.ones_like(forecast_outflow['inflow'])*initial_sa
    
    ######### If dels_scenario = CL (Custom Water Level ############ 
    # delS = I - E till custom water level is reached.
    # Then delS = 0 (if I > Evap else I - Evap), O = I - E
    # delSA and delH is iteratively estimated  
    if(dels_scenario is None or dels_scenario == 'CL'):
        if cust_wl is not None:
            forecast_outflow_cust_wl = forecast_scenario_custom_wl(forecast_outflow, initial_sa, aec_data, cust_wl)
            forecast_outflow = forecast_outflow.merge(forecast_outflow_cust_wl)

    ######### If dels_scenario = RC (Rule Curve Scenario ############ 
    # S/Smax ratio for base date and base date + 15 days is computed.
    # delS/Smax is obtained and multiplied with known Smax to get permissible delS.
    # if delS is +ve, inflow is accumulated. If delS is -ve, outflow is produced by evenly releasing required inflow over the 15 day period.
    if(dels_scenario is None or dels_scenario == 'RC'):
        if rule_curve is not None:
            forecast_outflow_rule_curve = forecast_scenario_rule_curve(forecast_outflow, initial_sa, base_date, base_date_15lead, rule_curve, s_max, aec_data)
            forecast_outflow = forecast_outflow.merge(forecast_outflow_rule_curve)

    ######### If dels_scenario = ST (Storage Scenario ############ 
    # Storage change scenarios take a list of %Smax values as permissible change in storage. Eg. [1,2.5,5] (1%, 2.5%, and 5% of Smax)
    # Outflows, delH, delS, sarea are generated for all %Smax values as per similar logic as RC scenario.
    # Custom storage value in MCM may also be provided. (release: -ve or store: +ve)
    if(dels_scenario is None or dels_scenario == 'ST'):
        forecast_outflow_st = forecast_scenario_st(forecast_outflow, initial_sa, cust_st, s_max, aec_data, st_percSmax)
        forecast_outflow = forecast_outflow.merge(forecast_outflow_st)

    if(output_path is not None):
        # read existing file, and append new columns
        new_data = forecast_outflow.to_pandas().drop(columns=['evaporation', 'inflow']).reset_index()
        new_data['date'] = pd.to_datetime(new_data['date'])
        if Path(output_path).exists():
            old_data = pd.read_csv(output_path, parse_dates = ['date'])
            combined_data = pd.merge(new_data, old_data, how='outer', on='date', suffixes=(None, "_old"))
            combined_data = combined_data.loc[:, ~combined_data.columns.str.endswith('_old')] # drop columns that are old, replace them with newly generated scenario outputs
            combined_data.to_csv(output_path, index=False)
        else:
            new_data.to_csv(output_path, index=False)

    return forecast_outflow

def forecast_outflow(
    basedate, lead_time, basin_data_dir, reservoir_shpfile, reservoir_shpfile_column_dict,
    forecast_reservoir_shpfile_column_dict, rule_curve_dir,
    scenarios = ['GC', 'GO', 'RC', 'ST'],
    st_percSmaxes = [0.5, 1, 2.5],
    actual_st_left = False
):
    reservoirs = gpd.read_file(reservoir_shpfile)
    reservoirs['Inflow_filename'] = reservoirs[reservoir_shpfile_column_dict['unique_identifier']].astype(str)

    for res_name in reservoirs['Inflow_filename'].tolist():
        res_scenarios = scenarios.copy()
        observed_outflow_fp = basin_data_dir / 'final_outputs' / 'outflow' / f'{res_name}.csv'
        # Calculate forecast outflow only if observed outflow exists. Will skip for guages.
        if (observed_outflow_fp.exists()):

            forecast_inflow_fp = basin_data_dir / 'rat_outputs' / 'forecast_inflow' / f'{basedate:%Y%m%d}' / f'{res_name}.csv'
            forecast_evap_fp = basin_data_dir / 'rat_outputs' / 'forecast_evaporation' / f'{basedate:%Y%m%d}' / f'{res_name}.csv'
            sarea_fp = basin_data_dir / 'gee' / 'gee_sarea_tmsos' / f'{res_name}.csv'
            aec_fp = basin_data_dir / 'final_outputs' / 'aec' / f'{res_name}.csv'
            output_fp = basin_data_dir / 'rat_outputs' / 'forecast_outflow' / f'{basedate:%Y%m%d}' / f'{res_name}.csv'
            output_fp.parent.mkdir(parents=True, exist_ok=True)

            s_max = reservoirs[reservoirs[reservoir_shpfile_column_dict['unique_identifier']] == res_name][forecast_reservoir_shpfile_column_dict['column_capacity']].values[0]
            reservoir_id = reservoirs[reservoirs[reservoir_shpfile_column_dict['unique_identifier']] == res_name][forecast_reservoir_shpfile_column_dict['column_id']].values[0] # reservoir_id will correspond to GRAND_ID if you're using RAT provided rule curves. 

            if np.isnan(s_max) and 'ST' in res_scenarios:
                res_scenarios.remove('ST')

            if (np.isnan(reservoir_id) or rule_curve_dir is None) and 'RC' in res_scenarios: # if reservoir_id is nan or if rule_curve_dir is not passed, remove RC scenario
                res_scenarios.remove('RC')
                rule_curve_fp = None
            else:
                rule_curve_fp = rule_curve_dir / f'{int(reservoir_id)}.txt' # if reservoir_id is float, coerce it to int

            for scenario in res_scenarios:       
                forecast_outflow_for_res(
                    base_date = basedate,
                    forecast_lead_time = lead_time,
                    forecast_inflow_fp = forecast_inflow_fp,
                    evap_fp = forecast_evap_fp,
                    sarea_fp = sarea_fp,
                    aec_fp = aec_fp,
                    dels_scenario = scenario,
                    s_max = s_max,
                    rule_curve = rule_curve_fp,
                    st_percSmax = st_percSmaxes[:],
                    output_path = output_fp,
                    actual_st_left = actual_st_left
                )

def convert_forecast_inflow(forecast_inflow_dir, reservoir_shpfile, reservoir_shpfile_column_dict,  final_out_forecast_inflow_dir, basedate):
    # Inflow
    reservoirs = gpd.read_file(reservoir_shpfile)
    reservoirs['Inflow_filename'] = reservoirs[reservoir_shpfile_column_dict['unique_identifier']].astype(str)

    forecast_inflow_paths = list(Path(forecast_inflow_dir).glob('*.csv'))
    final_out_forecast_inflow_dir.mkdir(exist_ok=True)

    for forecast_inflow_path in forecast_inflow_paths:
        res_name = os.path.splitext(os.path.split(forecast_inflow_path)[-1])[0]

        savepath = final_out_forecast_inflow_dir / forecast_inflow_path.name

        forecast_df = pd.read_csv(forecast_inflow_path, parse_dates=['date'])
        forecast_df['inflow (m3/d)'] = forecast_df['streamflow'] * (24*60*60)        # indicate units, convert from m3/s to m3/d
        forecast_df = forecast_df[['date', 'inflow (m3/d)']]

        print(f"Converting [Inflow]: {res_name}")
        forecast_df = forecast_df[forecast_df['date'] >= basedate]
        forecast_df.to_csv(savepath, index=False)
        print(forecast_df.tail())


def convert_forecast_evaporation(evap_dir, final_evap_dir):
    # Evaporation
    evap_paths = [os.path.join(evap_dir, f) for f in os.listdir(evap_dir) if f.endswith(".csv")]
    evap_dir.mkdir(exist_ok=True)

    for evap_path in evap_paths:
        res_name = os.path.splitext(os.path.split(evap_path)[-1])[0]
        savename = res_name

        savepath = os.path.join(final_evap_dir , f"{savename}.csv")

        df = pd.read_csv(evap_path)
        df = df[['time', 'OUT_EVAP']]
        df.rename({'time':'date', 'OUT_EVAP':'evaporation (mm)'}, axis=1, inplace=True)

        print(f"Converting [Evaporation]: {res_name}, {savepath}")
        df.to_csv(savepath, index=False)


def convert_forecast_outflow_states(outflow_dir, final_outflow_dir, final_dels_dir, final_sarea_dir):
    """
    Convert forecast outflow states (outflow, dels and sarea) to RAT's final format (step-14).

    Parameters:
    - outflow_dir (Path): input outflow directory (rat_outputs).
    - final_outflow_dir (Path): directory where final outflow files will be saved.
    - final_dels_dir (Path): directory where final storage change files will be saved.
    - final_sarea_dir (Path): directory where final sarea files will be saved.
    """

    # Outflow
    outflow_paths = list(sorted(outflow_dir.glob("*.csv")))

    for outflow_path in outflow_paths:
        res_name = outflow_path.name.split('.')[0]

        savename = res_name

        df = pd.read_csv(outflow_path, parse_dates=['date'])

        # outflow
        outflow_df = df.copy()
        outflow_savefp = final_outflow_dir / f"{savename}.csv"
        outflow_cols = list(filter(lambda x: 'outflow' in x, outflow_df.columns))
        col_names = []
        for outflow_col in outflow_cols:
            if outflow_col.startswith('ST'):
                outflow_case = outflow_col.split('_')[0] + ' ' + outflow_col.split('_')[-1]
            elif outflow_col.startswith('GO') or outflow_col.startswith('GC'):
                outflow_case = outflow_col.split('_')[0]
            
            converted_col_name = f'outflow (m3/d) [case: {outflow_case}]'
            col_names.append(converted_col_name)
            outflow_df.loc[outflow_df[outflow_col]<0, outflow_col] = 0
            outflow_df[converted_col_name] = outflow_df[outflow_col] # Units are already in m3/d
        outflow_df = outflow_df[['date', *col_names]]
        final_outflow_dir.mkdir(parents=True, exist_ok=True)
        outflow_df.to_csv(outflow_savefp, index=False)

        # dels
        dels_df = df.copy()
        dels_savefp = final_dels_dir / f"{savename}.csv"
        dels_cols = list(filter(lambda x: 'delS' in x, dels_df.columns))
        col_names = []
        for dels_col in dels_cols:
            if dels_col.startswith('ST'):
                dels_case = dels_col.split('_')[0] + ' ' + dels_col.split('_')[-1]
            elif dels_col.startswith('GO') or dels_col.startswith('GC') or dels_col.startswith('RC'):
                dels_case = dels_col.split('_')[0]
            
            converted_col_name = f'delS (m) [case: {dels_case}]'
            col_names.append(converted_col_name)
            dels_df[converted_col_name] = dels_df[dels_col]
        dels_df = dels_df[['date', *col_names]]
        final_dels_dir.mkdir(parents=True, exist_ok=True)
        dels_df.to_csv(dels_savefp, index=False)

        # sarea
        sarea_df = df.copy()
        sarea_savefp = final_sarea_dir / f"{savename}.csv"
        sarea_cols = list(filter(lambda x: 'sarea' in x, sarea_df.columns))
        col_names = []
        for sarea_col in sarea_cols:
            if sarea_col.startswith('ST'):
                sarea_case = sarea_col.split('_')[0] + ' ' + sarea_col.split('_')[-1]
            elif sarea_col.startswith('GO') or sarea_col.startswith('GC'):
                sarea_case = sarea_col.split('_')[0]
            
            converted_col_name = f'area (km2) [case: {sarea_case}]'
            col_names.append(converted_col_name)
            sarea_df[converted_col_name] = sarea_df[sarea_col]
        sarea_df = sarea_df[['date', *col_names]]
        final_sarea_dir.mkdir(parents=True, exist_ok=True)
        sarea_df.to_csv(sarea_savefp, index=False)


