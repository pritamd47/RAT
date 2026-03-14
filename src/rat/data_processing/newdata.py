import subprocess
import requests
import numpy as np
from datetime import datetime, timedelta
import calendar
import os
import time
import shutil
import tempfile
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import xarray as xr
from logging import getLogger
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dask.distributed import Semaphore
import dask
import pandas as pd

from rat.utils.logging import LOG_NAME, LOG_LEVEL, NOTIFICATION
from rat.utils.utils import create_directory
import configparser

log = getLogger(LOG_NAME)
log.setLevel(LOG_LEVEL)

def run_command(cmd,shell_bool=False):
    """Safely runs a command, and returns the returncode silently in case of no error. Otherwise,
    raises an Exception
    """
    res = subprocess.run(cmd, check=True, capture_output=True,shell=shell_bool)

    if res.returncode != 0:
        log.error(f"Error with return code {res.returncode}")
        raise Exception
    return res.returncode


def _determine_precip_version(date):
    """Determines which version of IMERG to download. Most preferred is IMERG Late, followed by
    IMERG Early. IMERG-Final has some issues. Currently only running using IMERG Early and Late
    """
    version = None
    # if date < (datetime.today() - timedelta(days=4*30)):
    #     version = "IMERG-FINAL"
    # elif date < (datetime.today() - timedelta(days=10)):
    #     version = "IMERG-LATE"
    # else:
    #     version = "IMERG-EARLY"
    if date < (datetime.today() - timedelta(days=2)):
        version = "IMERG-LATE"
    else:
        version = "IMERG-EARLY"
    return version

def _determine_precip_link_and_version(date):
    version = _determine_precip_version(date)
    if version == "IMERG-FINAL":
        link = f"ftp://arthurhou.pps.eosdis.nasa.gov/gpmdata/{date.strftime('%Y')}/{date.strftime('%m')}/{date.strftime('%d')}/gis/3B-DAY-GIS.MS.MRG.3IMERG.{date.strftime('%Y%m%d')}-S000000-E235959.0000.V06A.tif"
    elif version == "IMERG-LATE":
        # if date >= datetime(2024, 6, 1): # Version was changed from V06E to V07B
        link = f"https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/{date.strftime('%Y')}/{date.strftime('%m')}/3B-HHR-L.MS.MRG.3IMERG.{date.strftime('%Y%m%d')}-S233000-E235959.1410.V07B.1day.tif"
    else:
        link = f"https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/{date.strftime('%Y')}/{date.strftime('%m')}/3B-HHR-E.MS.MRG.3IMERG.{date.strftime('%Y%m%d')}-S233000-E235959.1410.V07B.1day.tif"
    return (link,version)

def _get_cmd_precip_download(outputpath,link,version,secrets):
    # Define the command (different for FINAL, same for EARLY and LATE)
    if version == "IMERG-FINAL":
        cmd = [
            "curl",
            '-o',
            outputpath,
            '--ssl-reqd',
            '-u',
            f'{secrets["imerg"]["username"]}:{secrets["imerg"]["pwd"]}',
            link
        ]
    else:
        cmd = [
            "wget",
            "-O",
            outputpath,
            "--user",
            f'{secrets["imerg"]["username"]}',
            '--password',
            f'{secrets["imerg"]["pwd"]}',
            link,
            '--no-proxy'
        ]
    return cmd

def download_precip(date, version, outputpath, secrets, interpolate=True):
    """
    Parameters:
        date: datetime object that defines the date for which data is required
        version: which version of data to download - IMERG-LATE or IMERG-EARLY
        outputpath: path where the data should be downloaded
    Returns:
        STATUS: Can have the following values:
            STARTED: Download has started
            SUCCESS: Download was successful
            FAILED: Download failed
            INTERPOLATION_STARTED: Download failed, but interpolation was started
            INTERPOLATED: Download failed, but interpolation was successful
            INTERPOLATION_FAILED: Download failed, and interpolation failed
            NOT_FOUND: Download link not found. Data can be interpolated if `interpolate` is set to True
    =======
    TODO: Add ability to select either CHIRPS or IMERG data
    """
    STATUS = "STARTED"
    if not(os.path.exists(outputpath)):
        link,version_ = _determine_precip_link_and_version(date)
        response = requests.head(link,auth=(secrets["imerg"]["username"], secrets["imerg"]["pwd"]))
        
        if response.status_code == 200 :  
        # Define the command (different for FINAL, same for EARLY and LATE)
            cmd = _get_cmd_precip_download(outputpath,link,version,secrets)
            log.debug("Downloading precipitation 1-day file: %s (%s)", date.strftime('%Y-%m-%d'), version)
            exit_code = run_command(cmd)
            STATUS = "SUCCESS" if exit_code == 0 else "FAILED"
        else:
            if(interpolate):
                STATUS="INTERPOLATION_STARTED"
                log.debug('Link for 1day file does not exist. Trying to interpolate data using previous and next date.')
                pre_date = date - timedelta(days=1)
                pre_link, pre_version = _determine_precip_link_and_version(pre_date)
                pre_response = requests.head(pre_link,auth=(secrets["imerg"]["username"], secrets["imerg"]["pwd"]))

                post_date = date + timedelta(days=1)
                post_link, post_version = _determine_precip_link_and_version(post_date)
                post_response = requests.head(post_link,auth=(secrets["imerg"]["username"], secrets["imerg"]["pwd"]))
                if(pre_response.status_code==200 and post_response.status_code==200):
                    pre_outputpath = ('').join(outputpath.split('.')[:-1])+'_pretemp.tif'
                    pre_cmd = _get_cmd_precip_download(pre_outputpath,pre_link,pre_version,secrets)
                    log.debug("Downloading pre-precipitation 1-day file: %s (%s)", date.strftime('%Y-%m-%d'), version)
                    exit_code1 = run_command(pre_cmd)

                    post_outputpath = ('').join(outputpath.split('.')[:-1])+'_posttemp.tif'
                    post_cmd = _get_cmd_precip_download(post_outputpath,post_link,post_version,secrets)
                    log.debug("Downloading post-precipitation 1-day file: %s (%s)", date.strftime('%Y-%m-%d'), version)
                    exit_code2 = run_command(post_cmd)

                    STATUS = "INTERPOLATED" if (exit_code1 == 0) & (exit_code2 == 0) else "INTERPOLATION_FAILED"

                    ## taking mean of pre and post date
                    pre_precip = rxr.open_rasterio(pre_outputpath)
                    post_precip = rxr.open_rasterio(post_outputpath)
                    precip = (pre_precip+post_precip)/2 
                    precip.rio.to_raster(outputpath)

                    ## Removing pre and post date files after
                    if os.path.isfile(pre_outputpath):
                        os.remove(pre_outputpath)
                    if os.path.isfile(post_outputpath):
                        os.remove(post_outputpath)
                    log.debug(f"Precipitation file interpolated using previous and next date : {date.strftime('%Y-%m-%d')}")

                elif (pre_response.status_code==200):
                    pre_cmd = _get_cmd_precip_download(outputpath,pre_link,pre_version,secrets)
                    log.debug("Being Replaced by pre-precipitation 1-day file: %s (%s)", date.strftime('%Y-%m-%d'), version)
                    exit_code = run_command(pre_cmd)
                    STATUS = "INTERPOLATED" if exit_code == 0 else "INTERPOLATION_FAILED"
                
                elif (post_response.status_code==200):
                    post_cmd = _get_cmd_precip_download(outputpath,post_link,post_version,secrets)
                    log.debug("Being Replaced by post-precipitation 1-day file: %s (%s)", date.strftime('%Y-%m-%d'), version)
                    exit_code = run_command(post_cmd)
                    STATUS = "INTERPOLATED" if exit_code == 0 else "INTERPOLATION_FAILED"
                else:
                    log.warning(f"Precipitation file cannnot be interpolated from pre/post date. Skipping downloading: {date.strftime('%Y-%m-%d')}")
                    STATUS = "INTERPOLATION_FAILED"
            else:
                log.warning(f"Precipitation download link not found. Skipping downloading: {date.strftime('%Y-%m-%d')}")
                STATUS = "NOT_FOUND"
    else:
        STATUS = "SKIPPED"
        log.debug(f"Precipitation file already exits: {date.strftime('%Y-%m-%d')}")

    return STATUS

def download_tmax(year, outputpath):
    """
    Parameters:
        year: year for which data is to be downloaded, as a string
        outputpath: path where the data has to be saved
    """
    ## New data will keep on coming in the year going on i.e. recent year
    if (not(os.path.exists(outputpath))):
        cmd = [
            'wget', 
            '-O', 
            f'{outputpath}', 
            f'https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.{year}.nc'
        ]
        log.debug("Downloading tmax: %s", year)
        return run_command(cmd)
    else:
        log.info("File already exists tmax: %s", year)
        # Checking days in year and in file. 
        days_in_year=366 if calendar.isleap(int(year)) else 365
        with xr.open_dataset(outputpath, engine='netcdf4') as nc_data:
            days_in_file = len(nc_data['time'])
        if(days_in_file<days_in_year):
            log.info(f"The file has only data for {days_in_file} days which is less than the days in the year ({days_in_year}).So, updating the tmax file: {year}")
            cmd = [
                    'wget', 
                    '-O', 
                    f'{outputpath}', 
                    f'https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.{year}.nc'
                ]
            return run_command(cmd)
        else:
            log.info(f"The tmax file has complete data for {days_in_file} days of the year. No need to download for {year}.")
                

def download_tmin(year, outputpath):
    """
    Parameters:
        year: year for which data is to be downloaded, as a string
        outputpath: path where the data has to be saved
    """
    ## New data will keep on coming in the year going on i.e. recent year
    if (not(os.path.exists(outputpath))):
        cmd = [
            'wget', 
            '-O', 
            f'{outputpath}', 
            f'https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.{year}.nc'
        ]
        log.debug("Downloading tmin: %s", year)
        return run_command(cmd)
    else:
        log.info("File already exists tmin: %s", year)
        # Checking days in year and in file. 
        days_in_year=366 if calendar.isleap(int(year)) else 365
        with xr.open_dataset(outputpath) as nc_data:
            days_in_file = len(nc_data['time'])
        if(days_in_file<days_in_year):
            log.info(f"The file has only data for {days_in_file} days which is less than the days in the year ({days_in_year}).So, updating the tmin file: {year}")
            cmd = [
                    'wget', 
                    '-O', 
                    f'{outputpath}', 
                    f'https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.{year}.nc'
                ]
            return run_command(cmd)
        else:
            log.info(f"The tmin file has complete data for {days_in_file} days of the year. No need to download for {year}.")

def download_uwnd(year, outputpath):
    """
    Parameters:
        year: year for which data is to be downloaded, as a string
        outputpath: path where the data has to be saved
    """
    ## New data will keep on coming in the year going on i.e. recent year
    if (not(os.path.exists(outputpath))):
        cmd = ['wget', '-O', outputpath, f'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/uwnd.10m.gauss.{year}.nc']
        log.debug("Downloading uwnd: %s", year)
        return run_command(cmd)
    else:
        log.info("File already exists uwnd: %s", year)
        # Checking days in year and in file.
        days_in_year=366 if calendar.isleap(int(year)) else 365
        with xr.open_dataset(outputpath) as nc_data:
            days_in_file = int(len(nc_data['time'])/4)  # daily 4 times data at 6hr frequency
        if(days_in_file<days_in_year):
            log.info(f"The file has only data for {days_in_file} days which is less than the days in the year ({days_in_year}).So, updating the uwnd file: {year}")
            cmd = ['wget', '-O', outputpath, f'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/uwnd.10m.gauss.{year}.nc']
            return run_command(cmd)
        else:
            log.info(f"The uwnd file has complete data for {days_in_file} days of the year. No need to download for {year}.")

def download_vwnd(year, outputpath):
    """
    Parameters:
        year: year for which data is to be downloaded, as a string
        outputpath: path where the data has to be saved
    """
    ## New data will keep on coming in the year going on i.e. recent year
    if (not(os.path.exists(outputpath))):
        cmd = ['wget', '-O', outputpath, f'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/vwnd.10m.gauss.{year}.nc']
        log.debug("Downloading vwnd: %s", year)
        return run_command(cmd)
    else:
        log.info("File already exists vwnd: %s", year)
        # Checking days in year and in file.
        days_in_year=366 if calendar.isleap(int(year)) else 365
        with xr.open_dataset(outputpath) as nc_data:
            days_in_file = int(len(nc_data['time'])/4) # daily 4 times data at 6hr frequency
        if(days_in_file<days_in_year):
            log.info(f"The file has only data for {days_in_file} days which is less than the days in the year ({days_in_year}).So, updating the vwnd file: {year}")
            cmd = ['wget', '-O', outputpath, f'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface_gauss/vwnd.10m.gauss.{year}.nc']
            return run_command(cmd)
        else:
            log.info(f"The vwnd file has complete data for {days_in_file} days of the year. No need to download for {year}.")

def download_data(begin, end, datadir, secrets):
    """Downloads the data between dates defined by begin and end

    Parameters:
        begin: Data will start downloading from this date, including this date
        end: Data will be downloaded until this date, including this date
        datedir: Base directory for downloading data
    """

    # Obtain list of dates to be downloaded
    # required_dates = [begin+timedelta(days=n) for n in range((end-begin).days)]
    required_dates = pd.date_range(begin, end)
    required_years = list(set([d.strftime("%Y") for d in required_dates]))
    # Download Precipitation
    log.debug("Downloading Precipitation")

    precip_statuses = {
        "Date": [],
        "Status": []
    }

    sem = Semaphore(max_leases=4, name="IMERG-Downloader")
    def download_precip_with_semaphore(date, version, outputpath, secrets, sem):
        with sem:
            STATUS = download_precip(date, version, outputpath, secrets)
            return date, STATUS
    
    futures = []
    for date in required_dates:
        data_version = _determine_precip_version(date)
        outputpath = os.path.join(datadir, "precipitation", f"{date.strftime('%Y-%m-%d')}_IMERG.tif")
        # if not os.path.isfile(outputpath):
        future = dask.delayed(download_precip_with_semaphore)(date, data_version, outputpath, secrets, sem)
        futures.append(future)

    results = dask.compute(*futures) # downloading precipitation files first
    precip_statuses["Date"] = [r[0] for r in results]
    precip_statuses["Status"] = [r[1] for r in results]
    precip_statuses = pd.DataFrame(precip_statuses)
    precip_statuses.sort_values(by="Date", inplace=True)
    log.debug("Precipitation download statuses:\n%s", precip_statuses.to_string(index=False, col_space=25))

    futures = []

    # Download other forcing data
    log.debug("Downloading TMax, TMin, UWnd, and VWnd")
    for year in required_years:
        futures.append(dask.delayed(download_tmax)(year, os.path.join(datadir, "tmax", year+'.nc')))
        futures.append(dask.delayed(download_tmin)(year, os.path.join(datadir, "tmin", year+'.nc')))
        futures.append(dask.delayed(download_uwnd)(year, os.path.join(datadir, "uwnd", year+'.nc')))
        futures.append(dask.delayed(download_vwnd)(year, os.path.join(datadir, "vwnd", year+'.nc')))
    
    results = dask.compute(*futures)

def _clip_scale_save_precip(basin_bounds, srcpath, dstpath):
    """Clip IMERG tif to basin_bounds, resample to 0.0625 deg, scale by 0.1, write as GeoTiff."""
    NODATA = -9999.0
    RES = 0.0625
    minx, miny, maxx, maxy = basin_bounds
    nx = round((maxx - minx) / RES)
    ny = round((maxy - miny) / RES)
    da = rxr.open_rasterio(srcpath, masked=True)
    da = da * 0.1
    da = da.rio.reproject(
        da.rio.crs,
        shape=(ny, nx),
        transform=rasterio.transform.from_origin(minx, maxy, RES, RES),
        resampling=Resampling.nearest,
        nodata=NODATA
    )
    da = da.assign_coords(x=np.array(da.x.data).round(5), y=np.array(da.y.data).round(5))
    da = da.round(4)
    da.rio.to_raster(dstpath, driver='GTiff')


def process_precip(basin_bounds, srcpath, dstpath, secrets=None, temp_datadir=None, itry=0):
    """For any IMERG Precipitation file located at `srcpath` is clipped, scaled and converted to
    ASCII grid file and saved at `dstpath`.
    """
    src_fn = Path(srcpath)
    date = pd.to_datetime(src_fn.stem.split('_')[0])
    if temp_datadir is not None and not os.path.isdir(temp_datadir):
        log.warning(f"ERROR: {temp_datadir} directory doesn't exist")
        return date, 'Precipitaion', 'FAILED'

    if not os.path.exists(dstpath):
        log.debug("Processing Precipitation file: %s", srcpath)
        STATUS = 'STARTED'
        try:
            _clip_scale_save_precip(basin_bounds, srcpath, dstpath)
            STATUS = 'SUCCESS'
        except Exception as e:
            log.error(f"Processing error in {date}: {e}")
            # delete old precipitation file and redownload, retry once
            try:
                src_fn.unlink(missing_ok=True)
                version = _determine_precip_version(date)
                download_precip(date, version, srcpath, secrets)
                _clip_scale_save_precip(basin_bounds, srcpath, dstpath)
                STATUS = 'SUCCESS'
            except Exception:
                log.warning('Processing failed. Downloaded file might be corrupted.')
                STATUS = 'FAILED'
    else:
        STATUS = 'SKIPPED'
        log.debug(f"Processing Precipitation file exist: {srcpath}")
    return date, 'Precipitaion', STATUS

def _clip_save_nc(basin_bounds, srcpath, date, var, dstpath):
    """Extract a date slice from a NetCDF file, wrap longitudes to -180/180,
    resample to basin_bounds at 0.0625 deg, and write as GeoTiff."""
    NODATA = -9999.0
    RES = 0.0625

    ds = xr.open_dataset(srcpath)
    # Select time slice by day-of-year index (matches gdal_translate -b behaviour)
    da = ds[var].isel(time=date.dayofyear - 1)

    # Wrap longitudes from 0-360 to -180-180
    da = da.assign_coords(lon=((da.lon + 180) % 360 - 180)).sortby('lon')

    da = da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    da = da.rio.write_crs('EPSG:4326')

    # Reproject to an output grid defined exactly by basin_bounds and RES.
    # Mirrors gdalwarp -te -tr: all variables land on the same pixel grid
    # regardless of source grid alignment (regular or Gaussian).
    minx, miny, maxx, maxy = basin_bounds
    nx = round((maxx - minx) / RES)
    ny = round((maxy - miny) / RES)
    da = da.rio.reproject(
        'EPSG:4326',
        shape=(ny, nx),
        transform=rasterio.transform.from_origin(minx, maxy, RES, RES),
        resampling=Resampling.nearest,
        nodata=NODATA
    )
    da = da.assign_coords(x=np.array(da.x.data).round(5), y=np.array(da.y.data).round(5))
    da = da.round(4)
    da.attrs.pop('_FillValue', None)
    da.encoding.pop('_FillValue', None)
    da.rio.write_nodata(NODATA, inplace=True)
    da.rio.to_raster(dstpath, driver='GTiff')


def process_nc(basin_bounds, date, srcpath, dstpath, temp_datadir=None, var='---'):
    """For TMax, TMin, UWnd and VWnd, the processing steps are same, and can be performed using
    this function.

    Parameters:
        date: Datetime object of the date of data
        srcpath: path of the nc file
        dstpath: path where the final ascii file will be saved
        temp_datadir: directory where the temporary data will be stored
    """
    os.environ['PROJ_DATA'] = str(Path(rasterio.__file__).parent / 'proj_data')
    if temp_datadir is not None and not os.path.isdir(temp_datadir):
        log.warning(f"ERROR: {temp_datadir} directory doesn't exist")
        return date, var, 'FAILED'

    if not os.path.exists(dstpath):
        log.debug("Processing NC file: %s for date %s", srcpath, date.strftime('%Y-%m-%d'))
        STATUS = 'STARTED'
        try:
            _clip_save_nc(basin_bounds, srcpath, date, var, dstpath)
            STATUS = 'SUCCESS'
        except Exception as e:
            log.error(f"Processing error for {var} on {date.strftime('%Y-%m-%d')}: {e}")
            STATUS = 'FAILED'
    else:
        STATUS = 'SKIPPED'
        log.debug(f"Processing NC file exists: {srcpath} for date {date.strftime('%Y-%m-%d')}")
    return date, var, STATUS

def process_data(basin_bounds,raw_datadir, processed_datadir, begin, end, secrets, temp_datadir):
    if not os.path.isdir(temp_datadir):
        os.makedirs(temp_datadir)

    #### Process precipitation ####
    log.debug("Processing Precipitation")
    raw_datadir_precip = os.path.join(raw_datadir, "precipitation")
    processed_datadir_precip = os.path.join(processed_datadir, "precipitation")

    # with tqdm(os.listdir(raw_datadir_precip)) as pbar:
    ds = pd.date_range(begin, end)

    futures = []
    processed_statuses = []

    for srcname in os.listdir(raw_datadir_precip):
        if datetime.strptime(srcname.split(os.sep)[-1].split("_")[0], "%Y-%m-%d") in ds:
            srcpath = os.path.join(raw_datadir_precip, srcname)
            dstpath = os.path.join(processed_datadir_precip, srcname)

            # pbar.set_description(f"Precipitation: {srcname.split('_')[0]}")
            future = dask.delayed(process_precip)(basin_bounds, srcpath, dstpath, secrets, temp_datadir)
            futures.append(future)

    #### Process NC files ####
    # required_dates = [begin+timedelta(days=n) for n in range((end-begin).days)]
    required_dates = pd.date_range(begin, end)
    #### Process TMAX ####
    log.debug("Processing TMAX")
    raw_datadir_tmax = os.path.join(raw_datadir, "tmax")
    processed_datadir_tmax = os.path.join(processed_datadir, "tmax")

    for date in required_dates:
        srcpath = os.path.join(raw_datadir_tmax, date.strftime('%Y')+'.nc')
        dstpath = os.path.join(processed_datadir_tmax, f"{date.strftime('%Y-%m-%d')}_TMAX.tif")

        future = dask.delayed(process_nc)(basin_bounds,date, srcpath, dstpath, temp_datadir, "tmax")
        futures.append(future)

    #### Process TMin ####
    log.debug("Processing TMIN")
    raw_datadir_tmin = os.path.join(raw_datadir, "tmin")
    processed_datadir_tmin = os.path.join(processed_datadir, "tmin")

    for date in required_dates:
        srcpath = os.path.join(raw_datadir_tmin, date.strftime('%Y')+'.nc')
        dstpath = os.path.join(processed_datadir_tmin, f"{date.strftime('%Y-%m-%d')}_TMIN.tif")

        future = dask.delayed(process_nc)(basin_bounds,date, srcpath, dstpath, temp_datadir, "tmin")
        futures.append(future)

    #### Process UWND ####
    log.debug("Processing UWND")
    raw_datadir_uwnd = os.path.join(raw_datadir, "uwnd")
    daily_datadir_uwnd = os.path.join(raw_datadir, "uwnd_daily")
    processed_datadir_uwnd = os.path.join(processed_datadir, "uwnd")

    uwnd_files = [os.path.join(raw_datadir_uwnd, f) for f in os.listdir(raw_datadir_uwnd)]

    for uwnd_f in uwnd_files:
        xr.open_dataset(uwnd_f).resample(time='1D').mean().to_netcdf(os.path.join(daily_datadir_uwnd, uwnd_f.split(os.sep)[-1]))
        # xr.open_dataset(vwnd_f).resample(time='1D').mean().to_netcdf(os.path.join(vwnd_outdir, vwnd_f.split(os.sep)[-1]))

    for date in required_dates:
        srcpath = os.path.join(daily_datadir_uwnd, date.strftime('%Y')+'.nc')
        dstpath = os.path.join(processed_datadir_uwnd, f"{date.strftime('%Y-%m-%d')}_UWND.tif")

        future = dask.delayed(process_nc)(basin_bounds,date, srcpath, dstpath, temp_datadir, "uwnd")
        futures.append(future)

    #### Process VWND ####
    log.debug("Processing VWND")
    raw_datadir_vwnd = os.path.join(raw_datadir, "vwnd")
    daily_datadir_vwnd = os.path.join(raw_datadir, "vwnd_daily")
    processed_datadir_vwnd = os.path.join(processed_datadir, "vwnd")

    vwnd_files = [os.path.join(raw_datadir_vwnd, f) for f in os.listdir(raw_datadir_vwnd)]

    for vwnd_f in vwnd_files:
        xr.open_dataset(vwnd_f).resample(time='1D').mean().to_netcdf(os.path.join(daily_datadir_vwnd, vwnd_f.split(os.sep)[-1]))

    for date in required_dates:
        srcpath = os.path.join(daily_datadir_vwnd, date.strftime('%Y')+'.nc')
        dstpath = os.path.join(processed_datadir_vwnd, f"{date.strftime('%Y-%m-%d')}_VWND.tif")

        future = dask.delayed(process_nc)(basin_bounds,date, srcpath, dstpath, temp_datadir, "vwnd")
        futures.append(future)

    processed_statuses = dask.compute(*futures)
    processed_statuses = pd.DataFrame(processed_statuses, columns=['Date','Variable','Status'])
    processed_statuses.sort_values(by=["Date","Variable"], inplace=True)
    log.debug("Files processed statuses:\n%s", processed_statuses.to_string(index=False, col_space=25))

def get_newdata(basin_name,basin_bounds,data_dir, basin_data_dir,startdate, enddate, secrets_file, download=True, process=True):
    raw_datadir = os.path.join(data_dir,"raw",'')
    processed_datadir = os.path.join(basin_data_dir,"pre_processing","processed",'')
    temp_datadir = os.path.join(basin_data_dir,"pre_processing","temp",'')
    
    ####Creating the required directories####

    dir_paths=[raw_datadir,processed_datadir]
    #Creating data directory if does not exist
    create_directory(basin_data_dir)

    #Creating temp directory if does not exist
    create_directory(temp_datadir)
    data_vars_folder_names=['precipitation','tmax','tmin','uwnd','vwnd']
    raw_data_vars_folder_names=['uwnd_daily','vwnd_daily']
    
    for dir_path in dir_paths:
        
        # Making directories with each name in data_vars_folder_names in dir_paths
        for data_var_name in data_vars_folder_names:
            temp_dir_path_var=os.path.join(dir_path,data_var_name,'')
            create_directory(temp_dir_path_var)

        # Making directories with each name in raw_data_vars_folder_names in raw_datadir
        if(dir_path==raw_datadir):
            for data_var_name in raw_data_vars_folder_names:
                temp_dir_path_var=os.path.join(dir_path,data_var_name,'')
                create_directory(temp_dir_path_var)
    
    secrets = configparser.ConfigParser()
    secrets.read(secrets_file)  # assuming there's a secret ini file with user/pwd

    enddate = enddate

    startdate_str = startdate.strftime("%Y-%m-%d")
    enddate_str = enddate.strftime("%Y-%m-%d")

    log.log(NOTIFICATION, "Started Downloading and Processing data from %s -> %s", startdate_str, enddate_str)
    log.debug("Raw data directory: %s", raw_datadir)
    log.debug("Processed data directory: %s", processed_datadir)

    #### DATA DOWNLOADING ####
    if download:
        download_data(startdate, enddate, raw_datadir, secrets)

    #### DATA PROCESSING ####
    if process:
        process_data(basin_bounds,raw_datadir, processed_datadir, startdate, enddate, secrets, temp_datadir)