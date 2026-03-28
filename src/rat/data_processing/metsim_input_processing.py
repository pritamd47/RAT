import xarray as xr
import datetime
import os
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import rasterio as rio
import pandas as pd
import shutil
import dask
from pathlib import Path
from dateutil.relativedelta import relativedelta

from rat.utils.logging import LOG_NAME, LOG_LEVEL, NOTIFICATION

log = getLogger(LOG_NAME)
log.setLevel(LOG_LEVEL)

_NETCDF_ENCODING = {
    'precip': {'dtype': 'float32'},
    'tmax':   {'dtype': 'float32'},
    'tmin':   {'dtype': 'float32'},
    'wind':   {'dtype': 'float32'},
}

_NETCDF_ENCODING_WITH_EXTENT = {**_NETCDF_ENCODING, 'extent': {'dtype': 'float32'}}

class CombinedNC:
    def __init__(self, start, end, datadir, basingridpath, outputdir, use_previous, forecast_dir=None, forecast_basedate=None, climatological_data=None, z_lim=3, low_latency_dir=None, n_workers=1):
        """
        Parameters:
            start: Start date in YYYY-MM-DD format
            :
            :
            datadir: Directory path of ascii format files
        """
        log.debug("Started combining forcing data for MetSim processing")
        self._start = start
        self._end = end
        self._total_days = (self._end - self._start).days
        self._datadir = datadir
        self._outputpath = outputdir
        self._use_previous = use_previous
        self._n_workers = n_workers

        self._rast = rio.open(basingridpath)
        self._ar = self._rast.read(1, masked=True)
        self._gridvalue = self._ar.flatten()

        self._longitudes1d, self._latitudes1d = self._get_lat_long_1d()
        self._latitudes, self._longitudes = self._get_lat_long_meshgrid()

        if forecast_dir:
            log.debug("Combining forcing data for forecasting mode.")
            self.read_forecast(forecast_dir, forecast_basedate)
            self._write()
        elif low_latency_dir:
            log.debug("Combining forcing data for low latency mode.")
            self.read_low_latency(low_latency_dir)
            self._write()
        else:
            self._read_and_write_in_chunks()
            # self._read()
            # self._write()

        # If climatological data is passed, then run the climatological data correction
        if climatological_data:
            self._climatological_data = climatological_data
            self._z_lim = z_lim
            log.debug(f"Running climatological corection of satellite precipitation")
            self.satellite_precip_correction()

    def satellite_precip_correction(self):
        # UPDATE (2023-06-02): Seems like this function isn't able to filter out some very high precipitation values (Jan 6, Jan 7 2022).
        # from https://github.com/pritamd47/2022_05_04-extreme_precip/blob/1baa1319513abbecfb89f7a7269a1214b50cdca0/notebooks/Extreme-Precip.ipynb
        daily_precip = xr.open_dataset(self._climatological_data)
        log_precip = daily_precip.copy()
        # Take the natural log to convert to normal from log-normal distribution
        log_precip['tp'] = np.log(log_precip['tp'], where=log_precip['tp']>0)

        weekly_log_precip_mean = log_precip.groupby(log_precip['time'].dt.isocalendar().week).mean()
        weekly_log_precip_std = log_precip.groupby(log_precip['time'].dt.isocalendar().week).std()

        forcings = xr.open_dataset(self._outputpath)
        forcings_precip = forcings[['precip']]

        log_forcings_precip = forcings_precip.copy()
        log_forcings_precip['precip'] = np.log(forcings_precip['precip'], where=forcings_precip['precip']>0)

        # Align both the datasets, so that the final z-scores, means and std devs can be calculated and compared easily.
        weekly_log_precip_mean_sampled = weekly_log_precip_mean.sel(longitude=log_forcings_precip.lon, latitude=log_forcings_precip.lat, method='nearest')
        weekly_log_precip_std_sampled = weekly_log_precip_std.sel(longitude=log_forcings_precip.lon, latitude=log_forcings_precip.lat, method='nearest')

        log_forcings_precip['weekly_mean_precip'] = weekly_log_precip_mean_sampled['tp']
        log_forcings_precip['weekly_std_precip'] = weekly_log_precip_std_sampled['tp']

        # build daily clim precip
        times = []
        clim_precip_mean = []
        clim_precip_std = []

        for t in log_forcings_precip.time:
            time = pd.to_datetime(t.values)
            times.append(time)
            # clim_precip_std.append(log_forcings_precip['weekly_std_precip'].sel(week=time.weekofyear))
            clim_precip_mean.append(log_forcings_precip['weekly_mean_precip'].sel(week=time.weekofyear).values)
            clim_precip_std.append(log_forcings_precip['weekly_std_precip'].sel(week=time.weekofyear).values)

        mean = xr.DataArray(np.array(clim_precip_mean), coords=({'time': times, 'lon': log_forcings_precip.lon, 'lat': log_forcings_precip.lat}), dims=["time", "lat", "lon"])
        std = xr.DataArray(np.array(clim_precip_std), coords=({'time': times, 'lon': log_forcings_precip.lon, 'lat': log_forcings_precip.lat}), dims=["time", "lat", "lon"])

        log_forcings_precip['climatological_precip_mean'] = mean
        log_forcings_precip['climatological_precip_std'] = std

        # calculate z-scores
        z_scores = (log_forcings_precip['precip'] - log_forcings_precip['climatological_precip_mean'])/log_forcings_precip['climatological_precip_std']

        high_precip = log_forcings_precip['climatological_precip_mean']
        precip_extremes_handled = np.where((z_scores.data>self._z_lim)|(z_scores.data<-self._z_lim), high_precip.data, log_forcings_precip['precip'].data)
        log_forcings_precip['precip'].data = precip_extremes_handled
        forcings['precip'].data = np.exp(log_forcings_precip['precip'].data)  # Convert to precipitation
        forcings.attrs['precip_filtering'] = f">{self._z_lim}sigma | <-{self._z_lim}sigma"

        forcings.to_netcdf(self._outputpath.replace(".nc", "_precip_handled.nc"))
        forcings.close()
        daily_precip.close()
        shutil.move(self._outputpath.replace(".nc", "_precip_handled.nc"), self._outputpath)

    def _get_lat_long_1d(self):
        x_res, y_res = self._rast.res
        r_lon_0, r_lat_0 = self._rast.xy(self._rast.height-1, 0)
        longitudes1d = (np.arange(0, self._rast.shape[1])*x_res + r_lon_0).round(5)
        latitudes1d = (np.arange(0, self._rast.shape[0])*y_res + r_lat_0).round(5)

        return (longitudes1d, latitudes1d)

    def _get_lat_long_meshgrid(self):
        xy = np.meshgrid(self._latitudes1d, self._longitudes1d, indexing='ij')
        longitudes, latitudes = xy[1].flatten(), xy[0].flatten()
        # Need to flip latutudes on horizontal axis
        latitudes = np.flip(latitudes, 0)

        return latitudes, longitudes

    def read_forecast(self, forecast_dir, basedate):
        forecast_dir = Path(forecast_dir)
        basedate = pd.to_datetime(basedate)

        # define data arrays
        self.precips = np.zeros((15, self._rast.height, self._rast.width))
        self.tmaxes = np.zeros((15, self._rast.height, self._rast.width))
        self.tmins = np.zeros((15, self._rast.height, self._rast.width))
        self.winds = np.zeros((15, self._rast.height, self._rast.width))

        self.dates = pd.date_range(basedate + datetime.timedelta(days=1), basedate + datetime.timedelta(days=15))

        for day, date in enumerate(self.dates):
            fileDate = date
            reqDate = fileDate.strftime("%Y-%m-%d")
            # pbar.set_description(reqDate)

            precipfilepath = forecast_dir / f'gefs-chirps/processed' / f'{basedate:%Y%m%d}' / f'{date:%Y%m%d}.asc'
            precipitation = rio.open(precipfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)#.flatten()[self.gridvalue==0.0]

            #Reading Maximum Temperature ASCII file contents
            tmaxfilepath = forecast_dir / f'gfs/processed/{basedate:%Y%m%d}/tmax/{date:%Y%m%d}.asc'
            tmax = rio.open(tmaxfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)#.flatten()[self.gridvalue==0.0]

            #Reading Minimum Temperature ASCII file contents
            tminfilepath = forecast_dir / f'gfs/processed/{basedate:%Y%m%d}/tmin/{date:%Y%m%d}.asc'
            tmin = rio.open(tminfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)#.flatten()[self.gridvalue==0.0]

            #Reading Average Wind Speed ASCII file contents
            uwndfilepath = forecast_dir / f'gfs/processed/{basedate:%Y%m%d}/uwnd/{date:%Y%m%d}.asc'
            uwnd = rio.open(uwndfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)

            # #Reading Average Wind Speed ASCII file contents
            vwndfilepath = forecast_dir / f'gfs/processed/{basedate:%Y%m%d}/vwnd/{date:%Y%m%d}.asc'
            vwnd = rio.open(vwndfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)
            wind = (0.75*np.sqrt(uwnd**2 + vwnd**2))#.flatten()[self.gridvalue==0.0]

            # self.dates.append(fileDate)
            self.precips[day, :, :] = precipitation
            self.tmaxes[day, :, :] = tmax
            self.tmins[day, :, :] = tmin
            self.winds[day, :, :] = wind
            # pbar.update(1)

    def read_low_latency(self, low_latency_dir):
        low_latency_dir = Path(low_latency_dir)
        start = self._start
        end = self._end

        # define data arrays
        self.precips = np.zeros((self._total_days+1, self._rast.height, self._rast.width))
        self.tmaxes = np.zeros((self._total_days+1, self._rast.height, self._rast.width))
        self.tmins = np.zeros((self._total_days+1, self._rast.height, self._rast.width))
        self.winds = np.zeros((self._total_days+1, self._rast.height, self._rast.width))

        self.dates = pd.date_range(start, end)

        for day, date in enumerate(self.dates):
            fileDate = date
            reqDate = fileDate.strftime("%Y-%m-%d")
            log.debug("Combining data: %s", reqDate)
            # pbar.set_description(reqDate)

            precipfilepath = low_latency_dir / f'gefs-chirps/processed' / f'{date:%Y%m%d}' / f'{date:%Y%m%d}.asc'
            precipitation = rio.open(precipfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)#.flatten()[self.gridvalue==0.0]

            #Reading Maximum Temperature ASCII file contents
            tmaxfilepath = low_latency_dir / f'gfs/processed/{date:%Y%m%d}/tmax/{date:%Y%m%d}.asc'
            tmax = rio.open(tmaxfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)#.flatten()[self.gridvalue==0.0]

            #Reading Minimum Temperature ASCII file contents
            tminfilepath = low_latency_dir / f'gfs/processed/{date:%Y%m%d}/tmin/{date:%Y%m%d}.asc'
            tmin = rio.open(tminfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)#.flatten()[self.gridvalue==0.0]

            #Reading Average Wind Speed ASCII file contents
            uwndfilepath = low_latency_dir / f'gfs/processed/{date:%Y%m%d}/uwnd/{date:%Y%m%d}.asc'
            uwnd = rio.open(uwndfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)

            # #Reading Average Wind Speed ASCII file contents
            vwndfilepath = low_latency_dir / f'gfs/processed/{date:%Y%m%d}/vwnd/{date:%Y%m%d}.asc'
            vwnd = rio.open(vwndfilepath).read(1, masked=True).astype(np.float32).filled(np.nan)
            wind = (0.75*np.sqrt(uwnd**2 + vwnd**2))#.flatten()[self.gridvalue==0.0]

            # self.dates.append(fileDate)
            self.precips[day, :, :] = precipitation
            self.tmaxes[day, :, :] = tmax
            self.tmins[day, :, :] = tmin
            self.winds[day, :, :] = wind
            # pbar.update(1)

    # Imputes missing data by interpolation in the order of dimensions time, lon, lat.
    def _impute_basin_missing_data(self, combined_data):
        combine_nomiss_data = combined_data
        try:
            #Interpolation using time dimension - if unsuccesful values will still be NaN
            combine_nomiss_data = combine_nomiss_data.interpolate_na(dim="time", method="linear", fill_value="extrapolate", limit = 30)
            #Interpolation using lon dimension - if unsuccesful values will still be NaN
            combine_nomiss_data = combine_nomiss_data.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")
            #Interpolation using lat dimension - if unsuccesful values will still be NaN
            combine_nomiss_data = combine_nomiss_data.interpolate_na(dim="lat", method="linear", fill_value="extrapolate", use_coordinate=False)
        except:
            print("No inter or extra polation is possible.")
        #Restoring original values outside basin extent. This ensures that ocean tiles remain to be NaN/-9999
        combine_nomiss_data = combine_nomiss_data.where(combined_data['extent']==1,combined_data)
        return combine_nomiss_data

    def _min_max_temperature_check(self, combined_data):
        # Create a mask where tmax is less than tmin
        mask = combined_data['tmax'] < combined_data['tmin']

        # Use the mask to interchange tmax and tmin values efficiently
        # Apply the swap using xarray's where function to avoid temporary arrays

        tmax_swapped = xr.where(mask, combined_data['tmin'], combined_data['tmax'])
        tmin_swapped = xr.where(mask, combined_data['tmax'], combined_data['tmin'])

        # Assign the swapped arrays back to the original dataset
        combined_data['tmax'] = tmax_swapped
        combined_data['tmin'] = tmin_swapped

        return combined_data

    def _write(self):
        precip_da = xr.DataArray(
            data = self.precips,
            coords=[self.dates, np.flip(self._latitudes1d), self._longitudes1d],
            dims=['time', 'lat', 'lon']
        )

        tmax_da = xr.DataArray(
            data = self.tmaxes,
            coords=[self.dates, np.flip(self._latitudes1d), self._longitudes1d],
            dims=['time', 'lat', 'lon']
        )

        tmin_da = xr.DataArray(
            data = self.tmins,
            coords=[self.dates, np.flip(self._latitudes1d), self._longitudes1d],
            dims=['time', 'lat', 'lon']
        )

        wind_da = xr.DataArray(
            data = self.winds,
            coords=[self.dates, np.flip(self._latitudes1d), self._longitudes1d],
            dims=['time', 'lat', 'lon']
        )

        extent_da = xr.DataArray(
            data = self._ar,
            coords=[np.flip(self._latitudes1d), self._longitudes1d],
            dims=['lat', 'lon']
        )

        ds = xr.Dataset(
            data_vars=dict(
                precip = precip_da,
                tmax = tmax_da,
                tmin = tmin_da,
                wind = wind_da,
                extent = extent_da
            )
        )

        if self._use_previous:
            if os.path.isfile(self._outputpath):
                log.debug(f"Found existing file at {self._outputpath} -- Updating in-place")
                # Assuming the existing file structure is same as the one generated now. Basically
                #   assuming that the previous file was also created by MetSimRunner
                existing = xr.open_dataset(self._outputpath)#.load()
                existing.close()
                last_existing_time = existing.time[-1]
                log.debug("Existing data: %s", last_existing_time)
                existing_to_append = existing.sel(time=slice(ds.time[0] - np.timedelta64(120,'D') , last_existing_time))
                ds = ds.sel(time=slice(last_existing_time + np.timedelta64(1,'D') , ds.time[-1]))
                # ds = ds.isel(time=slice(1, None))
                write_ds = xr.merge([existing_to_append, ds])
                ds = self._impute_basin_missing_data(write_ds)
                ds = self._min_max_temperature_check(ds)
            else:
                raise Exception('Previous combined dataset not found. Please run RAT without state files first.')
        else:
            log.debug(f"Creating new file at {self._outputpath}")
            ds = self._impute_basin_missing_data(ds)
            ds = self._min_max_temperature_check(ds)
        ds.to_netcdf(self._outputpath)
        # log.debug(f"Saving {len(paths)} files at {self._outputdir}")
        # xr.save_mfdataset(datasets, paths)

    def _read_day_raw(self, date):
        reqDate = date.strftime("%Y-%m-%d")
        log.debug("Combining data: %s", reqDate)
        d = self._datadir

        def _open(path):
            with rio.open(path) as src:
                return src.read(1, masked=True).astype(np.float32).filled(np.nan)

        precip = _open(os.path.join(d, f'precipitation/{reqDate}_IMERG.tif'))
        tmax   = _open(os.path.join(d, f'tmax/{reqDate}_TMAX.tif'))
        tmin   = _open(os.path.join(d, f'tmin/{reqDate}_TMIN.tif'))
        uwnd   = _open(os.path.join(d, f'uwnd/{reqDate}_UWND.tif'))
        vwnd   = _open(os.path.join(d, f'vwnd/{reqDate}_VWND.tif'))
        wind   = 0.75 * np.sqrt(uwnd**2 + vwnd**2)

        return precip, tmax, tmin, wind

    def _read_chunk(self, chunk_start, chunk_end):
        dates = pd.date_range(chunk_start, chunk_end)
        n, H, W = len(dates), self._rast.height, self._rast.width
        arrays = {v: np.empty((n, H, W), dtype=np.float32) for v in ('precip', 'tmax', 'tmin', 'wind')}

        with ThreadPoolExecutor(max_workers=min(self._n_workers, len(dates))) as pool:
            future_to_idx = {pool.submit(self._read_day_raw, date): i for i, date in enumerate(dates)}
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                p, tx, tn, w = future.result()
                arrays['precip'][i] = p
                arrays['tmax'][i]   = tx
                arrays['tmin'][i]   = tn
                arrays['wind'][i]   = w

        lats = np.flip(self._latitudes1d)
        coords = {'time': dates.to_pydatetime(), 'lat': lats, 'lon': self._longitudes1d}
        return xr.Dataset({
            k: xr.DataArray(v, dims=['time', 'lat', 'lon'], coords=coords)
            for k, v in arrays.items()
        })

    def _write_chunk(self, ds_chunk, chunk_path, existing_to_append=None, last_existing_time=None):
        if existing_to_append is not None:
            ds_chunk_sel = ds_chunk.sel(time=slice(last_existing_time + np.timedelta64(1, 'D'), ds_chunk.time[-1]))
            if 'extent' in existing_to_append.data_vars:
                # Drop the extent variable from existing_to_append if it already exists in the file for concat operation with other chunks. It will be added after writing all chunks in _apply_dataset_operations
                existing_to_append = existing_to_append.drop_vars('extent')
            write_ds = xr.concat([existing_to_append, ds_chunk_sel], dim='time')
            write_ds.to_netcdf(chunk_path, mode='w', unlimited_dims=['time'], encoding=_NETCDF_ENCODING)
        else:
            ds_chunk.to_netcdf(chunk_path, mode='w', unlimited_dims=['time'], encoding=_NETCDF_ENCODING)

    def _read_and_write_in_chunks(self):

        start_date = pd.to_datetime(self._start).to_pydatetime()
        start_date_pd = pd.to_datetime(self._start)
        end_date = pd.to_datetime(self._end).to_pydatetime()

        tmp_dir = Path(self._outputpath).parent / '.tmp_combinednc_chunks'
        tmp_dir.mkdir(exist_ok=True)

        chunk_start = start_date
        loop_number = 0

        try:
            while chunk_start < end_date:
                # Calculate the chunk end date (1 month later)
                chunk_end = min(chunk_start + relativedelta(months=1) - relativedelta(days=1), end_date)

                chunk_start_str = chunk_start.strftime("%Y-%m-%d")
                chunk_end_str = chunk_end.strftime("%Y-%m-%d")
                print(f"Running combinedNC for {chunk_start_str} and {chunk_end_str}")

                ds_chunk = self._read_chunk(chunk_start, chunk_end)
                chunk_path = tmp_dir / f"chunk_{loop_number:04d}.nc"

                # For first chunk
                if loop_number == 0:
                    # If using previously stored data, merge last 120 days from the start date to the chunk_data
                    if self._use_previous:
                        if os.path.isfile(self._outputpath):
                            log.debug(f"Found existing file at {self._outputpath} -- Updating in-place")
                            # Assuming the existing file structure is same as the one generated now. Basically
                            #   assuming that the previous file was also created by MetSimRunner

                            # Open the existing NetCDF file
                            with xr.open_dataset(self._outputpath) as existing:
                                last_existing_time = existing.time[-1]
                                log.debug("Existing data: %s", last_existing_time)

                                # Select the time slice you need before closing the dataset
                                existing_to_append = existing.sel(
                                    time=slice(start_date_pd.to_datetime64() - np.timedelta64(120, 'D'), last_existing_time)
                                ).load()  # Load the data into memory
                            # Now that the dataset is closed, pass the sliced data to the write function
                            self._write_chunk(ds_chunk, chunk_path, existing_to_append, last_existing_time)
                        else:
                            raise Exception('Previous combined dataset not found. Please run RAT with spin-up or use state file paths.')
                    # Else simply write the chunk
                    else:
                        self._write_chunk(ds_chunk, chunk_path)
                # For all other chunks, just write them
                else:
                    self._write_chunk(ds_chunk, chunk_path)

                loop_number += 1
                chunk_start = chunk_end + relativedelta(days=1)
                print("Chunk written.")

            print("Written complete CombinedNC and now applying data imputation and other operations like min-max temperature check.")
            da = xr.open_mfdataset(sorted(tmp_dir.glob("chunk_*.nc")), combine='by_coords', chunks='auto')
            self._apply_dataset_operations(da, self._outputpath)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _apply_dataset_operations(self, da, output_path):
        extent_da = xr.DataArray(
                data=self._ar,
                coords={'lat': np.flip(self._latitudes1d), 'lon': self._longitudes1d},
                dims=['lat', 'lon'],
            ).rename('extent')
        da['extent'] = extent_da
        clean_da = self._impute_basin_missing_data(da)
        clean_da = self._min_max_temperature_check(clean_da)
        clean_da.to_netcdf(output_path, encoding=_NETCDF_ENCODING_WITH_EXTENT)
        da.close()

def generate_state_and_inputs(forcings_startdate, forcings_enddate, combined_datapath, out_dir):
    # Generate state. Assuming `nc_fmt_data` contains all the data, presumably containing enough data
    # to create state file (upto 90 days prior data from forcings_startdate)
    combined_data = xr.open_dataset(combined_datapath)

    state_startdate = forcings_startdate - datetime.timedelta(days=90)
    state_enddate = forcings_startdate - datetime.timedelta(days=1)

    state_ds = combined_data.sel(time=slice(state_startdate, state_enddate))
    state_outpath = os.path.join(out_dir, "state.nc")
    log.debug(f"Saving state at: {state_outpath}")
    state_ds.to_netcdf(state_outpath)

    # Generate the metsim input
    forcings_ds = combined_data.sel(time=slice(forcings_startdate, forcings_enddate))
    forcings_outpath = os.path.join(out_dir, "metsim_input.nc")
    log.debug(f"Saving forcings: {forcings_outpath}")
    forcings_ds.to_netcdf(forcings_outpath)

    return state_outpath, forcings_outpath

def generate_metsim_state_and_inputs_with_multiple_nc_files(
        nc_file_paths,         # List of paths to NetCDF files to concatenate
        start_dates,           # List of start dates for each subsequent file in nc_file_paths
        out_dir,               # Output directory
        forcings_start_date,   # Start date for forcings
        forcings_end_date,     # End date for forcings
        forecast_mode=False    # Flag to indicate forecast mode
    ):
    """
    Generate MetSim state and input files by concatenating multiple NetCDF files along the time dimension.

    Parameters:
    - nc_file_paths (list of str): Paths to the NetCDF files to be concatenated.
    - start_dates (list of datetime.datetime): Start dates for each file in nc_file_paths from the second file onwards.
      The list should have one fewer elements than nc_file_paths.
    - out_dir (str): Directory where the output NetCDF files will be saved.
    - forcings_start_date (datetime.datetime): Start date for the forcings time slice.
    - forcings_end_date (datetime.datetime): End date for the forcings time slice.
    - forecast_mode (bool): If True, output files are prefixed with 'forecast_'.

    Returns:
    - state_outpath (Path): Path to the generated state NetCDF file.
    - forcings_outpath (Path): Path to the generated MetSim input NetCDF file.
    """

    out_dir = Path(out_dir)
    nc_datasets = []

    # Load and slice the datasets
    for i, nc_file_path in enumerate(nc_file_paths):
        ds = xr.open_dataset(nc_file_path)

        if i == 0:
            # First dataset: Slice from the start to the first start_date
            ds = ds.sel(time=slice(None, start_dates[0] - pd.Timedelta(days=1)))
        else:
            # Subsequent datasets: Slice from the day after the previous start_date
            if i < len(start_dates):
                ds = ds.sel(time=slice(start_dates[i-1], start_dates[i] - pd.Timedelta(days=1)))
            else:
                ds = ds.sel(time=slice(start_dates[i-1], None))

        nc_datasets.append(ds)

    # Concatenate all datasets along the time dimension
    combined_data = xr.concat(nc_datasets, dim="time")

    # Generate state NetCDF file
    state_startdate = forcings_start_date - pd.Timedelta(days=90)
    state_enddate = forcings_start_date - pd.Timedelta(days=1)
    state_ds = combined_data.sel(time=slice(state_startdate, state_enddate))

    state_filename = "forecast_state.nc" if forecast_mode else "state.nc"
    state_outpath = out_dir / state_filename
    state_ds.to_netcdf(state_outpath)

    # Generate MetSim input NetCDF file
    forcings_ds = combined_data.sel(time=slice(forcings_start_date, forcings_end_date))

    forcings_filename = "forecast_metsim_input.nc" if forecast_mode else "metsim_input.nc"
    forcings_outpath = out_dir / forcings_filename
    print(f"Saving forcings: {forcings_outpath}")
    forcings_ds.to_netcdf(forcings_outpath)

    return state_outpath, forcings_outpath
