"""
Utility functions for downloading and caching data files.

This module provides functions to download files from URLs and cache them locally
to avoid redundant downloads. Files are stored in a 'data_cache' directory.
"""

# Standard library
import calendar
import gzip
from pathlib import Path

# Third-party
import pandas as pd
import requests
import xarray as xr


CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


def open_or_download(url, local_name=None):
    """
    Download a CSV file from URL to local cache, or return cached DataFrame if exists.
    
    Checks if the file already exists in the cache directory. If it exists,
    loads and returns it as a DataFrame without downloading. If not, downloads
    the file, saves it to the cache, and returns it as a DataFrame.
    
    Parameters
    ----------
    url : str
        URL of the CSV file to download.
    local_name : str, optional
        Custom filename for the cached file. If None, uses the filename
        extracted from the URL.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the CSV data.
    
    Raises
    ------
    requests.HTTPError
        If the HTTP request fails (e.g., 404, 500).
    pd.errors.EmptyDataError
        If the CSV file is empty or cannot be parsed.
    
    Examples
    --------
    >>> df = open_or_download("https://example.com/data.csv")
    >>> type(df)
    <class 'pandas.core.frame.DataFrame'>
    """
    filename = local_name or url.split('/')[-1]
    local_path = CACHE_DIR / filename
    
    if local_path.exists():
        print(f"loaded {filename}")
        return pd.read_csv(local_path, low_memory=False)
    
    print(f"downloading {filename}...")
    resp = requests.get(url)
    resp.raise_for_status()
    local_path.write_bytes(resp.content)
    return pd.read_csv(local_path, low_memory=False)


def open_or_download_gz(url, local_name=None):
    """
    Download a gzipped CSV file, decompress, cache, and return as DataFrame.
    
    Downloads a gzipped file from URL, decompresses it, and caches the
    decompressed CSV file. If the decompressed file already exists in cache,
    loads and returns it directly without downloading. The gzip extension
    is automatically removed from the cached filename.
    
    Parameters
    ----------
    url : str
        URL of the gzipped CSV file to download.
    local_name : str, optional
        Custom filename for the cached decompressed CSV file. If None,
        uses the filename from the URL with '.gz' extension removed.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the CSV data.
    
    Raises
    ------
    requests.HTTPError
        If the HTTP request fails (e.g., 404, 500).
    pd.errors.EmptyDataError
        If the CSV file is empty or cannot be parsed.
    
    Examples
    --------
    >>> df = open_or_download_gz("https://example.com/data.csv.gz")
    >>> type(df)
    <class 'pandas.core.frame.DataFrame'>
    """
    filename = local_name or url.split('/')[-1].replace('.gz', '')
    local_path = CACHE_DIR / filename
    
    if local_path.exists():
        print(f"loaded {filename}")
        return pd.read_csv(local_path, low_memory=False)
    
    print(f"downloading {filename}...")
    resp = requests.get(url)
    resp.raise_for_status()
    
    decompressed = gzip.decompress(resp.content)
    local_path.write_bytes(decompressed)
    
    return pd.read_csv(local_path, low_memory=False)


def get_era5_url(base_url, year, month, file_code):
    """
    Generate ERA5 OPeNDAP URL for surface data given year, month, and file code.
    
    Parameters
    ----------
    base_url : str
        Base OPeNDAP URL for ERA5 surface data.
    year : int
        Year (e.g., 2024).
    month : int
        Month (1-12).
    file_code : str
        ERA5 file code (e.g., '128_059_cape').
    
    Returns
    -------
    str
        Full OPeNDAP URL for the ERA5 monthly surface dataset.
    
    Examples
    --------
    >>> get_era5_url("https://example.com/base", 2024, 5, '128_059_cape')
    'https://example.com/base/202405/e5.oper.an.sfc.128_059_cape.ll025sc.2024050100_2024053123.nc'
    """
    yyyymm = f"{year}{month:02d}"
    last_day = calendar.monthrange(year, month)[1]
    filename = (
        f"e5.oper.an.sfc.{file_code}.ll025sc."
        f"{yyyymm}0100_{yyyymm}{last_day:02d}23.nc"
    )
    return f"{base_url}/{yyyymm}/{filename}"


def open_or_download_era5(url, lat_min, lat_max, lon_min, lon_max, hour):
    """
    Open ERA5 dataset from OPeNDAP URL, subset spatially and temporally, cache locally.
    
    Checks if a cached subsetted netCDF file exists. If it exists, opens and returns it.
    If not, opens the dataset via OPeNDAP, subsets by latitude/longitude and hour,
    saves the subsetted data to cache, and returns the dataset.
    
    Parameters
    ----------
    url : str
        OPeNDAP URL for the ERA5 dataset.
    lat_min, lat_max : float
        Latitude bounds (degrees N). ERA5 latitude is descending, so max comes first.
    lon_min, lon_max : float
        Longitude bounds (0-360 format).
    hour : int
        Hour to filter (UTC, e.g., 18 for 18 UTC).
    
    Returns
    -------
    xr.Dataset
        Subsetted xarray Dataset.
    
    Examples
    --------
    >>> ds = open_or_download_era5(
    ...     "https://example.com/data.nc",
    ...     lat_min=37.0, lat_max=40.0,
    ...     lon_min=258.0, lon_max=265.5,
    ...     hour=18
    ... )
    """
    filename = url.split('/')[-1]
    local_nc = CACHE_DIR / filename
    
    if local_nc.exists():
        print(f"loaded {filename}")
        return xr.open_dataset(local_nc)
    
    print(f"downloading {filename}...")
    ds = xr.open_dataset(url)
    ds = ds.sel(
        latitude=slice(lat_max, lat_min),  # ERA5 lat is descending
        longitude=slice(lon_min, lon_max)
    )
    ds = ds.sel(time=ds.time.dt.hour == hour)
    ds.to_netcdf(local_nc)
    return ds


def open_or_download_era5_pl_monthly(
    base_url, year, month, file_code, nc_var, level, grid_code, lat_min, lat_max, lon_min, lon_max, hour
):
    """
    Load ERA5 pressure level data for a month by aggregating daily files.
    
    Parameters
    ----------
    base_url : str
        Base OPeNDAP URL for ERA5 pressure level data.
    year : int
        Year.
    month : int
        Month (1-12).
    file_code : str
        ERA5 file code (e.g., '128_131_u').
    nc_var : str
        NetCDF variable name (e.g., 'U').
    level : int
        Pressure level in hPa (e.g., 500).
    grid_code : str
        Grid code: 'll025uv' for U/V winds, 'll025sc' for other variables.
    lat_min, lat_max : float
        Latitude bounds.
    lon_min, lon_max : float
        Longitude bounds.
    hour : int
        Hour to filter (UTC).
    
    Returns
    -------
    xr.DataArray
        Monthly aggregated data at specified pressure level.
    """
    yyyymm = f"{year}{month:02d}"
    last_day = calendar.monthrange(year, month)[1]
    
    # Load daily files and aggregate to monthly
    daily_data = []
    for day in range(1, last_day + 1):
        yyyymmdd = f"{year}{month:02d}{day:02d}"
        filename = f"e5.oper.an.pl.{file_code}.{grid_code}.{yyyymmdd}00_{yyyymmdd}23.nc"
        url = f"{base_url}/{yyyymm}/{filename}"
        
        ds = open_or_download_era5(url, lat_min, lat_max, lon_min, lon_max, hour)
        if 'level' in ds.dims:
            var_at_level = ds[nc_var].sel(level=level, method='nearest')
        else:
            var_at_level = ds[nc_var]
        daily_data.append(var_at_level)
    
    if not daily_data:
        raise ValueError(f"Failed to load any daily files for {file_code} in {year}-{month:02d}")
    
    return xr.concat(daily_data, dim='time')

