import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

def save_forecast_to_netcdf(
    rain_rate_forecast_ensemble: np.ndarray,
    reflectivity_forecast_ensemble: np.ndarray,
    pysteps_metadata: Dict[str, Any],
    output_path: str,
    n_leadtimes: int, # This is the number of forecast steps
    user_netcdf_specs: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Saves the mean of a forecast ensemble (rain rate and reflectivity) to a NetCDF file.

    Args:
        rain_rate_forecast_ensemble: 4D NumPy array (ensemble_member, lead_time, 
                                     height, width) of rain rate forecasts (mm/hr).
        reflectivity_forecast_ensemble: 4D NumPy array (ensemble_member, lead_time,
                                        height, width) of reflectivity forecasts (dBZ).
        pysteps_metadata: Metadata dictionary. Expected keys: 'y1', 'y2' (lat bounds),
                          'x1', 'x2' (lon bounds), 'timestamps' (list of datetimes),
                          optionally 'timestep' (forecast interval in minutes),
                          'projection', 'site_coords', 'institution'.
        output_path: Full path (including filename) for the output NetCDF file.
        n_leadtimes: Integer, number of forecast lead times (length of lead_time dim).
        user_netcdf_specs: Optional dictionary for user-defined NetCDF specifications.
                           Can override default variable names, units, etc.

    Returns:
        True if saving was successful, False otherwise.
    """
    if user_netcdf_specs is None:
        user_netcdf_specs = {}

    try:
        # Step 1: Calculate Ensemble Means
        # Input shape: (ensemble_member, lead_time, height, width)
        # Output shape: (lead_time, height, width)
        rain_rate_mean = np.mean(rain_rate_forecast_ensemble, axis=0)
        reflectivity_mean = np.mean(reflectivity_forecast_ensemble, axis=0)

        if rain_rate_mean.shape[0] != n_leadtimes:
            raise ValueError(f"Rain rate mean leadtime dimension ({rain_rate_mean.shape[0]}) "
                             f"does not match n_leadtimes ({n_leadtimes}).")
        if reflectivity_mean.shape[0] != n_leadtimes:
            raise ValueError(f"Reflectivity mean leadtime dimension ({reflectivity_mean.shape[0]}) "
                             f"does not match n_leadtimes ({n_leadtimes}).")

        # Step 2: Create Coordinate Arrays
        num_ensemble_members, _, height, width = rain_rate_forecast_ensemble.shape
        
        # Determine forecast timestep in minutes
        timestep_minutes = pysteps_metadata.get('timestep') # Explicit timestep if provided
        if timestep_minutes is None:
            if len(pysteps_metadata.get('timestamps', [])) >= 2:
                # Calculate from the input radar scan interval
                dt = pysteps_metadata['timestamps'][1] - pysteps_metadata['timestamps'][0]
                if isinstance(dt, timedelta):
                    timestep_minutes = dt.total_seconds() / 60.0
                elif isinstance(dt, (int, float)): # If already numeric
                    timestep_minutes = float(dt)
                else:
                    print("Warning: Timestep from timestamps has unknown type, using default 6 min.")
                    timestep_minutes = 6.0 
            else:
                print("Warning: Not enough timestamps and no explicit 'timestep' in metadata. Using default 6 min for lead_time coordinate.")
                timestep_minutes = 6.0  # Default if not calculable

        if timestep_minutes <= 0:
            print(f"Warning: Invalid timestep_minutes ({timestep_minutes}), using default 6 min.")
            timestep_minutes = 6.0

        # Lead time coordinates (e.g., [0, 6, 12, ...])
        # The first forecast step is often T+timestep, but STEPS output includes T+0 implicitly if the first "prediction" is the last input.
        # However, n_leadtimes usually refers to actual future predictions.
        # If STEPS output has shape (n_leadtimes, H, W), these are for T+dt, T+2dt, ... T+n_leadtimes*dt
        # Let's assume lead times are 0-indexed multiples of timestep_minutes
        lead_time_coords = np.arange(n_leadtimes) * timestep_minutes
        
        latitude_coords = np.linspace(pysteps_metadata['y1'], pysteps_metadata['y2'], height)
        longitude_coords = np.linspace(pysteps_metadata['x1'], pysteps_metadata['x2'], width)

        # Step 3: Create xarray.DataArray Objects
        precip_var_name = user_netcdf_specs.get("precip_var_name", "precipitation_forecast")
        precip_units = user_netcdf_specs.get("precip_units", "mm/hr")
        precip_long_name = user_netcdf_specs.get("precip_long_name", "Mean Ensemble Precipitation Forecast")

        refl_var_name = user_netcdf_specs.get("refl_var_name", "reflectivity_forecast")
        refl_units = user_netcdf_specs.get("refl_units", "dBZ")
        refl_long_name = user_netcdf_specs.get("refl_long_name", "Mean Ensemble Reflectivity Forecast")

        lead_time_dim_name = user_netcdf_specs.get("lead_time_dim_name", "lead_time")
        latitude_dim_name = user_netcdf_specs.get("latitude_dim_name", "latitude")
        longitude_dim_name = user_netcdf_specs.get("longitude_dim_name", "longitude")

        common_coords = {
            lead_time_dim_name: lead_time_coords,
            latitude_dim_name: latitude_coords,
            longitude_dim_name: longitude_coords
        }
        common_dims = (lead_time_dim_name, latitude_dim_name, longitude_dim_name)

        da_precip = xr.DataArray(
            data=rain_rate_mean,
            coords=common_coords,
            dims=common_dims,
            name=precip_var_name,
            attrs={"units": precip_units, "long_name": precip_long_name}
        )

        da_refl = xr.DataArray(
            data=reflectivity_mean,
            coords=common_coords,
            dims=common_dims,
            name=refl_var_name,
            attrs={"units": refl_units, "long_name": refl_long_name}
        )
        
        # Add attributes to coordinate variables
        da_precip[lead_time_dim_name].attrs['units'] = 'minutes'
        da_precip[lead_time_dim_name].attrs['long_name'] = 'forecast lead time from initial scan'
        da_precip[latitude_dim_name].attrs['units'] = 'degrees_north'
        da_precip[latitude_dim_name].attrs['long_name'] = 'Latitude'
        da_precip[longitude_dim_name].attrs['units'] = 'degrees_east'
        da_precip[longitude_dim_name].attrs['long_name'] = 'Longitude'


        # Step 4: Combine into xarray.Dataset
        dataset = xr.Dataset({precip_var_name: da_precip, refl_var_name: da_refl})

        # Step 5: Add Global Attributes
        initial_scan_time = pysteps_metadata.get('timestamps', [datetime.utcnow()])[0]
        if isinstance(initial_scan_time, datetime):
            dataset.attrs["initial_scan_time"] = initial_scan_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else: # if it's already a string or other format
            dataset.attrs["initial_scan_time"] = str(initial_scan_time)

        dataset.attrs["projection"] = str(pysteps_metadata.get("projection", "Unknown"))
        site_coords_val = pysteps_metadata.get("site_coords", "Unknown")
        if isinstance(site_coords_val, (list, tuple)) and len(site_coords_val) == 2:
            dataset.attrs["site_latitude"] = float(site_coords_val[0])
            dataset.attrs["site_longitude"] = float(site_coords_val[1])
        else:
            dataset.attrs["site_coords"] = str(site_coords_val)
            
        dataset.attrs["institution"] = str(pysteps_metadata.get("institution", "Unknown"))
        dataset.attrs["history"] = f"Created by pysteps-based nowcasting workflow on {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}."
        dataset.attrs["ensemble_members_count"] = num_ensemble_members
        dataset.attrs["description"] = "Mean ensemble forecast of precipitation and reflectivity."

        # Step 6: Save to NetCDF
        # Define encoding for variables to handle _FillValue, scale_factor, etc. if needed.
        # For basic saving, this might not be strictly necessary but is good practice.
        encoding = {}
        for var_name in dataset.data_vars:
            encoding[var_name] = {'zlib': True, 'complevel': 4} # Example compression

        dataset.to_netcdf(output_path, encoding=encoding)
        print(f"Successfully saved forecast to {output_path}")
        return True

    except Exception as e:
        print(f"Error in save_forecast_to_netcdf: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("output_module.py created. Contains `save_forecast_to_netcdf` function.")

    # Example of how to call the function (requires actual data and metadata)
    # This is a placeholder and will not run without proper inputs.

    # 1. Create dummy forecast ensembles (ensemble_member, lead_time, height, width)
    n_ens_val = 5
    n_lead_val = 10 # 10 lead times
    height_val, width_val = 128, 128
    
    dummy_rain_forecast = np.random.rand(n_ens_val, n_lead_val, height_val, width_val) * 10 # 0-10 mm/hr
    dummy_refl_forecast = np.random.rand(n_ens_val, n_lead_val, height_val, width_val) * 60 - 10 # -10 to 50 dBZ

    # 2. Create dummy pysteps_metadata
    current_time = datetime.utcnow()
    dummy_metadata = {
        "zr_a": 223.0,
        "zr_b": 1.53,
        "zerovalue": -15.0,
        "threshold": 0.0,
        "timestamps": [current_time - timedelta(minutes=i*6) for i in range(2, -1, -1)], # 3 input timestamps
        "timestep": 6.0, # Forecast timestep in minutes
        "x1": 100.0, "x2": 101.27, # Lon extent for 128 pixels at ~0.01 deg/pix
        "y1": 30.0,  "y2": 31.27,  # Lat extent for 128 pixels at ~0.01 deg/pix
        "xpixelsize": 0.01, 
        "ypixelsize": 0.01,
        "cartesian_unit": "degrees",
        "projection": "+proj=longlat +datum=WGS84 +no_defs",
        "site_coords": (30.5, 100.5), # Lat, Lon
        "institution": "My Weather Service"
    }

    # 3. Output path
    output_file_path = "dummy_forecast.nc" # Will be created in the current directory

    # 4. User NetCDF specs (optional)
    custom_specs = {
        "precip_var_name": "rainfall_rate_forecast",
        "precip_units": "mm h-1", # Example of slightly different unit string
        "precip_long_name": "Mean Rainfall Rate Forecast from Ensemble",
        "refl_var_name": "radar_reflectivity_forecast",
        "refl_long_name": "Mean Radar Reflectivity Forecast from Ensemble",
    }

    print(f"\nAttempting to run save_forecast_to_netcdf with dummy data...")
    success = save_forecast_to_netcdf(
        dummy_rain_forecast,
        dummy_refl_forecast,
        dummy_metadata,
        output_file_path,
        n_leadtimes=n_lead_val, # Number of forecast steps in the ensemble arrays
        user_netcdf_specs=custom_specs 
    )

    if success:
        print(f"Dummy forecast saved to {output_file_path}. Check its contents using a NetCDF viewer.")
        # Verify by loading back (optional)
        try:
            ds_loaded = xr.open_dataset(output_file_path)
            print("\nLoaded dataset summary:")
            print(ds_loaded)
            # Clean up dummy file
            import os
            os.remove(output_file_path)
            print(f"Cleaned up dummy file: {output_file_path}")
        except Exception as e:
            print(f"Error loading back or deleting dummy file: {e}")
    else:
        print("\nFailed to save dummy forecast.")
    
    print("\nNote: The example run uses dummy data. For actual use, provide real forecast data and accurate metadata.")
