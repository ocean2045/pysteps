import numpy as np
import pysteps
from pysteps.utils import conversion, transformation
from pysteps.motion import lucaskanade
from pysteps.nowcasts import get_method
from datetime import timedelta

def generate_steps_nowcast(
    input_reflectivity_dBZ: np.ndarray,
    pysteps_metadata: dict,
    n_leadtimes: int,
    n_ens_members: int,
    num_input_frames_for_motion: int
) -> Optional[np.ndarray]:
    """
    Generates a STEPS nowcast from radar reflectivity data.

    Args:
        input_reflectivity_dBZ: 3D NumPy array (time, height, width) of radar 
                                reflectivity in dBZ.
        pysteps_metadata: Metadata dictionary compatible with pysteps. 
                          Expected keys: 'zr_a', 'zr_b' (for Z-R conversion),
                          'zerovalue' (for dBZ), 'threshold' (for dBZ), 
                          'timestamps' (list of datetime objects), 
                          'xpixelsize', 'ypixelsize' (in degrees),
                          'cartesian_unit'.
        n_leadtimes: Integer, number of forecast lead times.
        n_ens_members: Integer, number of ensemble members for STEPS.
        num_input_frames_for_motion: Integer, number of recent frames to use for
                                     motion estimation and STEPS input.

    Returns:
        A 4D NumPy array (ensemble_member, lead_time, height, width) of rain rate
        forecasts in mm/hr. Returns None if an error occurs.
    """
    try:
        # Step 1: Convert Reflectivity to Rain Rate (mm/hr)
        # pysteps.utils.conversion.to_rainrate uses zr_a and zr_b from metadata if present
        # Ensure they are floats if fetching manually
        zr_a = float(pysteps_metadata.get("zr_a", 200.0))
        zr_b = float(pysteps_metadata.get("zr_b", 1.6))
        
        # Create a temporary metadata copy for to_rainrate if needed, or ensure original is not modified
        metadata_for_conversion = pysteps_metadata.copy()
        metadata_for_conversion['zr_a'] = zr_a
        metadata_for_conversion['zr_b'] = zr_b

        # The to_rainrate function might fill NaNs if instructed by metadata,
        # but we should handle NaNs explicitly before transformation.
        # Let's assume input_reflectivity_dBZ might contain NaNs.
        # Pysteps functions generally prefer NaNs to be filled with the zerovalue.
        dbz_zerovalue = pysteps_metadata.get("zerovalue", -15.0) # Original dBZ zerovalue
        input_reflectivity_dBZ_filled = np.nan_to_num(input_reflectivity_dBZ, nan=dbz_zerovalue)

        rain_rate_mmhr, _ = conversion.to_rainrate(
            input_reflectivity_dBZ_filled, 
            metadata=metadata_for_conversion
        )

        # Step 2: Transform Rain Rate to dBR
        # For dBR transformation, we need a threshold and zerovalue specific to dBR.
        # These are typically different from the dBZ zerovalue.
        dbr_threshold = float(pysteps_metadata.get("dbr_threshold", -10.0)) # Threshold for dBR
        dbr_zerovalue = float(pysteps_metadata.get("dbr_zerovalue", -15.0)) # Zerovalue for dBR (transformed data)
        
        # Metadata for dB_transform
        metadata_for_dbr = pysteps_metadata.copy()
        metadata_for_dbr["threshold"] = dbr_threshold
        metadata_for_dbr["zerovalue"] = dbr_zerovalue
        
        # Handle NaNs/infs in rain_rate_mmhr before dB_transform.
        # Values <= 0 will result in -inf, map them to a value that results in dbr_zerovalue.
        # If rain_rate_mmhr is 0, it becomes -inf. dB_transform handles this if `zerovalue` is correctly set.
        # Smallest detectable rain rate, maps to dbr_threshold.
        # For dB_transform, input values <= threshold are set to zerovalue in dBR.
        # So, ensure rain_rate_mmhr has no NaNs/infs.
        # Values very close to 0 can cause issues. If threshold is 0.01 mm/hr, values below this become dbr_zerovalue.
        rain_rate_threshold_for_dbr = float(pysteps_metadata.get("rain_rate_threshold_for_dbr", 0.01)) # e.g. 0.01 mm/hr
        rain_rate_mmhr_filled = np.nan_to_num(rain_rate_mmhr, nan=0.0) # Replace NaNs with 0
        rain_rate_mmhr_filled[rain_rate_mmhr_filled < rain_rate_threshold_for_dbr] = 0.0 # Apply floor

        dbr_data, _ = transformation.dB_transform(
            rain_rate_mmhr_filled, 
            metadata=metadata_for_dbr, 
            threshold=rain_rate_threshold_for_dbr, # This is R_thr for the transform
            zerovalue=dbr_zerovalue      # This is the output value for R < R_thr
        )
        # Ensure no NaNs/infs after transform (dB_transform should handle this)
        dbr_data = np.nan_to_num(dbr_data, nan=dbr_zerovalue, posinf=dbr_zerovalue, neginf=dbr_zerovalue)

        # Step 3: Estimate Motion Field
        # Use the most recent `num_input_frames_for_motion` frames of dBR data
        if dbr_data.shape[0] < num_input_frames_for_motion:
            print(f"Warning: Not enough frames ({dbr_data.shape[0]}) for motion estimation. Need {num_input_frames_for_motion}.")
            return None
        
        dbr_data_for_motion = dbr_data[-num_input_frames_for_motion:, :, :]
        
        # Check for sufficient variance in input data for Lucas-Kanade
        if np.allclose(dbr_data_for_motion, dbr_data_for_motion[0, :, :]): # Check if all frames are identical
             print("Warning: All input frames for Lucas-Kanade are identical. Motion field might be zero.")
             # Proceed, Lucas-Kanade might return zero vectors, which is acceptable.

        motion_field = lucaskanade.dense_lucaskanade(dbr_data_for_motion)

        # Step 4: Generate STEPS Nowcast
        nowcast_method_steps = get_method("steps")

        # Determine kmperpixel
        # This is a placeholder. For geographic coordinates (degrees), proper projection is needed.
        # 1 degree lat ~ 111 km. 1 degree lon ~ 111 km * cos(latitude).
        # Using site_coords for latitude if available, else a mid-latitude default.
        site_lat = pysteps_metadata.get("site_coords", [35.0, 0.0])[0] # Default to 35 deg lat
        x_pixel_size_km = pysteps_metadata['xpixelsize'] * 111.0 * np.cos(np.deg2rad(site_lat))
        y_pixel_size_km = pysteps_metadata['ypixelsize'] * 111.0
        kmperpixel_val = (x_pixel_size_km + y_pixel_size_km) / 2.0
        if kmperpixel_val <= 0:
            print(f"Warning: kmperpixel calculated as {kmperpixel_val}. Using default 1.0 km.")
            kmperpixel_val = 1.0 # Default to 1km if calculation fails

        # Determine timestep in minutes
        if len(pysteps_metadata['timestamps']) < 2:
            print("Warning: Not enough timestamps to determine timestep. Using default 6 minutes.")
            timestep_val = 6.0
        else:
            # Assuming timestamps are sorted
            dt = pysteps_metadata['timestamps'][1] - pysteps_metadata['timestamps'][0]
            if isinstance(dt, timedelta): # If datetime objects
                 timestep_val = dt.total_seconds() / 60.0
            elif isinstance(dt, (int, float)): # If already numeric (e.g. minutes)
                 timestep_val = float(dt)
            else:
                 print(f"Warning: Unknown timestamp difference type: {type(dt)}. Using default 6 minutes.")
                 timestep_val = 6.0
        
        if timestep_val <= 0:
            print(f"Warning: timestep calculated as {timestep_val}. Using default 6 minutes.")
            timestep_val = 6.0


        # STEPS parameters
        # precip_thr for STEPS should be in dBR units (same as dbr_threshold)
        steps_precip_thr = dbr_threshold 

        # The input to STEPS should be the same frames used for motion estimation
        # STEPS expects input shape (num_input_frames, height, width)
        dbr_forecast_ensemble = nowcast_method_steps(
            precip=dbr_data_for_motion,  # Input dBR data
            velocity=motion_field,
            timesteps=n_leadtimes,
            n_ens_members=n_ens_members,
            n_cascade_levels=6,
            kmperpixel=kmperpixel_val,
            timestep=timestep_val,
            seed=None,
            R_thr=steps_precip_thr # Threshold for rain/no rain, in dBR units
            # Other parameters like `vel_pert_method`, `mask_method`, `probmatching_method` use defaults
        )
        # Output shape: (n_ens_members, n_leadtimes, height, width)

        # Step 5: Inverse Transform Forecast
        # Use the same dbr_threshold and dbr_zerovalue for inverse transform
        # The metadata for inverse transform should reflect the dBR domain
        metadata_for_inverse_dbr = metadata_for_dbr # Uses dbr_threshold and dbr_zerovalue
        
        rain_rate_forecast_ensemble, _ = transformation.dB_transform(
            dbr_forecast_ensemble, 
            metadata=metadata_for_inverse_dbr, 
            inverse=True,
            threshold=rain_rate_threshold_for_dbr, # R_thr for inverse transform
            zerovalue=dbr_zerovalue # Input zerovalue for inverse transform
        )
        # Ensure no negatives in rain rate, set to 0
        rain_rate_forecast_ensemble[rain_rate_forecast_ensemble < 0] = 0.0
        rain_rate_forecast_ensemble = np.nan_to_num(rain_rate_forecast_ensemble, nan=0.0)


        # Step 6: Return
        # Expected output shape: (ensemble_member, lead_time, height, width)
        return rain_rate_forecast_ensemble

    except Exception as e:
        print(f"Error in generate_steps_nowcast: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    print("nowcasting_workflow.py created. Contains `generate_steps_nowcast` function.")
    
    # Example of how to call the function (requires actual data and metadata)
    # This is a placeholder and will not run without proper inputs.
    
    # 1. Create dummy input_reflectivity_dBZ (time, height, width)
    # Typically, you might have 3-5 input frames for STEPS.
    num_frames = 3
    height, width = 128, 128 # Example dimensions
    dummy_input_dBZ = np.random.rand(num_frames, height, width) * 50 - 10 # Random dBZ values
    dummy_input_dBZ[dummy_input_dBZ < 0] = -15.0 # Apply a zerovalue

    # 2. Create dummy pysteps_metadata
    from datetime import datetime, timedelta
    current_time = datetime.utcnow()
    dummy_metadata = {
        "zr_a": 223.0,  # Example Z-R params
        "zr_b": 1.53,
        "zerovalue": -15.0, # For input dBZ
        "threshold": 0.0,   # For input dBZ
        "timestamps": [current_time - timedelta(minutes=i*6) for i in range(num_frames-1, -1, -1)], # Timestamps for each frame
        "xpixelsize": 0.01, # degrees (approx 1km)
        "ypixelsize": 0.01, # degrees (approx 1km)
        "cartesian_unit": "degrees",
        "site_coords": (34.0, 108.0), # Example site lat/lon for kmperpixel calc
        # For dBR transform and STEPS
        "dbr_threshold": -10.0, # dBR threshold (transformed data)
        "dbr_zerovalue": -15.0, # dBR zerovalue (transformed data)
        "rain_rate_threshold_for_dbr": 0.01, # mm/hr, values below this become dbr_zerovalue
        "timestep": 6.0 # Timestep in minutes, can also be calculated from timestamps
    }

    # 3. Set other parameters
    n_leadtimes_val = 10  # e.g., 10 * 6 min = 1 hour forecast
    n_ens_members_val = 5
    num_input_frames_for_motion_val = num_frames # Use all available frames for motion

    print(f"\nAttempting to run generate_steps_nowcast with dummy data...")
    print(f"Input dBZ shape: {dummy_input_dBZ.shape}")
    print(f"Metadata timestamps: {dummy_metadata['timestamps']}")
    print(f"Metadata xpixelsize: {dummy_metadata['xpixelsize']}, ypixelsize: {dummy_metadata['ypixelsize']}")


    forecast_ensemble = generate_steps_nowcast(
        dummy_input_dBZ,
        dummy_metadata,
        n_leadtimes=n_leadtimes_val,
        n_ens_members=n_ens_members_val,
        num_input_frames_for_motion=num_input_frames_for_motion_val
    )

    if forecast_ensemble is not None:
        print(f"\nSuccessfully generated STEPS nowcast ensemble.")
        print(f"Output forecast shape: {forecast_ensemble.shape}")
        # (ensemble_member, lead_time, height, width)
        # Expected: (5, 10, 128, 128)
        if forecast_ensemble.shape == (n_ens_members_val, n_leadtimes_val, height, width):
            print("Output shape is as expected.")
        else:
            print(f"Warning: Output shape {forecast_ensemble.shape} is not as expected {(n_ens_members_val, n_leadtimes_val, height, width)}.")
        
        # Check some values
        print(f"Min forecast rain rate: {forecast_ensemble.min():.4f} mm/hr")
        print(f"Max forecast rain rate: {forecast_ensemble.max():.4f} mm/hr")
        print(f"Mean forecast rain rate: {forecast_ensemble.mean():.4f} mm/hr")
    else:
        print("\nFailed to generate STEPS nowcast.")

    print("\nNote: The example run uses dummy data. For actual use, provide real radar data and accurate metadata.")

from typing import Optional # Added at the end to ensure it's available for the function hint.
# It's better practice to have all imports at the top.
# Re-arranging imports in the next step if possible, or ensuring linters/formatters handle it.
