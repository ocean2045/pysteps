import cinrad.io
import cinrad.calc
import xarray as xr
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any # Added for type hinting

def read_cinrad_reflectivity(file_path: str) -> Optional[Tuple[xr.Dataset, cinrad.io.StandardData]]:
    """
    Reads a CINRAD radar data file, extracts the first available 'REF' (reflectivity)
    data, and converts it to a Cartesian grid.

    Args:
        file_path: Path to the CINRAD FMT radar data file.

    Returns:
        A tuple containing:
            - cartesian_data: xarray.Dataset with reflectivity data on a Cartesian grid.
            - file_object: The cinrad.io.StandardData object.
        Returns None if any error occurs during processing.

    Raises: (Note: Original raises are now handled internally and logged, returning None)
        FileNotFoundError: If the specified file_path does not exist.
        ValueError: If the 'REF' product is not available in any tilt,
                    or if no tilts are available in the file.
    """
    try:
        f = cinrad.io.StandardData(file_path)
    except FileNotFoundError:
        # print(f"Warning: Radar data file not found: {file_path}")
        raise # Re-raise to be caught by load_fmt_time_series
    except Exception as e:
        # print(f"Warning: Error opening CINRAD file {file_path}: {e}")
        raise # Re-raise

    available_tilts = f.get_avail_tilt()
    if not available_tilts:
        # print(f"Warning: No tilts available in the file: {file_path}")
        raise ValueError(f"No tilts available in the file: {file_path}") # Re-raise

    ref_tilt_index = None
    for tilt_num in available_tilts:
        try:
            f.get_data(tilt_num, 230, 'REF') # Check if REF exists for this tilt
            ref_tilt_index = tilt_num
            break
        except Exception:
            continue

    if ref_tilt_index is None:
        # print(f"Warning: 'REF' product not available for any tilt in file: {file_path}")
        raise ValueError(f"'REF' product not available for any tilt in file: {file_path}") # Re-raise

    try:
        polar_ref_data = f.get_data(tilt_num=ref_tilt_index, drange=230, data_type='REF')
        cartesian_data = cinrad.calc.polar_to_xy(polar_ref_data, grid_shape=(500, 500))
    except Exception as e:
        # print(f"Warning: Error processing data (polar to xy) for {file_path}: {e}")
        raise # Re-raise

    return cartesian_data, f

def extract_pysteps_metadata_from_cinrad(
    cartesian_ds: xr.Dataset, 
    cinrad_file_obj: cinrad.io.StandardData
) -> Dict[str, Any]:
    """
    Extracts metadata from CINRAD Cartesian data and file object for pysteps.

    Args:
        cartesian_ds: xarray.Dataset containing gridded radar data.
        cinrad_file_obj: The cinrad.io.StandardData object.

    Returns:
        A dictionary containing pysteps-compatible metadata.
    """
    metadata: Dict[str, Any] = {}
    metadata["timestamps"] = [cinrad_file_obj.scantime]
    metadata["unit"] = "dBZ"
    metadata["transform"] = None
    metadata["zerovalue"] = -15.0 
    metadata["threshold"] = metadata["zerovalue"]
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    x1 = cartesian_ds.longitude.min().item()
    y1 = cartesian_ds.latitude.min().item()
    x2 = cartesian_ds.longitude.max().item()
    y2 = cartesian_ds.latitude.max().item()
    metadata["x1"] = x1
    metadata["y1"] = y1
    metadata["x2"] = x2
    metadata["y2"] = y2

    if 'longitude' not in cartesian_ds.dims or 'latitude' not in cartesian_ds.dims:
        raise ValueError("cartesian_ds must have 'longitude' and 'latitude' dimensions.")
    if cartesian_ds.dims['longitude'] == 0 or cartesian_ds.dims['latitude'] == 0:
        raise ValueError("cartesian_ds dimensions cannot be zero.")
        
    metadata["xpixelsize"] = (x2 - x1) / cartesian_ds.dims['longitude']
    metadata["ypixelsize"] = (y2 - y1) / cartesian_ds.dims['latitude']
    
    metadata["cartesian_unit"] = "degrees"
    metadata["yorigin"] = "lower"
    metadata["projection"] = "+proj=longlat +datum=WGS84 +no_defs"
    metadata["site_coords"] = (cinrad_file_obj.site_latitude, cinrad_file_obj.site_longitude)
    metadata["institution"] = "Unknown CINRAD site" 
    metadata["product"] = "REF" 
    return metadata

def load_fmt_time_series(fmt_file_paths: List[str]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Loads a time series of CINRAD FMT radar data.

    Processes a list of CINRAD file paths, reads each one, extracts reflectivity
    data and metadata, and then consolidates them into a 3D NumPy array 
    (time, latitude, longitude) and a single metadata dictionary for pysteps.

    Args:
        fmt_file_paths: A list of strings, where each string is a path to a 
                        CINRAD FMT radar data file.

    Returns:
        A tuple containing:
            - reflectivity_stack: A 3D NumPy array (time, lat, lon) of reflectivity
                                  data. Returns None if no files are successfully processed.
            - consolidated_metadata: A dictionary containing metadata for the time
                                     series. The 'timestamps' field will be a list of
                                     all timestamps from processed files. Other metadata
                                     is taken from the first successfully processed file.
                                     Returns None if no files are successfully processed.
    """
    reflectivity_data_list: List[np.ndarray] = []
    timestamps_list: List[datetime] = []
    consolidated_metadata: Optional[Dict[str, Any]] = None

    for file_path in fmt_file_paths:
        try:
            # read_cinrad_reflectivity now raises exceptions on failure
            result = read_cinrad_reflectivity(file_path)
            # If result is None, it means an error handled within read_cinrad_reflectivity (though current impl re-raises)
            # This check is more for if read_cinrad_reflectivity was modified to return None on error
            if result is None: 
                print(f"Warning: Skipping file {file_path} due to read error (returned None).")
                continue

            cartesian_ds, cinrad_file_obj = result
            
            if 'REF' not in cartesian_ds:
                print(f"Warning: 'REF' data not found in {file_path}. Skipping.")
                continue

            data_array = cartesian_ds['REF'].values.astype(np.float32) # Ensure float32 for pysteps
            
            # It's good practice to fill NaNs, pysteps might expect this.
            # Use zerovalue from metadata, but metadata isn't fully formed yet.
            # We'll use a placeholder and then use the one from consolidated_metadata later if needed.
            # For now, let's assume -15.0 is a reasonable fill value for dBZ.
            data_array = np.nan_to_num(data_array, nan=-15.0) 

            reflectivity_data_list.append(data_array)
            
            current_metadata = extract_pysteps_metadata_from_cinrad(cartesian_ds, cinrad_file_obj)
            timestamps_list.append(current_metadata['timestamps'][0])

            if consolidated_metadata is None:
                consolidated_metadata = current_metadata

        except FileNotFoundError:
            print(f"Warning: File not found {file_path}. Skipping.")
        except ValueError as ve:
            print(f"Warning: Value error processing {file_path}: {ve}. Skipping.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred processing {file_path}: {e}. Skipping.")

    if not reflectivity_data_list:
        print("Error: No valid CINRAD files were processed.")
        return None, None

    try:
        reflectivity_stack = np.stack(reflectivity_data_list, axis=0)
    except ValueError as e:
        print(f"Error: Could not stack reflectivity data. Arrays might have inconsistent shapes. {e}")
        # Check shapes for debugging
        # for i, arr in enumerate(reflectivity_data_list):
        #    print(f"Shape of array {i} from file {fmt_file_paths[i] if i < len(fmt_file_paths) else 'unknown'}: {arr.shape}")
        return None, None


    if consolidated_metadata is not None:
        consolidated_metadata["timestamps"] = sorted(timestamps_list) # Keep timestamps sorted
        # Ensure zerovalue used for np.nan_to_num is consistent with metadata
        # This requires re-processing the stack if the first file's zerovalue was different,
        # or ensuring all files use the same processing. For simplicity, we assume
        # the initial nan_to_num was sufficient or that zerovalue is consistent.
        # If data was filled with -15.0 and metadata['zerovalue'] is different,
        # it might be an issue for pysteps.
        # A more robust solution would be to collect all data, then fill NaNs based on final metadata.
        # For now, we assume the -15.0 fill is okay or that metadata['zerovalue'] will be -15.0.
        
        # If we want to ensure the fill value matches the metadata:
        fill_value = consolidated_metadata.get("zerovalue", -15.0)
        # Check if any element in the stack is the original NaN fill value (-15.0) and needs re-filling
        # This is tricky because -15.0 could also be a valid data point.
        # The np.nan_to_num should ideally happen AFTER consolidated_metadata is set,
        # or ensure that the default fill value matches the one in metadata.
        # For now, we will assume the initial fill was using the correct zerovalue.
        # If reflectivity_stack still contains NaNs (if astype(np.float32) re-introduced them from original NaNs),
        # we should fill them here.
        reflectivity_stack = np.nan_to_num(reflectivity_stack, nan=fill_value)


    return reflectivity_stack, consolidated_metadata

if __name__ == '__main__':
    print("cinrad_utils.py updated. Contains `read_cinrad_reflectivity`, `extract_pysteps_metadata_from_cinrad`, and `load_fmt_time_series`.")
    # Example usage for load_fmt_time_series (requires sample files and os module):
    # import os
    #
    # # Create dummy files for testing if they don't exist
    # # This requires a more complex setup to create valid dummy CINRAD-like files.
    # # For now, assume actual files or more robust test setup.
    #
    # # sample_files_dir = "sample_cinrad_data"
    # # os.makedirs(sample_files_dir, exist_ok=True)
    # # file_paths_to_test = [os.path.join(sample_files_dir, f"sample_radar_echo_{i:02d}.fmt") for i in range(3)]
    #
    # # Create dummy files (VERY simplified, not real CINRAD files)
    # # for fp in file_paths_to_test:
    # #     with open(fp, "w") as f:
    # #         f.write("This is a dummy CINRAD file.") # Pycinrad will fail to read this.
    #
    # # To test this properly, you need actual or mock CINRAD files.
    # # print(f"\nTesting load_fmt_time_series with (mocked/non-existent) files: {file_paths_to_test}")
    # # R_stack, R_meta = load_fmt_time_series(file_paths_to_test)
    #
    # # if R_stack is not None and R_meta is not None:
    # #     print(f"Successfully loaded time series data. Shape: {R_stack.shape}")
    # #     print("Metadata for the series:")
    # #     for key, value in R_meta.items():
    # #         if key == "timestamps":
    # #             print(f"  {key}: {len(value)} timestamps, first: {value[0]}, last: {value[-1] if len(value) > 0 else 'N/A'}")
    # #         else:
    # #             print(f"  {key}: {value}")
    # # else:
    # #     print("Failed to load time series data.")
    #
    # # Example of how to use the previously defined functions (from before)
    # # try:
    # #     # Replace "path_to_your_sample_cinrad_file.fmt" with an actual file path for testing
    # #     sample_file = "path_to_your_sample_cinrad_file.fmt" 
    # #     if not os.path.exists(sample_file):
    # #         print(f"\nSample file {sample_file} not found. Skipping single file example run.")
    # #     else:
    # #         print(f"\nAttempting to process single file: {sample_file}")
    # #         cart_data, radar_file_obj = read_cinrad_reflectivity(sample_file) # Assuming this doesn't return None on success
    # #         if cart_data and radar_file_obj: # Check if read was successful
    # #             print("Successfully processed radar data with read_cinrad_reflectivity.")
    # #             print("Cartesian Data (REF) snippet:")
    # #             print(cart_data['REF'].isel(latitude=slice(0,5), longitude=slice(0,5)))
    # #            
    # #             metadata = extract_pysteps_metadata_from_cinrad(cart_data, radar_file_obj)
    # #             print("\nSuccessfully extracted pysteps metadata:")
    # #             for key, value in metadata.items():
    # #                 print(f"  {key}: {value}")
    # # except FileNotFoundError as e:
    # #     print(f"Error in single file example: {e}")
    # # except ValueError as e:
    # #     print(f"Error processing CINRAD data in single file example: {e}")
    # # except Exception as e:
    # #     print(f"An unexpected error occurred in single file example: {e}")
    pass
