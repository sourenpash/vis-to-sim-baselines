# Tools and Utilities

## Data Tools

### Zarr Image Resizing

Location: `scripts/data_tools/resize_zarr_images.py`

This utility script resizes image or depth arrays stored in zarr format datasets using Hydra configuration. It supports batch processing of multiple datasets with regex pattern matching and parallel processing capabilities.

#### Usage

```bash
# Using default configuration
python scripts/data_tools/resize_zarr_images.py

# Specify zarr file(s)
python scripts/data_tools/resize_zarr_images.py zarr_paths=/path/to/dataset.zarr

# Process multiple files
python scripts/data_tools/resize_zarr_images.py zarr_paths=[dataset1.zarr,dataset2.zarr]

# Override resize dimensions
python scripts/data_tools/resize_zarr_images.py resize.height=128 resize.width=128

# Adjust parallel processing
python scripts/data_tools/resize_zarr_images.py parallel.max_workers=16
```

#### Configuration

The tool uses Hydra configuration. You can override parameters directly via command line. Key parameters:

```yaml
# Target dimensions
resize:
  height: 256
  width: 256

# Parallel processing
parallel:
  enabled: true
  max_workers: 4

# Dataset mappings with regex support
datasets:
  # Resize all camera views
  - input_pattern: ^/data/obs/images/camera_(\d+)$
    output_pattern: /data/obs/images_resized/camera_\1
  
  # Resize depth data
  - input_pattern: ^/data/obs/depth/camera_(\d+)$
    output_pattern: /data/obs/depth_resized/camera_\1
```

#### Examples

```bash
# Resize all camera images to 128x128 using regex patterns
python scripts/data_tools/resize_zarr_images.py \
    zarr_paths=/path/to/dataset.zarr \
    resize.height=128 \
    resize.width=128

# Process multiple datasets in parallel
python scripts/data_tools/resize_zarr_images.py \
    zarr_paths=[train.zarr,val.zarr,test.zarr] \
    parallel.max_workers=8 \
    resize.height=224 \
    resize.width=224

# Use custom configuration
python scripts/data_tools/resize_zarr_images.py \
    --config-name=my_resize_config
```

#### Technical Details

- **Regex Pattern Matching**: Automatically finds and processes all matching keys in zarr files
- **Parallel Processing**: Configurable multi-process support for handling multiple datasets
- **Interpolation Methods**:
  - **Depth data**: Uses `INTER_NEAREST` to preserve depth values
  - **RGB images**: Uses `INTER_AREA` for downsampling and `INTER_CUBIC` for upsampling
- **Chunk Processing**: Configurable chunk size (default: 10 frames) for memory efficiency
- **Progress Tracking**: Real-time progress bars with detailed processing statistics

<!-- ---

*Additional tools and utilities will be documented here as they are developed.* -->