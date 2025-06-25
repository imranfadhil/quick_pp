# SEG-Y Diagnostic Module

This module provides a comprehensive `SEGYDiagnostic` class for diagnosing and plotting SEG-Y files, particularly useful for identifying and resolving "Inlines inconsistent" errors.

## Features

- **Comprehensive Diagnosis**: Analyze SEG-Y file geometry, identify issues, and test different field combinations
- **Visualization**: Plot seismic traces and geometry analysis
- **Detailed Reporting**: Generate comprehensive diagnostic reports
- **Recommendations**: Get actionable recommendations based on diagnostic results

## Quick Start

```python
from segy_diagnostic import SEGYDiagnostic

# Create diagnostic object
diagnostic = SEGYDiagnostic('path/to/your/file.segy')

# Run comprehensive diagnosis
results = diagnostic.diagnose_file()

# Plot seismic traces
diagnostic.plot_traces(num_traces=20, save_path='traces.png')

# Plot geometry analysis
diagnostic.plot_geometry_analysis(save_path='geometry.png')

# Get recommendations
recommendations = diagnostic.get_recommendations()
for rec in recommendations:
    print(rec)
```

## Class Methods

### Core Methods

- `diagnose_file(verbose=True)`: Perform comprehensive diagnosis
- `check_file_exists()`: Verify file exists
- `get_recommendations()`: Get actionable recommendations

### Plotting Methods

- `plot_traces(num_traces=20, save_path=None, figsize=(15, 10))`: Plot seismic traces
- `plot_geometry_analysis(save_path=None)`: Plot geometry analysis

### Analysis Methods

- `_get_basic_info()`: Get basic file information
- `_analyze_geometry()`: Analyze inline/crossline geometry
- `_try_field_combinations()`: Test different field combinations
- `_test_data_access()`: Test data access capabilities

### Reporting Methods

- `generate_report(output_file=None)`: Generate detailed diagnostic report

## Example Usage

See `segy_example.py` for comprehensive examples including:

1. **Basic Diagnosis**: Simple file diagnosis
2. **Plotting**: Visualizing seismic data and geometry
3. **Advanced Usage**: Accessing specific diagnostic information
4. **Custom Analysis**: Performing custom analysis on geometry data

## Diagnostic Information

The `diagnose_file()` method returns a dictionary with:

- `basic_info`: Number of traces, samples, sample interval
- `geometry_analysis`: Inline/crossline statistics and duplicate detection
- `field_combinations`: Results of testing different field combinations
- `data_access`: Data access test results

## Common Issues and Solutions

### "Inlines inconsistent" Error

This error typically occurs when:
- Duplicate inline/crossline values exist
- Field combinations are incorrect
- File format is non-standard

**Solutions:**
1. Use `ignore_geometry=True` for basic data access
2. Try alternative field combinations (automatically tested)
3. Manually reconstruct 3D geometry if needed

### Duplicate Values

The diagnostic will detect and report:
- Duplicate inline values
- Duplicate crossline values
- Gaps in coverage

## Output Files

The module can generate:
- `seismic_traces.png`: Plot of seismic traces
- `geometry_analysis.png`: Geometry analysis plots
- `detailed_segy_report.txt`: Comprehensive diagnostic report

## Dependencies

- `segyio`: SEG-Y file reading
- `numpy`: Numerical operations
- `matplotlib`: Plotting
- `pathlib`: File path handling

## Running Examples

```bash
# Run the example script
python segy_example.py

# Or run the main diagnostic
python segy_diagnostic.py
```

## Notes

- The class automatically handles file existence checks
- Geometry analysis requires reading all trace headers (may be slow for large files)
- Plotting methods return matplotlib Figure objects for further customization
- All methods include comprehensive error handling 