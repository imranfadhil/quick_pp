#!/usr/bin/env python3
"""
SEG-Y File Diagnostic and Plotting Module
This module provides a class to diagnose and fix "Inlines inconsistent" errors in SEG-Y files.
"""

import segyio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Uncomment to run module as a standalone script
import os
import sys
sys.path.append(os.getcwd())

from quick_pp import logger


class SEGYDiagnostic:
    """
    A class for diagnosing and plotting SEG-Y files.

    This class provides methods to:
    - Diagnose SEG-Y file geometry issues
    - Analyze inline/crossline geometry
    - Try different field combinations
    - Plot seismic data
    - Generate diagnostic reports
    """

    def __init__(self, segy_file_path: str):
        """
        Initialize the SEGYDiagnostic with a SEG-Y file path.

        Args:
            segy_file_path (str): Path to the SEG-Y file
        """
        self.segy_file_path = Path(segy_file_path)
        self.file_exists = self.segy_file_path.exists()
        self.diagnostic_results = {}
        self.geometry_data = {}

    def check_file_exists(self) -> bool:
        """Check if the SEG-Y file exists."""
        if not self.file_exists:
            logger.error(f"File not found: {self.segy_file_path}")
        return self.file_exists

    def diagnose_file(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive diagnosis of the SEG-Y file.

        Args:
            verbose (bool): Whether to print diagnostic information

        Returns:
            Dict containing diagnostic results
        """
        if verbose:
            logger.info(f"Diagnosing SEG-Y file: {self.segy_file_path}")
            logger.info("=" * 60)

        if not self.check_file_exists():
            return {"success": False, "error": "File not found"}

        results = {
            "file_path": str(self.segy_file_path),
            "file_exists": True,
            "basic_info": self._get_basic_info(),
            "geometry_analysis": self._analyze_geometry(),
            "field_combinations": self._try_field_combinations(),
            "data_access": self._test_data_access()
        }

        self.diagnostic_results = results

        if verbose:
            self._print_diagnostic_summary(results)

        return results

    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the SEG-Y file."""
        try:
            with segyio.open(self.segy_file_path, ignore_geometry=True) as f:
                info = {
                    "num_traces": len(f.trace),
                    "num_samples": len(f.samples),
                    "sample_interval": f.bin[segyio.BinField.Interval],
                    "success": True
                }

                # Get first few trace headers
                trace_headers = []
                for i in range(min(5, len(f.trace))):
                    inline_val = f.header[i][segyio.tracefield.TraceField.INLINE_3D]
                    xline_val = f.header[i][segyio.tracefield.TraceField.CROSSLINE_3D]
                    trace_headers.append({
                        "trace": i,
                        "inline": inline_val,
                        "crossline": xline_val
                    })
                info["sample_headers"] = trace_headers

                return info
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_geometry(self) -> Dict[str, Any]:
        """Analyze the geometry of the SEG-Y file."""
        try:
            with segyio.open(self.segy_file_path, ignore_geometry=True) as f:
                inlines = []
                xlines = []

                for i in range(len(f.trace)):
                    inline_val = f.header[i][segyio.tracefield.TraceField.INLINE_3D]
                    xline_val = f.header[i][segyio.tracefield.TraceField.CROSSLINE_3D]
                    inlines.append(inline_val)
                    xlines.append(xline_val)

                inlines = np.array(inlines)
                xlines = np.array(xlines)

                analysis = {
                    "total_traces": len(f.trace),
                    "unique_inlines": len(np.unique(inlines)),
                    "unique_crosslines": len(np.unique(xlines)),
                    "inline_range": [int(inlines.min()), int(inlines.max())],
                    "crossline_range": [int(xlines.min()), int(xlines.max())],
                    "has_duplicate_inlines": len(np.unique(inlines)) != len(inlines),
                    "has_duplicate_crosslines": len(np.unique(xlines)) != len(xlines),
                    "success": True
                }

                # Find duplicate inlines
                if analysis["has_duplicate_inlines"]:
                    unique_inlines, counts = np.unique(inlines, return_counts=True)
                    duplicates = unique_inlines[counts > 1]
                    analysis["duplicate_inlines"] = duplicates.tolist()[:10]  # First 10

                # Find duplicate crosslines
                if analysis["has_duplicate_crosslines"]:
                    unique_xlines, counts = np.unique(xlines, return_counts=True)
                    duplicates = unique_xlines[counts > 1]
                    analysis["duplicate_crosslines"] = duplicates.tolist()[:10]  # First 10

                self.geometry_data = {"inlines": inlines, "xlines": xlines}
                return analysis

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _try_field_combinations(self) -> Dict[str, Any]:
        """Try different inline/crossline field combinations."""
        field_combinations = [
            ("INLINE_3D", "CROSSLINE_3D",
             segyio.tracefield.TraceField.INLINE_3D,
             segyio.tracefield.TraceField.CROSSLINE_3D),
            ("TRACE_SEQUENCE_FILE", "TRACE_SEQUENCE_FILE",
             segyio.tracefield.TraceField.TRACE_SEQUENCE_FILE,
             segyio.tracefield.TraceField.TRACE_SEQUENCE_FILE),
            ("CDP", "CDP_TRACE",
             segyio.tracefield.TraceField.CDP,
             segyio.tracefield.TraceField.CDP_TRACE),
        ]

        results = {"successful_combinations": [], "all_tested": []}

        for name1, name2, field1, field2 in field_combinations:
            try:
                with segyio.open(self.segy_file_path, iline=field1, xline=field2) as f:
                    combination_result = {
                        "inline_field": name1,
                        "crossline_field": name2,
                        "num_inlines": len(f.iline_numbers),
                        "num_crosslines": len(f.xline_numbers),
                        "inline_numbers": f.iline_numbers.tolist(),
                        "crossline_numbers": f.xline_numbers.tolist(),
                        "success": True
                    }
                    results["successful_combinations"].append(combination_result)
                    results["all_tested"].append({
                        "inline_field": name1,
                        "crossline_field": name2,
                        "success": True
                    })
            except Exception as e:
                results["all_tested"].append({
                    "inline_field": name1,
                    "crossline_field": name2,
                    "success": False,
                    "error": str(e)[:100]
                })

        return results

    def _test_data_access(self) -> Dict[str, Any]:
        """Test data access capabilities."""
        try:
            with segyio.open(self.segy_file_path, ignore_geometry=True) as f:
                # Read first few traces to test access
                test_traces = np.array([f.trace[i] for i in range(min(10, len(f.trace)))])

                return {
                    "success": True,
                    "test_traces_shape": test_traces.shape,
                    "data_range": [float(test_traces.min()), float(test_traces.max())],
                    "data_mean": float(test_traces.mean()),
                    "data_std": float(test_traces.std())
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _print_diagnostic_summary(self, results: Dict[str, Any]):
        """Print a summary of diagnostic results."""
        logger.info("DIAGNOSTIC SUMMARY:")
        logger.info("=" * 60)

        # Basic info
        if results["basic_info"]["success"]:
            info = results["basic_info"]
            logger.info("✓ File opened successfully")
            logger.info(f"  Traces: {info['num_traces']}")
            logger.info(f"  Samples per trace: {info['num_samples']}")
            logger.info(f"  Sample interval: {info['sample_interval']} ms")

        # Geometry analysis
        if results["geometry_analysis"]["success"]:
            geom = results["geometry_analysis"]
            logger.info("Geometry Analysis:")
            logger.info(f"  Unique inlines: {geom['unique_inlines']}")
            logger.info(f"  Unique crosslines: {geom['unique_crosslines']}")

            if geom["has_duplicate_inlines"]:
                logger.warning("  ⚠️  WARNING: Duplicate inlines detected")
            if geom["has_duplicate_crosslines"]:
                logger.warning("  ⚠️  WARNING: Duplicate crosslines detected")

        # Field combinations
        field_results = results["field_combinations"]
        if field_results["successful_combinations"]:
            logger.info("✓ Working field combinations found:")
            for combo in field_results["successful_combinations"]:
                logger.info(f"  {combo['inline_field']} + {combo['crossline_field']}")
                logger.info(f"    Inlines: {combo['num_inlines']}, Crosslines: {combo['num_crosslines']}")
        else:
            logger.warning("  ⚠️  No working field combinations found")

        # Data access
        if results["data_access"]["success"]:
            data = results["data_access"]
            logger.info("✓ Data access successful")
            logger.info(f"  Data range: {data['data_range'][0]:.3f} to {data['data_range'][1]:.3f}")

    def plot_traces(self, num_traces: int = 20, save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (15, 10)) -> Optional[plt.Figure]:
        """
        Plot seismic traces from the SEG-Y file.

        Args:
            num_traces (int): Number of traces to plot
            save_path (str, optional): Path to save the plot
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib Figure object or None if failed
        """
        if not self.check_file_exists():
            return None

        try:
            with segyio.open(self.segy_file_path, ignore_geometry=True) as f:
                num_traces = min(num_traces, len(f.trace))

                fig, axes = plt.subplots(4, 5, figsize=figsize)
                axes = axes.flatten()

                for i in range(num_traces):
                    if i < len(axes):
                        axes[i].plot(f.trace[i])
                        axes[i].set_title(f'Trace {i}')
                        axes[i].grid(True, alpha=0.3)
                        axes[i].set_xlabel('Sample')
                        axes[i].set_ylabel('Amplitude')

                # Hide unused subplots
                for i in range(num_traces, len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    logger.info(f"Plot saved as '{save_path}'")

                return fig

        except Exception as e:
            logger.error(f"Error plotting traces: {e}")
            return None

    def plot_geometry_analysis(self, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot geometry analysis results.

        Args:
            save_path (str, optional): Path to save the plot

        Returns:
            matplotlib Figure object or None if failed
        """
        if not self.geometry_data:
            logger.warning("No geometry data available. Run diagnose_file() first.")
            return None

        try:
            inlines = self.geometry_data["inlines"]
            xlines = self.geometry_data["xlines"]

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Inline distribution
            axes[0, 0].hist(inlines, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('Inline Distribution')
            axes[0, 0].set_xlabel('Inline Number')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)

            # Crossline distribution
            axes[0, 1].hist(xlines, bins=50, alpha=0.7, color='red')
            axes[0, 1].set_title('Crossline Distribution')
            axes[0, 1].set_xlabel('Crossline Number')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)

            # Scatter plot of inline vs crossline
            axes[1, 0].scatter(inlines, xlines, alpha=0.5, s=1)
            axes[1, 0].set_title('Inline vs Crossline')
            axes[1, 0].set_xlabel('Inline Number')
            axes[1, 0].set_ylabel('Crossline Number')
            axes[1, 0].grid(True, alpha=0.3)

            # Statistics
            axes[1, 1].text(0.1, 0.9, f'Total Traces: {len(inlines)}',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.8, f'Unique Inlines: {len(np.unique(inlines))}',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.7, f'Unique Crosslines: {len(np.unique(xlines))}',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f'Inline Range: {inlines.min()} - {inlines.max()}',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.5, f'Crossline Range: {xlines.min()} - {xlines.max()}',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Statistics')
            axes[1, 1].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Geometry analysis plot saved as '{save_path}'")

            return fig

        except Exception as e:
            logger.error(f"Error plotting geometry analysis: {e}")
            return None

    def get_recommendations(self) -> List[str]:
        """
        Get recommendations based on diagnostic results.

        Returns:
            List of recommendation strings
        """
        if not self.diagnostic_results:
            return ["Run diagnose_file() first to get recommendations"]

        recommendations = []

        # Check if file can be opened with proper geometry
        field_results = self.diagnostic_results.get("field_combinations", {})
        if field_results.get("successful_combinations"):
            recommendations.append(
                "✓ File can be opened with proper geometry using alternative field combinations")
        else:
            recommendations.append(
                "⚠️  File has geometry issues but can be worked with using ignore_geometry=True")
            recommendations.append(
                "   This means you can access the data but may need to manually reconstruct the 3D geometry")

        # Check for duplicates
        geom_analysis = self.diagnostic_results.get("geometry_analysis", {})
        if (geom_analysis.get("has_duplicate_inlines") or geom_analysis.get("has_duplicate_crosslines")):
            recommendations.append("⚠️  Duplicate inline/crossline values detected - this may cause geometry issues")

        # General recommendations
        recommendations.extend([
            "- Use ignore_geometry=True for basic data access",
            "- Manually reconstruct 3D geometry if needed",
            "- Check with your data provider about the SEG-Y file format"
        ])

        return recommendations

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive diagnostic report.

        Args:
            output_file (str, optional): Path to save the report

        Returns:
            Report string
        """
        if not self.diagnostic_results:
            self.diagnose_file(verbose=False)

        report = []
        report.append("SEG-Y File Diagnostic Report")
        report.append("=" * 60)
        report.append(f"File: {self.segy_file_path}")
        report.append(f"Date: {Path().cwd()}")
        report.append("")

        # Basic info
        basic_info = self.diagnostic_results.get("basic_info", {})
        if basic_info.get("success"):
            report.append("BASIC INFORMATION:")
            report.append(f"  Number of traces: {basic_info['num_traces']}")
            report.append(f"  Samples per trace: {basic_info['num_samples']}")
            report.append(f"  Sample interval: {basic_info['sample_interval']} ms")
            report.append("")

        # Geometry analysis
        geom_analysis = self.diagnostic_results.get("geometry_analysis", {})
        if geom_analysis.get("success"):
            report.append("GEOMETRY ANALYSIS:")
            report.append(f"  Unique inlines: {geom_analysis['unique_inlines']}")
            report.append(f"  Unique crosslines: {geom_analysis['unique_crosslines']}")
            report.append(f"  Inline range: {geom_analysis['inline_range']}")
            report.append(f"  Crossline range: {geom_analysis['crossline_range']}")
            report.append(f"  Has duplicate inlines: {geom_analysis['has_duplicate_inlines']}")
            report.append(f"  Has duplicate crosslines: {geom_analysis['has_duplicate_crosslines']}")
            report.append("")

        # Field combinations
        field_results = self.diagnostic_results.get("field_combinations", {})
        report.append("FIELD COMBINATIONS TESTED:")
        for combo in field_results.get("all_tested", []):
            status = "✓" if combo.get("success") else "✗"
            report.append(f"  {status} {combo['inline_field']} + {combo['crossline_field']}")
            if not combo.get("success"):
                report.append(f"    Error: {combo.get('error', 'Unknown error')}")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        recommendations = self.get_recommendations()
        for rec in recommendations:
            report.append(f"  {rec}")

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_file}")

        return report_text


def main(file_objects):
    """Example usage of the SEGYDiagnostic class."""
    segy_file = file_objects[0].name

    # Create diagnostic object
    diagnostic = SEGYDiagnostic(segy_file)

    # Run comprehensive diagnosis
    diagnostic.diagnose_file()

    folder = Path(r'notebooks\outputs')
    # Plot traces
    diagnostic.plot_traces(num_traces=20, save_path=folder / 'seismic_traces.png')
    
    # Plot geometry analysis
    diagnostic.plot_geometry_analysis(save_path=folder / 'geometry_analysis.png')

    # Generate report
    diagnostic.generate_report(output_file=folder / 'segy_diagnostic_report.txt')

    # Print recommendations
    logger.info("RECOMMENDATIONS:")
    for rec in diagnostic.get_recommendations():
        logger.info(rec)


if __name__ == "__main__":
    from tkinter import Tk, filedialog

    root = Tk()
    file_objects = filedialog.askopenfiles(title='Choose SEG-Y files to be diagnosed',
                                           filetypes=(('SEG-Y Files', '*.segy'), ('All Files', '*.*')),
                                           mode='rb')
    root.destroy()
    if file_objects:
        # Test read_las_file function
        main(file_objects)
