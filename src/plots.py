"""
plots.py

This file contains functions for plotting data.

@Author: Ofir Paz
@Version: 14.08.2024
"""


# ================================== Imports ================================= #
from pathlib import Path
from typing import Any, Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import zoom
from src.dataset.spine_dataset import load_dicom_series
# ============================== End Of Imports ============================== #


# ============================= Helper Functions ============================= #
def prepare_volume_for_visualization(volume: np.ndarray) -> np.ndarray:
    """
    Prepare a 3D volume for visualization by normalizing it to the range [0, 255].

    Args:
        volume (np.ndarray): The 3D image to prepare.

    Returns:
        np.ndarray: The prepared 3D image.
    """
    volume = volume.astype(np.float32)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume)) * 255
    
    return volume


def interpolate_volume(
        volume: np.ndarray, 
        shape: Tuple[int, int, int] = (50, 180, 180), 
        order: int = 5
    ) -> np.ndarray:
    """
    Interpolate a 3D volume to a fixed shape.

    Args:
        volume (np.ndarray): The 3D volume to interpolate.
        shape (Tuple[int, int, int]): The fixed shape to interpolate to.
        order (int): The order of the spline interpolation.

    Returns:
        np.ndarray: The interpolated 3D volume.
    """
    zoom_factors = [t / s for t, s in zip(shape, volume.shape)]
    interpolated_volume = zoom(volume, zoom_factors, order=order)

    return interpolated_volume


def frame_args(duration: float) -> dict[str, Any]:
    """
    Create frame arguments for an animation.

    Args:
        duration (float): The duration of the frame.

    Returns:
        dict[str, Any]: The frame arguments
    """
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }


def make_3d_dicom_plot(
        volume: np.ndarray, 
        num_layers: Optional[int] = None,
        width: int = 800,
        height: int = 600,
        plot_title = 'Slices in volumetric DICOM data'
    ) -> None:
    """
    Make an interactive 3D plot of a DICOM series.

    Args:
        volume (np.ndarray): The 3D volume to plot.
        num_layers (Optional[int]): The number of layers in the volume.
        width (int): The width of the plot.
        height (int): The height of the plot.
        plot_title (string): The plot title
    """
    if num_layers is None:
        num_layers = volume.shape[0]

    nb_frames, r, c = volume.shape

    # Define frames for the animation.
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=(num_layers - 0.1 - k * (num_layers / nb_frames)) * np.ones((r, c)),
        surfacecolor=volume[nb_frames - 1 - k].T,
        cmin=0, cmax=255  # Adjust cmin and cmax to match your normalized volume
        ),
        name=str(k)  # Name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])

    # Add initial data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(num_layers - 0.1) * np.ones((r, c)),
        surfacecolor=volume[nb_frames - 1].T,
        colorscale='Gray',
        cmin=0, cmax=255,  # Adjust to match the intensity range of the volume
        #colorbar=dict(thickness=20, ticklen=4)
        ))
    
    # Slider configuration
    sliders = [
                {
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],  # type: ignore
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout configuration
    fig.update_layout(
            title=plot_title,
            width=width,
            height=height,
            scene=dict(
                        zaxis=dict(range=[-.1, num_layers + 0.1], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(15)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )

    # Display the interactive 3D animation
    fig.show()
# ========================== End Of Helper Functions ========================= #


# ============================== Plots Functions ============================= #
def plot_pixel_array(
        pixel_array: np.ndarray,
        title: str = "Example DICOM Image",
        figsize: Tuple[int, int] = (6, 6)
    ) -> None:
    """
    Plot a DICOM pixel array.

    Args:
        pixel_array (np.ndarray): The pixel array to plot.
        title (str): The title of the plot.
    """
    plt.figure(figsize=figsize)
    plt.imshow(pixel_array, cmap='gray')
    plt.title(title)
    plt.show()


def plot_dicom_series(
        path: Union[str, Path], 
        interp_shape: Tuple[int, int, int] = (50, 180, 180),
        interpol_order: int = 5,
        width: int = 800,
        height: int = 600,
        plot_title = 'Slices in volumetric DICOM data'
    ) -> None:
    """
    Plot a DICOM series from a directory.

    Args:
        path (Union[str, Path]): The directory containing the DICOM series.
        interp_shape (Tuple[int, int, int]): The shape to interpolate the volume to.
        interpol_order (int): The order of the spline interpolation.
        width (int): The width of the plot.
        height (int): The height of the plot.
        plot_title (string): The plot title
    """
    volume = load_dicom_series(path)
    volume = prepare_volume_for_visualization(volume)
    interpolated_volume = interpolate_volume(volume, interp_shape, interpol_order)
    make_3d_dicom_plot(interpolated_volume, volume.shape[0], width, height, plot_title)
# ========================== End Of Plots Functions ========================== #
