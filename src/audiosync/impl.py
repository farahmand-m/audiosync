import heapq

import numpy as np


def fourier_magnitudes(signal):
    """
    Calculates the magnitudes of the Fourier transform of a signal.

    The function only returns the magnitudes of the first half of the Fourier transform, since the second half is
    symmetric to the first half for real-valued signals.

    Parameters
    ----------
    signal: array_like
        Audio signal.

    Returns
    -------
    numpy.ndarray
        Magnitudes of the Fourier transform.
    """
    fft = np.fft.fft(signal)
    length = len(fft)
    mag = np.abs(fft)
    mag = mag[:length // 2]  # Real-valued signal; spectrum is symmetric.
    mag = np.round(mag, 2)
    return mag


def rolling_window(arr, window_size, overlap, pad=True):
    """
    Creates a rolling window view of an array.

    Parameters
    ----------
    arr: array_like
        Input array.
    window_size: int
        Size of the window.
    overlap: int
        Number of data points to overlap between windows.
    pad: bool
        Whether to pad the array to ensure that the last window is the same size as the others.

    Returns
    -------
    numpy.ndarray
        A view of the input array with shape (num_windows, window_size).
    int
        The number of windows.
    """
    length = len(arr)
    step_size = window_size - overlap
    num_windows = (length - window_size) // step_size + 1
    if pad:
        padding = max(0, (num_windows - 1) * step_size + window_size - len(arr))
        arr = np.pad(arr, (0, padding), constant_values=0)
    num_bytes, = arr.strides
    shape = (num_windows, window_size)
    strides = (step_size * num_bytes, num_bytes)
    rolled_view = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return rolled_view, num_windows


def to_peaks(signal, window_size, overlap, spectral_band, temporal_band, peaks_per_bin):
    """
    Bins the signal and finds the peaks in each bin.

    Parameters
    ----------
    signal: array_like
        The audio signal.
    window_size: int
        The number of data points in each window.
    overlap: int
        The number of data points to overlap between windows.
    spectral_band: int
        Size of the spectral band in each bin.
    temporal_band: int
        Size of the temporal band in each bin.
    peaks_per_bin: int
        The number of peaks to find in each bin.

    Returns
    -------
    dict
        A dictionary where the keys are spectro-temporal bins and the values are tuples, containing the intensity of
        the frequency at that bin, the window number (temporal), and its frequency index (spectral).
    """
    # Calculate the magnitudes of the signal's Fourier transform.
    windows, num_windows = rolling_window(signal, window_size, overlap, pad=True)
    magnitudes = np.apply_along_axis(fourier_magnitudes, axis=1, arr=windows)
    windows_indices = np.arange(num_windows)
    # Create spectro-temporal bins.
    bins = {}
    for window_index, values in zip(windows_indices, magnitudes):
        for frequency_index, value in enumerate(values):
            tup = (value, window_index, frequency_index)
            bin = (window_index // temporal_band, frequency_index // spectral_band)
            bins.setdefault(bin, []).append(tup)
    # Identify the peaks in each bin.
    peaks = {}
    for bin, bin_magnitudes in bins.items():
        max_intensities = heapq.nlargest(peaks_per_bin, bin_magnitudes, key=lambda tup: tup[0])
        for value, window_index, frequency_index in max_intensities:
            peaks.setdefault(frequency_index, []).append(window_index)
    return peaks


def synchronize(base_signal, other_signal, window_size=1024, overlap=0, spectral_band=512, temporal_band=43, peaks_per_bin=7):
    """
    Finds the delay between two audio signals.

    Based on Allison Deal's algorithm, available at https://github.com/allisonnicoledeal/VideoSync.

    Parameters
    ----------
    base_signal: array_like
        The first audio signal.
    other_signal: array_like
        The second audio signal.
    window_size: int
        The number of data points in each window.
    overlap: int
        The number of data points to overlap between windows.
    spectral_band: int
        Size of the spectral band in each bin.
    temporal_band: int
        Size of the temporal band in each bin.
    peaks_per_bin: int
        The number of peaks to find in each bin.

    Returns
    -------
    int
        The number of samples by which the second signal is delayed relative to the first.
    """
    base_peaks = to_peaks(base_signal, window_size, overlap, spectral_band, temporal_band, peaks_per_bin)
    other_peaks = to_peaks(other_signal, window_size, overlap, spectral_band, temporal_band, peaks_per_bin)
    # Calculate the differences between peaks in each frequency.
    pairs = []
    for key in other_peaks:
        if key in base_peaks:
            pairs.extend((a, b) for a in other_peaks[key] for b in base_peaks[key])
    # Find the most common difference between peaks.
    frequencies = {}
    for a, b in pairs:
        frequencies[a - b] = frequencies.get(a - b, 0) + 1
    return max(frequencies, key=frequencies.get)
