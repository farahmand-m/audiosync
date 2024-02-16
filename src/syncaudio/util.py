import numpy as np
from scipy.io import wavfile


def read_audio(audio_file):
    """
    Reads a WAV file and returns the audio data (as a 1D vector) and the sample rate.

    Parameters
    ----------
    audio_file: str
        Path to the WAV file.

    Returns
    -------
    numpy.ndarray
        Audio signal.
    int
        Sample rate.
    """
    sample_rate, data = wavfile.read(audio_file)  # Return the sample sample_rate (in samples/sec) and data from a WAV file
    return np.mean(data, axis=1).astype(np.float32), sample_rate
