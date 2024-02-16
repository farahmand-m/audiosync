# AudioSync

This is a simple tool to synchronize audio files. The tool is based on Allison Deal's [algorithm](https://github.com/allisonnicoledeal/VideoSync).

![Allison Deal's Audio Sync Algorithm](https://github.com/allisonnicoledeal/VideoSync/raw/master/screenshots/diagram.png)

## Installation

```bash
pip install audiosync
```

## Usage

```bash
audiosync audio1.wav audio2.wav
```

## API

```python
from audiosync import synchronize
from audiosync import read_audio

self = read_audio('audio1.wav')
other = read_audio('audio2.wav')

delay = synchronize(self, other, window_size=1024, overlap=0, spectral_band=512, temporal_band=43, peaks_per_bin=7)

print(delay, 'seconds')
```
