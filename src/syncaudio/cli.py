import argparse
import datetime

from .impl import synchronize
from .util import read_audio


def run():
    parser = argparse.ArgumentParser(description="Synchronize two audio files.")
    parser.add_argument("self", help="Path to the first audio file.")
    parser.add_argument("other", help="Path to the second audio file.")
    args = parser.parse_args()

    self_audio, sample_rate = read_audio(args.self)
    other_audio, other_sample_rate = read_audio(args.other)
    assert sample_rate == other_sample_rate, "Sample rates must be the same."
    delay = synchronize(self_audio, other_audio)

    seconds = round(delay / (sample_rate / 1024), 4)
    term = "ahead" if seconds > 0 else "behind"
    print(f'"self" is {datetime.timedelta(seconds=abs(seconds))} {term} of "other".')
