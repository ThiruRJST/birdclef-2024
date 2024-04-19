from librosa.feature import melspectrogram
import librosa
import numpy as np
from pydub import AudioSegment as AS
import IPython
import IPython.display as ipd
from dataclasses import dataclass
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import os
from PIL import Image

train_data = pd.read_csv("train_metadata.csv")


@dataclass
class cfg:
    MAXLEN: int = 1000000
    CHUNK_SIZE: int = 1000000
    NMELS: int = 256
    sample_audio_path: str = f"train_audio/{train_data.loc[0, 'filename']}"


# Helper Functions


def normalize(x):
    return np.float32(x) / 2**15


def read(file, norm=False):
    try:
        a = AS.from_ogg(file)
    except:
        return np.zeros(cfg.MAXLEN)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if norm:
        return a.frame_rate, normalize(y)
    if not norm:
        return a.frame_rate, np.float32(y)


def to_imagenet(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = (V - norm_min) / (norm_max - norm_min)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return np.stack([V] * 3, axis=-1)


def create_melspec(path: tuple):
    spec_fname = "/".join(path[0].split("/")[-2:])
    save_path = f"{path[1]}/{spec_fname.replace('.ogg', '.png')}"
    sr, y = read(path[0])
    y = librosa.util.fix_length(y, size=cfg.MAXLEN, mode="edge")
    melspec = melspectrogram(y=y, n_mels=cfg.NMELS)
    melspec = Image.fromarray(librosa.power_to_db(melspec).astype(np.uint8))
    melspec.save(f"{save_path}")


if __name__ == "__main__":
    # create the species folders
    species = [x.strip() for x in open("labels.txt", "r").readlines()]

    ROOT = "/home/tensorthiru/CLEF/birdclef-2024/melspecs"
    for s in species:
        if not os.path.exists(f"{ROOT}/{s}"):
            os.makedirs(f"{ROOT}/{s}")

    audio_paths = [
        (f"/home/tensorthiru/CLEF/birdclef-2024/train_audio/{x}", ROOT)
        for x in train_data.filename.values
    ]
    p = Pool(16)
    with p:
        for i in tqdm(
            p.imap_unordered(create_melspec, audio_paths), total=len(audio_paths)
        ):
            pass
