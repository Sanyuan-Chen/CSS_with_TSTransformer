import os
import numpy as np
import soundfile as sf
import pathlib
import warnings


def suffix_of_path(path):
    """
    Return unique index from a file path
    """
    fname = os.path.basename(path)
    if not fname:
        raise ValueError(f"{path}: is directory path?")
    token = fname.split(".")
    if len(token) == 1:
        return token[0]
    else:
        return '.'.join(token[:-1])


def read_wav(fname, beg=0, end=None, normalize=True, transpose=True, sr=16000):
    """
    Read wave files using soundfile (support multi-channel)
    Args:
        fname: file name or object
        beg, end: begin and end index for chunk-level reading
        normalize: normalized samples between -1 and 1
        sr: sample rate
    Return:
        samps: in shape C x N
    """
    # samps: N x C or N
    #   N: number of samples
    #   C: number of channels
    samps, wav_sr = sf.read(fname,
                            start=beg,
                            stop=end,
                            dtype="float32" if normalize else "int16")
    if wav_sr != sr:
        raise ValueError(f"sr mismatch: {sr} vs {wav_sr}")
    if not normalize:
        samps = samps.astype("float32")
    # put channel axis first
    # N x C => C x N
    if samps.ndim != 1 and transpose:
        samps = np.transpose(samps)
    return samps


class WaveReader(object):
    """
    WaveReader class
    """
    def __init__(self, wav_stats_dict, sr=16000, norm=True, channel=-1):
        self.wav_stats_dict = wav_stats_dict
        self.sr, self.norm = sr, norm
        self.channel = channel

    def __len__(self):
        return len(self.wav_stats_dict)

    def _load(self, addr):
        wav = read_wav(addr, normalize=self.norm, sr=self.sr)
        if self.channel >= 0 and wav.ndim == 2:
            wav = wav[self.channel]
        return wav

    def __iter__(self):
        for key, addr in self.wav_stats_dict.items():
            yield key, self._load(addr)


class WaveListReader(WaveReader):
    """
    Wave reader from a wave list
    """
    def __init__(self, wav_list, sr=16000, norm=True, channel=-1):
        wav_stats_dict = {}
        wav_list = pathlib.Path(wav_list)
        with open(str(wav_list), "r") as wl:
            for raw_line in wl:
                addr = raw_line.strip()
                suffix = suffix_of_path(addr)
                suffix = os.path.join(os.path.basename(os.path.dirname(addr)), suffix)
                if suffix in wav_stats_dict:
                    warnings.warn(
                        f"Seems duplicated utterance exists: {raw_line}")
                wav_stats_dict[suffix] = addr
        super(WaveListReader, self).__init__(wav_stats_dict,
                                             sr=sr,
                                             norm=norm,
                                             channel=channel)


def write_wav(fname, samps, sr=16000, normalize=True):
    """
    Write wav files, support single/multi-channel
    Args:
        fname: file name
        samps: Numpy's ndarray
        sr: sample rate
        normalize: normalize used in read_wav
    """
    samps = samps.astype("float32" if normalize else "int16")
    # for multi-channel, accept ndarray [num_samples, num_channels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # make dirs
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    sf.write(fname, samps, sr)

