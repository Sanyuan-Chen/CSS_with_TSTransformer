#!/usr/bin/env python

import pathlib
import argparse
import torch as th
import numpy as np
import yaml
from nnet import support_nnet as support_sep_nnet
from feature_transform import support_transform
from utils.audio_util import write_wav, WaveListReader
from utils.mvdr_util import make_mvdr


class Separator(object):
    def __init__(self, cpt_dir, device_id=-1):
        # load nnet
        self.epoch, self.nnet, self.conf = self._load(cpt_dir)
        # offload to device
        if device_id < 0:
            self.device = th.device("cpu")
        else:
            self.device = th.device(f"cuda:{device_id:d}")
            self.nnet.to(self.device)
        # set eval model
        self.nnet.eval()

        print(f"Load checkpoint of {self.conf['nnet']} from {cpt_dir}: " + f"epoch {self.epoch}, device_id = {device_id}")
        num_params = sum([param.nelement() for name, param in self.nnet.named_parameters() if 'enh_transform' not in name]) / 10.0 ** 6
        print(f"#param: {num_params:.2f}M")

    def _load(self, cpt_dir):
        cpt_dir = pathlib.Path(cpt_dir)
        # load checkpoint
        cpt = th.load(cpt_dir / "best.pt.tar", map_location="cpu")
        with open(cpt_dir / "train.yaml", "r") as f:
            conf = yaml.full_load(f)
            net_cls = support_sep_nnet(conf["nnet"])
        enh_transform = None
        if "enh_transform" in conf:
            conf["enh_transform"]["center"] = True
            enh_transform = support_transform("enh")(**conf["enh_transform"])
        if enh_transform:
            nnet = net_cls(enh_transform=enh_transform, **conf["nnet_conf"])
        else:
            nnet = net_cls(**conf["nnet_conf"])

        nnet.load_state_dict(cpt["model_state_dict"])
        return cpt["epoch"], nnet, conf

    def run(self, src, output_mask=False):
        """
        Args:
            src (Array): (C) x S
        """
        src = th.from_numpy(src).to(self.device)
        return self.nnet.infer(src, mode="freq" if output_mask else "time")


def run(args):
    separator = Separator(args.checkpoint, device_id=args.device_id)
    mix_reader = WaveListReader(args.wav_list,
                                sr=args.sr,
                                channel=args.channel)
    sep_dir = pathlib.Path(args.sep_dir)

    print(f"Start Separation " + ("w/ mvdr" if args.mvdr else "w/o mvdr"))
    for key, mix in mix_reader:
        print(f"Processing utterance {key}...")

        if args.mvdr == 'true':
            masks = separator.run(mix, output_mask=True)
            masks = [m.cpu().numpy() for m in masks]
            sep = make_mvdr(mix.T, masks)
            sep = [s * 0.9 / np.max(np.abs(s)) for s in sep]
        else:
            sep = separator.run(mix)
            sep = [s.cpu().numpy() for s in sep]
            sep = [s * 0.9 / np.max(np.abs(s)) for s in sep]

        for i in range(args.num_spks):
            write_wav(str(sep_dir / f"{key}_{i}.wav"), sep[i], sr=args.sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do speech separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the separation model")
    parser.add_argument("--wav_list",
                        type=str,
                        help="Mixture input wave list")
    parser.add_argument("--sep_dir",
                        type=str,
                        help="Directory to dump separated output")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the source audio")
    parser.add_argument("--channel",
                        type=int,
                        default=-1,
                        help="Channel index for source audio")
    parser.add_argument("--num_spks",
                        type=int,
                        default=2,
                        help="Number of the speakers")
    parser.add_argument("--mvdr",
                        type=str,
                        choices=["true", "false"],
                        default="false",
                        help="If true, use mvdr")
    args = parser.parse_args()
    run(args)
