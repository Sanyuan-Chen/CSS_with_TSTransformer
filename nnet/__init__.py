from .transformers import FreqTransformer


nnet_cls = {
    "transformer": FreqTransformer,
}


def support_nnet(nnet_type):
    if nnet_type not in nnet_cls:
        raise RuntimeError(f"Unsupported network type: {nnet_type}")
    return nnet_cls[nnet_type]
