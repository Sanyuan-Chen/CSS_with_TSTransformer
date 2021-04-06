from .feature_transform import FeatureTransform

trans_cls = {"enh": FeatureTransform}


def support_transform(trans_type):
    if trans_type not in trans_cls:
        raise RuntimeError(f"Unsupported transform type: {trans_type}")
    return trans_cls[trans_type]
