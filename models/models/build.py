from utils.registry import Registry
import models
import yaml

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoder
"""

__all__ = ['get_model', 'build_encoder']

def get_model(args,trainset = None):
    num_classes=get_numclasses(args,trainset)
    print("=> Creating model '{}'".format(args.arch))
    print("Model Option")
    print(" 1) use_pretrained =", args.use_pretrained)
    print(" 2) No_transfer_learning =", args.No_transfer_learning)
    print(" 3) use_bn =", args.use_bn)
    print(" 4) use_pre_fc =", args.use_pre_fc)
    print(" 5) use_bn_layer =", args.use_bn_layer)
    model = models.__dict__[args.arch](num_classes=num_classes,l2_norm=args.l2_norm, use_pretrained = args.use_pretrained, transfer_learning = not(args.No_transfer_learning), use_bn = args.use_bn, use_pre_fc = args.use_pre_fc, use_bn_layer = args.use_bn_layer)
    return model

def build_encoder(args):
    with open('datasets/configs.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)[args.set]
    if args.verbose:
        print(ENCODER_REGISTRY)
    print("=> Creating model '{}'".format(args.arch))
    print("Model Option")
    print(" 1) use_pretrained =", args.use_pretrained)
    encoder = ENCODER_REGISTRY.get(args.arch)(args, num_classes=dataset_config['num_classes']) if len(args.arch) > 0 else None

    return encoder
