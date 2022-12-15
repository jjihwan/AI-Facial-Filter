from .ehanet import EHANet18
from .ehanet2 import EHANet18_2
def get_model(model, url=None, n_classes=19, pretrained=True):
    if model == 'EHANet18':
        net = EHANet18(num_classes=n_classes, pretrained=pretrained)
    elif model == 'EHANet18_2':
        net = EHANet18_2(num_classes=n_classes, pretrained=pretrained)
    else:
        raise ValueError("No corresponding model was found...")
    return net