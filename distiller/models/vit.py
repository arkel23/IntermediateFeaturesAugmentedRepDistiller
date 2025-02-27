import torch.nn as nn
import torchvision.models as models

from pretrained_vit import ViT, PRETRAINED_CONFIGS, ViTConfigExtended


def vit(model_name, num_classes, image_size, pretrained):

    def_config = PRETRAINED_CONFIGS['{}'.format(model_name)]['config']
    config = ViTConfigExtended(**def_config)
    config.num_classes = num_classes
    config.image_size = image_size

    model = ViT(config, name=model_name, pretrained=pretrained, ret_interm_repr=True)

    print('Loading pt={} {} model with {} classes output head'.format(
        pretrained, model_name, num_classes
    ))

    return model
