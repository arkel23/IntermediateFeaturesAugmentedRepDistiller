
import math
import torch
import torch.nn as nn
import torchvision.models.feature_extraction as feature_extraction
from einops.layers.torch import Rearrange

class Extractor(nn.Module):
    def __init__(self, model, model_name):
        super(Extractor, self).__init__()
        return_nodes = self.get_return_nodes(model_name)
        self.model = feature_extraction.create_feature_extractor(model, return_nodes=return_nodes)

        self.pool = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        Rearrange('b c 1 1 -> b c')
                    )
    
    def forward(self, x, classify_only=True):
        interm_features = list(self.model(x).values())
        if classify_only:
            return interm_features[-1]
        return [self.pool(feats) for feats in interm_features[:-1]] + [interm_features[-1]]  
    
    def get_return_nodes(self, model_name):
        # train_nodes, eval_nodes = feature_extraction.get_graph_node_names(model)
        if model_name == 'resnet8':
            return_nodes = {
                'layer2.0.relu': 'layerminus4',
                'layer2.0.relu_2': 'layerminus3',
                'layer3.0.relu': 'layerminus2',
                'layer3.0.relu_3': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'resnet14':
            return_nodes = {
                'layer3.0.relu': 'layerminus4',
                'layer3.0.relu_5': 'layerminus3',
                'layer3.1.relu': 'layerminus2',
                'layer3.1.relu_6': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'resnet20':
            return_nodes = {
                'layer3.0.relu': 'layerminus6',
                'layer3.0.relu_7': 'layerminus5',
                'layer3.1.relu': 'layerminus4',
                'layer3.1.relu_8': 'layerminus3',
                'layer3.2.relu': 'layerminus2',
                'layer3.2.relu_9': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'resnet32':
            return_nodes =  {
                'layer3.0.relu': 'layerminus10',
                'layer3.0.relu_11': 'layerminus9',
                'layer3.1.relu': 'layerminus8',
                'layer3.1.relu_12': 'layerminus7',
                'layer3.2.relu': 'layerminus6',
                'layer3.2.relu_13': 'layerminus5',
                'layer3.3.relu': 'layerminus4',
                'layer3.3.relu_14': 'layerminus3',
                'layer3.4.relu': 'layerminus2',
                'layer3.4.relu_15': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'resnet44':
            return_nodes =  {
                'layer3.0.relu_15': 'layerminus14',
                'layer3.0.relu': 'layerminus13',
                'layer3.1.relu': 'layerminus12',
                'layer3.1.relu_16': 'layerminus11',
                'layer3.2.relu': 'layerminus10',
                'layer3.2.relu_17': 'layerminus9',
                'layer3.3.relu': 'layerminus8',
                'layer3.3.relu_18': 'layerminus7',
                'layer3.4.relu': 'layerminus6',
                'layer3.4.relu_19': 'layerminus5',
                'layer3.5.relu': 'layerminus4',
                'layer3.5.relu_20': 'layerminus3',
                'layer3.6.relu': 'layerminus2',
                'layer3.6.relu_21': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'resnet56':
            return_nodes =  {
                'layer3.0.relu': 'layerminus18',
                'layer3.0.relu_19': 'layerminus17',
                'layer3.1.relu': 'layerminus16',
                'layer3.1.relu_20': 'layerminus15',
                'layer3.2.relu': 'layerminus14',
                'layer3.2.relu_21': 'layerminus13',
                'layer3.3.relu': 'layerminus12',
                'layer3.3.relu_22': 'layerminus11',
                'layer3.4.relu': 'layerminus10',
                'layer3.4.relu_23': 'layerminus9',
                'layer3.5.relu': 'layerminus8',
                'layer3.5.relu_24': 'layerminus7',
                'layer3.6.relu': 'layerminus6',
                'layer3.6.relu_25': 'layerminus5',
                'layer3.7.relu': 'layerminus4',
                'layer3.7.relu_26': 'layerminus3',
                'layer3.8.relu': 'layerminus2',
                'layer3.8.relu_27': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'resnet110':
            return_nodes =  {
                'layer3.9.relu': 'layerminus18',
                'layer3.9.relu_46': 'layerminus17',
                'layer3.10.relu': 'layerminus16',
                'layer3.10.relu_47': 'layerminus15',
                'layer3.11.relu': 'layerminus14',
                'layer3.11.relu_48': 'layerminus13',
                'layer3.12.relu': 'layerminus12',
                'layer3.12.relu_49': 'layerminus11',
                'layer3.13.relu': 'layerminus10',
                'layer3.13.relu_50': 'layerminus9',
                'layer3.14.relu': 'layerminus8',
                'layer3.14.relu_51': 'layerminus7',
                'layer3.15.relu': 'layerminus6',
                'layer3.15.relu_52': 'layerminus5',
                'layer3.16.relu': 'layerminus4',
                'layer3.16.relu_53': 'layerminus3',
                'layer3.17.relu': 'layerminus2',
                'layer3.17.relu_54': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'resnet8x4':
            return_nodes = {
                'layer2.0.relu': 'layerminus4',
                'layer2.0.relu_2': 'layerminus3',
                'layer3.0.relu': 'layerminus2',
                'layer3.0.relu_3': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'resnet32x4':
            return_nodes =  {
                'layer3.0.relu': 'layerminus10',
                'layer3.0.relu_11': 'layerminus9',
                'layer3.1.relu': 'layerminus8',
                'layer3.1.relu_12': 'layerminus7',
                'layer3.2.relu': 'layerminus6',
                'layer3.2.relu_13': 'layerminus5',
                'layer3.3.relu': 'layerminus4',
                'layer3.3.relu_14': 'layerminus3',
                'layer3.4.relu': 'layerminus2',
                'layer3.4.relu_15': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name == 'ResNet50':
            return_nodes =  {
                'layer4.0.relu_40': 'layerminus9',
                'layer4.0.relu_41': 'layerminus8',
                'layer4.0.relu_42': 'layerminus7',
                'layer4.1.relu_43': 'layerminus6',
                'layer4.1.relu_44': 'layerminus5',
                'layer4.1.relu_45': 'layerminus4',
                'layer4.2.relu_46': 'layerminus3',
                'layer4.2.relu_47': 'layerminus2',
                'layer4.2.relu_48': 'layerminus1',
                'linear': 'layerminus0'
            }
        elif model_name in ['wrn_16_1', 'wrn_16_2']:
            return_nodes = {
                'block3.layer.0.relu1': 'layerminus5',
                'block3.layer.0.relu2': 'layerminus4',
                'block3.layer.1.relu1': 'layerminus3',
                'block3.layer.1.relu2': 'layerminus2',
                'relu': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name in ['wrn_40_1', 'wrn_40_2']:
            return_nodes = {
                'block3.layer.0.relu1': 'layerminus13',
                'block3.layer.0.relu2': 'layerminus12',
                'block3.layer.1.relu1': 'layerminus11',
                'block3.layer.1.relu2': 'layerminus10',
                'block3.layer.2.relu1': 'layerminus9',
                'block3.layer.2.relu2': 'layerminus8',
                'block3.layer.3.relu1': 'layerminus7',
                'block3.layer.3.relu2': 'layerminus6',
                'block3.layer.4.relu1': 'layerminus5',
                'block3.layer.4.relu2': 'layerminus4',
                'block3.layer.5.relu1': 'layerminus3',
                'block3.layer.5.relu2': 'layerminus2',
                'relu': 'layerminus1',
                'fc': 'layerminus0'
            }
        elif model_name in ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
            return_nodes = {
                'relu_1': 'layerminus4',
                'relu_2': 'layerminus3',
                'relu_3': 'layerminus2',
                'relu_4': 'layerminus1',
                'classifier': 'layerminus0'
            }
        elif model_name == 'MobileNetV2':
            return_nodes = {
                'blocks.6.0.conv.0': 'layerminus11',
                'blocks.6.0.conv.1': 'layerminus10',
                'blocks.6.0.conv.2': 'layerminus9',
                'blocks.6.0.conv.3': 'layerminus8',
                'blocks.6.0.conv.4': 'layerminus7',
                'blocks.6.0.conv.5': 'layerminus6',
                'blocks.6.0.conv.6': 'layerminus5',
                'blocks.6.0.conv.7': 'layerminus4',
                'conv2.0': 'layerminus3',
                'conv2.1': 'layerminus2',
                'conv2.2': 'layerminus1',
                'classifier.0': 'layerminus0'                
            }
        elif model_name == 'ShuffleV1':
            return_nodes = {
                'layer3.0.relu_37': 'layerminus12',
                'layer3.0.relu_38': 'layerminus11',
                'layer3.0.relu_39': 'layerminus10',
                'layer3.1.relu_40': 'layerminus9',
                'layer3.1.relu_41': 'layerminus8',
                'layer3.1.relu_42': 'layerminus7',
                'layer3.2.relu_43': 'layerminus6',
                'layer3.2.relu_44': 'layerminus5',
                'layer3.2.relu_45': 'layerminus4',
                'layer3.3.relu_46': 'layerminus3',
                'layer3.3.relu_47': 'layerminus2',
                'layer3.3.relu_48': 'layerminus1',
                'linear': 'layerminus0' 
            }
        elif model_name == 'ShuffleV1':
            return_nodes = {
                'layer3.0.relu_27': 'layerminus10',
                'layer3.0.relu_28': 'layerminus9',
                'layer3.0.relu_20': 'layerminus8',
                'layer3.1.relu_30': 'layerminus7',
                'layer3.1.relu_31': 'layerminus6',
                'layer3.2.relu_32': 'layerminus5',
                'layer3.2.relu_33': 'layerminus4',
                'layer3.3.relu_34': 'layerminus3',
                'layer3.3.relu_35': 'layerminus2',
                'relu_36': 'layerminus1',
                'linear': 'layerminus0' 
            }
        else:
            raise NotImplementedError

        return return_nodes
