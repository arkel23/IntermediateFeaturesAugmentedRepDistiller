
import math
import torch.nn as nn
import torchvision.models.feature_extraction as feature_extraction
from einops.layers.torch import Rearrange

class Extractor(nn.Module):
    def __init__(self, model, model_name, layers='default'):
        super(Extractor, self).__init__()
        self.model_name = model_name
        self.layers = layers
        return_nodes = self.get_return_nodes(model, model_name, layers)
        self.model = feature_extraction.create_feature_extractor(model, return_nodes=return_nodes)
        
        if layers not in ['default', 'preact', 'last_only']:
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), Rearrange('b c 1 1 -> b c'))

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        if self.model_name in ['resnet8', 'resnet14', 'resnet32', 'resnet44', 
                'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4']:
            feat_m.append(self.model.conv1)
            feat_m.append(self.model.bn1)
            feat_m.append(self.model.relu)
            feat_m.append(self.model.layer1)
            feat_m.append(self.model.layer2)
            feat_m.append(self.model.layer3)
        elif self.model_name in ['ResNet18', 'ResNet34', 'ResNet50']:
            feat_m.append(self.model.conv1)
            feat_m.append(self.model.bn1)
            feat_m.append(self.model.relu)
            feat_m.append(self.model.layer1)
            feat_m.append(self.model.layer2)
            feat_m.append(self.model.layer3)
            feat_m.append(self.model.layer4)
        elif self.model_name in ['wrn_40_2', 'wrn_40_1', 'wrn_16_2']:
            feat_m.append(self.model.conv1)
            feat_m.append(self.model.block1)
            feat_m.append(self.model.block2)
            feat_m.append(self.model.block3)
        elif self.model_name in ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
            feat_m.append(self.model.block0)
            feat_m.append(self.model.pool0)
            feat_m.append(self.model.block1)
            feat_m.append(self.model.pool1)
            feat_m.append(self.model.block2)
            feat_m.append(self.model.pool2)
            feat_m.append(self.model.block3)
            feat_m.append(self.model.pool3)
            feat_m.append(self.model.block4)
            feat_m.append(self.model.pool4)
        elif self.model_name in ['ShuffleV1']:
            feat_m.append(self.model.conv1)
            feat_m.append(self.model.bn1)
            feat_m.append(self.model.layer1)
            feat_m.append(self.model.layer2)
            feat_m.append(self.model.layer3)
        elif self.model_name in ['ShuffleV2']:
            feat_m.append(self.model.conv1)
            feat_m.append(self.model.bn1)
            feat_m.append(self.model.layer1)
            feat_m.append(self.model.layer2)
            feat_m.append(self.model.layer3)
        elif self.model_name in ['MobileNetV2']:
            feat_m.append(self.model.conv1)
            feat_m.append(self.model.blocks)
        else:
            raise NotImplementedError    
        return feat_m
  
    def forward(self, x, classify_only=True):
        if self.layers == 'last_only':
            return self.model(x)
        x = list(self.model(x).values())
        if classify_only:
            return x[-1]
        else:
            if hasattr(self, 'pool'):
                return [self.pool(feats) for feats in x[:-1]] + [x[-1]]  
            else:
                return x
            
    def get_return_nodes(self, model, model_name, layers):
        # train_nodes, eval_nodes = feature_extraction.get_graph_node_names(model)
        if layers == 'last':
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
            elif model_name == 'ResNet18':
                return_nodes =  {
                    'layer4.0.relu_13': 'layerminus4',
                    'layer4.0.relu_14': 'layerminus3',
                    'layer4.1.relu_15': 'layerminus2',
                    'layer4.1.relu_16': 'layerminus1',
                    'linear': 'layerminus0'
                }
            elif model_name == 'ResNet34':
                return_nodes =  {
                    'layer4.0.relu_27': 'layerminus6',
                    'layer4.0.relu_28': 'layerminus5',
                    'layer4.1.relu_29': 'layerminus4',
                    'layer4.1.relu_30': 'layerminus3',
                    'layer4.2.relu_31': 'layerminus2',
                    'layer4.2.relu_32': 'layerminus1',
                    'linear': 'layerminus0'
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
            elif model_name == 'ShuffleV2':
                return_nodes = {
                    'layer3.0.relu_27': 'layerminus10',
                    'layer3.0.relu_28': 'layerminus9',
                    'layer3.0.relu_29': 'layerminus8',
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
        
        elif layers == 'blocks':
            if model_name == 'resnet8':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.0.relu_1': 'layerminus4',
                    'layer2.0.relu_2': 'layerminus3',
                    'layer3.0.relu_3': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet14':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.1.relu_2': 'layerminus4',
                    'layer2.1.relu_4': 'layerminus3',
                    'layer3.1.relu_6': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet20':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.2.relu_3': 'layerminus4',
                    'layer2.2.relu_6': 'layerminus3',
                    'layer3.2.relu_9': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet32':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.4.relu_5': 'layerminus4',
                    'layer2.4.relu_10': 'layerminus3',
                    'layer3.4.relu_15': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet44':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.6.relu_7': 'layerminus4',
                    'layer2.6.relu_14': 'layerminus3',
                    'layer3.6.relu_21': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet56':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.8.relu_9': 'layerminus4',
                    'layer2.8.relu_18': 'layerminus3',
                    'layer3.8.relu_27': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet110':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.17.relu_18': 'layerminus4',
                    'layer2.17.relu_36': 'layerminus3',
                    'layer3.17.relu_54': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet8x4':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.0.relu_1': 'layerminus4',
                    'layer2.0.relu_2': 'layerminus3',
                    'layer3.0.relu_3': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet32x4':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.4.relu_5': 'layerminus4',
                    'layer2.4.relu_10': 'layerminus3',
                    'layer3.4.relu_15': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name == 'ResNet18':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.1.relu_4': 'layerminus5',
                    'layer2.1.relu_8': 'layerminus4',
                    'layer3.1.relu_12': 'layerminus3',
                    'layer4.1.relu_16': 'layerminus2',
                    'linear': 'layerminus0'
                }
            elif model_name == 'ResNet34':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.relu_6': 'layerminus5',
                    'layer2.3.relu_14': 'layerminus4',
                    'layer3.5.relu_26': 'layerminus3',
                    'layer4.2.relu_32': 'layerminus2',
                    'linear': 'layerminus0'
                }
            elif model_name == 'ResNet50':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.relu_9': 'layerminus5',
                    'layer2.3.relu_21': 'layerminus4',
                    'layer3.5.relu_39': 'layerminus3',
                    'layer4.2.relu_48': 'layerminus2',
                    'linear': 'layerminus0'
                }
            elif model_name in ['wrn_16_1', 'wrn_16_2']:
                return_nodes = {
                    'conv1': 'layerminus5',
                    'block1.layer.1.add_1': 'layerminus4',
                    'block2.layer.1.add_3': 'layerminus3',
                    'block3.layer.1.add_5': 'layerminus2',
                    'fc': 'layerminus0'
                }
            elif model_name in ['wrn_40_1', 'wrn_40_2']:
                return_nodes = {
                    'conv1': 'layerminus5',
                    'block1.layer.5.add_5': 'layerminus4',
                    'block2.layer.5.add_11': 'layerminus3',
                    'block3.layer.5.add_17': 'layerminus2',
                    'fc': 'layerminus0'                }
            elif model_name in ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                return_nodes = {
                    'relu': 'layerminus6',
                    'relu_1': 'layerminus5',
                    'relu_2': 'layerminus4',
                    'relu_3': 'layerminus3',
                    'relu_4': 'layerminus2',
                    'classifier': 'layerminus0'
                }
            elif model_name == 'MobileNetV2':
                return_nodes = {
                    'conv1.2': 'layerminus6',
                    'blocks.1.1.add': 'layerminus5',
                    'blocks.2.2.add_2': 'layerminus4',
                    'blocks.4.2.add_7': 'layerminus3',
                    'blocks.6.0.conv.7': 'layerminus2',
                    'classifier.0': 'layerminus0'                
                }
            elif model_name == 'ShuffleV1':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.3.relu_12': 'layerminus4',
                    'layer2.7.relu_36': 'layerminus3',
                    'layer3.3.relu_48': 'layerminus2',
                    'linear': 'layerminus0' 
                }
            elif model_name == 'ShuffleV2':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.3.shuffle.reshape_3': 'layerminus4',
                    'layer2.7.shuffle.reshape_11': 'layerminus3',
                    'layer3.3.shuffle.reshape_15': 'layerminus2',
                    'linear': 'layerminus0'
                }
            else:
                raise NotImplementedError
                            
        elif layers == 'all':
            train_nodes, eval_nodes = feature_extraction.get_graph_node_names(model)
            if model_name in ['resnet8', 'resnet14', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 
                              'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2']:
                return_nodes = {node:i for i, node in enumerate(train_nodes) if 'relu' in train_nodes}
                return_nodes['fc'] = 'layerminus0' 
            elif model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ShuffleV1', 'ShuffleV2']:
                return_nodes = {node:i for i, node in enumerate(train_nodes) if 'relu' in train_nodes}
                return_nodes['linear'] = 'layerminus0' 
            elif model_name in ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                return_nodes = {node:i for i, node in enumerate(train_nodes) if 'relu' in train_nodes}
                return_nodes['classifier'] = 'layerminus0' 
            elif model_name == 'MobileNetV2':
                return_nodes = {node:i for i, node in enumerate(train_nodes) if 'conv' in train_nodes}
                return_nodes['classifier.0'] = 'layerminus0' 
            else:
                raise NotImplementedError
        
        elif layers == 'default':
            if model_name == 'resnet8':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.0.relu_1': 'layerminus4',
                    'layer2.0.relu_2': 'layerminus3',
                    'layer3.0.relu_3': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet14':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.1.relu_2': 'layerminus4',
                    'layer2.1.relu_4': 'layerminus3',
                    'layer3.1.relu_6': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet20':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.2.relu_3': 'layerminus4',
                    'layer2.2.relu_6': 'layerminus3',
                    'layer3.2.relu_9': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet32':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.4.relu_5': 'layerminus4',
                    'layer2.4.relu_10': 'layerminus3',
                    'layer3.4.relu_15': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet44':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.6.relu_7': 'layerminus4',
                    'layer2.6.relu_14': 'layerminus3',
                    'layer3.6.relu_21': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet56':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.8.relu_9': 'layerminus4',
                    'layer2.8.relu_18': 'layerminus3',
                    'layer3.8.relu_27': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet110':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.17.relu_18': 'layerminus4',
                    'layer2.17.relu_36': 'layerminus3',
                    'layer3.17.relu_54': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet8x4':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.0.relu_1': 'layerminus4',
                    'layer2.0.relu_2': 'layerminus3',
                    'layer3.0.relu_3': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet32x4':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.4.relu_5': 'layerminus4',
                    'layer2.4.relu_10': 'layerminus3',
                    'layer3.4.relu_15': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'ResNet18':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.1.relu_4': 'layerminus5',
                    'layer2.1.relu_8': 'layerminus4',
                    'layer3.1.relu_12': 'layerminus3',
                    'layer4.1.relu_16': 'layerminus2',
                    'view': 'layerminus1',
                    'linear': 'layerminus0'
                }
            elif model_name == 'ResNet34':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.relu_6': 'layerminus5',
                    'layer2.3.relu_14': 'layerminus4',
                    'layer3.5.relu_26': 'layerminus3',
                    'layer4.2.relu_32': 'layerminus2',
                    'view': 'layerminus1',
                    'linear': 'layerminus0'
                }
            elif model_name == 'ResNet50':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.relu_9': 'layerminus5',
                    'layer2.3.relu_21': 'layerminus4',
                    'layer3.5.relu_39': 'layerminus3',
                    'layer4.2.relu_48': 'layerminus2',
                    'view': 'layerminus1',
                    'linear': 'layerminus0'
                }
            elif model_name in ['wrn_16_1', 'wrn_16_2']:
                return_nodes = {
                    'conv1': 'layerminus5',
                    'block1.layer.1.add_1': 'layerminus4',
                    'block2.layer.1.add_3': 'layerminus3',
                    'block3.layer.1.add_5': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name in ['wrn_40_1', 'wrn_40_2']:
                return_nodes = {
                    'conv1': 'layerminus5',
                    'block1.layer.5.add_5': 'layerminus4',
                    'block2.layer.5.add_11': 'layerminus3',
                    'block3.layer.5.add_17': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'                }
            elif model_name in ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                return_nodes = {
                    'relu': 'layerminus6',
                    'relu_1': 'layerminus5',
                    'relu_2': 'layerminus4',
                    'relu_3': 'layerminus3',
                    'relu_4': 'layerminus2',
                    'view': 'layerminus1',
                    'classifier': 'layerminus0'
                }
            elif model_name == 'MobileNetV2':
                return_nodes = {
                    'conv1.2': 'layerminus6',
                    'blocks.1.1.add': 'layerminus5',
                    'blocks.2.2.add_2': 'layerminus4',
                    'blocks.4.2.add_7': 'layerminus3',
                    'blocks.6.0.conv.7': 'layerminus2',
                    'view': 'layerminus1',
                    'classifier.0': 'layerminus0'                
                }
            elif model_name == 'ShuffleV1':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.3.relu_12': 'layerminus4',
                    'layer2.7.relu_36': 'layerminus3',
                    'layer3.3.relu_48': 'layerminus2',
                    'view_16': 'layerminus1',
                    'linear': 'layerminus0' 
                }
            elif model_name == 'ShuffleV2':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.3.shuffle.reshape_3': 'layerminus4',
                    'layer2.7.shuffle.reshape_11': 'layerminus3',
                    'layer3.3.shuffle.reshape_15': 'layerminus2',
                    'view_16': 'layerminus1',
                    'linear': 'layerminus0'
                }
            else:
                raise NotImplementedError
        
        
        elif layers == 'preact':
            if model_name == 'resnet8':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.0.add': 'layerminus4',
                    'layer2.0.add_1': 'layerminus3',
                    'layer3.0.add_2': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet14':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.1.add_1': 'layerminus4',
                    'layer2.1.add_3': 'layerminus3',
                    'layer3.1.add_5': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet20':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.2.add_2': 'layerminus4',
                    'layer2.2.add_5': 'layerminus3',
                    'layer3.2.add_8': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet32':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.4.add_4': 'layerminus4',
                    'layer2.4.add_9': 'layerminus3',
                    'layer3.4.add_14': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet44':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.6.add_6': 'layerminus4',
                    'layer2.6.add_13': 'layerminus3',
                    'layer3.6.add_20': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet56':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.8.add_8': 'layerminus4',
                    'layer2.8.add_17': 'layerminus3',
                    'layer3.8.add_26': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet110':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.17.add_17': 'layerminus4',
                    'layer2.17.add_35': 'layerminus3',
                    'layer3.17.add_53': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet8x4':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.0.add': 'layerminus4',
                    'layer2.0.add_1': 'layerminus3',
                    'layer3.0.add_2': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'resnet32x4':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.4.add_4': 'layerminus4',
                    'layer2.4.add_9': 'layerminus3',
                    'layer3.4.add_14': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name == 'ResNet18':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.1.add_1': 'layerminus5',
                    'layer2.1.add_3': 'layerminus4',
                    'layer3.1.add_5': 'layerminus3',
                    'layer4.1.add_7': 'layerminus2',
                    'view': 'layerminus1',
                    'linear': 'layerminus0'
                }
            elif model_name == 'ResNet34':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.add_2': 'layerminus5',
                    'layer2.3.add_6': 'layerminus4',
                    'layer3.5.add_12': 'layerminus3',
                    'layer4.2.add_15': 'layerminus2',
                    'view': 'layerminus1',
                    'linear': 'layerminus0'
                }
            elif model_name == 'ResNet50':
                return_nodes = {
                    'relu': 'layerminus6',
                    'layer1.2.add_2': 'layerminus5',
                    'layer2.3.add_6': 'layerminus4',
                    'layer3.5.add_12': 'layerminus3',
                    'layer4.2.add_15': 'layerminus2',
                    'view': 'layerminus1',
                    'linear': 'layerminus0'
                }
            elif model_name in ['wrn_16_1', 'wrn_16_2']:
                return_nodes = {
                    'conv1': 'layerminus5',
                    'block2.layer.0.bn1': 'layerminus4',
                    'block3.layer.0.bn1': 'layerminus3',
                    'bn1': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'
                }
            elif model_name in ['wrn_40_1', 'wrn_40_2']:
                return_nodes = {
                    'conv1': 'layerminus5',
                    'block2.layer.0.bn1': 'layerminus4',
                    'block3.layer.0.bn1': 'layerminus3',
                    'bn1': 'layerminus2',
                    'view': 'layerminus1',
                    'fc': 'layerminus0'                }
            elif model_name == 'vgg8':
                return_nodes = {
                    'relu': 'layerminus6',
                    'block1.1': 'layerminus5',
                    'block2.1': 'layerminus4',
                    'block3.1': 'layerminus3',
                    'block4.1': 'layerminus2',
                    'view': 'layerminus1',
                    'classifier': 'layerminus0'
                }
            elif model_name == 'vgg11':
                return_nodes = {
                    'relu': 'layerminus6',
                    'block1.1': 'layerminus5',
                    'block2.4': 'layerminus4',
                    'block3.4': 'layerminus3',
                    'block4.4': 'layerminus2',
                    'view': 'layerminus1',
                    'classifier': 'layerminus0'
                }
            elif model_name == 'vgg13':
                return_nodes = {
                    'relu': 'layerminus6',
                    'block1.4': 'layerminus5',
                    'block2.4': 'layerminus4',
                    'block3.4': 'layerminus3',
                    'block4.4': 'layerminus2',
                    'view': 'layerminus1',
                    'classifier': 'layerminus0'
                }
            elif model_name == 'vgg16':
                return_nodes = {
                    'relu': 'layerminus6',
                    'block1.4': 'layerminus5',
                    'block2.4': 'layerminus4',
                    'block3.7': 'layerminus3',
                    'block4.7': 'layerminus2',
                    'view': 'layerminus1',
                    'classifier': 'layerminus0'
                }
            elif model_name == 'vgg19':
                return_nodes = {
                    'relu': 'layerminus6',
                    'block1.4': 'layerminus5',
                    'block2.10': 'layerminus4',
                    'block3.10': 'layerminus3',
                    'block4.10': 'layerminus2',
                    'view': 'layerminus1',
                    'classifier': 'layerminus0'
                }
            
            elif model_name == 'MobileNetV2':
                return_nodes = {
                    'conv1.2': 'layerminus6',
                    'blocks.1.1.add': 'layerminus5',
                    'blocks.2.2.add_2': 'layerminus4',
                    'blocks.4.2.add_7': 'layerminus3',
                    'blocks.6.0.conv.7': 'layerminus2',
                    'view': 'layerminus1',
                    'classifier.0': 'layerminus0'                
                }
            elif model_name == 'ShuffleV1':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.3.add_2': 'layerminus4',
                    'layer2.7.add_9': 'layerminus3',
                    'layer3.3.add_12': 'layerminus2',
                    'view_16': 'layerminus1',
                    'linear': 'layerminus0' 
                }
            elif model_name == 'ShuffleV2':
                return_nodes = {
                    'relu': 'layerminus5',
                    'layer1.3.cat_5': 'layerminus4',
                    'layer2.7.cat_20': 'layerminus3',
                    'layer3.3.cat_27': 'layerminus2',
                    'view_16': 'layerminus1',
                    'linear': 'layerminus0'
                }
            else:
                raise NotImplementedError
            
        elif layers == 'last_only':
            if model_name in ['resnet8', 'resnet14', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 
                              'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2']:
                return_nodes = {'fc': 'layerminus0'}            
            elif model_name in ['ResNet18', 'ResNet34', 'ResNet50', 'ShuffleV1', 'ShuffleV2']:
                return_nodes = {'linear': 'layerminus0'}            
            elif model_name in ['vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                return_nodes = {'classifier': 'layerminus0'}            
            elif model_name == 'MobileNetV2':
                return_nodes = {'classifier.0': 'layerminus0'}            
            else:
                raise NotImplementedError
        
        
        else:
            raise NotImplementedError      
        
        return return_nodes
