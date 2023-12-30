import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet101, ResNet101_Weights

class ResnetFeatureExtractor:
    def __init__(self, arch='resnet18'):

        if arch == 'resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.transforms = ResNet18_Weights.DEFAULT.transforms()
        elif arch == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)  
            self.transforms = ResNet50_Weights.DEFAULT.transforms()      
        elif arch == 'resnet101':
            self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
            self.transforms = ResNet101_Weights.DEFAULT.transforms()
        else :
            raise NotImplementedError('Not implemented arch')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

        # register forward hook
        self.forward_hook = self.model.layer4.register_forward_hook(self.getActivation())
    
    def getActivation(self,):        
        # the hook signature
        def hook(model, input, output):
            features = output.detach()
            self.features = features
        return hook

    
    def extract_features(self, X):
        # forward pass -- getting the outputs
        with torch.no_grad():
            y = self.model(X)
        
        return self.features

