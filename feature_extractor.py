import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import torch
import imagehash


class ImageHashExtractor(object):
    def __init__(self):
        self.feature_length = 256

    def extract(self, x):
        imghash = imagehash.phash(x, hash_size=16).hash.ravel()
        return imghash.astype(int)


class NetFeatureExtractor(object):
    def __init__(self, network='resnet'):
        if network == 'resnet':
            model_ft = models.resnet18(pretrained=True)
            model_ft.eval()
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft.fc.weight = torch.nn.Parameter(torch.eye(num_ftrs))
            self.model = model_ft
            self.feature_length = 512
        elif network == 'vgg':
            model_ft = models.vgg16(pretrained=True)
            model_ft.eval()
            self.model = lambda x: nn.functional.max_pool3d(model_ft.features(x), kernel_size=5, stride=3)
            self.feature_length = 170
        else:
            raise NotImplementedError('Unknown feature extraction network "{}"'.format(network))

        self.input_size = 224
        self.model_transformation = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, x):
        with torch.no_grad():
            transf_x = self.model_transformation(x)
            features = self.model(torch.stack((transf_x,)))
            features = features.flatten()

        return features.numpy()


if __name__ == '__main__':
    from torchvision import datasets

    ds = datasets.CIFAR10('cifar10',
                          train=True,
                          download=True)
    extractor1 = NetFeatureExtractor()
    f1 = extractor1.extract(ds[0][0])
    print(ds[0][0], f1.shape)

    extractor2 = ImageHashExtractor()
    f2 = extractor2.extract(ds[0][0])
    print(f2.shape)
