import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

# https://pytorch.org/docs/stable/torchvision/models.html
# transform needed for the ImageNet pre-trained VGG16 from PyTorch
image_net_tf = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VGG16Backbone(nn.Module):
    def __init__(self, weight_type, output_size):
        super().__init__()
        assert weight_type in ["random", "image_net", "genome"]
        self.weight_type = weight_type
        self.output_size = output_size

        pretrained = True if weight_type == "image_net" else False
        vgg16 = models.vgg16(pretrained=pretrained)
        vgg16 = list(vgg16.features.children())[:-1] + [
            nn.AdaptiveAvgPool2d(output_size=output_size)
        ]
        self.vgg16 = nn.Sequential(*vgg16)

        if weight_type == "genome":
            self.load_genome_weights()

    def forward(self, imgs):
        """
        imgs: (1, C, H, W) in scale of [0, 255].
        Do not need to normalize the image here.
        The normalization is defined in `preprocess`.
        """
        assert imgs.shape[0] == 1, (
            "This function does not support batch processing. "
            "Implement a forward batch instead."
        )
        assert imgs.shape[1] == 3
        assert min(imgs.shape[2:]) >= 224
        features = self.vgg16(imgs)
        return features

    def load_genome_weights(self):
        pass
