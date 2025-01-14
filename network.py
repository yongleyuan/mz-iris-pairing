import torch.nn as nn
import torchvision.models as models


class SiameseResnet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
    ) -> None:
        super(SiameseResnet, self).__init__()

        self.backbone = backbone
        if self.backbone == "resnet18":
            self.resnet = models.resnet18(pretrained=True)
        elif self.backbone == "resnet34":
            self.resnet = models.resnet34(pretrained=True)
        elif self.backbone == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
        elif self.backbone == "resnet101":
            self.resnet = models.resnet101(pretrained=True)
        else:
            raise ValueError(
                "Unsupported ResNet version. Choose from ['resnet18', 'resnet34', 'resnet50', 'resnet101']"
            )

        layers = list(self.resnet.children())

        # Change 1st layer to accept channel of 1 instead of 3
        layers[0] = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        # Remove the last fully connected layer
        fc_in = layers[-1].in_features
        layers = layers[:-1]

        # Update encoder
        self.resnet = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 128),
        )

        self.features = None  # placeholder for features (model saliency)

        self.hook_handle = None  # placeholder for hook handle

    def forward1(self, img):
        output = self.resnet(img)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def forward(self, img_left, img_right):
        output_left = self.forward1(img_left)
        output_right = self.forward1(img_right)
        return output_left, output_right

    def add_hook(self):
        def get_features():
            # Hook signature
            def hook(model, input, output):
                self.features = output

            return hook

        if self.backbone in ("resnet18", "resnet34"):
            self.resnet[-2][-1].conv2.register_forward_hook(get_features())
        else:  # ("resnet50", "resnet101")
            self.resnet[-2][-1].conv3.register_forward_hook(get_features())

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def get_params(self):
        return list(self.model.fc.parameters())[0]
