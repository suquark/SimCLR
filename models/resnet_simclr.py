import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RADSimCLR(nn.Module):
    def __init__(self, base_model, out_dim, obs_shape=(9, 84, 84), feature_dim=256, num_layers=4, num_filters=32):
        super().__init__()
        self.convs = nn.ModuleList(
             [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.fc = nn.Linear(num_filters * 35 * 35, feature_dim)

        # projection MLP
        self.l1 = nn.Linear(feature_dim, feature_dim)
        self.l2 = nn.Linear(feature_dim, out_dim)

    def forward(self, x):
        if x.max() > 1.:
             x = x / 255.
        for c in self.convs:
            x = c(x).relu()
        x = x.view(x.size(0), -1)
        h = self.fc(x).relu()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, obs_shape=(9, 84, 84)):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features
        layers = list(resnet.children())[:-1]
        layers[0] = nn.Conv2d(obs_shape[0], resnet.inplanes, kernel_size=7, stride=2, padding=3,
                              bias=False)
        nn.init.kaiming_normal_(layers[0].weight, mode='fan_out', nonlinearity='relu')
        self.features = nn.Sequential(*layers)

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
