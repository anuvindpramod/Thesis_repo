import torch
import torch.nn as nn
import monai.networks.nets as nets


class Agepredictionmodel(nn.Module):
    def __init__(self):
        super(Agepredictionmodel, self).__init__()
        '''self.base_model = monai.bundle.load(name="breast_density_classification",model_zoo_source="monai_io")'''
        self.base_model = nets.DenseNet121(spatial_dims=2, in_channels=3, out_channels=4)
        for param in self.base_model.parameters():
            param.requires_grad = False

        num_features = self.base_model.features.norm5.num_features
        self.age_predictor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        batch_size, num_views, channels, height, width = x.shape
        x = x.view(batch_size*num_views, channels, height, width)

        features = self.base_model.features(x)
        features = features.view(batch_size, num_views, -1)
        features = torch.mean(features, dim=1)  # Average across views
        age = self.age_predictor(features)
        return age
