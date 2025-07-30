import torch
import torch.nn as nn
import torch.nn.functional as F
import monai.networks.nets as nets

class Agepredictionmodel(nn.Module):
    def __init__(self):
        super(Agepredictionmodel, self).__init__()
        
        self.base_model = nets.DenseNet121(
            spatial_dims=2, in_channels=3, out_channels=4)
        
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
        
        batch_size = x.shape[0]
        num_views = x.shape[1]
        
        
        x = x.view(batch_size * num_views, x.shape[2], x.shape[3], x.shape[4])

       
        features = self.base_model.features(x)
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        pooled_features = torch.flatten(pooled_features, 1)
        features_vec = pooled_features.view(batch_size, num_views, -1)
        averaged_features = torch.mean(features_vec, dim=1)
        age = self.age_predictor(averaged_features)
        return age