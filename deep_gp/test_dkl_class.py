import torch
import gpytorch
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, latent_dim=10):
        super().__init__()
        self.add_module('l1', torch.nn.Linear(data_dim, 512))
        self.add_module('r1', torch.nn.ReLU())
        self.add_module('l2', torch.nn.Linear(512, 256))
        self.add_module('r2', torch.nn.ReLU())
        self.add_module('l3', torch.nn.Linear(256, 64))
        self.add_module('r3', torch.nn.ReLU())
        self.add_module('l4', torch.nn.Linear(64, latent_dim))


class SmallFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, latent_dim=10):
        super().__init__()
        self.add_module('l1', torch.nn.Linear(data_dim, 128))
        self.add_module('r1', torch.nn.ReLU())
        self.add_module('l2', torch.nn.Linear(128, 64))
        self.add_module('r2', torch.nn.ReLU())
        self.add_module('l3', torch.nn.Linear(64, latent_dim))

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 data_dim, latent_dim=10, extractor_type="small"):

        super().__init__(train_x, train_y, likelihood)

        if extractor_type == "large":
            self.feature_extractor = LargeFeatureExtractor(data_dim, latent_dim)
        else:
            self.feature_extractor = SmallFeatureExtractor(data_dim, latent_dim)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
        )

    def forward(self, x):
        z = self.feature_extractor(x)
        mean_x = self.mean_module(z)
        covar_x = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
