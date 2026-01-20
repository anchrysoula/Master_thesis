import torch
import gpytorch

#Parts of the code were adapted and modified from the GPyTorch documentation: 
#https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html

def get_activation(name):
    if name == "relu":
        return torch.nn.ReLU()
    else:
        return torch.nn.Tanh()
class FeatureExtractor1(torch.nn.Sequential):
    def __init__(self, data_dim, latent_dim, activation):
        super().__init__()
        act = get_activation(activation)
        self.add_module('l1', torch.nn.Linear(data_dim, 512))
        self.add_module('a1', act)
        self.add_module('l2', torch.nn.Linear(512, 256))
        self.add_module('a2', act)
        self.add_module('l3', torch.nn.Linear(256, 64))
        self.add_module('a3', act)
        self.add_module('l4', torch.nn.Linear(64, latent_dim))

class FeatureExtractor2(torch.nn.Sequential):
    def __init__(self, data_dim, latent_dim, activation):
        super().__init__()
        act = get_activation(activation)
        self.add_module('l1', torch.nn.Linear(data_dim, 128))
        self.add_module('a1', act)
        self.add_module('l2', torch.nn.Linear(128, 64))
        self.add_module('a2', act)
        self.add_module('l3', torch.nn.Linear(64, latent_dim))

class FeatureExtractor3(torch.nn.Sequential):  # from DKL paper
    def __init__(self, data_dim, latent_dim, activation):
        super().__init__()
        act = get_activation(activation)
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('a1', act)
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('a2', act)
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('a3', act)
        self.add_module('linear4', torch.nn.Linear(50, latent_dim))

class FeatureExtractor4(torch.nn.Sequential):
    def __init__(self, data_dim, latent_dim, activation):
        super().__init__()
        act = get_activation(activation)
        self.add_module('l1', torch.nn.Linear(data_dim, 64))
        self.add_module('a1', act)
        self.add_module('l2', torch.nn.Linear(64, latent_dim))

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 data_dim, latent_dim=10,
                 extractor_type="small",
                 activation="relu",
                 kernel_type="rbf_ard",
                 noise_value=0.05):

        super().__init__(train_x, train_y, likelihood)

        # feature extractors
        if extractor_type == "large":
            self.feature_extractor = FeatureExtractor1(data_dim, latent_dim, activation)
        elif extractor_type == "medium":
            self.feature_extractor = FeatureExtractor2(data_dim, latent_dim, activation)
        elif extractor_type == "dkl":
            self.feature_extractor = FeatureExtractor3(data_dim, latent_dim, activation)
        else:
            self.feature_extractor = FeatureExtractor4(data_dim, latent_dim, activation)

      
        # kernels
        
        if kernel_type == "rbf_ard":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)

        elif kernel_type == "matern_15":
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=latent_dim)

        else:
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=latent_dim)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        # Set noise
        if noise_value is not None:
            likelihood.noise = noise_value
        # else: leave noise free to be learned


    def forward(self, x):
        z = self.feature_extractor(x)
        mean_x = self.mean_module(z)
        covar_x = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
