import torch
import gpytorch
from gpytorch.models.deep_gps import DeepGPLayer
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGP
from gpytorch.likelihoods import GaussianLikelihood

# This class builds a single deep GP layer.

class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        # Inducing points
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        # Variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)

        # Mean function
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        # Kernel
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)




# The following class builds a two‑layer deep GP model.

class DeepGPModel(DeepGP):
    def __init__(self, input_dim, hidden_dim=10, num_inducing=128):
        super().__init__()

        # First GP layer
        self.hidden_layer = DeepGPHiddenLayer(
            input_dims=input_dim,
            output_dims=hidden_dim,
            num_inducing=num_inducing,
            mean_type='linear'
        )

        # Output GP layer
        self.output_layer = DeepGPHiddenLayer(
            input_dims=hidden_dim,
            output_dims=None,
            num_inducing=num_inducing,
            mean_type='constant'
        )

        # Likelihood
        self.likelihood = GaussianLikelihood()

    def forward(self, x):
        hidden_rep = self.hidden_layer(x)
        output = self.output_layer(hidden_rep)
        return output
    
    def predict(self, test_loader, num_likelihood_samples=10):
        self.eval()
        self.likelihood.eval()

        device = next(self.parameters()).device  # <--- NEW

        mus = []
        vars_ = []

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(num_likelihood_samples):
            for xb, _ in test_loader:
                xb = xb.to(device)  # <--- NEW

                preds = self.likelihood(self(xb))
                mus.append(preds.mean)
                vars_.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(vars_, dim=-1)

