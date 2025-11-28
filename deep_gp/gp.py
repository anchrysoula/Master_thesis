import numpy as np

class GaussianProcess:
    def __init__(self, a=-1, b=1, num_points=100):
        """
        Initialize GaussianProcess object.

        Parameters
        ----------
        a : float, default=-1
            Lower bound of the interval.
        b : float, default=1
            Upper bound of the interval.
        num_points : int, default=100
            Number of points to sample from the interval.

        Attributes
        ----------
        a : float
            Lower bound of the interval.
        b : float
            Upper bound of the interval.
        z : numpy.ndarray
            Array of points sampled from the interval.
        X_test : numpy.ndarray
            Array of points to be used for prediction.

        """
        self.a = a
        self.b = b
        self.z = np.linspace(a, b, num=num_points)
        self.X_test = np.linspace(a, b, num=num_points)

    def kernel_matrix_vectorized(self, z, theta_0, theta_1):
        """
        Compute the kernel matrix using the given parameters and points.

        Parameters
        ----------
        z : numpy.ndarray
            Array of points sampled from the interval.
        theta_0 : float
            Amplitude of the kernel.
        theta_1 : float
            Variance of the kernel.

        Returns
        -------
        kernel_matrix : numpy.ndarray
            Kernel matrix computed from the given parameters and points.
        """
        diff = z[:, None] - z[None, :]
        return theta_0 * np.exp(-theta_1 / 2 * diff**2)


    def sample_gp(self, kernel, num_samples=10):
        """
        Sample from the Gaussian process prior distribution.

        Parameters
        ----------
        kernel : numpy.ndarray
            Kernel matrix computed from the given parameters and points.
        num_samples : int, default=10
            Number of samples to draw from the GP prior distribution.

        Returns
        -------
        samples : numpy.ndarray
            Array of samples drawn from the GP prior distribution.
        """
        return np.random.multivariate_normal(mean=np.zeros(len(self.z)), 
        cov=kernel, size=num_samples)

    def kernel_funct_matrix(self, x1, x2, theta_0, theta_1):
        """
        Compute the kernel matrix between two sets of points.

        Parameters
        ----------
        x1 : numpy.ndarray
            Array of points to compute the kernel matrix with.
        x2 : numpy.ndarray
            Array of points to compute the kernel matrix with.
        theta_0 : float
            Amplitude of the kernel.
        theta_1 : float
            Variance of the kernel.

        Returns
        -------
        kernel_matrix : numpy.ndarray
            Kernel matrix computed from the given parameters and points.
        """
        sq_differ = (np.subtract.outer(x1, x2)) ** 2
        return theta_0 * np.exp(-theta_1 / 2 * sq_differ)

    def compute_cov_matrices(self, X_train, X_test, theta_0, theta_1, sigma_2):
        """
        Compute covariance matrices for Gaussian process prediction.

        Parameters
        ----------
        X_train : numpy.ndarray
            Array of training points.
        X_test : numpy.ndarray
            Array of test points.
        theta_0 : float
            Amplitude of the kernel.
        theta_1 : float
            Variance of the kernel.
        sigma_2 : float
            Variance of the noise.

        Returns
        -------
        C_N : numpy.ndarray
            Covariance matrix of the training points.
        K_mat : numpy.ndarray
            Covariance matrix between the training points and test points.
        D : numpy.ndarray
            Covariance matrix of the test points.
        """
        C_N = self.kernel_funct_matrix(X_train, X_train, theta_0, theta_1) 
        + sigma_2 * np.eye(len(X_train))
        K_mat = self.kernel_funct_matrix(X_train, X_test, theta_0, theta_1)
        D = self.kernel_funct_matrix(X_test, X_test, theta_0, theta_1)
        return C_N, K_mat, D

    def compute_gp_posterior(self, X_train, y_train, theta_0, theta_1, sigma_2):
        """
        Compute the posterior mean and covariance matrix for Gaussian process
        prediction.

        Parameters
        ----------
        X_train : numpy.ndarray
            Array of training points.
        y_train : numpy.ndarray
            Array of corresponding labels corresponding corresponding to the 
            training points.
        theta_0 : float
            Amplitude of the kernel.
        theta_1 : float
            Variance of the kernel.
        sigma_2 : float
            Variance of the noise.

        Returns
        -------
        mean_vector : numpy.ndarray
            Mean vector of the posterior distribution.
        cov_matrix : numpy.ndarray
            Covariance matrix of the posterior distribution.
        std_dev : numpy.ndarray
            Standard deviation of the posterior distribution.
        """
        C_N, K_mat, D = self.compute_cov_matrices(X_train, self.X_test, 
        theta_0, theta_1, sigma_2)
        z = np.linalg.solve(C_N, y_train)
        mean_vector = K_mat.T @ z
        W = np.linalg.solve(C_N, K_mat)
        cov_matrix = D - K_mat.T @ W
        std_dev = np.sqrt(np.diag(cov_matrix))
        return mean_vector, cov_matrix, std_dev

    def diagnostics(self, C_N):
        """
        Compute a diagnostic for the kernel matrix, checking how close it is 
        to the identity matrix.

        Parameters
        ----------
        C_N : numpy.ndarray
            Kernel matrix computed from the training points.

        Returns
        -------
        diagnostic : float
            Norm of the difference between the kernel matrix and the identity 
            matrix.
        """
        identity_check = C_N @ np.linalg.solve(C_N, np.eye(C_N.shape[0]))
        return np.linalg.norm(identity_check - np.eye(C_N.shape[0]))
