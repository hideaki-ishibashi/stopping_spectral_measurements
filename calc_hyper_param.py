from core.stopping_criteria import *
from utils import get_dataset
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from core.GPR import GPR
import random
import os


def main():
    np.random.seed(1)
    random.seed(1)
    data_names = ["Ni2+", "Co2+", "Mn2+", "Ni_3_exp", "Ni_4_exp", "Ni_6_exp"]
    for i, data_name in enumerate(data_names):
        param_dir = "param/" + data_name + "/"
        os.makedirs(param_dir, exist_ok=True)
        [X, y] = get_dataset.get_dataset(data_name)
        train_size = 900

        X_train = X[:train_size]
        y_train = y[:train_size]

        length_scale_bounds = (1e-3, 1e3)
        noise_level_bounds = (1e-5, 1e3)
        pow = 1.0
        Length = (length_scale_bounds[1] - length_scale_bounds[0]) * np.random.rand() + length_scale_bounds[0]
        noise_level = (noise_level_bounds[1] - noise_level_bounds[0]) * np.random.rand() + noise_level_bounds[0]
        kernel = pow * RBF(length_scale=Length, length_scale_bounds=length_scale_bounds) \
                 + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        gp = GPR(kernel=kernel, alpha=0.0, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=5)
        gp.fit(X_train, y_train)
        params = np.exp(gp.kernel_.theta)
        np.savetxt(param_dir + "params.txt", params)


if __name__ == "__main__":
    main()
