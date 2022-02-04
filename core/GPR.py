import numpy as np
import random
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.utils.optimize import _check_optimize_result
import warnings
import numpy as np
from scipy.linalg import solve_triangular
import scipy.optimize
import re

GPR_CHOLESKY_LOWER = True

class GPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=2e05, gtol=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                                              options={'maxiter': self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min


    def predict_noiseless(self, X, return_std=False, return_cov=False):
        # sklearnのgpではノイズまで含めた事後分布推定をしているため，ノイズを除去した予測を定義
        # 実装はsklearnのGPregressorのpredictの分散，標準偏差の計算を修正したものになっている
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )

        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        X = self._validate_data(X, ensure_2d=ensure_2d, dtype=dtype, reset=False)

        if not hasattr(self, "X_train_"):
            if self.kernel is None:
                kernel = C(1.0, constant_value_bounds="fixed") * RBF(
                    1.0, length_scale_bounds="fixed"
                )
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans @ self.alpha_

            y_mean = self._y_train_std * y_mean + self._y_train_mean

            V = solve_triangular(
                self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
            )

            if return_cov:
                noise_var = self.get_noise_level()

                y_cov = self.kernel_(X) - V.T @ V - noise_var*np.eye(X.shape[0])


                y_cov = np.outer(y_cov, self._y_train_std ** 2).reshape(
                    *y_cov.shape, -1
                )

                if y_cov.shape[2] == 1:
                    y_cov = np.squeeze(y_cov, axis=2)

                return y_mean, y_cov
            elif return_std:
                noise_var = self.get_noise_level()
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ij,ji->i", V.T, V)
                y_var -= noise_var

                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn(
                        "Predicted variances smaller than 0. "
                        "Setting those variances to 0."
                    )
                    y_var[y_var_negative] = 0.0

                y_var = np.outer(y_var, self._y_train_std ** 2).reshape(
                    *y_var.shape, -1
                )

                if y_var.shape[1] == 1:
                    y_var = np.squeeze(y_var, axis=1)

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def get_var(self, X_ast):
        [mu,std] = self.predict(X_ast,return_std=True)
        return std

    def change_optimizer(self,optimizer):
        self.optimizer = optimizer
        self.kernel = self.kernel_

    def get_noise_level(self):
        params = self.kernel_.get_params()
        for key in sorted(params):
            if re.search("noise_level", key) and not re.search("bounds", key):
                return params[key]
        # White kernelがないときは予測分布からnoise varianceを引く必要がないため0を設定している
        return 0.
