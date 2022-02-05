import numpy as np
from core.GPR import GPR
from utils import utils
from tqdm import tqdm
import random
import warnings


class AL(object):
    def __init__(self, X_pool, y_pool, init_sample_size, stopping_criteria, kernel, acq_func_type="max_var", isEarlystopping=False, normalize_y=False,
                 optimizer=None, n_restarts_optimizer=0, copy_X_train=False, max_iter=2e05,random_state=None):
        self.gp_params = {"kernel": kernel, "alpha": 0, "optimizer": optimizer,
                          "n_restarts_optimizer": n_restarts_optimizer, "normalize_y": normalize_y,
                          "copy_X_train": copy_X_train, "max_iter": max_iter, "random_state": None}
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.init_sample_size = init_sample_size
        self.pool_size = self.X_pool.shape[0]
        [self.sampled_indecies, self.pool_indecies] = utils.get_init_samples(init_sample_size, self.pool_size)
        self.acquired_indecies = []
        self.X_sampled = self.X_pool[self.sampled_indecies]
        self.y_sampled = self.y_pool[self.sampled_indecies]
        self.stopping_criteria = stopping_criteria
        self.acq_func_type = acq_func_type
        self.gp_history = []
        self.isEarlystopping = isEarlystopping
        warnings.simplefilter('ignore')

    def data_acquire(self, gp, pool_data, pool_indecies, current_time):
        [mu, var] = gp.predict_noiseless(pool_data[pool_indecies], return_std=True)
        if self.acq_func_type == "max_var":
            index = pool_indecies[np.argmax(var)]
        elif self.acq_func_type == "adaptive":
            index = pool_indecies[np.argmax(var / var.max() + 1 / current_time * np.sqrt(np.abs(mu / np.max(mu))))]
        else:
            index = random.sample(pool_indecies, 1)[0]
        return index

    def explore(self, max_iters=100):
        gp = GPR(**self.gp_params)
        gp.fit(self.X_sampled, self.y_sampled)
        self.gp_history.append(gp)
        self.sample_size = len(self.pool_indecies)
        if max_iters < self.sample_size:
            self.sample_size = max_iters
        for t in tqdm(range(1, self.sample_size + 1)):
            # acquire new input
            new_data_index = self.data_acquire(gp, self.X_pool, self.pool_indecies, t)
            self.sampled_indecies.append(new_data_index)
            self.acquired_indecies.append(new_data_index)
            self.pool_indecies.remove(new_data_index)

            # update training dataset
            self.X_sampled = self.X_pool[self.sampled_indecies]
            self.y_sampled = self.y_pool[self.sampled_indecies]

            # update gp posterior
            gp = GPR(**self.gp_params)
            gp.fit(self.X_sampled, self.y_sampled)
            self.gp_history.append(gp)

            # calculate stopping criterion
            params = np.exp(gp.kernel_.theta)
            self.gp_params["kernel"] = gp.kernel_
            gp_params = self.gp_params.copy()
            gp_params["optimizer"] = None
            gp_old = GPR(**gp_params)
            gp_old.fit(self.X_sampled[:-1], self.y_sampled[:-1])
            pos_old = gp_old.predict_noiseless(self.X_sampled, return_cov=True)
            noise_var = gp.get_noise_level()
            KL_pq = utils.calcKL_pq_fast(pos_old, self.y_sampled, 1 / noise_var)
            KL_qp = utils.calcKL_qp_fast(pos_old, self.y_sampled, 1 / noise_var)
            KL_pq_min = utils.calcKL_pq_min(pos_old, 1 / params[-1])
            KL_qp_min = utils.calcKL_qp_min(pos_old, 1 / params[-1])
            for criterion in self.stopping_criteria:
                criterion.calcR_min(KL_pq_min, KL_qp_min, t)
                criterion.check_threshold(KL_pq, KL_qp, t)

            # confirm of stop conditions
            if self.isEarlystopping and all(list(map(lambda sc: sc.stop_flags, self.stopping_criteria))):
                break

    def get_posterior_history(self, node):
        pos_list = []
        for t, gp in enumerate(self.gp_history):
            pos = gp.predict_noiseless(node, return_std=True)
            pos_list.append(pos)
        return pos_list

    def get_test_error_history(self, X_test, y_test):
        test_error = []
        diff_test_error = []
        for i, gp in enumerate(self.gp_history):
            mu, std = gp.predict_noiseless(X_test, return_std=True)
            error = utils.calc_expected_squre_error(y_test, [mu, std ** 2])
            test_error = np.append(test_error, error)
            if i != 0:
                diff_test_error = np.append(diff_test_error, test_error[-2] - test_error[-1])
        return test_error, diff_test_error
