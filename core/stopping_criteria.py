import numpy as np
import mpmath as mp


class base_criterion(object):
    def __init__(self, name):
        self.name = name
        self.stop_flags = False
        self.stop_timings = np.nan


class error_stability_criterion(base_criterion):
    def __init__(self, threshold, validate_size=1):
        super(error_stability_criterion, self).__init__(r"$\lambda={}$".format(threshold))
        self.error_ratio = np.empty(0, float)
        self.min_values = np.empty(0, float)
        self.threshold = threshold
        self.validate_size = validate_size
        self.R = np.empty(0, float)
        self.R_min = np.empty(0, float)

    def check_threshold(self, KL_pq, KL_qp, current_time):
        self.R = np.append(self.R, np.sqrt(0.5 * KL_pq) + np.sqrt(0.5 * KL_qp))
        if self.validate_size <= current_time:
            error_ratio = self.R[-1] / self.R[:self.validate_size + 1].min()
            if self.validate_size == current_time:
                error_ratio = 1
            self.error_ratio = np.append(self.error_ratio, error_ratio)
            if self.error_ratio[-1] <= self.threshold and not self.stop_flags:
                self.stop_timings = current_time
                print("{} : {}".format(self.name, current_time))
                self.stop_flags = True
        else:
            self.error_ratio = np.append(self.error_ratio, 1)

    def calcR_min(self, KL_pq_min, KL_qp_min, current_time):
        self.R_min = np.append(self.R_min, np.sqrt(0.5 * KL_pq_min) + np.sqrt(0.5 * KL_qp_min))
        if self.validate_size <= current_time:
            min_values = self.R_min[-1] / self.R[:self.validate_size + 1].min()
            if self.validate_size == current_time:
                min_values = 1
            self.min_values = np.append(self.min_values, min_values)
        else:
            self.min_values = np.append(self.min_values, 1)


class error_stability_criterion_lambert(base_criterion):
    def __init__(self, threshold, validate_size=1):
        super(error_stability_criterion_lambert, self).__init__(r"$\lambda={}$".format(threshold))
        self.error_ratio = np.empty(0, float)
        self.min_values = np.empty(0, float)
        self.threshold = threshold
        self.validate_size = validate_size
        self.R = np.empty(0, float)
        self.R_min = np.empty(0, float)

    def check_threshold(self, KL_pq, KL_qp, current_time):
        tol = 1e-10
        u_pq = (KL_pq - 1) / (np.exp(1))
        if u_pq > -1 / mp.e + tol:
            Lambda_upper = float(mp.lambertw(u_pq) + 1)
        else:
            Lambda_upper = 0
        u_qp = (KL_qp - 1) / (np.exp(1))
        if u_qp > -1 / mp.e + tol:
            Lambda_lower = float(mp.lambertw(u_qp) + 1)
        else:
            Lambda_lower = 0
        self.R = np.append(self.R, np.exp(Lambda_upper) + np.exp(Lambda_lower) - 2)
        if self.validate_size <= current_time:
            error_ratio = self.R[-1] / self.R[:self.validate_size + 1].min()
            if self.validate_size == current_time:
                error_ratio = 1
            self.error_ratio = np.append(self.error_ratio, error_ratio)
            if self.error_ratio[-1] <= self.threshold and not self.stop_flags:
                self.stop_timings = current_time
                print("{} : {}".format(self.name, current_time))
                self.stop_flags = True
        else:
            self.error_ratio = np.append(self.error_ratio, 1)

    def calcR_min(self, KL_pq_min, KL_qp_min, current_time):
        tol = 1e-10
        u_pq = (KL_pq_min - 1) / (np.exp(1))
        if u_pq > -1 / mp.e + tol:
            Lambda_upper = float(mp.lambertw(u_pq) + 1)
        else:
            Lambda_upper = 0
        u_qp = (KL_qp_min - 1) / (np.exp(1))
        if u_qp > -1 / mp.e + tol:
            Lambda_lower = float(mp.lambertw(u_qp) + 1)
        else:
            Lambda_lower = 0
        R = np.exp(Lambda_upper) - 1 + np.exp(Lambda_lower) - 1
        self.R_min = np.append(self.R_min, R)
        if self.validate_size <= current_time:
            min_values = self.R_min[-1] / self.R[:self.validate_size + 1].min()
            if self.validate_size == current_time:
                min_values = 1
            self.min_values = np.append(self.min_values, min_values)
        else:
            self.min_values = np.append(self.min_values, 1)
