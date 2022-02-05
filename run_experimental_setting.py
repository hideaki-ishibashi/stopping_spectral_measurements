from core.stopping_criteria import *
from utils import utils, get_dataset
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from core.active_learning import AL
import random
import os


def main():
    np.random.seed(1)
    random.seed(1)
    data_names = ["Ni2+", "Co2+", "Mn2+", "MnO2_1_exp", "MnO2_4_exp", "MnO2_7_exp", "Co_2_exp", "Co_3_exp", "Co_5_exp",
                  "Ni_3_exp", "Ni_4_exp", "Ni_6_exp"]
    param_names = ["Ni2+", "Co2+", "Mn2+", "Ni_3_exp", "Ni_4_exp", "Ni_6_exp", "Ni_3_exp", "Ni_4_exp", "Ni_6_exp",
                   "Ni_3_exp", "Ni_4_exp", "Ni_6_exp"]
    for i, data_name in enumerate(data_names):
        save_dir = "result/" + data_name + "/"
        save_name = save_dir + "AL_experiment.dat"
        os.makedirs(save_dir, exist_ok=True)
        param_dir = "param/" + param_names[i] + "/"

        # get dataset
        [X, y] = get_dataset.get_dataset(data_name)
        print(X.shape)
        train_size = 900
        init_sample_size = 10
        sample_size = 200
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        np.savetxt(save_dir + "X_test.txt", X_test)
        np.savetxt(save_dir + "y_test.txt", y_test)
        np.savetxt(save_dir + "X_range.txt", [X.min(), X.max()])
        np.savetxt(save_dir + "y_range.txt", [y.min(), y.max()])

        # set stopping criterion
        threshold = 0.1
        validate_size = 10
        error_stability1 = error_stability_criterion_lambert(threshold, validate_size)
        threshold = 0.05
        error_stability2 = error_stability_criterion_lambert(threshold, validate_size)
        threshold = 0.025
        error_stability3 = error_stability_criterion_lambert(threshold, validate_size)
        stopping_criteria = [error_stability1, error_stability2, error_stability3]

        # calculate active learning
        params = np.loadtxt(param_dir + "params.txt")
        kernel = params[0] * RBF(length_scale=params[1]) + WhiteKernel(noise_level=params[2])
        al = AL([X_train, y_train], init_sample_size, stopping_criteria, acq_func_type="adaptive", kernel=kernel)
        al.explore(max_iters=sample_size)

        # set a stop timing to budget when the stopping criterion does not stop the AL
        for criterion in al.stopping_criteria:
            if criterion.stop_timings is np.nan:
                criterion.stop_timings = sample_size

        # save result
        utils.serialize(al, save_name=save_name)


if __name__ == "__main__":
    main()
