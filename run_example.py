from core.stopping_criteria import *
from utils import utils, get_dataset
from core.active_learning import AL
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import os


def main():
    # np.random.seed(1)
    # random.seed(1)
    data_names = ["Ni2+", "Co2+", "Mn2+", "MnO2_1_exp", "MnO2_4_exp", "MnO2_7_exp", "Co_2_exp", "Co_3_exp", "Co_5_exp",
                  "Ni_3_exp", "Ni_4_exp", "Ni_6_exp"]
    param_names = ["Ni2+", "Co2+", "Mn2+", "Ni_3_exp", "Ni_4_exp", "Ni_6_exp", "Ni_3_exp", "Ni_4_exp", "Ni_6_exp",
                   "Ni_3_exp", "Ni_4_exp", "Ni_6_exp"]
    for i, data_name in enumerate(data_names):
        save_dir = "result/" + data_name + "/"
        os.makedirs(save_dir, exist_ok=True)
        param_dir = "param/" + param_names[i] + "/"
        save_name = save_dir + "AL_example.dat"

        # get dataset
        [X, y] = get_dataset.get_dataset(data_name)
        train_size = 900
        init_sample_size = 20
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
        threshold = 0.05
        validate_size = 10
        error_stability = error_stability_criterion(threshold, validate_size)
        stopping_criteria = [error_stability]

        # calculate active learning
        params = np.loadtxt(param_dir + "params.txt")
        kernel = params[0] * RBF(length_scale=params[1]) + WhiteKernel(noise_level=params[2])
        al = AL(X_train, y_train, init_sample_size, stopping_criteria, kernel=kernel)
        # kernel = 1 * RBF(length_scale=1) + WhiteKernel(noise_level=0.1)
        # al = AL(X_train, y_train, init_sample_size, [error_stability], isEarlystopping=True, kernel=kernel,
        #         optimizer="fmin_l_bfgs_b", max_iter=10, n_restarts_optimizer=2)
        al.explore(max_iters=sample_size)

        # set a stop timing to budget when the stopping criterion does not stop the AL
        for criterion in al.stopping_criteria:
            if criterion.stop_timings is np.nan:
                criterion.stop_timings = sample_size

        # save result
        utils.serialize(al, save_name=save_name)


if __name__ == "__main__":
    main()
