import matplotlib.pylab as plt
from core.stopping_criteria import *
from utils import utils, plot_results
import os


def main():
    data_names = ["Ni2+", "Co2+", "Mn2+", "MnO2_1_exp", "MnO2_4_exp", "MnO2_7_exp", "Co_2_exp", "Co_3_exp", "Co_5_exp",
                  "Ni_3_exp", "Ni_4_exp", "Ni_6_exp"]
    plt.figure(1, [15, 5])
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    x_label = "Photon energy (eV)"
    y_label = "Intensity (arb. units)"
    for i, data_name in enumerate(data_names):
        save_dir = "result/" + data_name + "/"
        os.makedirs(save_dir, exist_ok=True)
        save_name = save_dir + "AL_example.dat"
        # save_name = save_dir + "AL_experiment.dat"
        sample_size = 200

        X_test = np.loadtxt(save_dir + "X_test.txt")[:, None]
        y_test = np.loadtxt(save_dir + "y_test.txt")
        X_range = np.loadtxt(save_dir + "X_range.txt")
        y_range = np.loadtxt(save_dir + "y_range.txt")

        # load result of active learning
        al = utils.deserialize(save_name=save_name)
        node_size = 100
        node = np.linspace(X_range[0], X_range[1], node_size)[:, None]
        test_error, diff_test_error = al.get_test_error_history(X_test, y_test)

        # draw result
        for t in range(1,sample_size+1):
            title = "Data size = {0}\n".format(t + al.init_sample_size)
            plot_results.draw_pred_model(ax1, al, node, t, y_range, title, x_label, y_label, True, True)
            plot_results.draw_test_error(ax2, al, test_error, diff_test_error, t)
            plot_results.draw_error_ratio(ax3, al, t)
            plt.tight_layout()
            plt.savefig(save_dir + "fitting_" + data_name + str(t).zfill(3) + ".png")
            plt.pause(0.001)


if __name__ == "__main__":
    main()
