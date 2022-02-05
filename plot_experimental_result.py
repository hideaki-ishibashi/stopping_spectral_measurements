import matplotlib.pylab as plt
from core.stopping_criteria import *
from utils import utils, plot_results
import os


def main():
    data_names = ["Ni2+", "Co2+", "Mn2+", "MnO2_1_exp", "MnO2_4_exp", "MnO2_7_exp", "Co_2_exp", "Co_3_exp", "Co_5_exp",
                  "Ni_3_exp", "Ni_4_exp", "Ni_6_exp"]
    x_label = "Photon energy (eV)"
    y_label = "Intensity (arb. units)"
    fig1 = plt.figure(1, [15, 10])
    axes = [plt.subplot(231), plt.subplot(232), plt.subplot(233), plt.subplot(234), plt.subplot(235), plt.subplot(236)]
    fig2 = plt.figure(2)
    ax2 = plt.subplot(111)
    fig3 = plt.figure(3)
    ax3 = plt.subplot(111)
    for i, data_name in enumerate(data_names):
        save_dir = "result/" + data_name + "/"
        save_name = save_dir + "AL_experiment.dat"
        os.makedirs(save_dir, exist_ok=True)
        init_sample_size = 10

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
        t_list = [0, 10, 50]
        label_list = ["Data size = {}".format(t_list[0] + init_sample_size),
                      "Data size = {}".format(t_list[1] + init_sample_size),
                      "Data size = {}".format(t_list[2] + init_sample_size)]
        for criterion in al.stopping_criteria:
            st = int(criterion.stop_timings)
            t_list.append(st)
            label = "Data size = {}\n".format(st + init_sample_size) + "(" + criterion.name + ")"
            label_list.append(label)
        plt.figure(fig1.number)
        for i, t in enumerate(t_list):
            plot_results.draw_pred_model(axes[i], al, node, t, y_range, label_list[i], x_label, y_label, False, True)
        fig1.tight_layout()
        fig1.savefig(save_dir + "fitting_" + data_name + ".pdf")
        plt.pause(0.001)
        plt.figure(fig2.number)
        plot_results.draw_test_error(ax2, al, test_error, diff_test_error)
        fig2.savefig(save_dir + "test_error_" + data_name + ".pdf")
        plt.pause(0.001)
        plt.figure(fig3.number)
        plot_results.draw_error_ratio(ax3, al)
        fig3.savefig(save_dir + "error_ratio_" + data_name + ".pdf")
        plt.pause(0.001)


if __name__ == "__main__":
    main()
