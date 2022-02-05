import matplotlib.pylab as plt
from core.stopping_criteria import *
from utils import utils, plot_results
import os


def main():
    data_names = ["Ni2+", "Co2+", "Mn2+", "MnO2_1_exp", "MnO2_4_exp", "MnO2_7_exp", "Co_2_exp", "Co_3_exp", "Co_5_exp",
                  "Ni_3_exp", "Ni_4_exp", "Ni_6_exp"]
    plt.figure(1, [8, 8])
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    x_label = "Photon energy (eV)"
    y_label = "Intensity (arb. units)"
    for i, data_name in enumerate(data_names):
        save_dir = "result/" + data_name + "/"
        os.makedirs(save_dir, exist_ok=True)
        save_name = save_dir + "AL_example.dat"
        init_sample_size = 20
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
        for criterion in al.stopping_criteria:
            title = "Data size = {0}\n".format(criterion.stop_timings + init_sample_size) + "(" + criterion.name + ")"
            st = int(criterion.stop_timings)
            plot_results.draw_pred_model(ax1, al, node, st, y_range, title, x_label, y_label, False, True)
        title = "Data size = {0}\n".format(sample_size + init_sample_size)
        plot_results.draw_pred_model(ax3, al, node, y_range=y_range, title=title, x_label=x_label, y_label=y_label,
                                     plotCurrentData=False, isLegend=True)
        plot_results.draw_test_error(ax2, al, test_error, diff_test_error)
        plot_results.draw_error_ratio(ax4, al)
        plt.tight_layout()
        plt.savefig(save_dir + "fitting_" + data_name + ".pdf")
        plt.pause(0.001)


if __name__ == "__main__":
    main()
