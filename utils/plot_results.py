import matplotlib.pylab as plt
import numpy as np


def draw_pred_model(axes, al, node, time=None, y_range=None, title=None, x_label=None, y_label=None, isDisplay=False):
    if time is None:
        time = len(al.gp_history)-1
    axes.cla()
    axes.set_title(title)
    dataset = [al.X_sampled[:al.init_sample_size + time], al.y_sampled[:al.init_sample_size + time]]
    pos = al.gp_history[time].predict_noiseless(node, return_std=True)
    axes.scatter(dataset[0][:,0], dataset[1],c="k")
    axes.plot(node, pos[0],c="b",label="Mean function")
    axes.fill_between(node[:,0], pos[0]-2*pos[1], pos[0]+2*pos[1], color="b",alpha=0.5, label="Covariance function")
    if y_range is not None:
       axes.set_ylim(y_range[0]-0.1,y_range[1]+0.1)
    if x_label is not None:
        axes.set_xlabel(x_label)
    if y_label is not None:
        axes.set_ylabel(y_label)
    if isDisplay:
        axes.legend(bbox_to_anchor=(1, 1), loc='upper right').get_frame().set_alpha(1.0)

def draw_test_error(axes,al,test_error,diff_test_error=None,time=None):
    if time is None:
        time = al.stopping_criteria[0].error_ratio.shape[0]
    axes.cla()
    axes.plot(range(al.init_sample_size+1,test_error[:time].shape[0]+al.init_sample_size+1), test_error[:time],c="k", label="Test error")
    if diff_test_error is not None:
        axes.plot(range(al.init_sample_size+1,diff_test_error[:time].shape[0]+al.init_sample_size+1),diff_test_error[:time],c="b",label="Diff test errors")
    color = ["r","g","b"]
    for c,criterion in enumerate(al.stopping_criteria):
        if time >= criterion.stop_timings:
            axes.axvline(x=criterion.stop_timings+al.init_sample_size, ymin=-1e10, ymax=1e10, c=color[c],
                     linestyle="dashed", label=criterion.name)
    axes.legend(bbox_to_anchor=(1, 1), loc='upper right').get_frame().set_alpha(1.0)
    axes.set_xlim(al.init_sample_size+1,al.stopping_criteria[0].error_ratio.shape[0]+al.init_sample_size+1)
    axes.set_xlabel("Data size")
    axes.set_ylabel("Test error")

def draw_error_ratio(axes,al,time=None):
    if time is None:
        time = al.stopping_criteria[0].error_ratio.shape[0]
    axes.cla()
    axes.plot(range(al.init_sample_size+1,al.stopping_criteria[0].error_ratio[:time].shape[0]+al.init_sample_size+1), al.stopping_criteria[0].error_ratio[:time], c="k", label="Error ratio")
    axes.plot(range(al.init_sample_size+1,al.stopping_criteria[0].error_ratio[:time].shape[0]+al.init_sample_size+1), al.stopping_criteria[0].min_values[:time],c="r",label="Error ratio's minimum value")
    color = ["r","g","b"]
    for c,criterion in enumerate(al.stopping_criteria):
        if time >= criterion.stop_timings:
            axes.axvline(x=criterion.stop_timings+al.init_sample_size, ymin=-0.1, ymax=1.1, c=color[c],
                     linestyle="dashed", label=criterion.name)
    axes.legend(bbox_to_anchor=(1, 1), loc='upper right').get_frame().set_alpha(1.0)
    axes.set_xlim(al.init_sample_size+1,al.stopping_criteria[0].error_ratio.shape[0]+al.init_sample_size+1)
    axes.set_ylim(-0.1,1.1)
    axes.set_xlabel("Data size")
    axes.set_ylabel("Error ratio")
