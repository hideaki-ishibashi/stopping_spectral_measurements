# Stopping spectral measurements

This is the code for our paper [Automated stopping criterion for spectral measurements with active learning](https://www.nature.com/articles/s41524-021-00606-5)

When you use this code in your publication, please cite the above paper as
```
@article{ueno2021,
    author={Ueno,T. and Ishibashi,H. and Hino,H. and Ono,K.},
    year={2021},
    title={Automated stopping criterion for spectral measurements with active learning},
    journal={npj Computational Materials},
    volume={7},
    number={1},
}
```

## Installation
Our code uses the following packages:
- cycler          0.11.0
- fonttools       4.29.1
- joblib          1.1.0
- kiwisolver      1.3.2
- matplotlib      3.5.1
- mpmath          1.2.1
- numpy           1.22.2
- packaging       21.3
- pandas          1.4.0
- pillow          9.0.1
- pyparsing       3.0.7
- python-dateutil 2.8.2
- pytz            2021.3
- scikit-learn    1.0.2
- scipy           1.7.3
- setuptools-scm  6.4.2
- six             1.16.0
- sklearn         0.0
- threadpoolctl   3.1.0
- tomli           2.0.0
- tqdm            4.62.3


If you use poetry, you can install the packages by running:
```
git clone https://github.com/hideaki-ishibashi/stopping_spectral_measurements.git
cd stopping_spectral_measurements
poetry install
```
Otherwise, you can install the packages by running:
```
git clone https://github.com/hideaki-ishibashi/stopping_spectral_measurements.git
cd stopping_spectral_measurements
pip install -r requirements.txt
```


## A brief overview of construction of our code

- `calc_hyper_param.py`
    - main code for calculating the hyperparameter of GP.
- `run_example.py` and `run_experimental_setting.py`
    - main code for stopping specral measurements.
- `plot_example.py`, `plot_animation.py` and `plot_experimental_result.py`
    - main code for visualization.
- `core`
    - codes defining active learning with GP and its stopping criterion.
- `utils`
    - utilities for accessing to dataset and result, plotting results and calculating some statistics are defined.
- `dataset`
    - spectrum dataset are placed. spectrum data in /dataset/simulation/ are calculated using CTM4XAS [E. Stavitski and F.M.F. de Groot, Micron 41, 687 (2010)].
- `param`
    - hyperparameter of GP caclulated by `calc_hyper_param.py` are placed.

## Usage

- Reproducing the experimental result of our article.
    - You can reproduce the result by executing `run_experimental_setting.py` after executing `calc_hyper_param.py`.
    - The figures of our article is reproduced by executing the `run_experimental_setting.py`.
    - The animation of active learning can be reproduced by executing the `plot_animation.py`.

- Applying the code to other dataset.
    - Active learning is implemented as `core.active_learning.AL`, whose arguments `pool_data`, `initial_sample_size`, `stopping_criteria`, `acq_funcition_type` and `isEarlystopping` in addition to the arguments of Gaussian process regressor of scikit-learn.
        - scikit-learn's GP can be seen [here](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).
        - `pool_data` is a list of training input and output.
        - `stopping_criteria` is a list of instances defined by `core.stopping_criteria.py`.
        - `isEarlystopping` is a flag whether to stop early the active learning. If it is `True`, the active learning is stopped when satisfying the stopping conditions. Otherwise, the active learning is not stopped. When `stopping_criteria` has mulitple stopping critrion, the active learning is stopped when satisfying the all conditions, but we recommend that the number of stopping criteria is one.
        - Possible value of `acq_func_type` is `"max_var"` or `"adaptive"`. When `acq_func_type="max_var"`, the acquisition function becomes maximum variance criterion. When `acq_func_type="adaptive"`, the acqusition function becomes the adaptive acquisition function used for our article. Other than that, next sample is selected randomly.
    - Active learning is executed by `explore`, whose argument is pre-determined budget.
    - In our code, the hyperparameter of GP can be estimated as a option, but we recommend using predetermined hyperparameter since this method tends to make the stop timing unstable.
        - KL divergence between GPs with different hyperparameters cannot be calculated exactly. So we calculate an approximated KL-divergence instead of the exact KL-divergence. Specifically, let <img src="https://latex.codecogs.com/svg.image?\theta_t" title="\theta_t" /> and <img src="https://latex.codecogs.com/svg.image?p_t(f&space;|&space;\rho)" title="p_t(f | \rho)" /> be a hyperparamter of GP posterior at time t and GP posterior with hyperparameter <img src="https://latex.codecogs.com/svg.image?\rho" title="\rho" /> at time t, respectively. In our code, we calculate <img src="https://latex.codecogs.com/svg.image?D_{\rm&space;KL}[p_t(f|\rho_t)||p_{t-1}(f|\rho_t)]" title="D_{\rm KL}[p_t(f|\rho_t)||p_{t-1}(f|\rho_t)]" /> and <img src="https://latex.codecogs.com/svg.image?D_{\rm&space;KL}[p_{t-1}(f|\rho_t)||p_t(f|\rho_t)]" title="D_{\rm KL}[p_{t-1}(f|\rho_t)||p_t(f|\rho_t)]" /> instead of <img src="https://latex.codecogs.com/svg.image?D_{\rm&space;KL}[p_t(f|\rho_t)||p_{t-1}(f|\rho_{t-1})]" title="D_{\rm KL}[p_t(f|\rho_t)||p_{t-1}(f|\rho_{t-1})]" /> and <img src="https://latex.codecogs.com/svg.image?D_{\rm&space;KL}[p_{t-1}(f|\rho_{t-1})||p_t(f|\rho_t)]" title="D_{\rm KL}[p_{t-1}(f|\rho_{t-1})||p_t(f|\rho_t)]" />.
    - Please refer to `run_example.py` and `plot_example.py` for implementation example.

- Stopping criterion
    - The implemented stopping criteria are `error_stability_criterion` and `error_stability_criterion_lambert`.
        - The difference of the two criteria is how the upper bound of the gap between expected generalization errors is evaluated. While `error_stability_criterion_lambert` calculates the upper bound by using the lambert function, `error_stability_criterion` calculates the upper bound by using Pinsker's inequality.
            - Pinsker-type upper bound is proposed by [D. Russo and B. V. Roy](https://www.jmlr.org/papers/volume17/14-087/14-087.pdf).
                - Daniel Russo and Benjamin Van Roy, ''An Information-Theoretic Analysis of Thompson Sampling'', *Journal of Machine Learning Research*, Vol. 17, No. 68, pp. 1 -- 30, 2016.
        - Although `error_stability_criterion_lambert` is used in our article, we recommend `error_stability_criterion` in practice since the pinsker-type upper bound is tighter than lamber-type upper bound. 
    - `check_threshold` calculates the error ratio and determines stopping timing.
    - `calcR_min` calculates the minimum value of KL-divergence and the function does not affect the stopping timing.


- GP regression
    - GP is implemented by using Gaussian process regressor of scikit-learn, but it is slightly modified, which is placed to `core.GPR`.
        - In order to estimate the noise variance as with other hyperparameters, we assumes that the kernel of GP is added white kernel and alpha of Gaussian process regressor is set to zero.
        - In scikit-learn, a covariance of GP posterior is added the noise variance when the noise variance is calculated as white kernel. So, we added `predict_noiseless` function to predict a noiseless posterior.

## License
The source code is licensed GNU General Public License v3.0,see LICENSE.
