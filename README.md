# Stopping spectral measurements

This is the code for our paper [Automated stopping criterion for spectral measurements with active learning](https://www.nature.com/articles/s41524-021-00606-5)

Please cite us:
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

- calc_hyper_param.py
    - main code for calculating the hyperparameter of GP
- run_example.py and run_experimental_setting.py
    - main code for stopping specral measurements
- plot_example.py, plot_animation.py and plot_experimental_result.py
    - main code for visualization
- core
    - codes defining active learning with GP and its stopping criterion.
- utils
    - utilities for accessing to dataset and result, plot results and calculation some statistics are defined.
- dataset
    - spectrum dataset are placed.
- param
    - hyperparameter of GP caclulated by calc_hyper_param.py are placed.

## Usage

- Reproducing the experimental result of our article.
    - You can reproduce the result by executing run_experimental_setting.py after you executing calc_hyper_param.py.
    - The figures of our article is reproduced by executing the run_experimental_setting.py.
    - The animation of active learning can be reproduced by executing the plot_animation.py.

- Applying the code to other dataset.
    - 能動学習はcore.active_learningのALで実装されており，ALクラスの引数はscikit learnのGPの引数に加え，学習データのプールデータ，初期サンプルサイズ，stopping_criteria，獲得関数の種類を必要としている．
        - stopping_criteriaはcore.stopping_criteria.pyで定義されたクラスのインスタンスのリスト．
        - isEarlystoppingは停止基準を使って早期停止するかどうかのフラグ．Trueなら停止条件を満たしたら早期停止し，Falseなら停止条件を満たしても早期停止しない．停止基準のリストが複数個あった場合，全ての条件を満たす時に学習を停止する．通常は停止基準の数は１つを推奨．
        - 獲得関数は分散最大化基準（acq_func_type="max_var"）と論文で用いているadaptiveな獲得関数（acq_func_type="adaptive"）の２種類が実装されている．acq_func_typeを"max_var"と"adaptive"以外にするとランダム探索になる．
        - GP of scikit learn is denoted [here](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).
    - 能動学習の探索はALクラスのexplore関数で行える．explore関数はあらかじめ与えられたbudgetを引数としている．
    - 能動学習の過程でGPのハイパーパラメータも周辺尤度最大化により推定することができる．ただし，停止基準の停止タイミングが不安定になりやすいため，あらかじめハイパーパラメータを決めておく方法を推奨. 
        - ハイパーパラメータが異なるGP間のKLダイバージェンスは厳密には計算できないが，このコードではGPのハイパーパラメータも逐次推定するときは更新されたハイパーパラメータを使ってGP間のKLダイバージェンスを近似的に計算している．具体的には，新しくデータを取得しGPのハイパーパラメータを更新した後，そのハイパーパラメータを用いてデータを取得する前後の事後分布を再度計算し直し，その事後分布間のKLダイバージェンスを使って停止基準を計算している．

- Stopping criterion
    - 停止基準のクラスはcore.stopping_criteria.pyに定義されている．
    - 実装されている停止基準はerror_stability_criterionとerror_stability_criterion_lambertの２つの停止基準がある．
        - ２つの停止基準の違いは期待汎化誤差の差分の上界の違いであり，error_stability_criterion_lambertはランベルト関数を用いた上界に基づく停止基準であり，error_stability_criterionはPinskerの不等式に基づく停止基準である． 
        - 論文ではランベルト関数を用いた上界を実験で用いていたがPinskerの不等式に基づく上界がよりタイトな上界になるため実際の運用ではerror_stability_criterionを用いることを推奨．
    - 停止基準の計算と停止条件を満たしたかどうかの判定はcheck_threshold関数によって行う．
    - calcR_min関数はKLダイバージェンスの最小値を計算している関数であり，停止基準には影響を与えない．


- GP regression
    - GPはscikit learnのGaussian process regressorで実装されているが以下の理由でGaussian process regressorを継承したクラスをcore.GPRに定義し，それを用いている．
    - このコードではGPのノイズの分散も周辺尤度最大化できるようにするためにwhiteカーネルが加算されていることを前提としており，Gaussian process regressorのalphaはalpha=0に設定している．
    - scikit learnのコードではノイズの分散をwhiteカーネルとして計算する場合，事後分散にノイズが加わる．そのため，ハイパラの最適化とノイズなしの予測分布の推定を両立できないため，ノイズなしの予測分布を計算する関数を追加している．

## License
The source code is licensed GNU General Public License v3.0,see LICENSE.