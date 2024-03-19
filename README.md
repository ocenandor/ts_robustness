# Robustness of time-series models to adversarial attacks
## Abstract :memo:

Adversarial attacks are special techniques used to modify input data samples, aiming to changing model predictions and disrupt its work. 
This topic is very popular in computer vision, but can also be transferred to the time-series domain. 
Different machine learning models have different levels of sensitivity to adversarial attacks, the so-called robustness.

In our project we compare the robustness of models used in time-series binary classification task against 3 adversarial strategies: DeepFool, SimBA and BIM. 
Specifically, we train 3 state-of-the-art neural networks (LSTM, CNN, Transformer) with custom architectures for FordA dataset classification task, subjecting them to attacks, and compare the level of robustness. 

## Quick start :rocket:
1. Download repo:
```bash
git clone https://github.com/ocenandor/ts_robustness.git
```

2. Build docker image:
```bash
docker build . -t ts_robustness
```

3. Run image (don't forget to use gpu if available)
 ```bash
docker run  --gpus device=0 -it --rm ts_robustness
```

4. Run main.py to download data, train models and get some statistics of the models' robustness in report
```bash
python main.py
```


## Usage :joystick: (TODO)

data/downloadFordA.sh - download FordA dataset from source

tools:
  - train.py - script to train model
  - attack.py - base script to test model with attack
  - hypersearch.py - script for tuning models' hyperparameters with wandb sweeps

models - directory with models' weights
configs - directory with models' configuration files
||||
| -------------| ------------- |------------- |
| [`data/downloadFordA.sh`](./data/downloadFordA.sh) |download FordA dataset from source to [`data`](./data)|
| [`train.py`](./train.py)   | train model from [`src.models`](./src/models) with json config. Configurations for results in tables are in [`configs`](./configs) folder  |
| [`deep_fool_test.py`](./deep_fool_test.py)  | attack model from [`src.models`](./src/models.py) with deepfool attack from [`src.attacks.deepfool.py`](./src/attacks/deepfool.py) and return distribution of iterations for inversion of labels  |
| [`hyper_search`](hyper_search/transformer_search.py)  |  folder with scripts for hyper search for models |


## Results :bar_chart:
### Test Accuracy, %
| Model / Dataset | FordA (length=500)|
| :-------------:| :-------------: |
| CNN   | **86.80 CI 95% [85.94; 87.43]**  | 
| LSTM  | **79.56 CI 95% [79.50; 79.61]**  |
| Transformer  | **75.96 CI 95% [75.76; 76.15]** | 

### Test Accuracy (Max), %
| Model / Dataset | FordA (length=500)| FordA (length=150)|
| :-------------:| :-------------: |:-------------:|
| CNN   | **90.45**  |**89.48** |
| LSTM  | **81.36**|**88.09**|
| Transformer  | **77.80** | **89.11**|

### Implementation
| Model / Attack | DeepFool | SimBA | IFGSM (BIM)|
| :-------------:| :-------------: | :-------------: | :-------------: |
| CNN   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| LSTM  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| Transformer  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
