# Robustness of time-series models to adversarial attacks
## Abstract :memo:

Adversarial attacks are specialized techniques used to modify original data, aiming to degrade the accuracy of model predictions. 
While prevalent in computer vision, these attacks are also relevant in the time-series domain. 
Models exhibit varying susceptibility levels to such attacks. 

In our project we assess the robustness of time-series models against 3 adversarial strategies (DeepFool, SimBA, BIM). 
Specifically, the project entails implementing and training 3 simple custom architectures (transformer, CNN, LSTM) for time series classification tasks, subjecting them to various attacks, and evaluating the impact on performance metrics. 

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

(In progress)4. Run main.sh to download data, train models and get some statistics in report
```bash
bash main.sh
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
### Test Quality
| Model / Dataset | FordA (acc, %)|
| :-------------:| :-------------: |
| CNN   | **89**  | 
| LSTM  | **78**  |
| Transformer  | **88** | 

### Implementation
| Model / Attack | DeepFool | SimBA | IFGSM |
| :-------------:| :-------------: | :-------------: | :-------------: |
| CNN   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| LSTM  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| Transformer  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
