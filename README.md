# Robustness of time-series models to adversarial attacks
## Abstract :memo:

Adversarial attacks are specialized techniques used to modify original data, aiming to degrade the accuracy of model predictions. 
While prevalent in computer vision, these attacks are also relevant in the time-series domain. 
Models exhibit varying susceptibility levels to such attacks. 

In our project we assess the robustness of time-series models against 3 adversarial strategies (DeepFool, SimBA, BIM). 
Specifically, the project entails implementing and training 3 simple custom architectures (transformer, CNN, LSTM) for time series classification tasks, subjecting them to various attacks, and evaluating the impact on performance metrics. 

## Usage :rocket:
To create virtual enviroment run

```bash
conda env create --file env.yml -n ts_robustness
```

### Files description

|||
| -------------| ------------- |
| [`train.py`](./train.py)   | train model from [`src.models`](./src/models) with json config. Configurations for results in tables are in [`configs`](./configs) folder  | 
| [`deep_fool_test.py`](./deep_fool_test.py)  | attack model from [`src.models`](./src/models) with deepfool attack from [`src.attacks.deepfool.py`](./src/attacks/deepfool.py) and return distribution of iterations for inversion of labels  |
| [`hyper_search`](hyper_search/transformer_search.py)  |  folder with scripts for hyper search for models | 


## Results :bar_chart:
### Test Quality
| Model / Dataset | FordA (acc, %)|
| :-------------:| :-------------: |
| CNN   | :x:  | 
| LSTM  | :x:  |
| Transformer  | 89 | 

### Implementation
| Model / Attack | DeepFool | SimBA | IFGSM |
| :-------------:| :-------------: | :-------------: | :-------------: |
| CNN   | :x:  | :x: | :x: |
| LSTM  | :x:  | :x: | :x: |
| Transformer  | :heavy_check_mark:  | :x: | :x: |
