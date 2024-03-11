# Robustness of time-series models to adversarial attacks
## Usage :memo:

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
