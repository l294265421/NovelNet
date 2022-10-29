# NovelNet
Modeling User Repeat Consumption Behavior for Online Novel Recommendation (RecSys 2022)

We implemented our model based on the session-recommedndation framework [session-rec](https://github.com/rn5l/session-rec), and detailed usage can be found there.

## Requirements
anaconda
python==3.7.1
torch==1.4.0 
scipy==1.6.2 
python-dateutil==2.8.1 
pytz==2021.1 
certifi==2020.12.5 
numpy==1.20.2 
dill==0.3.3 
pyyaml==5.4.1 
networkx==2.5.1 
scikit-learn==0.24.2 
numexpr==2.7.3 
keras==2.3.1 
six==1.15.0 
theano==1.0.3 
pandas==1.2.4 
psutil==5.8.0 
pympler==0.9 
tensorflow==2.3.0 
tables==3.6.1 
scikit-optimize==0.8.1 
python-telegram-bot==13.5

## Dataset
Datasets can be downloaded from: https://www.dropbox.com/sh/ur9amfhf9mag213/AAAtI7SWJft1WZZiR03nyDNCa?dl=0.
- Unzip the dataset file to the data folder.

## Running NovelNet
python run_confg.py conf/in_for_fiction_rec_more/84.yml conf/out

## Citation
```
@inproceedings{10.1145/3523227.3546762,
    author = {Li, Yuncong and Yin, Cunxiang and he, yancheng and Xu, Guoqiang and Cai, Jing and luo, leeven and Zhong, Sheng-hua},
    title = {Modeling User Repeat Consumption Behavior for Online Novel Recommendation},
    year = {2022},
    isbn = {9781450392785},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3523227.3546762},
    doi = {10.1145/3523227.3546762},
    abstract = {Given a user’s historical interaction sequence, online novel recommendation suggests the next novel the user may be interested in. Online novel recommendation is important but underexplored. In this paper, we concentrate on recommending online novels to new users of an online novel reading platform, whose first visits to the platform occurred in the last seven days. We have two observations about online novel recommendation for new users. First, repeat novel consumption of new users is a common phenomenon. Second, interactions between users and novels are informative. To accurately predict whether a user will reconsume a novel, it is crucial to characterize each interaction at a fine-grained level. Based on these two observations, we propose a neural network for online novel recommendation, called NovelNet. NovelNet can recommend the next novel from both the user’s consumed novels and new novels simultaneously. Specifically, an interaction encoder is used to obtain accurate interaction representation considering fine-grained attributes of interaction, and a pointer network with a pointwise loss is incorporated into NovelNet to recommend previously-consumed novels. Moreover, an online novel recommendation dataset is built from a well-known online novel reading platform and is released for public use as a benchmark. Experimental results on the dataset demonstrate the effectiveness of NovelNet 1.},
    booktitle = {Proceedings of the 16th ACM Conference on Recommender Systems},
    pages = {14–24},
    numpages = {11},
    keywords = {online novel recommendation, repeat consumption, interaction understanding},
    location = {Seattle, WA, USA},
    series = {RecSys '22}
}
```