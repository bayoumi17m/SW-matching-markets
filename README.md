# README
Optimizing Rankings for Recommendation in Matching Markets

This repository contains various recommendation algorithms for two-sided or matching markets. The code accompanies the paper "Optimizing Rankings for Recommendation in Matching Markets" [WWW]() [arXiv]() where we firstly define the matching market, secondly show the underperformance of other methods, and propose a novel optimization setup to tackle this problem along with societal considerations.

If you find any module of this repository helpful for your own research, please consider citing the below WWW'22 paper. Thanks!

```bibtex
@inproceedings{SuBayoumiJoachims22,
  author = {Yi Su, Magd Bayoumi, and Thorsten Joachims},
  title = {Optimizing Rankings for Recommendation in Matching Markets},
  booktitle = {Proceedings of The Web Conference (WWW)},
  year = {2022}
}
```

## Accessing networking dataset
The networking dataset relevances can be found in `data/networking/networking_relevances.pkl`. If you would like the full imputed table of information then this is accessible [here](https://drive.google.com/file/d/1sdHNMRKCYnV9yzEoD5sWS6EpCOcuY7hZ/view?usp=sharing). Additionally, we provide a dataset README within `data/networking/README.md`

## Installation
All packages required can be installed via poetry, conda, or pip. Any of the following commands will work:

```bash
$ poetry install
$ pip install -r requirements.txt
$ conda create env -f environment.yml
```

## Reproducing experimental results
The experiments were conducted using PyTorch 1.7.1 and CVXPY 1.1.7 on a server with 16 cores and 64 GB of RAM.

### Synthetic experiments
To generate a synthetic relevance table and optimize, one can execute the `market_match_sw_optim.py` script. The following are command line arguments:

```
FLAGS
    --job_num
        Default: 50
        Number of employers
    --cand_num
        Default: 75
        Number of candidates
    --seed
        Default: 621
        seed used for RNG
    --environment
        Default: 'random'
        asymmetric level of relevance between jobs and cands
    --v_cand_type
        Default: 'inv'
        type of examination function to use for candidates
    --v_job_type
        Default: 'inv'
        type of examination function to use for employer
    --lr_sch
        Default: 'constant'
        learning rate schedule, one of "constant" or "decay"
    --noise_level
        Default: 0.2
        Sigma (Stdev) of noise to add to the asymmetric relevance
    --lambda_value
        Default: 0
        Level of non-congestion where 0 means full congestion
    --min_rel
        Default: 0.0
        Minimum relevance for congestion factor. The congestion will approach this value as it approaches less popular cands/jobs
    --epoch_num
        Default: 50
        Maximum number of epochs to update
```

This information can also be viewed via `python market_match_sw_optim.py --help`. To reproduce the case where we have:
* 100 jobs
* 150 candidates
* $v(x)=\frac{1}{x}$ for both sides
* minimum relevance was 0
* Non-Congestion level is 1 (ie. no congestion)
* Asymmetric level is random for both sides
* Max Epochs is 50

We can execute the following command:

```bash
python market_match_sw_optim.py \
    --job_num=100 \
    --cand_num=150 \
    --v_cand_type=inv \
    --v_job_type=inv  \
    --min_rel=0 \
    --lambda_value=1 \
    --environment=random \
    --epoch_num=50
```

This will output a json file with the candidate relevance table, employer relevance table, lower bounds over updates, and lastly the ranking policies for each candidate. The ranking policies can be used within a Monte carlo simulation. The simulation is run via `market_match_mc.py`.

For the simulation, it has the following options:

```
POSITIONAL ARGUMENTS
    JSON_PATH
        path to a json file with a candidate relevance table, job relevance table, and some ranking policies as matrices.

FLAGS
    --v_cand_type
        Default: 'inv'
        Examination function of the candidate
    --v_job_type
        Default: 'inv'
        Examination function of the job
    --runs
        Default: 1000
        Number of monte carlo runs to do for each ranking
    --seed
        Default: 621
        seed used for RNG
    --output_path
        Default: ''
        Path to output results to. Defaults to json_path with 'MC_' concatenated to the beginning of the name
```

The script always requires an input json which is the output of `market_match_sw_optim.py`. To simulate the environment we had above for 10,000 steps, one can execute:

```bash
python market_match_mc.py --json_path=<json output path> --v_cand_type=inv --v_job_type=inv --runs=10000 --output_path=Monte_carlo_output.json --seed=621
```
The experimental results from the paper for all synthetic experiments is listed below. The tables are associated with each of the subplots from Figure 2 in the paper.

**Asymmetic relevance**|**Examination function**|**Congestion ($\lambda$)**|**Market size**|**Naive-Relevance Ranking**|**Reciprocal-Relevance Ranking**|**Social-Welfare Ranking**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
Random|$\frac{1}{x}$|0|100|156.3 ± 0.03|199.9 ± 0.07|198.5 ± 0.05
Random|$\frac{1}{x}$|0.25|100|128.7 ± 0.06|166.6 ± 0.05|173.5 ± 0.05
Random|$\frac{1}{x}$|0.5|100|106.9 ± 0.04|131.0 ± 0.05|152.7 ± 0.03
Random|$\frac{1}{x}$|0.75|100|95.0 ± 0.04|105.0 ± 0.05|134.2 ± 0.02
Random|$\frac{1}{x}$|1|100|92.1 ± 0.03|92.1 ± 0.03|118.2 ± 0.03

**Asymmetic relevance**|**Examination function**|**Congestion ($\lambda$)**|**Market size**|**Naive-Relevance Ranking**|**Reciprocal-Relevance Ranking**|**Social-Welfare Ranking**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
Similar|$\frac{1}{x}$|0.5|100|126.2 ± 0.07|143.8 ± 0.05|167.4 ± 0.05
Reverse|$\frac{1}{x}$|0.5|100|65.1 ± 0.05|104.5 ± 0.04|128.4 ± 0.03
Random|$\frac{1}{x}$|0.5|100|106.9 ± 0.04|131.0 ± 0.05|152.7 ± 0.03

**Asymmetic relevance**|**Examination function**|**Congestion ($\lambda$)**|**Market size**|**Naive-Relevance Ranking**|**Reciprocal-Relevance Ranking**|**Social-Welfare Ranking**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
Random|$\frac{1}{e^{x-1}}$|0.5|100|25.9 ± 0.03|47.8 ± 0.02|84.4 ± 0.04
Random|$\frac{1}{x}$|0.5|100|106.9 ± 0.04|131.0 ± 0.05|152.7 ± 0.03
Random|$\frac{1}{\log(x+1)}$|0.5|100|639.2 ± 0.04|669.7 ± 0.03|673.7 ± 0.10

**Asymmetic relevance**|**Examination function**|**Congestion ($\lambda$)**|**Market size**|**Naive-Relevance Ranking**|**Reciprocal-Relevance Ranking**|**Social-Welfare Ranking**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
Random|$\frac{1}{x}$|0.5|20|18.7 ± 0.02|21.2 ± 0.02|23.2 ± 0.02
Random|$\frac{1}{x}$|0.5|50|50.7 ± 0.03|61.4 ± 0.03|68.9 ± 0.05
Random|$\frac{1}{x}$|0.5|100|106.9 ± 0.04|131.0 ± 0.05|152.7 ± 0.03
Random|$\frac{1}{x}$|0.5|200|220.1 ± 0.03|274.3 ± 0.06|332.6 ± 0.07

### Real world dating data experiments
We utilize the Libimseti [1] dating site dataset of ratings. We do a preprocessing step to select a subset of the data for computation purposes (the dataset contains about 200k users), along with an in-filling procedure. We assume there is a `dating_data` folder that contains `ratings.dat` and `gender.dat` from the libimseti dataset.

**Below are more specific implementation details but are not neccessary to run the code.**
The rest is very similar to the synthetic experiments with an additional step. To reduce computational complexity, we use a two-stage ranking procedure. We first identify the top 100 results based on their reciprocal relevance, and only re-rank those to maximize social welfare. The ranking after the top 100 is by reciprocal relevance.

#### Data preprocessing
To preprocess the data we can execute:

```bash
python libimseti_preprocessing.py
```

This will create the female relevance table used for our experiments (`dating_data/female_to_male_rel_500.pkl`) as well as the male relevance table used for our experiments (`dating_data/male_to_female_rel_500.pkl`).

#### Experiments
With the relevance tables given from our preprocessing step, we perform the optimization using `dating_market_sw_optim.py`. The following are command line arguments:

```
POSITIONAL ARGUMENTS
    MALE_REL_PATH
        path to a pickle file with a male relevance table
    FEMALE_REL_PATH
        path to a pickle file with a female relevance table

FLAGS
    --seed
        Default: 621
        seed used for RNG
    --v_cand_type
        Default: 'inv'
        type of examination function to use for candidates
    --v_job_type
        Default: 'inv'
        type of examination function to use for employer
    --lr_sch
        Default: 'constant'
        learning rate schedule, one of "constant" or "decay"
    --epoch_num
        Default: 50
        Maximum number of epochs to update
```
This information can also be viewed via `python dating_match_sw_optim.py --help`. To reporduce table 1 results, one can execute the following command:

```bash
python dating_match_sw_optim.py \
    --male_rel_path=dating_data/male_to_female_rel_500.pkl \
    --female_rel_path=dating_data/female_to_male_rel_500.pkl \
    --v_cand_type=inv \
    --v_job_type=inv  \
    --epoch_num=50
```

This will output a json file with the candidate relevance table, employer relevance table, lower bounds over updates, and lastly the ranking policies for each candidate. It will additionally output a pickle file with the ranking policies per epoch. The ranking policies can be used within a Monte carlo simulation. The simulation is run via `dating_match_mc.py`.

For the simulation, it has the following options:

```
POSITIONAL ARGUMENTS
    MALE_REL_PATH
        path to a pickle file with a male relevance table
    FEMALE_REL_PATH
        path to a pickle file with a female relevance table
    PC_MATRICES
        path to a pickle file with some ranking policies as matrices

FLAGS
    --v_cand_type
        Default: 'inv'
        Examination function of the candidate
    --v_job_type
        Default: 'inv'
        Examination function of the job
    --runs
        Default: 1000
        Number of monte carlo runs to do for each ranking
    --seed
        Default: 621
        seed used for RNG
    --output_path
        Default: ''
        Path to output results to. Defaults to json_path with 'MC_' concatenated to the beginning of the name
```

The script always requires a few input paths which is the output of `dating_match_sw_optim.py` and `libimseti_preprocessing.py`. To simulate the environment we had above for 10,000 steps, one can execute:

```bash
python dating_match_mc.py --male_rel_path="dating_data/male_to_female_rel_500.pkl" --female_rel_path="dating_data/female_to_male_rel_500.pkl" --pc_matrices="dating_500_inv_epoch_50.pkl" --v_cand_type=inv --v_job_type=inv --runs=10000 --output_path=Monte_carlo_output.json --seed=621
```
The results from the paper are below:

**Dataset**|**Networking Recommendation**|**Online Dating (Limimseti)**
:-----:|:-----:|:-----:
Naive-Relevance Ranking|604.0 ± 0.11|844.0 ± 0.10
Reciprocal-Relevance Ranking|763.9 ± 0.06|957.2 ± 0.12
Social-Welfare Ranking|824.1 ± 0.18| 1199.2 ± 0.14


## References
[1] Peng Xia, Benyuan Liu, Yizhou Sun, and Cindy Chen. Reciprocal recommendation system for online dating. InProceedings of the 2015 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining 2015, ASONAM ’15, page 234–241, New York, NY, USA, 2015. Association for Computing Machinery.

[2] Arkadiusz Paterek. Improving regularized singular value decomposition for collaborative filtering. 2007

