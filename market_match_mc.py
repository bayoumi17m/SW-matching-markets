import concurrent.futures as cf
import copy
import warnings
warnings.filterwarnings("ignore")
import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from sklearn import preprocessing
import cvxpy as cp
from tqdm import tqdm, trange


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def to_permutation_matrix(vec):
    """to_permutation_matrix creates a permuation matrix from a sorted array.

    :param vec: array representing the desired 
    :type vec: 1D np.ndarray of dtype np.integer
    :returns: permutation matrix from the sorted array
    :rtype: np.ndarry of dtype np.float
    """
    return np.eye(vec.size)[vec].T

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def init_probmat(d, uniform_init=True, log=False):
    """init_probmat initializes the problem matrix for optimization.

    The problem matrix is a doubly stochastic matrix used as a stochastic
    ranking policy. 

    :param d: Dimension of the problem matrix, ie. |J|
    :type d: int
    :param uniform_init: Whether to initialize uniformly or randomly
    :type uniform_init: bool
    :param log: Whether to take the log of the doubly stochastic matrix
    :type log: bool
    """
    if uniform_init:
        init_mat = np.ones((d,d))/d
    else:
        init_mat = np.random.rand(d,d)
        init_mat = preprocessing.normalize(init_mat, norm="l1", axis=0)
        init_mat = preprocessing.normalize(init_mat, norm="l1", axis=1)
        init_mat = preprocessing.normalize(init_mat, norm="l1", axis=0)
        init_mat = preprocessing.normalize(init_mat, norm="l1", axis=1)
        init_mat = preprocessing.normalize(init_mat, norm="l1", axis=0)
        init_mat = preprocessing.normalize(init_mat, norm="l1", axis=1)
    if log:
        mat = np.log(init_mat)
    else:
        mat = init_mat
    return torch.tensor(mat, requires_grad=True)

def simulate_market(runs, cand_rel, job_rel, Pc_list, v_cand_type, v_job_type):
    """simulate_market runs a monte carlo simulation of a matching market.

    This function assumes a two-sided market using 

    :param runs: Number of times to run the MC simulation
    :type runs: int
    :param cand_rel: Relevance table denoting for candidate c the relevance of job j.
        Each row is a candidate c and each column denotes a job j. Denoted f_c(j)
        in paper.
    :type cand_rel: torch.tensor of shape (|C|, |J|) with all values in [0,1].
    :param job_rel: Relevance table denoting for job j the relevance of candidate c.
        Each row is a job j and each column denotes a candidate c. Denoted g_j(c)
        in paper.
    :type job_rel: torch.tensor of shape (|J|, |C|) with all values in [0,1].
    :param Pc_list: Ranking policies for each candidate where Pc_list[c] is the
        ranking policy for candidate c.
    :type Pc_list: list of torch.tensor of shape (|J|, |J|) that are doubly
        stochastic matrices
    :param v_cand_type: The type of examination function to be used for the
        candidate. Must be one of {inv, log, inv3, exp}.
    :type v_cand_type: str 
    :param v_job_type: The type of examination function to be used for the
        employer. Must be one of {inv, log, inv3, exp}.
    :type v_job_type: str
    """
    cand_num, job_num = cand_rel.shape

    Pc_sim = [Pc.detach().numpy() for Pc in Pc_list]
    if v_cand_type == "inv":
        v_cand = torch.tensor(1./(np.arange(job_num, dtype=np.float32)+1), dtype=torch.float32)
    elif v_cand_type == "log":
        v_cand = torch.tensor(1./np.log(np.arange(job_num)+2), dtype=torch.float32)
    elif v_cand_type == "inv3":
        v_cand = torch.tensor(1./(np.power(3,np.arange(job_num, dtype=np.float32))), dtype=torch.float32)
    elif v_cand_type == "exp":
        v_cand = torch.tensor(1./(np.exp(np.arange(job_num, dtype=np.float32))), dtype=torch.float32)
    
    if v_job_type == "inv":
        v_job = torch.tensor(1./(np.arange(cand_num, dtype=np.float32)+1), dtype=torch.float32)
    elif v_job_type == "log":
        v_job = torch.tensor(1./np.log(np.arange(cand_num)+2), dtype=torch.float32)
    elif v_job_type == "inv3":
        v_job = torch.tensor(1./(np.power(3,np.arange(cand_num, dtype=np.float32))), dtype=torch.float32)
    elif v_job_type == "exp":
        v_job = torch.tensor(1./(np.exp(np.arange(cand_num, dtype=np.float32))), dtype=torch.float32)

    interviews = np.zeros((runs, cand_num, job_num))

    # Get expected examination and application probability 
    exam_prob = np.array([np.dot(ranking_policy, v_cand) for ranking_policy in Pc_sim])
    apply_prob = np.multiply(cand_rel, exam_prob)
   
    # Sometimes precision can cause errors when sampling so clip values 
    apply_prob = np.clip(apply_prob, 0, 1)

    for epoch in trange(runs, leave=False, desc="Sim Count"):
        # Application simulation
        apply_sim = np.random.binomial(n=1, p=apply_prob)
        # Application relevance table
        appl_rel_table = np.multiply(apply_sim.T, job_rel)

        interview_sim = np.zeros((job_num, cand_num))
        for job in range(job_num):
            bool_filter = appl_rel_table[job] > 0
            # Correcting the view to match applications recieved and their
            # relevance ordering
            view_ord = np.argsort(-appl_rel_table[job])
            view_corr = np.argsort(view_ord)
            # v_job for job j based on the applications and their relevance
            v_job_temp = np.where(bool_filter, v_job[view_corr], 0)
            interview_probs = np.multiply(job_rel[job], v_job_temp)
            # Sometimes precision can cause errors when sampling so clip values
            interview_probs = np.clip(interview_probs, 0, 1)
            # probability of interview being given
            interview_sim_j = np.random.binomial(n=1, p=interview_probs) # Prob of interview
            interview_sim[job] = interview_sim_j

        interviews[epoch] = interview_sim.T
    # SW objective
    interview_counts_per_sim = interviews.sum(axis=-1).sum(axis=-1)
    expectation, Sx = interview_counts_per_sim.mean(), interview_counts_per_sim.std()
    candidate_expected_utility = interviews.sum(axis=-1).sum(axis=0) / runs
    job_expected_utility = interviews.sum(axis=1).sum(axis=0) / runs

    return expectation, Sx, candidate_expected_utility, job_expected_utility 


def main(json_path, v_cand_type="inv", v_job_type="inv", runs=1000, seed=621, output_path=""):
    """main runs a full market simulation experiment. It will
    run the simulation for Greedy ranking, Reciprocal ranking and the ranking
    policy given within the json file.

    :param json_path: path to a json file with a candidate relevance table,
        job relevance table, and some ranking policies as matrices.
    :type json_path: str
    :param v_cand_type: Examination function of the candidate
    :type v_cand_type: str
    :param v_job_type: Examination function of the job
    :type v_job_type: str
    :param runs: Number of monte carlo runs to do for each ranking
    :type runs: int
    :param seed: seed used for RNG
    :type seed: int
    :param output_path: Path to output results to. Defaults to json_path with
        'MC_' concatenated to the beginning of the name
    :type output_path: str, optional
    """
    if not output_path:
        # If no output is given, assign the path to be same but with MC_
        # concatenated to the name
        output_path = Path(json_path).parent / ("MC_" + Path(json_path).name)
    
    # Load data and place in variables
    with Path(json_path).open() as fp:
        json_data = json.load(fp)

    cand_rel = np.array(json_data["cand_rel"])
    job_rel = np.array(json_data["job_rel"])
    cand_mat = np.argsort(-cand_rel, axis=1)

    cand_num, job_num = cand_mat.shape

    # Greedy rank
    Pc_list = [init_probmat(job_num, True, False) for i in range(cand_num)]
    for idx, c_i in enumerate(Pc_list):
        c_i.data = torch.tensor(to_permutation_matrix(cand_mat[idx]))
    
    set_seed(seed)
    naive_expectation, naive_Sx, naive_candidates, naive_employers = \
        simulate_market(runs, cand_rel, job_rel, Pc_list, v_cand_type, v_job_type)

    # Reciprocal rank
    Pc_list = [init_probmat(job_num, True, False) for i in range(cand_num)]
    reci_rel = np.multiply(cand_rel, job_rel.T)
    reci_mat = (-reci_rel).argsort(axis=1) #.argsort(axis=1)
    for idx, c_i in enumerate(Pc_list):
        c_i.data = torch.tensor(to_permutation_matrix(reci_mat[idx]))
    
    set_seed(seed)
    reci_expectation, reci_Sx, reci_candidates, reci_employers = \
        simulate_market(runs, cand_rel, job_rel, Pc_list, v_cand_type, v_job_type)
    
    # Ours
    Pc_list = [init_probmat(job_num,-1, False) for i in range(cand_num)]
    for idx, c_i in enumerate(Pc_list):
        Pc = json_data["Pc"][str(idx)]
        c_i.data = torch.tensor(Pc)
    
    set_seed(seed)
    ours_expectation, ours_Sx, ours_candidates, ours_employers = \
        simulate_market(runs, cand_rel, job_rel, Pc_list, v_cand_type, v_job_type)

    output = {
        "Greedy": {
            "Expectation": naive_expectation,
            "Stdev": naive_Sx,
            "SE": naive_Sx / np.sqrt(runs),
            "Individual Utility Cand": naive_candidates,
            "Individual Utility Job": naive_employers,
        },
        "Reciprocal": {
            "Expecation": reci_expectation,
            "Stdev": reci_Sx,
            "SE": reci_Sx / np.sqrt(runs),
            "Individual Utility Cand": reci_candidates,
            "Individual Utility Job": reci_employers,
        },
        "Ours": {
            "Expectation": ours_expectation,
            "Stdev": ours_Sx,
            "SE": ours_Sx / np.sqrt(runs),
            "Individual Utility Cand": ours_candidates,
            "Individual Utility Job": ours_employers,
        }
    }

    with Path(output_path).open("w") as fp:
        json.dump(output, fp, indent=4, cls=NumpyEncoder)

    return 


if __name__=='__main__':
    import fire
    fire.Fire(main)
