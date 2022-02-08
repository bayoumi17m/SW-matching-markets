import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.linalg import circulant
from sklearn import preprocessing
import warnings
import cvxpy as cp
import copy
warnings.filterwarnings("ignore")

import json
import pickle
import time
from pathlib import Path
import fire

topn = 100

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

def set_seed(seed):
    """set_seed sets the random seed for numpy and torch.

    :param seed: seed used for RNG
    :type seed: int
    """
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

def to_permutation_matrix(vec):
    """to_permutation_matrix creates a permuation matrix from a sorted array.

    :param vec: array representing the desired 
    :type vec: 1D np.ndarray of dtype np.integer
    :returns: permutation matrix from the sorted array
    :rtype: np.ndarry of dtype np.float
    """
    return np.eye(vec.size)[vec].T

def get_v_cand(job_num, v_cand_type):
    """get_v_cand retuns the values from the exam. function for every rank

    :param job_num: Number of ranks to be ranked
    :type job_num: int
    :param v_cand_type: type of examination function to use
    :type v_cand_type: str
    :returns: Candidate examination probabilities for each rank
    :rtype: torch.tensor of shape (|J|,)
    """
    if v_cand_type == "inv":
        v_cand = torch.tensor(1./(np.arange(job_num)+1))
    elif v_cand_type == "log":
        v_cand = torch.tensor(1./np.log(np.arange(job_num)+2))
    elif v_cand_type == "exp":
        v_cand = torch.tensor(1./(np.power(np.e, np.arange(job_num))))
    return v_cand


def get_v_job(v_job_type, job_rel, cand_exprank, j):
    """get_v_job retuns the values from the exam. function for every rank.
    Additionally, it will multiply by the relevance of candidate / job pair.

    :param v_job_type: type of examination function to use for employer
    :type v_job_type: str
    :param job_rel: Relevance table denoting for job j the relevance of candidate c.
        Each row is a job j and each column denotes a candidate c. Denoted g_j(c)
        in paper.
    :type job_rel: torch.tensor of shape (|J|, |C|) with all values in [0,1].
    :param cand_exprank: cand_exprank[j,c] is expected rank of candidate c for
        job j. 
    :type cand_exprank: torch.tensor of shape (|J|, |C|) of torch.float64
    :param j: index of the specified job
    :type j: int
    :returns: The probability job j will interact with all candidates c
    :rtype: torch.tensor of shape (|C|,)
    """
    if v_job_type == "inv":
        temp = torch.div(job_rel[j,:], cand_exprank[j,:])
    elif v_job_type == "log":
        exa = 1./torch.log(cand_exprank[j,:]+1)
        temp = torch.mul(job_rel[j,:], exa)
    elif v_job_type == "exp":
        temp = torch.div(job_rel[j, :], torch.exp(cand_exprank[j,:]-1))
    return temp

def get_click_rank(Pc_list, job_mat, cand_rel, v_cand, prod_correction):
    """get_click_rank retrieves the expected rank of candidate c for job j

    :param Pc_list: Ranking policies for each candidate where Pc_list[c] is the
        ranking policy for candidate c.
    :type Pc_list: list of torch.tensor of shape (|J|, |J|) that are doubly
        stochastic matrices
    :param job_mat: ranking of all jobs by relevance where index 0 is rank 1 and
        the value represents which candidate is the most relevant
    :type job_mat: np.ndarray as shape (|J|, |C|) of type np.integer
    :param cand_rel: Relevance table denoting for candidate c the relevance of job j.
        Each row is a candidate c and each column denotes a job j. Denoted f_c(j)
        in paper.
    :type cand_rel: torch.tensor of shape (|C|, |J|) with all values in [0,1].
    :param v_cand: candidates examination function values for each rank
    :type v_cand: torch.tensor of shape (|J|,) 
    """
    job_num, cand_num = np.shape(job_mat)
    cand_exprank = torch.zeros((job_num, cand_num)) 
    cand_click = torch.zeros((job_num, cand_num), dtype=torch.float64) 
    
    for j in range(job_num):
        k = np.argsort(job_mat[j,:])
        temp = 1
        for i in k:
            cand_exprank[j,i] = temp
            # cand_eprob = torch.dot(Pc_list[i][j], v_cand)
            cand_eprob = torch.dot(Pc_list[i][prod_correction[i,j]], v_cand)
            cand_cprob = cand_eprob * cand_rel[i,j]
            cand_click[j,i] = cand_cprob
            temp += cand_cprob
    return cand_exprank, cand_click

def fw_step(grad, exprank):
    """fw_step is a step of Frank-Wolfe conditional gradient descent.

    This method will find 

    :param grad: gradient of the objective function SW for all candidates ranking
        policies
    :type grad: torch.tensor of shape (cand_num*job_num, job_num)
    :param cand_exprank: cand_exprank[j,c] is expected rank of candidate c for
        job j. 
    :type cand_exprank: torch.tensor of shape (|J|, |C|) of torch.float64
    """
    job_num, cand_num = np.shape(exprank)
    
    m_ones = cp.Constant(np.ones(topn))
    u_ones = cp.Constant(np.ones(cand_num))
    ones = cp.Constant(np.ones(topn*cand_num))
    x = cp.Variable((topn*cand_num,topn))
    grad_ord = np.concatenate([grad[job_num*(i):job_num*(i)+topn, :topn] for i in range(cand_num)])
    objective = cp.Minimize(objec(x, grad_ord))
    constr = [x>=0] # Non-negative entries
    
    # Sum of rows are equal to 1
    constr += [x@m_ones == ones]
    for t in range(cand_num):
        # Sum of columns are equal to 1
        constr += [m_ones@(x[t*topn:(t+1)*topn, :]) == m_ones]

    prob = cp.Problem(objective, constr)
    prob.solve(solver=cp.SCS, verbose=True)
    return x.value

def objec(x, grad):
    """objec is the objective for the Frank Wolfe direction finding subproblem.

    :param x: variable to solve the LP for
    :type x: cvxpy.Variable
    :param grad: gradient of the SW function
    :type grad: np.ndarray
    :returns: Expression of x^T * gradient of SW
    :rtype: cvxpy.Expression
    """
    total_sum = cp.multiply(x,grad)
    total_sum = cp.sum(total_sum)
    return total_sum
     

def main(male_rel_path, female_rel_path, seed=621, v_cand_type="inv", v_job_type="inv", lr_sch="constant", epoch_num=50):
    """main runs the optimization procedure as well as calculating lower bounds.

    :param male_rel_path: path to a pickle file with a male relevance table.
    :type male_rel_path: str
    :param female_rel_path: path to a pickle file with a female relevance table.
    :type female_rel_path: str
    :param seed: seed used for RNG
    :type seed: int
    :param v_cand_type: type of examination function to use for candidates
    :type v_cand_type: str
    :param v_job_type: type of examination function to use for employer
    :type v_job_type: str
    :param lr_sch: learning rate schedule, one of "constant" or "decay"
    :type lr_sch: str
    :param epoch_num: Maximum number of epochs to update
    :type epoch_num: int
    """
    set_seed(seed)

    with open(male_rel_path, "rb") as fp:
        male_rel = cand_rel = pickle.load(fp)

    with open(female_rel_path, "rb") as fp:
        female_rel = job_rel = pickle.load(fp)

    job_num = job_rel.shape[0]
    cand_num = cand_rel.shape[0]

    user_rel = np.block(
        [
            [np.zeros((cand_num, cand_num)), male_rel],
            [female_rel, np.zeros((job_num, job_num))]
        ]
    )

    mask = np.ones_like(user_rel) # 

    cand_rel = np.where(mask, user_rel, 0)
    job_rel = user_rel.copy()

    job_num = job_rel.shape[0]
    cand_num = cand_rel.shape[0]


    cand_mat = np.argsort(-cand_rel, axis=1)
    job_mat = np.argsort(-job_rel, axis=1)

    cand_rel = torch.tensor(cand_rel)
    job_rel = torch.tensor(job_rel)
    
    torch.set_default_tensor_type('torch.DoubleTensor')

    # set examination model (cand and job examination model could be different)
    v_cand = get_v_cand(job_num, v_cand_type)
    print(f"cand attention/examination vector = {v_cand.data}", flush=True) 

    
    ## Relevance-based ie. Greedy
    with torch.no_grad():
        # initiate the doubly stochastic matrix P_c.
        c_list = [init_probmat(job_num, True, False) for i in range(cand_num)]
        for idx, c_i in enumerate(c_list):
            c_i.data = torch.tensor(to_permutation_matrix(cand_mat[idx]))

        # for each job, what is the cand's expected rank
        cand_exprank = torch.zeros((job_num, cand_num))
        cand_click = torch.zeros((job_num, cand_num), dtype=torch.float64) #job row
        
        for j in range(job_num):
            k = np.argsort(job_mat[j,:])
            temp = 1
            for i in k:
                cand_exprank[j,i] = temp
                cand_eprob = torch.dot(c_list[i][j], v_cand)
                cand_cprob = cand_eprob * cand_rel[i,j]
                cand_click[j,i] = cand_cprob
                temp += cand_cprob

        greedy_mat = np.zeros((job_num,cand_num))
        total_sum = torch.zeros(1)
        
        for j in range(job_num):
            temp = get_v_job(v_job_type, job_rel, cand_exprank, j)
            partial_sum = torch.dot(temp, cand_click[j,:])
            greedy_mat[j,:] = np.multiply(temp, cand_click[j,:])
            total_sum -= partial_sum
        print(f"The expected number of matches for greedy naive solution is: {-total_sum}", flush=True)
        
        
    ## Reciprocal-Based
    with torch.no_grad():
        # initiate the doubly stochastic matrix P_c.
        c_list = [init_probmat(job_num, True, False) for i in range(cand_num)]
        # job_rel = cand_rel = torch.tensor(user_rel)
        # prod_mat = np.argsort(-np.where(mask, np.multiply(user_rel.numpy(), user_rel.numpy().T), 0), axis=1)
        prod_mat = np.argsort(-np.multiply(cand_rel.numpy(), job_rel.numpy().T), axis=1)
        for idx, c_i in enumerate(c_list):
            c_i.data = torch.tensor(to_permutation_matrix(prod_mat[idx]))

        # for each job, what is the cand's expected rank
        cand_exprank = torch.zeros((job_num, cand_num))
        cand_click = torch.zeros((job_num, cand_num), dtype=torch.float64) #job row
        
        for j in range(job_num):
            k = np.argsort(job_mat[j,:])
            temp = 1
            for i in k:
                cand_exprank[j,i] = temp
                cand_eprob = torch.dot(c_list[i][j], v_cand)
                cand_cprob = cand_eprob * cand_rel[i,j]
                cand_click[j,i] = cand_cprob
                temp += cand_cprob

        greedy_mat = np.zeros((job_num,cand_num))
        total_sum = torch.zeros(1)
        
        for j in range(job_num):
            temp = get_v_job(v_job_type, job_rel, cand_exprank, j) 
            partial_sum = torch.dot(temp, cand_click[j,:])
            greedy_mat[j,:] = np.multiply(temp, cand_click[j,:])
            total_sum -= partial_sum
        print(f"The expected number of matches for product naive solution is: {-total_sum}", flush=True)
        
        
    ## Social Welfare based
    
    S = time.time()
    # initilization/parameters
    c_list = [init_probmat(job_num, True, False) for i in range(cand_num)]
    prod_mat = (-np.multiply(cand_rel, job_rel.T)).argsort(axis=1)
    prod_correction = prod_mat.argsort(axis=1)

    lower_left_block = (np.array(prod_mat[:, topn:])).argsort(axis=1)
    for idx, c_i in enumerate(c_list):
        # c_i.data = torch.tensor(one_hot(prod_mat[idx]), dtype=torch.float32)
        c_i.data = torch.zeros_like(c_i.data)
        c_i.data[:topn, :topn] = torch.ones((topn, topn)) / topn
        c_i.data[topn:, topn:] = torch.tensor(to_permutation_matrix(lower_left_block[idx]), dtype=torch.float32)
    # optimization (Frank-wolfe/conditional gradient descent)
    ls = []
    diff=1e10
    prev_sum=0
    epoch=0
    while np.abs(diff)>=1e-3 and epoch<= epoch_num:
        print(f"Epoch: {epoch:02}", flush=True)
        cand_exprank, cand_click = get_click_rank(c_list, job_mat, cand_rel, v_cand, prod_correction)
        total_sum = torch.zeros(1)
        
        for j in range(job_num):
            temp = get_v_job(v_job_type, job_rel, cand_exprank, j)
            partial_sum = torch.dot(temp, cand_click[j,:])
            total_sum -= partial_sum
            
        ls.append(-total_sum.data.numpy())
        print(f"The expected number of matches for our proposed solution is: {-total_sum}", flush=True)
        total_sum.backward()
        diff = -total_sum.data.numpy()-prev_sum
        prev_sum = -total_sum.data.numpy()
        print(f"current diff is: {diff}", flush=True)
        print(f"Backward pass complete", flush=True)
        

        with torch.no_grad():
            new_grads = [0]*len(c_list)
            x = torch.zeros((job_num*cand_num,job_num))
            
            for p in range(len(c_list)):
                x[p*job_num:(p+1)*job_num,:] = c_list[p].grad
                
            print(f"Before finding gradient", flush=True)
            grad_step = fw_step(x, cand_exprank)
            print(f"Finished gradient", flush=True)
            
            for p in range(len(c_list)):
                new_grads[p] = grad_step[p*topn:(p+1)*topn,:]
                # new_grads[p] = grad_step[p*job_num:(p+1)*job_num,:]
            
            if lr_sch == "decay":
                lr = 1/(epoch+2)
            else:
                lr = 0.2
                
            print(f"Epoch={epoch}, lr={lr}")
                
            p = 0
            for c_i in c_list:
                c_i[:topn, :topn].mul_(1-lr)
                c_i[:topn, :topn].add_(lr * torch.tensor(new_grads[p]))
                p+=1
                
        for c_i in c_list:
            c_i.grad.zero_()
        
        if epoch % 5 == 0:
            final_output = {}
            final_output["epoch"] = epoch
            final_output["cand_rel"] = cand_rel.numpy()
            final_output["job_rel"] = job_rel.numpy()
            final_output["Pc"] = {}
            final_output["loss"] = ls
            for uid, Pc in enumerate(c_list):
                final_output["Pc"][uid] = Pc.data.numpy()
            name_str = f"_500_user_seed_{seed}_env_dating_male_cand_female_employer_lr_{lr_sch}_candexa_{v_cand_type}_jobexa_{v_job_type}"
            with Path("dating_data/LI_top100_"+name_str+".json").open("w") as fp:
                fp.write(json.dumps(final_output, cls=NumpyEncoder, indent=4))
                
        epoch += 1
        with open(f"dating_500_{v_cand_type}_epoch_{epoch-1}.pkl", "wb") as fp:
            pickle.dump(c_list, fp)
                
                
    print(f"Took {time.time() - S} sec.", flush=True)
    print(f"Final loss: {-total_sum.data}", flush=True)

    final_output = {}
    final_output["cand_rel"] = cand_rel.numpy()
    final_output["job_rel"] = job_rel.numpy()
    final_output["Pc"] = {}
    final_output["epoch"] = epoch
    final_output["loss"] = ls
    for uid, Pc in enumerate(c_list):
        final_output["Pc"][uid] = Pc.data.numpy()

    with Path("dating_data/LI_top100_"+name_str+".json").open("w") as fp:
        fp.write(json.dumps(final_output, cls=NumpyEncoder, indent=4))
        
        
if __name__ == "__main__":
    fire.Fire(main)

