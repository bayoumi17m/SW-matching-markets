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
import time
from pathlib import Path
import fire

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

def get_click_rank(Pc_list, job_mat, cand_rel, v_cand):
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
            cand_eprob = torch.dot(Pc_list[i][j], v_cand)
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
    
    m_ones = cp.Constant(np.ones(job_num))
    u_ones = cp.Constant(np.ones(cand_num))
    ones = cp.Constant(np.ones(job_num*cand_num))
    x = cp.Variable((job_num*cand_num,job_num))
    objective = cp.Minimize(objec(x, grad))
    constr = [x>=0] # Non-negative entries
    
    # Sum of rows are equal to 1
    constr += [x@m_ones == ones]
    for t in range(cand_num):
        # Sum of columns are equal to 1
        constr += [m_ones@(x[t*job_num:(t+1)*job_num, :]) == m_ones]

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
     

def main(job_num=50, cand_num=75, seed=621, environment="random", v_cand_type="inv", v_job_type="inv", lr_sch="constant", noise_level=0.2, lambda_value=0, min_rel=0.0, epoch_num=50):
    """main runs the optimization procedure as well as calculating lower bounds.

    :param job_num: Number of employers
    :type job_num: int
    :param cand_num: Number of candidates
    :type cand_num: int
    :param seed: seed used for RNG
    :type seed: int
    :param environment: asymmetric level of relevance between jobs and cands
    :type environment: str
    :param v_cand_type: type of examination function to use for candidates
    :type v_cand_type: str
    :param v_job_type: type of examination function to use for employer
    :type v_job_type: str
    :param lr_sch: learning rate schedule, one of "constant" or "decay"
    :type lr_sch: str
    :param noise_level: Sigma (Stdev) of noise to add to the asymmetric relevance
    :type noise_level: float
    :param lambda_value: Level of non-congestion where 0 means full congestion
    :type lambda_value: float, [0-1]
    :param min_rel: Minimum relevance for congestion factor. The congestion will
        approach this value as it approaches less popular cands/jobs
    :type min_rel: float, [0-1)
    :param epoch_num: Maximum number of epochs to update
    :type epoch_num: int
    """
    set_seed(seed)
    cand_rel = np.random.random((cand_num, job_num))

    print(f"Environment: {environment}", flush=True)

    if environment == "random":
        job_rel = np.random.random((job_num, cand_num,))
    elif environment == "similar":
        noise = np.random.normal(loc=0, scale=noise_level, size=(job_num, cand_num))
        job_rel = np.minimum(np.maximum(cand_rel.T + noise, 0), 1) #
    elif environment == "reverse":
        noise = np.random.normal(loc=0, scale=noise_level, size=(job_num, cand_num))
        job_rel = np.minimum(np.maximum((1 - cand_rel.T) + noise, 0), 1)
    else:
        raise RuntimeError("Using a non-defined environment")

    # Setting congestion
    job_congestion_value = np.array([np.linspace(1, min_rel, cand_num) for _ in range(job_num)])
    job_rel = np.clip((1 - lambda_value)*job_rel + lambda_value*job_congestion_value, 0, 1)

    cand_congestion_value = np.array([np.linspace(1, min_rel, job_num) for _ in range(cand_num)])
    cand_rel = np.clip((1 - lambda_value)*cand_rel + lambda_value*cand_congestion_value, 0, 1)

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

    # optimization (Frank-wolfe/conditional gradient descent)
    ls = []
    diff=1e10
    prev_sum=0
    epoch=0
    while np.abs(diff)>=1e-3 and epoch<= epoch_num:
        print(f"Epoch: {epoch:02}", flush=True)
        cand_exprank, cand_click = get_click_rank(c_list, job_mat, cand_rel, v_cand)
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
                new_grads[p] = grad_step[p*job_num:(p+1)*job_num,:]
            
            if lr_sch == "decay":
                lr = 1/(epoch+2)
            else:
                lr = 0.2
                
            print(f"Epoch={epoch}, lr={lr}")
                
            p = 0
            for c_i in c_list:
                c_i.mul_(1-lr)
                c_i.add_(lr * torch.tensor(new_grads[p]))
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
            name_str = f"job_{job_num}_cand_{cand_num}_seed_{seed}_env_{environment}_lr_{lr_sch}_noise_{noise_level}_lambda_{lambda_value}_mrel_{min_rel}_epoch_{epoch_num}"
            with Path("LI"+name_str+".json").open("w") as fp:
                fp.write(json.dumps(final_output, cls=NumpyEncoder, indent=4))
                
        epoch += 1
                
                
    print(f"Took {time.time() - S} sec.", flush=True)
    print(f"Final loss: {-total_sum.data}", flush=True)

    final_output = {}
    final_output["cand_rel"] = cand_exprank_rel.numpy()
    final_output["job_rel"] = job_rel.numpy()
    final_output["Pc"] = {}
    final_output["epoch"] = epoch
    final_output["loss"] = ls
    for uid, Pc in enumerate(c_list):
        final_output["Pc"][uid] = Pc.data.numpy()

    with Path("LI"+name_str+".json").open("w") as fp:
        fp.write(json.dumps(final_output, cls=NumpyEncoder, indent=4))
        
        
if __name__ == "__main__":
    fire.Fire(main)
