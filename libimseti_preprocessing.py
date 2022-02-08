import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

gender = pd.read_csv("dating_data/gender.dat")
gender.columns = ["ID", "Gender"]

ratings = pd.read_csv("dating_data/ratings.dat")
ratings.columns = ["UserID", "ProfileID", "Rating"]

M_ids = np.array(ratings.loc[ratings['UserID'].isin(gender.loc[gender['Gender']=='M', 'ID']), 'UserID'].value_counts().index[:1000])

F_ids = np.array(ratings.loc[ratings['UserID'].isin(gender.loc[gender['Gender']=='F', 'ID']), 'UserID'].value_counts().index[:1000])

mid_to_row = {mid: row for row, mid in enumerate(M_ids)}
fid_to_row = {fid: row for row, fid in enumerate(F_ids)}

m_ratings = ratings.loc[(ratings['UserID'].isin(M_ids)) & (ratings['ProfileID'].isin(F_ids))]
m_ratings['UserID'] = m_ratings['UserID'].apply(lambda x: mid_to_row[x])
m_ratings['ProfileID'] = m_ratings['ProfileID'].apply(lambda x: fid_to_row[x])

m_A = np.zeros((1000, 1000), dtype=np.float32)
m_mask = np.zeros((1000, 1000), dtype=np.int8)

for _, r in tqdm(m_ratings.iterrows(), total=m_ratings.shape[0]):
    row, col, val = r['UserID'], r['ProfileID'], r['Rating']
    m_mask[row, col] = 1
    m_A[row, col] = val / 10.

f_ratings = ratings.loc[(ratings['UserID'].isin(F_ids)) & (ratings['ProfileID'].isin(M_ids))]
f_ratings['UserID'] = f_ratings['UserID'].apply(lambda x: fid_to_row[x])
f_ratings['ProfileID'] = f_ratings['ProfileID'].apply(lambda x: mid_to_row[x])

f_A = np.zeros((1000, 1000), dtype=np.float32)
f_mask = np.zeros((1000, 1000), dtype=np.int8)


for _, r in tqdm(f_ratings.iterrows(), total=f_ratings.shape[0]):
    row, col, val = r['UserID'], r['ProfileID'], r['Rating']
    f_mask[row, col] = 1
    f_A[row, col] = val / 10.


def pmf_solve(A, mask, k, mu, epsilon=1e-3, max_iterations=100):
    """
    Solve probabilistic matrix factorization using alternating least squares.
    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.
    [ Salakhutdinov and Mnih 2008 ]
    [ Hu, Koren, and Volinksy 2009 ]
    Parameters:
    -----------
    A : m x n array
        matrix to complete
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    k : integer
        how many factors to use
    mu : float
        hyper-parameter penalizing norm of factored U, V
    epsilon : float
        convergence condition on the difference between iterative results
    max_iterations: int
        hard limit on maximum number of iterations
    Returns:
    --------
    X: m x n array
        completed matrix
    """
    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)

    for _ in tqdm(range(max_iterations)):

        for i in range(m):
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([V.T, C_u[i], A[i,:]]))

        for j in range(n):
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([U.T, C_v[j], A[:,j]]))

        X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        if mean_diff < epsilon:
            break
        prev_X = X

    return X

imputed_m = pmf_solve(m_A, m_mask, 6, 1e-2)
imputed_f = pmf_solve(f_A, f_mask, 6, 1e-2)

clipped_f = np.clip(imputed_f, 0, 1)
clipped_m = np.clip(imputed_m, 0, 1)

with open("dating_data/female_to_male_rel_500.pkl", "wb") as fp:
    pickle.dump(clipped_f[:500, :500], fp)

with open("dating_data/male_to_female_rel_500.pkl", "wb") as fp:
    pickle.dump(clipped_m[:500, :500], fp)




