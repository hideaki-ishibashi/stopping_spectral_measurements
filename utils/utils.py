import numpy as np
import random
import joblib


def calcKL_gauss(pos1, pos2,epsilon=0.00):
    N = pos1[0].shape[0]
    f2 = pos2[0]
    f1 = pos1[0]
    S2 = pos2[1] + epsilon * np.eye(N)
    S1 = pos1[1] + epsilon * np.eye(N)
    S2_inv = np.linalg.inv(S2)
    S = S2_inv @ S1
    trace = np.trace(S)
    logdet = np.log(np.abs(np.linalg.det(S)))
    se = (f2 - f1).T @ S2_inv @ (f2 - f1)
    KL = 0.5 * (trace - logdet + se - N)
    return np.squeeze(KL)


def calcKL_pq_fast(pos_old,new_output,beta):
    m = pos_old[0][-1]
    k = pos_old[1][-1,-1]
    trace = beta*k
    logdet = np.log(1+beta*k)
    se = 1/(k+1/beta)**2*(k+k*beta*k)*(new_output[-1]-m)**2
    KL = 0.5*(trace - logdet + se)
    return np.squeeze(KL)


def calcKL_qp_fast(pos_old,new_output,beta):
    m = pos_old[0][-1]
    k = pos_old[1][-1,-1]
    trace = k/(k+1/beta)
    logdet = np.log(1+beta*k)
    se = k/((k+1/beta)**2)*(new_output[-1]-m)**2
    KL = 0.5*(-trace + logdet + se)
    return np.squeeze(KL)


def calcKL_pq_min(pos_old,beta):
    m = pos_old[0][-1]
    k = pos_old[1][-1,-1]
    trace = beta*k
    logdet = np.log(1+beta*k)
    KL = 0.5*(trace - logdet)
    return KL


def calcKL_qp_min(pos_old,beta):
    m = pos_old[0][-1]
    k = pos_old[1][-1,-1]
    trace = k/(k+1/beta)
    logdet = np.log(1+beta*k)
    KL = 0.5*(-trace + logdet)
    return KL


def calc_train_error_gauss(output,pos,var):
    [mean,std] = pos
    train_error = (((output - mean) ** 2).sum() + (std**2).sum()) / (2 * var * output.shape[0]) + 0.5*np.log(2*np.pi*var)
    # train_error = (((output - mean) ** 2).sum()) / (2 * output.shape[0])
    return train_error


def get_init_samples(init_sample_size, train_size):
    pool_indecies = set(range(train_size))
    sampled_indecies = set(random.sample(pool_indecies, init_sample_size))
    pool_indecies = list(pool_indecies - sampled_indecies)
    sampled_indecies = list(sampled_indecies)
    return [sampled_indecies, pool_indecies]


def calc_expected_squre_error(output,pos):
    [mean, var] = pos
    train_error = (((output - mean) ** 2).sum() + var.sum()) / (output.shape[0])
    return train_error


def calc_squre_error(output,f):
    train_error = (((output - f) ** 2).sum()) / output.shape[0]
    # train_error = (((output - mean) ** 2).sum()) / (2 * output.shape[0])
    return train_error


def serialize(obj,save_name):
    with open(save_name, mode='wb') as f:
        joblib.dump(obj, f, compress=3)


def deserialize(save_name):
    with open(save_name, mode='rb') as f:
        obj = joblib.load(f)
    return obj
