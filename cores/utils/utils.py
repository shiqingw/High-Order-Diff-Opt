import numpy as np
import pickle

def seed_everything(seed: int = 0):
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available(): 
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # if torch.backends.mps.is_available():
    #     torch.mps.manual_seed(seed)

def save_dict(dict_obj, fullname):
    with open(fullname, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(fullname):
    with open(fullname, 'rb') as handle:
        loaded_obj = pickle.load(handle)
    return loaded_obj

def dict2func(d):
    return lambda x: d[x]

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def solve_infinite_LQR(A, B, Q, R):
    '''
    A, B, Q and R are the matrices defining the OC problem
    QN is the matrix used for the terminal cost
    N is the horizon length
    '''
    tol = 1e-5
    P_old = np.zeros(Q.shape)
    P = np.eye(Q.shape[0])
    while np.linalg.norm(P - P_old) > tol: 
        P_old = P
        T = B.T @ P @ B + R
        K = - np.linalg.inv(T) @ B.T @ P @ A
        P = Q + A.T @ P @ A + A.T @ P @ B @ K
    return K

def solve_LQR_tracking(A_list, B_list, Q_list, R_list, x_bar, N):
    '''
    A_list, B_list, Q_list and R_list are the matrices defining the OC problem
    x_bar is the trajectory of desired states of size dim(x) x (N+1)
    N is the horizon length
    
    The function returns 1) a list of gains of length N and 2) a list of feedforward controls of length N
    '''
    Q = Q_list[-1]
    P = Q
    q = - Q @ x_bar[-1,:]
    p = q
    K_gains = np.empty((N, R_list[0].shape[0], Q_list[0].shape[0]))
    k_feedforward = np.empty((N, R_list[0].shape[0]))
    for i in range(N):
        A = A_list[N-1-i]
        B = B_list[N-1-i]
        Q = Q_list[N-1-i]
        R = R_list[N-1-i]

        T = B.T @ P @ B + R
        K = - np.linalg.inv(T) @ B.T @ P @ A
        k = - np.linalg.inv(T) @ B.T @ p
        q = - Q @ x_bar[N-1-i, :]
        p = q + A.T @ p + A.T @ P @ B @ k
        P = Q + A.T @ P @ A + A.T @ P @ B @ K
        k_feedforward[N-1-i] = k
        K_gains[N-1-i] = K

    return K_gains, k_feedforward

def points2d_to_ineq(corners):
    """
    corners: np.array of shape (n, 2), arranged in CLOCKWISE order
    """
    n = corners.shape[0]
    A = np.empty((n, 2))
    b = np.empty(n)
    for i in range(n):
        A[i, 0] = - corners[(i+1)%n, 1] + corners[i, 1]
        A[i, 1] = - corners[i, 0] + corners[(i+1)%n, 0]
        b[i] = -corners[i, 1]*corners[(i+1)%n, 0] + corners[(i+1)%n, 1]*corners[i, 0]
    norm_A = np.linalg.norm(A, axis=1)
    A = A / norm_A[:, None]
    b = b / norm_A
    return A, b