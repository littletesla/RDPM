def RDPM(X, num_mach, lam, delta, tau):
    from pathos.multiprocessing import ProcessingPool as Pool
    import numpy as np
    from scipy.special import comb
    np.set_printoptions(suppress=True)

    class Caculates:
        def __init__(self, data_XY, theta_old_init, theta_new_init, lambd, maxit, delta):
            self.X = data_XY[:,1:]
            self.Y = data_XY[:,0]
            self.theta_old = theta_old_init
            self.theta_new = theta_new_init
            self.alpha_old = 0
            self.alpha = 1
            self.lambd = lambd
            self.maxit = maxit
            self.niter = 0
            self.inter_length = np.sqrt(self.Y.shape[0])
            self.delta = delta
            self.n = self.Y.shape[0]

        # Soft-thresholding function for sparsity
        def softThres(x, delta, lamb):
            for i in range(len(x)):
                x_i = x[i,0]
                if x_i > delta * lamb:
                    temp = x_i - delta * lamb
                elif abs(x_i) <= delta * lamb:
                    temp = 0
                elif x_i < -delta * lamb:
                    temp = x_i + delta * lamb
                x[i,0] = temp
            return x

        # Update theta using gradient information
        def main_gradi(self, gradi):
            S = self.theta_new + (self.alpha_old - 1)/self.alpha * (self.theta_new - self.theta_old)
            self.theta_old = self.theta_new
            A = S - self.delta * gradi
            
            theta_ = Caculates.softThres(A, self.delta, self.lambd * self.inter_length)
            self.theta_new = theta_
            self.alpha_old = self.alpha
            self.alpha = (1+np.sqrt(4 + self.alpha_old**2))/2
            self.niter += 1
            return S
        
    # Truncate extreme values to improve robustness
    def truncate(X, tau):
        X[np.abs(X) > tau] = np.sign(X[np.abs(X) > tau]) * tau
        return X
    
    # Compute Qn scale estimator for robust variance estimation
    def Qn(x, d=1):
        n = len(x)
        x_1d = x.reshape(n)
        x = x_1d.tolist()
        if n < 2:
            raise ValueError("The input array must contain at least two elements.")
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(abs(x[i] - x[j]))
        distances = np.array(distances)
        distances.sort()
        h = n // 2 + 1
        k = int(comb(h, 2))
        Qn_value = distances[k] * d
        return Qn_value

    # Update precision matrix T
    def get_T(T, theta, column):
        for i in range(column):
            T[column,i] = -theta[i,0]
        return T

    maxit = 500
    n = int(len(X)/num_mach)
    p = X.shape[1]
    lambd = lam * np.sqrt((p)/n)
    T_matrix = np.eye(p)
    sigma_2 = []
    error_record = []
    
    for column in range(1, p):  # Process each column
        X_truncated = truncate(X, tau)
        Y_ = X_truncated[:,column].reshape(-1,1)
        X_ = X_truncated[:,0:column]

        data_node = []
        for i in range(num_mach):  # Split data across machines
            Y_machines = Y_[i*n:(i+1)*n].reshape(-1,1)
            X_machines = X_[i*n:(i+1)*n]
            data_X_Y = np.concatenate([Y_machines, X_machines], axis=1)
            data_node.append([Y_machines, X_machines])
        
        if column == 1:
            sigma_i_2 = Qn(data_X_Y[:,1], 3)
            sigma_2.append(sigma_i_2**2)

        theta_old_init = np.zeros((column,1))
        theta_new_init = np.zeros((column,1))
        S = np.zeros((column,1))
        caculate_workers = Caculates(data_X_Y, theta_old_init, theta_new_init, lambd, maxit, delta)

        # Compute gradient for optimization
        def grad(x, S):
            gradient = -x[1].T @ x[0] + x[1].T @ x[1] @ S
            return gradient / len(x[0])

        for j in range(caculate_workers.maxit):
            with Pool(processes=num_mach) as pool:
                results = pool.map(lambda chunk: grad(chunk, S), data_node)
            gradient_cons = np.mean(results, axis=0)
            S = caculate_workers.main_gradi(gradient_cons)
            delta_theta = np.linalg.norm((caculate_workers.theta_new - caculate_workers.theta_old), 2)
            
            if column == 100:
                error_record.append(delta_theta)
                iterr = caculate_workers.niter

            if (j == caculate_workers.maxit-1) or delta_theta < 1e-5:
                theta_i_final = caculate_workers.theta_new
                epsilon_i = data_X_Y[:,0:1] - data_X_Y[:,1:] @ theta_i_final
                sigma_i_2 = Qn(epsilon_i, 3)
                sigma_2.append(sigma_i_2**2)
                T_matrix = get_T(T_matrix, theta_i_final, column)
                break

    sigma_2_ = np.array(sigma_2)
    D = np.diag(sigma_2_)
    inverse_D = np.linalg.inv(D)
    Theta = T_matrix.T @ inverse_D @ T_matrix
    
    return Theta, error_record, iterr

if __name__ == "__main__":
    import numpy as np
    from Data_generate import Data_generation
    import matplotlib.pyplot as plt
    from pathos.multiprocessing import ProcessingPool as Pool

    n = 200; p = 200; mach_num = 5  # Number of machines
    scenario = 'MT'; structure = 'banded'
    N = n * mach_num
    X, Theta_true = Data_generation(N, p, scenario, structure)
    Theta, _, _ = RDPM(X, mach_num, lam=0.003, delta=0.05, tau=3)

    # Compute errors
    frob = np.linalg.norm((Theta_true - Theta), 'fro')
    spectral = np.linalg.norm((Theta_true - Theta), 2)
    max_err = np.max(np.abs(Theta_true - Theta))

    print('Frobenius norm:', frob)
    print('Spectral norm:', spectral)
    print('Max absolute error:', max_err)
    print(Theta.round(3))