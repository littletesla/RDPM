def Data_generation(n, p, scenario, structure):
    import numpy as np
    from scipy.stats import t

    # Function to generate a banded precision matrix
    def Banded_precision_matrix(n):
        W = np.zeros((n, n))
        for i in range(n):
            W[i, i] = 1
            if i + 1 < n:
                W[i, i + 1] = 0.6
            if i + 2 < n:
                W[i, i + 2] = 0.3 
            if i - 1 >= 0:
                W[i, i - 1] = 0.6
            if i - 2 >= 0:
                W[i, i - 2] = 0.3
        return W
    
    # Function to generate an autoregressive (AR) precision matrix
    def AR_precision_matrix(n):
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                W[i, j] = 0.7 ** abs(i - j)
        return W
    
    # Function to generate an Erdős–Rényi (ER) precision matrix
    def ER_precision_matrix(p):
        W2 = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                delta_ij = np.random.binomial(1, 0.01)
                u_ij = np.random.uniform(0.4, 0.8)
                W2[i, j] = u_ij * delta_ij
                W2[j, i] = W2[i, j]
        eigenvalues = np.linalg.eigvals(W2)
        lambda_min = np.min(np.real(eigenvalues))
        identity_matrix = np.eye(p)
        W1 = W2 + (abs(lambda_min) + 0.05) * identity_matrix
        return W1
    
    # Function to generate a random modular (RM) precision matrix
    def RM_precision_matrix(p):
        B = np.zeros((p, p))
        for i in range(p):
            for j in range(i + 1, p):
                if np.random.rand() < 0.1:
                    B[i, j] = 0.5
                    B[j, i] = 0.5
        I = np.eye(p)
        eigenvalues = np.linalg.eigvals(B)
        lambda_max = max(eigenvalues)
        lambda_min = min(eigenvalues)
        delta = (lambda_max - p * lambda_min) / (p - 1)
        W_0 = B + delta * I
        diag_elements = np.sqrt(np.diag(W_0))
        W = W_0 / (diag_elements[:, None] * diag_elements[None, :])
        return W

    # Function to generate skewed t-distributed random variables
    def skewed_t_rvs(df, skew, size):
        u0 = np.random.normal(size=size)
        v = t.rvs(df, size=size)
        delta = skew / np.sqrt(1 + skew**2)
        u1 = delta * np.abs(v) + np.sqrt(1 - delta**2) * u0
        return u1

    # Function to generate data matrix X based on a given covariance structure
    def X_generate(nSample, p, SIGMA, scenario):
        cholV = np.linalg.cholesky(SIGMA)
        if scenario == 'MVN':  # Multivariate normal distribution
            X = np.random.randn(nSample, p) @ cholV
        elif scenario == 'MT':  # Multivariate t-distribution
            df = 3.5
            X = (t.rvs(df, size=(nSample, p)) @ cholV) / np.sqrt(df)
        elif scenario == 'MST':  # Multivariate skew-t distribution
            df = 5
            alpha = 20
            X = (skewed_t_rvs(df, alpha, (nSample, p)) @ cholV) / np.sqrt(df)
        elif scenario == 'CMVN':  # Contaminated multivariate normal distribution
            mean = np.zeros(p)
            P = np.random.multivariate_normal(mean, SIGMA, nSample)
            contamination_cov = 20 * np.eye(p)
            Pc = np.random.multivariate_normal(mean, contamination_cov, nSample)
            X = 0.8 * P + 0.2 * Pc
        return X
    
    # Select the appropriate precision matrix structure
    if structure == 'banded':
        precision_matrix = Banded_precision_matrix(p)
    elif structure == 'ar':
        precision_matrix = AR_precision_matrix(p)
    elif structure == 'er':
        precision_matrix = ER_precision_matrix(p)
    elif structure == 'rm':
        precision_matrix = RM_precision_matrix(p)
        
    # Compute the covariance matrix as the inverse of the precision matrix
    cov_matrix = np.linalg.inv(precision_matrix)
    X = X_generate(n, p, cov_matrix, scenario)
    
    return X, precision_matrix

if __name__ == '__main__':
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    np.set_printoptions(suppress=True)

    n = 2**10
    p = 200
    X, precision_matrix = Data_generation(n, p, scenario='MVN', structure='rm')
    print(precision_matrix.round(4))

    # Visualize the precision matrix as a heatmap
    abs_covariance_matrix = np.abs(precision_matrix)
    cmap = LinearSegmentedColormap.from_list("black_white", [(1, 1, 1), (0, 0, 0)], N=256)
    sns.heatmap(abs_covariance_matrix, annot=False, fmt=".2f", cmap=cmap)
    plt.title('Black and White Precision Matrix Heatmap')
    plt.show()
