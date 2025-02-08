if __name__ == '__main__':
    from RDPM import RDPM
    import numpy as np
    from Data_generate import Data_generation
    import matplotlib.pyplot as plt

    n = 200; p = 200; mach_num = 5
    scenario = 'MT'; structure = 'banded'
    N = n * mach_num

    Lam = list(np.arange(0.001, 0.01, 0.001))

    Distance = []
    for i in range(len(Lam)):
        dist = []
        for times in range(50):
            X, Theta_true = Data_generation(N, p, scenario, structure)
            lam = Lam[i]
            tau_ = 3
            delta = 0.05
            thetas, _, _ = RDPM(X, mach_num, lam=lam, delta=delta, tau=tau_)
            distance = np.linalg.norm((Theta_true - thetas), 2)
            dist.append(distance)

        Distance.append(dist)
    for j in range(len(Distance)):
        print(f'{Lam[j].round(5)}: {np.mean(Distance[j]).round(2)}  +-{np.std(Distance[j]).round(2)}')

    mean = [np.mean(i) for i in Distance]
    x = Lam
    y = mean
    plt.plot(x, y)
    plt.title('iterations-error')
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()
