# 调用RDGM的主脚本，确保主函数保护结构
if __name__ == '__main__':
    from RDGM import RDGM
    import numpy as np
    from Data_generate import Data_generation
    import matplotlib.pyplot as plt

    # 参数设置
    n = 200; p = 200; mach_num = 10  # 机器数量
    scenario = 'type1'; structure = 'banded'  # 数据生成场景和结构
    N = n * mach_num  # 总样本数

    Lam = list(np.arange(0.0001, 0.03, 0.003))
    Tau = list(np.arange(1, 3, 0.2))
    Delta = list(np.arange(0.0001, 0.01, 0.0008))
    T = list(np.arange(3, 5, 2))
    D = list(np.arange(5.5, 9.5, 0.2))

    Distance = []  # 存储计算的距离结果
    for i in range(len(T)):
        dist = []
        for times in range(1):
            X, Theta_true = Data_generation(N, p, scenario, structure)
            lam = 0.004
            tau_ = 3
            delta = 0.05
            t = 2
            d = 2.5

            # 调用RDGM函数
            thetas, _, _ = RDGM(X, mach_num, lam=lam, delta=delta, tau=tau_, t=t, d=d)

            # 计算距离
            distance = np.linalg.norm((Theta_true - thetas), 2)
            dist.append(distance)

        Distance.append(dist)

    # 输出结果
    for j in range(len(Distance)):
        print(f'{T[j].round(5)}对应的结果是{np.mean(Distance[j]).round(2)}  +-{np.std(Distance[j]).round(2)}')

    # 绘制图形
    mean = [np.mean(i) for i in Distance]
    x = T
    y = mean
    plt.plot(x, y)
    plt.title('iterations-difference')
    plt.xlabel('iterations')
    plt.ylabel('difference')
    plt.grid(True)
    plt.show()
