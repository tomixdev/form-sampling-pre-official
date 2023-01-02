# Website
# https://linuxtut.com/en/2ac9e6eda39664ca8345/


from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import pylab as plt
import seaborn as sns
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def get_test_curves(view=False):
    T = 150
    t = .4

    A = np.sin(np.array(range(T)) / 10)
    B = np.sin((np.array(range(T)) / 10 + t * np.pi))
    C = np.zeros((T))
    if view:
        plt.plot(A)
        plt.plot(B)
        plt.plot(C)
        plt.legend(['sin(θ)', 'sin(θ+150*pi)', 'constant'], fontsize=10, loc=2)
        plt.show()

    return {'name': 'sin(θ)', 'data': A}, {'name': 'sin(θ+150*pi)', 'data': B}, {'name': 'constant', 'data': C}


def mse(a, b):
    return ((a-b)**2).mean()


def get_DWT_results(T, S, skip=1, view=False):
    T_data, S_data = T['data'], S['data']
    T_name, S_name = T['name'], S['name']
    distance, path = fastdtw(T_data, S_data, dist=euclidean)
    print("DTW({}, {}):".format(T_name, S_name), distance)
    if view:
        plt.plot(T_data)
        plt.plot(S_data)
        k = -1
        for i, j in path:
            k += 1
            if k % skip == 0:
                plt.plot([i, j], [T_data[i], S_data[j]],
                         color='gray', linestyle='dotted')
        plt.legend([T_name, S_name], fontsize=10, loc=2)
        plt.title('DWT plot result')
        plt.show()

    return distance, path


def get_derivative(T):
    diff = np.diff(T)
    next_diff = np.append(np.delete(diff, 0), 0)
    avg = (next_diff + diff) / 2
    avg += diff
    avg /= 2
    return np.delete(avg, -1)


def get_DDWT_results(T, S, skip=1, view=False):
    dT_data = get_derivative(T)
    dS_data = get_derivative(S)
    distance, path = fastdtw(dT_data, dS_data)  # dist=euclidean
    return distance, path


def get_test_curves_DDTWvsDWT(view=False):
    T = 150
    t = .4
    A = np.zeros((T))
    B = np.zeros((T))
    # A = np.sin(np.array(range(T)) / 10)
    # B = np.sin(np.array(range(T)) / 10+2)+50
    s_i = 50
    e_i = 60
    for i in range(s_i, e_i, 1):
        A[i] = np.sin(np.pi*(i-s_i)/(e_i-s_i))
    #     B[i] = -2.2
    if view:
        plt.plot(A)
        plt.plot(B)
        plt.legend(['sin(θ)', 'sin(θ+150*pi)'], fontsize=10, loc=2)
        plt.show()

    return {'name': 'down', 'data': A}, {'name': 'up', 'data': B}


def main():
    print("=== main ===")
    # A, B, C = get_test_curves()
    A, B = get_test_curves_DDTWvsDWT()

    # A["data"] = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0])
    # B["data"] = np.array([1, 1, 2, 4, 2, 1, 2, 0])

    print("{}-{}:".format(A['name'], B['name']), mse(A['data'], B['data']))
    # print("{}-{}:".format(A['name'], C['name']), mse(A['data'], C['data']))
    # Calculate DTW
    get_DWT_results(A, B, skip=5)
    get_DDWT_results(A, B, skip=5)


if __name__ == "__main__":
    main()
