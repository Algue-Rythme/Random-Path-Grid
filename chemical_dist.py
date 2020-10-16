import time
import numpy as np
from scipy.stats import linregress
from numba import jit, njit, prange
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


@njit
def from_edges_to_grid(random_edges):
    DOWN, RIGHT, UP, LEFT = 0, 1, 2, 3
    N = int(random_edges.shape[0])
    grid = np.full((N, N, 4), False)  # four neighbours
    for i in prange(N-1):
        for j in prange(N-1):
            grid[i  , j  , DOWN ] = random_edges[i, j, 0]
            grid[i+1, j  , UP   ] = random_edges[i, j, 0]
            grid[i  , j  , RIGHT] = random_edges[i, j, 1]
            grid[i  , j+1, LEFT ] = random_edges[i, j, 1]
        # last column for each line:
        grid[i  , N-1, DOWN ] = random_edges[i, N-1, 0]
        grid[i+1, N-1, UP   ] = random_edges[i, N-1, 0]
    for j in prange(N-1):  # last line:
        grid[N-1, j  , RIGHT] = random_edges[N-1, j, 1]
        grid[N-1, j+1, LEFT ] = random_edges[N-1, j, 1]
    return grid

def get_dual(grid):
    dual_grid = np.where(grid == 1., 0, 1.)
    return dual_grid

@njit
def get_random_edges(N, p):
    # 2 edges per node (i,j) are enough to span all neighbours
    # hence, only 2N² random numbers (at most) are required
    random_edges = np.random.rand(N, N, 2)
    random_edges = random_edges < p  # boolean array
    return random_edges

def get_random_grid(N, p):
    random_edges = get_random_edges(N, p)
    grid = from_edges_to_grid(random_edges)
    return grid

@njit
def efficient_bfs(grid):
    # corresponds to DOWN, RIGHT, UP, LEFT
    neighbours = np.array([[1,0], [0,1], [-1,0], [0,-1]])
    # all the edges have the same weight, hence BFS only is required
    # (no need for Dijkstra and expensive priority queue)
    # the queue of the BFS will contain atmost N² nodes
    # so the full queue can pre-allocated
    N = grid.shape[0]
    queue = np.zeros(shape=(N*N,2), dtype=np.int64)
    # matrix of distances from starting points
    dst = np.full(shape=(N,N), fill_value=np.inf)
    for start_i in range(N):  # left border
        dst[start_i, 0] = 0
        queue[start_i,0] = start_i
        queue[start_i,1] = 0
    head, tail = 0, N+1  # already N nodes in queue
    while head < tail:  # while queue not empty
        i, j = queue[head,0], queue[head,1]
        for k in prange(4):
            n_i, n_j = i+neighbours[k,0], j+neighbours[k,1]
            if grid[i,j,k] and dst[n_i, n_j] == np.inf:
                dst[n_i, n_j] = dst[i, j] + 1
                queue[tail,0] = n_i
                queue[tail,1] = n_j
                tail += 1
        head += 1
    shortest = np.inf
    for end_i in range(N):  # right border
        shortest = min(shortest, dst[end_i, N-1])
    return shortest  # can be np.inf if no path exists


if __name__ == '__main__':
    p = 0.5  # the probability of an edge to exist
    small_N = list(range(10, 100, 10))
    average_N = list(range(100, 200, 20))
    big_N = list(range(200, 400, 50))
    huge_N = list(range(400, 800, 100))
    N_range = small_N + average_N + big_N + huge_N  # the values of N (grid size) to try
    n_success = []
    l_n_success = []
    avg_lengths = []
    num_simulations = 1000  # number of simulations for each N
    start_time = time.time()
    for N in N_range:
        path_lengths = []
        for _ in tqdm(range(num_simulations),ascii=True,leave=False):
            grid = get_random_grid(N, p)
            path_length = efficient_bfs(grid)
            if path_length != np.inf:
                n_success.append(N)
                l_n_success.append(path_length)
                path_lengths.append(path_length)
        ratio_connected = len(path_lengths) / num_simulations
        avg_length = np.mean(path_lengths)
        std_length = np.std(path_lengths)
        avg_lengths.append(avg_length)
        print(f'N={N}\taverage length={avg_length:>8.2f}±{std_length:.2f}\tpath probability={ratio_connected:.5f}')
    print("--- %s seconds ---" % (time.time() - start_time))  # print runtime
    slope, intercept, _, _, _ = linregress(np.log(n_success), np.log(l_n_success))  # log-log to retrieve exponent
    factor = float(np.exp(intercept))
    print(f'Slope is {slope:.3f}')
    predicted = factor*np.array(N_range)**slope  # theoretical curv deduced from linear regression
    plt.scatter(n_success, l_n_success, c='g', marker='x', linewidths=0.1, label='empirical (N, path length) for each point')
    plt.plot(N_range, avg_lengths, color='r', label='empirical average length for each N')
    plt.plot(N_range, predicted, color='b', label='predicted average length for each N, using log-log regression')
    plt.xlabel('Grid size N')
    plt.ylabel('Average path length')
    plt.legend()
    patch_param = mpatches.Patch(color='g', label=f'{num_simulations} simulations per value of N')
    patch_avg   = mpatches.Patch(color='r', label=f'empirical average of path length as function of N')
    patch_slope = mpatches.Patch(color='b', label=f'path length$\\propto {factor:.2f} N^{{{slope:.3f}}}$ according to regression')
    plt.legend(handles=[patch_param, patch_avg, patch_slope])
    plt.show()
