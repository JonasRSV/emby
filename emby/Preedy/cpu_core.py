import numpy as np
import numba
from numba.typed import List
import heapq

epsilon = 0.00001


# @numba.jit(nopython=True, fastmath=True, forceobj=False)
def project(x: np.ndarray, X: np.ndarray):
    raise NotImplementedError()


@numba.jit(nopython=True)
def neighbour_distance(x: np.ndarray, y: np.ndarray):
    n = x.size
    d = 0.0
    for i in range(n):
        d += (y[i] - x[i]) * (y[i] - x[i])
    return d


@numba.jit(nopython=True)
def neighbours(i: int, X: np.ndarray, max_n: int):
    points, _ = X.shape

    heap = [(-100000.0, 1)]  # the heap is a min-heap
    for j in range(points):
        if i != j:
            heapq.heappush(heap, (-neighbour_distance(X[i], X[j]), j))

        if len(heap) > max_n:
            heapq.heappop(heap)

    n_heap = len(heap)
    neigh = []
    for j in range(n_heap):
        neigh.append(heap[j][1])

    return neigh


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def place_new(o: np.ndarray, weights: np.ndarray, beta: np.float32):
    n_o, dim = o.shape
    p = np.zeros(dim)
    for i in range(dim):
        p[i] = np.random.normal() * beta

    for i in range(dim):
        p[i] = np.sum(o[:, i] * weights) - p[i]

    return p


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def place_neighbour(n: np.ndarray, o: np.ndarray, weights: np.ndarray,
                    alpha: np.float32):
    n_n, dim = n.shape
    n_o, _ = o.shape

    # loss = (n - p - a)**2
    # d(loss) = 0 = -2(n - p - a)
    # 0 = p + a - n
    # n - a = p

    a = np.zeros(dim)
    for i in range(dim):
        a[i] = np.random.normal() * alpha

    p = np.zeros(dim)
    for i in range(dim):
        p[i] = np.sum(n[:, i] * weights) - a[i]

    return p


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def place(z: int, point: int, x_coords: np.ndarray, z_coords: np.ndarray,
          neigh: np.ndarray, place_mask: np.ndarray, alpha: np.float32,
          beta: np.float32):
    # let p be point
    # let n be neighbours
    # let o be others
    n = z_coords[(place_mask == 1) & (neigh == 1)]
    o = z_coords[(place_mask == 1) & (neigh != 1)]

    n_x = x_coords[(place_mask == 1) & (neigh == 1)]
    o_x = x_coords[(place_mask == 1) & (neigh != 1)]

    neighbour_weights = np.zeros(len(n_x))
    for i in range(len(n_x)):
        neighbour_weights[i] = (
            1 /
            (np.sqrt(neighbour_distance(x_coords[point], n_x[i]) + epsilon)))
    neighbour_weights = neighbour_weights / neighbour_weights.sum()

    other_weights = np.zeros(len(o_x))
    for i in range(len(o_x)):
        other_weights[i] = (
            1 /
            (np.sqrt(neighbour_distance(x_coords[point], o_x[i]) + epsilon)))
    other_weights = other_weights / other_weights.sum()

    if n.size == 0:
        return place_new(o, other_weights, beta)
    else:
        return place_neighbour(n, o, neighbour_weights, alpha)


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def fit(x: np.ndarray, z: int, alpha: np.float32, beta: np.float32, n: int,
        verbose: bool):
    points, dim = x.shape

    mat_neigh = np.zeros((points, points), dtype=np.int32)
    list_neigh = np.zeros((points, n), dtype=np.int32)
    for i in range(points):
        j = 0
        for k in neighbours(i, x, max_n=n):
            mat_neigh[i, k] = 1
            list_neigh[i, j] = k
            j += 1

        if verbose and i % (points // 100) == 0:
            print("neighbours", i, " / ", points)

    place_mask = np.zeros(points, dtype=np.int32)
    z_coords = np.zeros((points, z))

    mem = set()
    mem.add(0)
    queue = [0]

    placements = 0
    while placements < points:
        if len(queue) == 0:
            queue.append(place_mask.argmin())

        point = queue[0]
        queue = queue[1:]

        if place_mask[point] == 1:
            continue

        z_coord = place(z=z,
                        point=point,
                        x_coords=x,
                        z_coords=z_coords,
                        neigh=mat_neigh[point],
                        place_mask=place_mask,
                        alpha=alpha,
                        beta=beta)

        z_coords[point] = z_coord
        place_mask[point] = 1

        for n in list_neigh[point]:
            if n not in mem:
                queue.append(n)
                mem.add(n)

        placements += 1
        if verbose and placements % (points // 100) == 0:
            print("placing", placements, " / ", points)

    return z_coords
