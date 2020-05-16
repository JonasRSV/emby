import numpy as np
import numba
from numba.typed import List
import heapq


# @numba.jit(nopython=True, fastmath=True, forceobj=False)
def project(x: np.ndarray,
            X: np.ndarray):
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
def place_new(p: np.ndarray, o: np.ndarray, beta: np.float32):
    n_o, dim = o.shape
    for i in range(dim):
        p[i] = np.random.normal() * 100

    return p
    # Gradient #TODO
    # loss is (exp(-dist(o, p) / beta))
    # gradient ins -d(dist(o, p)) / beta * exp(-dist(o, p) / beta)
    # -(1 / 2(dist)) * -2(o - p)) / beta * exp(-dist(o, p) / beta)
    """
    movement = 10.0
    while movement > 0.001:
        dist_o_p = np.sqrt(np.square(o - p).sum(axis=1))
        d_dist = -(o - p)

        gradient = np.zeros(dim)
        for i in range(n_o):
            gradient += -(d_dist[i] * (1 / (2 * dist_o_p[i]))) / beta #* np.exp(-dist_o_p[i] / beta)

        p = p - 0.01 * gradient
        movement = 0.01 * np.abs(gradient).mean()

    return p
    """


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def place_neighbour(p: np.ndarray, n: np.ndarray, o: np.ndarray, alpha: np.float32):
    n_n, dim = n.shape
    n_o, _ = o.shape
    # Gradient
    # loss is (dist(n, p) - alpha))**2
    # d(dist(n, p)) * 2 * (dist(n, p) - 0.1))
    # d(dist(n, p)) = 1 / 2(dist(n, p)) * (-2(n, p))

    movement = 10.0
    while movement > 0.0001:
        dist_n_p = np.sqrt(np.square(n - p).sum(axis=1))
        d_dist = -(n - p)

        gradient = np.zeros(dim)
        for i in range(n_n):
            gradient += d_dist[i] * (1 / (2 * dist_n_p[i])) * 2 * (dist_n_p[i] - alpha)

        p = p - 0.01 * gradient
        movement = np.abs(gradient).mean()

    return p


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def place(z: int,
          point: int,
          x_coords: np.ndarray,
          z_coords: np.ndarray,
          neigh: np.ndarray,
          place_mask: np.ndarray,
          alpha: np.float32,
          beta: np.float32):
    # let p be point
    # let n be neighbours
    # let o be others

    p = (np.random.rand(z) - 0.5) * 40
    n = z_coords[(place_mask == 1) & (neigh == 1)]
    o = z_coords[place_mask == 1]

    if n.size == 0:
        return place_new(p, o, beta)
    else:
        return place_neighbour(p, n, o, alpha)


@numba.jit(nopython=True, fastmath=True, forceobj=False)
def fit(x: np.ndarray, z: int, alpha: np.float32, beta: np.float32, n: int, verbose: bool):
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
