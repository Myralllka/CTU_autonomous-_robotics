import numpy as np
import queue as Q


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.insert(0, current)
    return total_path


def astar(grid, begin, end):
    """
    :param grid: np array
    :param begin: 2-el structure
    :param end:   2-el structure
    """

    bx = begin[0]
    by = begin[1]
    ex = end[0]
    ey = end[1]
    h = grid.shape[0] - 1
    w = grid.shape[1] - 1
    comp_heuristic = lambda i, j: np.sqrt((ey - j)**2 +(ex - i)**2)
    open_set = Q.PriorityQueue()
    came_from = dict()
    g_score = np.ones_like(grid) * np.inf
    g_score[bx, by] = 0
    f_score = np.ones_like(grid) * np.inf
    f_score[bx, by] = comp_heuristic(bx, by)
    open_set.put((f_score[bx, by], (bx, by)))

    import matplotlib.pyplot as plt

    while not open_set.empty():
        current = open_set.get()
        current_fscore = current[0]
        cx, cy = current[1][0], current[1][1]

        if cx == ex and cy == ey:
            res = reconstruct_path(came_from, (cx, cy))
            return res
        neigbor = []
        if cx > 0 and grid[cx - 1, cy] == 0:  # top
            neigbor.append((cx - 1, cy))
        if cx < h and grid[cx + 1, cy] == 0:  # bottom
            neigbor.append((cx + 1, cy))
        if cy > 0 and grid[cx, cy - 1] == 0:  # left
            neigbor.append((cx, cy - 1))
        if cy < w and grid[cx, cy + 1] == 0:  # right
            neigbor.append((cx, cy + 1))

        for each in neigbor:
            tentative_gscore = g_score[cx, cy] + 1
            if tentative_gscore < g_score[each[0], each[1]]:
                came_from[each] = (cx, cy)
                g_score[each[0], each[1]] = tentative_gscore
                f_score[each[0], each[1]] = tentative_gscore + comp_heuristic(each[0], each[1])
                open_set.put((f_score[each[0], each[1]], each))



gr = np.array([[0, 0, 1, 0, 0, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1, 0]])

import matplotlib.pyplot as plt
if __name__ == "__main__":
    plt.imshow(gr)
    plt.show()
    plt.imshow(gr, origin="lower")
    plt.show()
    path = astar(gr, (0, 0), (4, 5))
    print(path)
    for each in path:
        gr[each[0], each[1]] = 50

    plt.imshow(gr, origin="lower")
    plt.show()
