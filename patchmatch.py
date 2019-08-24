"""
Implementation of "PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing", Barnet et al.
See https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf
"""

import multiprocessing
import multiprocessing.sharedctypes
import numpy as np


def patch_match(a, b, iters=5, patch_size=7, w=None, alpha=0.5):
    """ Performs the vanilla patch match algorithm.

    Args:
        a: Image to be matched. Should be a numpy array of shape [height, width, channels] with values in range [0, 1].
        b: Image to source the patches from. Should be a numpy array of shape [height, width, channels] with values in
            range [0, 1].
        iters: Number of iterations to run.
        patch_size: Width and height of a patch. Should be an uneven integer.
        w: Search window size for random search. If None it will be set to the maximum dimension of image b.
        alpha: Scale factor for random search. Should be a float in range (0, 1)
    """
    h_a, w_a, _ = a.shape
    h_b, w_b, _ = b.shape

    # Initialize NNF
    nnf = np.zeros([h_a, w_a, 2], dtype=object)
    distances = np.zeros([h_a, w_a])
    for y in range(h_a):
        for x in range(w_a):
            ry, rx = np.random.randint(0, h_b), np.random.randint(0, w_b)
            nnf[y, x] = [ry, rx]
            distances[y, x] = distance(a, b, y, x, ry, rx, patch_size)

    for i in range(iters):
        # On uneven iterations we go from left to right and top to bottom. On even ones
        # we go from right to left and bottom to top.
        if i % 2 == 0:
            range_y = range(h_a)
            range_x = range(w_a)
            change_x = 1
            change_y = 1
        else:
            range_y = range(h_a - 1, -1, -1)
            range_x = range(w_a - 1, -1, -1)
            change_x = -1
            change_y = -1

        for y in range_y:
            for x in range_x:
                best_y, best_x = nnf[y, x, 0], nnf[y, x, 1]
                best_dist = distances[y, x]

                # Propagation
                if 0 <= y - change_y < h_a:
                    patch_y, patch_x = nnf[y - change_y, x, 0], nnf[y - change_y, x, 1]
                    patch_y = patch_y + change_y
                    if 0 <= patch_y < h_b:
                        d = distance(a, b, y, x, patch_y, patch_x, patch_size)
                        if d < best_dist:
                            best_y, best_x, best_dist = patch_y, patch_x, d
                if 0 <= x - change_x < w_a:
                    patch_y, patch_x = nnf[y, x - change_x]
                    patch_x = patch_x + change_x
                    if 0 <= patch_x < w_b:
                        d = distance(a, b, y, x, patch_y, patch_x, patch_size)
                        if d < best_dist:
                            best_y, best_x, best_dist = patch_y, patch_x, d

                # Random search
                # If w is not given, set the search radius to the maximum image dimension
                if w is None:
                    radius = max(h_b, w_b)
                else:
                    radius = w

                while radius >= 1:
                    rand_y = np.random.randint(max(best_y - radius, 0), min(best_y + radius, h_b - 1))
                    rand_x = np.random.randint(max(best_x - radius, 0), min(best_x + radius, w_b - 1))
                    d = distance(a, b, y, x, rand_y, rand_x, patch_size)
                    if d < best_dist:
                        best_y, best_x, best_dist = rand_y, rand_x, d
                    radius = int(radius*alpha)

                # Update NNF
                nnf[y, x] = [best_y, best_x]
                distances[y, x] = best_dist
    return nnf


def patch_match_parallel(a, b, iters=5, patch_size=7, alpha=0.5, w=None, num_cores=4):
    """ Performs a parallelized version of the patch match algorithm on mutliple cores.

    As described in "The Generalized PatchMatchCorrespondence Algorithm", Barnes et al.
    See: https://gfx.cs.princeton.edu/pubs/Barnes_2010_TGP/generalized_pm.pdf

    Args:
        a: Image to be matched. Should be a numpy array of shape [height, width, channels] with values in range [0, 1].
        b: Image to source the patches from. Should be a numpy array of shape [height, width, channels] with values in
            range [0, 1].
        iters: Number of iterations to run.
        patch_size: Width and height of a patch. Should be an uneven integer.
        w: Search window size for random search. If None it will be set to the maximum dimension of image b.
        alpha: Scale factor for random search. Should be a float in range (0, 1).
        num_cores: Number of cores to run the algorithm on.
    """
    # Initialize NNF
    h_a, w_a, _ = a.shape
    h_b, w_b, _ = b.shape
    nnf = np.zeros([h_a, w_a, 2], dtype=object)
    distances = np.zeros([h_a, w_a])
    for y in range(h_a):
        for x in range(w_a):
            ry, rx = np.random.randint(0, h_b), np.random.randint(0, w_b)
            nnf[y, x] = [ry, rx]
            distances[y, x] = distance(a, b, y, x, ry, rx, patch_size)
    # Set up shared memory
    nnf_shared = multiprocessing.RawArray('i', nnf.reshape(-1))
    dists_shared = multiprocessing.RawArray('d', distances.reshape(-1))
    # Set up workers
    workers = []
    syncs = [multiprocessing.Event() for _ in range(num_cores)]
    h_tile = int(round(h_a / float(num_cores)))
    y_start = 0
    y_end = h_tile - 1
    for i in range(num_cores):
        # Last worker gets the rest if the numbers don't exactly match up
        if i == num_cores - 1:
            y_end = h_a - 1
        workers.append(multiprocessing.Process(
            target=_patch_match_tile,
            args=(a, b, y_start, y_end, nnf_shared, dists_shared, syncs, i, iters, patch_size, alpha, w)))
        y_start = y_end
        y_end += h_tile
    # Start
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # Retrieve shared memory
    return np.frombuffer(nnf_shared, dtype=np.int).reshape(h_a, w_a, 2)


def _patch_match_tile(a, b, y_start, y_end, nnf, distances, syncs, worker_id, iters=5, patch_size=7, alpha=0.5, w=None):
    """ Runs the algorithm on a horizontal tile from y_start to y_end and synchronizes the NNF after each iteration. """
    h_a, w_a, _ = a.shape
    h_b, w_b, _ = b.shape

    for i in range(iters):
        syncs[worker_id].clear()

        # On uneven iterations we go from left to right and top to bottom. On even ones
        # we go from right to left and bottom to top.
        if i % 2 == 0:
            range_y = range(y_start, y_end + 1)
            range_x = range(w_a)
            change_x = 1
            change_y = 1
        else:
            range_y = range(y_end, y_start, -1)
            range_x = range(w_a - 1, -1, -1)
            change_x = -1
            change_y = -1

        # Create local copy of shared memory
        nnf_copy = np.frombuffer(nnf, dtype=np.int).reshape(h_a, w_a, 2)
        dists_copy = np.frombuffer(distances, dtype=np.float).reshape(h_a, w_a)

        for y in range_y:
            for x in range_x:
                best_y, best_x = nnf_copy[y, x, 0], nnf_copy[y, x, 1]
                best_dist = dists_copy[y, x]

                # Propagation
                if 0 <= y - change_y < h_a:
                    patch_y, patch_x = nnf_copy[y - change_y, x]
                    patch_y = patch_y + change_y
                    if 0 <= patch_y < h_b:
                        d = distance(a, b, y, x, patch_y, patch_x, patch_size)
                        if d < best_dist:
                            best_y, best_x, best_dist = patch_y, patch_x, d
                if 0 <= x - change_x < w_a:
                    patch_y, patch_x = nnf_copy[y, x - change_x]
                    patch_x = patch_x + change_x
                    if 0 <= patch_x < w_b:
                        d = distance(a, b, y, x, patch_y, patch_x, patch_size)
                        if d < best_dist:
                            best_y, best_x, best_dist = patch_y, patch_x, d

                # Random search
                # If w is not given, set the search radius to the maximum image dimension
                if w is None:
                    radius = max(h_b, w_b)
                else:
                    radius = w

                while radius >= 1:
                    rand_y = np.random.randint(max(best_y - radius, 0), min(best_y + radius, h_b - 1))
                    rand_x = np.random.randint(max(best_x - radius, 0), min(best_x + radius, w_b - 1))
                    d = distance(a, b, y, x, rand_y, rand_x, patch_size)
                    if d < best_dist:
                        best_y, best_x, best_dist = rand_y, rand_x, d
                    radius = int(radius * alpha)

                # Update NNF
                nnf_copy[y, x] = [best_y, best_x]
                dists_copy[y, x] = best_dist

        # Synchronize with other workers
        syncs[worker_id].set()
        for event in syncs:
            event.wait()

        # Write back data
        start = np.ravel_multi_index((y_start, 0, 0), nnf_copy.shape)
        end = np.ravel_multi_index((y_end, w_a-1, 1), nnf_copy.shape)
        nnf[start:end+1] = nnf_copy[y_start:y_end+1].reshape(-1)

        start = np.ravel_multi_index((y_start, 0), dists_copy.shape)
        end = np.ravel_multi_index((y_end, w_a-1), dists_copy.shape)
        distances[start:end+1] = dists_copy[y_start:y_end+1].reshape(-1)


def distance(a, b, ay, ax, by, bx, patch_size):
    """ Returns the L2 distance between the patches at location (ay,ax) and (by,bx) in the images a and b. """
    radius0 = patch_size // 2
    radius1 = radius0 + 1
    dy0 = min(radius0, ay, by)
    dy1 = min(radius1, a.shape[0]-ay, b.shape[0]-by)
    dx0 = min(radius0, ax, bx)
    dx1 = min(radius1, a.shape[1]-ax, b.shape[1]-bx)
    squares = (a[ay-dy0:ay+dy1, ax-dx0:ax+dx1] - b[by-dy0:by+dy1, bx-dx0:bx+dx1]) ** 2
    return np.sum(squares) / ((dy0+dy1) * (dx0+dx1))


def reconstruct(b, matching):
    """ Reconstructs an image according to a patch matching for this image.

    Args:
         b: Image to source the patches from. Should be a numpy array of shape [height, width, channels] with values in
            range [0, 1].
         matching: Numpy array indicating which patches from 'b' to use to reconstruct the image. This should be the
            result of 'patch_match' or 'patch_match_parallel' on an input image and 'b'.
    """
    h_match, w_match = matching.shape[0], matching.shape[1]
    result = np.zeros([h_match, w_match, b.shape[2]])
    for y in range(h_match):
        for x in range(w_match):
            result[y, x] = b[matching[y, x, 0], matching[y, x, 1]]
    return result


# def reconstruct_avg(b, matching, patch_size=7):
#     """ Reconstructs an image according to a patch matching for this image by averaging over all overlapping patches.
#
#     Args:
#         b: Image to source the patches from. Should be a numpy array of shape [height, width, channels] with values in
#             range [0, 1].
#         matching: Numpy array indicating which patches from 'b' to use to reconstruct the image. This should be the
#             result of 'patch_match' or 'patch_match_parallel' on an input image and 'b'.
#         patch_size: Width and height of a patch. Should be an uneven integer.
#     """
#     radius0 = patch_size // 2
#     radius1 = radius0 + 1
#
#     h_match, w_match = matching.shape[0], matching.shape[1]
#     result = np.zeros([h_match, w_match, b.shape[2]])
#     for y in range(h_match):
#         for x in range(w_match):
#             dy0 = min(radius0, y)
#             dy1 = min(radius1, h_match - y)
#             dx0 = min(radius0, x)
#             dx1 = min(radius1, w_match - x)
#
#             patch = matching[y - dy0:y + dy1, x - dx0:x + dx1]
#             overlapping = np.zeros(shape=(patch.shape[0], patch.shape[1], b.shape[2]), dtype=np.float32)
#
#             for ay in range(overlapping.shape[0]):
#                 for ax in range(overlapping.shape[1]):
#                     py, px = patch[ay, ax, 0], patch[ay, ax, 1]
#                     overlapping[ay, ax] = b[py, px]
#
#             if overlapping.size > 0:
#                 result[y, x] = np.mean(overlapping, axis=(0, 1))
#     return result


def reconstruct_avg(b, matching, patch_size=7):
    """ Reconstructs an image according to a patch matching for this image by averaging over all overlapping patches.

    Args:
        b: Image to source the patches from. Should be a numpy array of shape [height, width, channels] with values in
            range [0, 1].
        matching: Numpy array indicating which patches from 'b' to use to reconstruct the image. This should be the
            result of 'patch_match' or 'patch_match_parallel' on an input image and 'b'.
        patch_size: Width and height of a patch. Should be an uneven integer.
    """
    radius0 = patch_size // 2
    radius1 = radius0 + 1

    h_match, w_match = matching.shape[0], matching.shape[1]
    h_b, w_b, c_b = b.shape
    result = np.zeros([h_match, w_match, b.shape[2]])
    for y in range(h_match):
        for x in range(w_match):
            dy0 = min(radius0, y)
            dy1 = min(radius1, h_match - y)
            dx0 = min(radius0, x)
            dx1 = min(radius1, w_match - x)

            total = np.zeros(c_b)
            count = 0
            for py in range(y-dy0, y+dy1):
                for px in range(x-dx0, x+dx1):
                    my, mx = matching[py, px, 0], matching[py, px, 1],
                    # Apply inverse of the translation from (y, x) to (py, py)
                    my += y-py
                    mx += x-px
                    if 0 <= my < h_b and 0 <= mx < w_b:
                        total += b[my, mx]
                        count += 1
            if count > 0:
                result[y, x] = total / count

    return result


# Usage example:
# if __name__ == "__main__":
#    import cv2
#    a = (cv2.imread("a.png", cv2.IMREAD_COLOR) / 255).astype(np.float32)
#    b = (cv2.imread("b.png", cv2.IMREAD_COLOR) / 255).astype(np.float32)
#    match = patch_match_parallel(a, b, 1, 7, num_cores=12)
#    img = reconstruct_avg(b, match)
#    cv2.imwrite("a_matched.png", (img*255).astype(np.int32))
