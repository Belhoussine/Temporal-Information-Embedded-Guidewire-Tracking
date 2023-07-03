import numpy as np
from pylab import plt
from copy import deepcopy
from utils.dataset import unstack_frames
from utils.io import write_image, create_folder


class RPCA:
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / \
                (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.frobenius_norm(self.D)

        # Principal component pursuit (PCP) algorithm
        # Table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)  # this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)  # this line implements step 4
            # this line implements step 5
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_results(self, thresholds, resolution=(512, 512)):
        L_frames = unstack_frames(self.L, resolution)
        S_frames = unstack_frames(self.S, resolution)
        for i, S_frame in enumerate(S_frames):
            for thresh in thresholds:
                frame = deepcopy(S_frame)
                frame[frame < thresh] = 0
                frame[frame >= thresh] = 255
                plt.imshow(frame, cmap="gray")
                plt.colorbar()
                plt.title(f"Sparse frame #{i}")
                plt.show()

    def save_results(self, dataset, frames, iterations, thresholds, result_folder, resolution=(512, 512)):
        # Creating result folder
        result_folder = f"{result_folder}/{dataset}/frames_{frames}"
        print(f"Creating folder: {result_folder}")
        create_folder(result_folder)

        # Unstacking frames and getting result frame
        S_frames = unstack_frames(self.S, resolution)
        for i, S_frame in enumerate(S_frames):
            for thresh in thresholds:
                frame = deepcopy(S_frame)
                frame[frame < thresh] = 0
                frame[frame >= thresh] = 255
                frame_name = f"{dataset}_f{frames}_i{iterations}_t{thresh}_{i:02d}.png"
                print(f"Writing frame to {frame_name}")
                write_image(frame, f"{result_folder}/{frame_name}", mode="L")

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')
