import numpy as np
import sys


def ctc_loss(y, p, alphabet):
    # Initialize z:
    z = []
    for i in range(2 * len(p) + 1):
        if i % 2 == 0:
            z.append("")
        else:
            z.append(p[i // 2])

    T, K = y.shape
    S = len(z)

    # Initialize alpha:
    alpha = np.zeros((S, T))
    alpha[0, 0] = y[0, 0]
    alpha[1, 0] = y[0, alphabet.index(z[1])]  # Maybe z1 - > z0
    alpha[2:, 0] = 0

    # Dynamic Programming:
    for t in range(1, T):
        for s in range(S):
            if s == 0:
                alpha[s, t] = alpha[s, t - 1] * y[t, 0]
            elif s == 1: # z[s] != "" (epsilon)
                alpha[s, t] = (alpha[s - 1, t - 1] + alpha[s, t - 1]) * y[t, alphabet.index(z[s])]
            else:
                if z[s] == '' or z[s] == z[s - 2]:
                    alpha[s, t] = (alpha[s - 1, t - 1] + alpha[s, t - 1]) * y[t, alphabet.index(z[s])]
                else:
                    alpha[s, t] = (alpha[s - 2, t - 1] + alpha[s - 1, t - 1] + alpha[s, t - 1]) * y[
                        t, alphabet.index(z[s])]

    return alpha[-1, -1]


def print_p(p: float):
    print("%.3f" % p)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python ex3.py <path_to_matrix> <labeling> <alphabet>")
        sys.exit(1)

    matrix_path = sys.argv[1]
    labeling = sys.argv[2]
    alphabet = sys.argv[3]

    # Load the matrix
    y = np.load(matrix_path)

    # Calculate P(p|y)
    p = ctc_loss(y, labeling, alphabet)

    # Print the result
    print_p(p)