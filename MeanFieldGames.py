import pandas as pd
from typing import Callable
from tqdm import tqdm

import numpy as np
from numpy.linalg import norm

from scipy import sparse
from scipy.sparse import diags, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, norm



##### For the 2 next lambda functions, 
##### should they be sent as parameters?

### Generate Example 1.2's H0  (numpy compatible)
# 0 <= c0, 0 < c1, 0 <= alpha <= 4(beta-1)/beta, 1 < beta
# p: float, 0 < mu
H0_generation: Callable = lambda c0, c1, alpha, beta: (
    lambda p, mu: norm(p) / (
        beta * (c0 + c1*mu)**alpha
    )
)  # USELESS ???

### Generate Example 2.1's Htilde
# 0 <= c0, 0 < c1, 0 <= alpha <= 4(beta-1)/beta, 1 < beta
# p in R^{2 x Nh} , 0 < mu
Htilde_generation: Callable = lambda c0, c1, alpha, beta: (
    lambda p, p2, mu: (
        np.maximum(0, p[0])**2 + np.minimum(0, p[1])**2
    )**(beta/2) / (
        beta * (c0 + c1*mu)**alpha
    )
)



class MFG:

    """
        Description
    """

    def __init__(
            self, 
            c0: float,
            c1: float,
            alpha: float,
            beta: float
    ) -> None:
        pass








if __name__ == "__main__":
    
    # SOME TESTS

    pass

else:
    print("EDP Project well imported.")
    print("CREDIT: Nathan SANGLIER, Ronan PÉCHEUL")
