import pandas as pd
from typing import Callable
from tqdm import tqdm

import numpy as np
from numpy.linalg import norm

from scipy import sparse
from scipy.sparse import diags, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, norm



NEWTON_STOPPING_CRITERIA: float = 1e-12
FIXED_POINT_STOPPING_CRITERIA: float = 1e-5  # normalized norm

pos: Callable = lambda x: np.maximum(0, x)
neg: Callable = lambda x: -np.minimum(0, x)


### Generate Example 2.1's Htilde and derivatives
# 0 <= c0, 0 < c1, 0 <= alpha <= 4(beta-1)/beta, 1 < beta
# p in R^{2 x Nh} , 0 < mu

Htilde_generation: Callable = lambda c0, c1, alpha, beta: (
    lambda p, mu: (
        neg(p[0])**2 + pos(p[1])**2
    )**(beta/2) / (
        beta * (c0 + c1*mu)**alpha
    )
)

Htilde_p1_generation: Callable = lambda c0, c1, alpha, beta: (
    lambda p, mu: -neg(p[0]) * (
        neg(p[0])**2 + pos(p[1])**2
    )**(beta/2 - 1) / (
        (c0 + c1*mu)**alpha
    )
)

Htilde_p2_generation: Callable = lambda c0, c1, alpha, beta: (
    lambda p, mu: pos(p[0]) * (
        neg(p[0])**2 + pos(p[1])**2
    )**(beta/2 - 1) / (
        (c0 + c1*mu)**alpha
    )
)



class MeanField:

    """
        Finish Description ....
    """

    def __init__(
        self,
        sigma: float,
        c0: float,
        c1: float,
        alpha: float,
        beta: float,
        f_tilde: Callable,
        g: Callable,
        m0: Callable,
        phi: Callable
    ) -> None:
        """
            Finish Description ....

            > 
            > 
            >
            >

        """
        self.nu: float = sigma**2 / 2
        self.f_tilde: Callable = f_tilde
        self.g: Callable = g
        self.m0: Callable = m0
        self.phi: Callable = phi

        self.Dt: Callable = lambda Delta_t, W: (W[1:] - W[:-1]) / Delta_t
        self.Dh: Callable = lambda h, W: (W[1:] - W[:-1]) / h
        self.Deltah: Callable = lambda h, W: -(2*W[2:] - W[1:-1] - W[:-2]) / h**2

        self.Htilde: Callable = Htilde_generation(c0, c1, alpha, beta)
        Htilde_p1: Callable = Htilde_p1_generation(c0, c1, alpha, beta)
        Htilde_p2: Callable = Htilde_p2_generation(c0, c1, alpha, beta)
        self.T: Callable = lambda i, h, U, M, Mtilde: (
            lambda nabla: (
                M[i]*Htilde_p1(nabla[i-1:i+1], Mtilde[i])
                - M[i-1]*Htilde_p1(nabla[i-2:i], Mtilde[i])
                + M[i+1]*Htilde_p2(nabla[i:i+2], Mtilde[i+1])
                - M[i]*Htilde_p2(nabla[i-2:i], Mtilde[i])
            )(self.Dh(h, U))
        ) / h  # I need to check indices !!!
        

    def MFGame_run(
        self,
        Nh: int,
        NT: int,
    ) -> np.array:
        """
            Finish description ....    

            > Nh:
            > NT:
            Return a 3D matrix (time x u x m) of the Mean Field Games.
        """

        # HJB

        # Fokker-Plank
        
        pass

    
    def MFControl_run(
        self,
        Nh: int,
        NT: int,
    ) -> np.array:
        """
            Finish description ....    
        
            > Nh: 
            > NT:
            Return a 3D matrix (time x u x m) of the Mean Field Games.
        """
        
        pass







if __name__ == "__main__":
    
    # SOME TESTS

    pass

else:
    print("EDP Project well imported.")
    print("CREDIT: Nathan SANGLIER, Ronan PÉCHEUL")
