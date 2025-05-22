import numpy as np
from MinGenEigCmp import min_gen_eig_cmp

def tdoafdoaloc_biasred(s: np.ndarray,
                        s_dot: np.ndarray,
                        rd: np.ndarray,
                        rd_dot: np.ndarray,
                        Q_alpha: np.ndarray):
    """
    Bias‐reduced TDOA/FDOA localization (Ho 2012).
    Inputs:
      s       : (N×M) sensor positions
      s_dot   : (N×M) sensor velocities
      rd      : (M-1,) range‐difference vector
      rd_dot  : (M-1,) range‐rate‐difference vector
      Q_alpha : ((2M-2)×(2M-2)) covariance of [rd; rd_dot]
    Returns:
      u       : (N,) source position
      u_dot   : (N,) source velocity
    """
    M = s.shape[1]
    N = s.shape[0]
    iQa = np.linalg.inv(Q_alpha)

    # First stage
    ht = rd**2 - np.sum(s[:,1:]**2, axis=0) + np.sum(s[:,0]**2)
    hf = 2*(rd*rd_dot
            - np.sum(s_dot[:,1:]*s[:,1:], axis=0)
            + np.dot(s_dot[:,0], s[:,0]))
    Gt = -2 * np.hstack([
        (s[:,1:]-s[:,[0]]).T,
        np.zeros((M-1, N)),
        rd.reshape(-1,1),
        np.zeros((M-1,1))
    ])
    Gf = -2 * np.hstack([
        (s_dot[:,1:]-s_dot[:,[0]]).T,
        (s[:,1:]-s[:,[0]]).T,
        rd_dot.reshape(-1,1),
        rd.reshape(-1,1)
    ])
    h1 = np.concatenate([ht, hf])
    G1 = np.vstack([Gt, Gf])
    W1 = iQa
    A = np.hstack([-G1, h1.reshape(-1,1)])

    # initial phi1
    phi1 = np.linalg.inv(G1.T @ W1 @ G1) @ (G1.T @ W1 @ h1)
    u     = phi1[:N]
    u_dot = phi1[N:-2]

    B_tilde    = np.diag(rd)
    Bdot_tilde = np.diag(rd_dot)

    # one update of W1
    b     = np.linalg.norm(u.reshape(-1,1)-s[:,1:], axis=0)
    b_dot = np.sum((u.reshape(-1,1)-s[:,1:])*(u_dot.reshape(-1,1)-s_dot[:,1:]), axis=0)/b
    B     = 2*np.diag(b)
    B_dot = 2*np.diag(b_dot)
    B1    = np.block([[B, np.zeros_like(B)],
                      [B_dot, B]])
    iB1   = np.linalg.inv(B1)
    W1    = iB1.T @ iQa @ iB1

    # build OmegaTilde
    E11 = np.zeros_like(W1)
    E11[M:, :M-1] = np.eye(M-1)
    E12 = np.zeros_like(W1)
    E12[:M-1,:M-1] = B_tilde
    E12[M:,:M-1] = Bdot_tilde
    E12[M:,M:]   = B_tilde
    O = lambda X,Y: np.trace(X @ Y @ Q_alpha)
    Omega = 4 * np.array([
        [ O(W1, np.eye(W1.shape[0])), O(W1, E11),              O(W1, E12)           ],
        [ 0,                         O(E11.T @ W1, E11),       O(E11.T @ W1, E12)   ],
        [ 0,                         0,                        O(E12.T @ W1, E12)   ]
    ])
    # symmetrize
    Omega[1,0] = Omega[0,1]
    Omega[2,0] = Omega[0,2]
    Omega[2,1] = Omega[1,2]

    # build At and solve min‐gen‐eig
    At = A.T @ W1 @ A
    A11 = At[:2*N, :2*N]
    A12 = At[:2*N, 2*N:]
    A21 = A12.T
    A22 = At[2*N:, 2*N:]
    Av1 = -np.linalg.inv(A11) @ A12
    Me  = A22 + A21 @ Av1
    v2, _ = min_gen_eig_cmp(Me, Omega)
    v1 = Av1 @ v2
    phi1 = np.concatenate([v1, v2[:-1]])

    u     = phi1[:N]
    u_dot = phi1[N:-2]
    icov_phi1 = G1.T @ W1 @ G1

    # Second stage
    h2 = np.concatenate([
        (u - s[:,0])**2,
        (u - s[:,0])*(u_dot - s_dot[:,0]),
        [phi1[-2]**2],
        [phi1[-2]*phi1[-1]]
    ])
    G2 = np.vstack([
        np.hstack([np.eye(N), np.zeros((N,1)), np.zeros((N,N)), np.zeros((N,1))]),
        np.hstack([np.zeros((N,N+1)), np.eye(N),  np.zeros((N,1))]),
        np.hstack([np.ones((1,N)),   np.zeros((1,N+1))]),
        np.hstack([np.zeros((1,N)),  np.ones((1,N+1))])
    ])
    K = np.block([
        [np.hstack([np.eye(N), np.zeros((N,1)), np.zeros((N,N)), np.zeros((N,1))])],
        [np.hstack([np.zeros((N,N)), np.zeros((N,1)), np.eye(N), np.zeros((N,1))])],
        [np.hstack([np.zeros((1,N)), [[1]], np.zeros((1,N)), [[0]]])],
        [np.hstack([np.zeros((1,N)), [[0]], np.zeros((1,N)), [[1]]])]
    ]).reshape(2*N+2,2*N+2)

    for _ in range(2):
        delta_u = u - s[:,0]
        Bt  = np.diag(np.concatenate([delta_u, [np.linalg.norm(delta_u)]])) 
        Btd = np.diag(np.concatenate([
            u_dot - s_dot[:,0],
            [(delta_u @ (u_dot - s_dot[:,0]))/np.linalg.norm(delta_u)]
        ]))
        B2 = np.block([[2*Bt,           np.zeros_like(Bt)],
                       [Btd,            Bt]])
        B2 = K @ B2 @ K.T
        iB2 = np.linalg.inv(B2)
        W2  = iB2.T @ icov_phi1 @ iB2
        phi2 = np.linalg.inv(G2.T @ W2 @ G2) @ (G2.T @ W2 @ h2)
        u      = np.sign(delta_u)*np.sqrt(np.abs(phi2[:N])) + s[:,0]
        u_dot  = phi2[N:] / delta_u + s_dot[:,0]

    return u, u_dot