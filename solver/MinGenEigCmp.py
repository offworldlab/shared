import numpy as np

def min_gen_eig_cmp(F: np.ndarray, D: np.ndarray):
    """
    Compute the minimum generalized eigenvalue and eigenvector of (F, D),
    where F and D are 3×3 numpy arrays.
    Returns (mn_eig_vec, mn_eig_val).
    """
    # preserve original F
    Forg = F.copy()
    # adjugate of D = det(D) * inv(D)
    detD = np.linalg.det(D)
    adjD = detD * np.linalg.inv(D)
    # update F
    Fm = adjD @ Forg

    # cubic coefficients for λ^3 + a*λ^2 + b*λ + c = 0
    a = -np.trace(Fm)
    b = (Fm[0,0]*Fm[1,1] + Fm[0,0]*Fm[2,2] + Fm[1,1]*Fm[2,2]
         - Fm[0,2]*Fm[2,0] - Fm[0,1]*Fm[1,0] - Fm[1,2]*Fm[2,1])
    c = (Fm[0,0]*Fm[1,2]*Fm[2,1] + Fm[0,1]*Fm[1,0]*Fm[2,2] + Fm[0,2]*Fm[1,1]*Fm[2,0]
         - Fm[0,0]*Fm[1,1]*Fm[2,2] - Fm[0,1]*Fm[1,2]*Fm[2,0] - Fm[0,2]*Fm[1,0]*Fm[2,1])

    # depressed cubic parameters
    q = (3*b - a**2) / 9
    r = (9*a*b - 27*c - 2*a**3) / 54
    d = q**3 + r**2

    # helper for real cube root
    def cbrt(x):
        return np.sign(x) * np.abs(x)**(1/3)

    if d < 0:
        rho = np.sqrt(-q**3)
        rho_r3 = rho**(1/3)
        theta = np.arccos(r / rho)
        th3 = theta / 3
        rt = np.array([
            -a/3 + 2*rho_r3*np.cos(th3),
            -a/3 - rho_r3*np.cos(th3) - np.sqrt(3)*rho_r3*np.sin(th3),
            -a/3 - rho_r3*np.cos(th3) + np.sqrt(3)*rho_r3*np.sin(th3),
        ])
    else:
        s = cbrt(r + np.sqrt(d))
        t = cbrt(r - np.sqrt(d))
        rt = np.array([-a/3 + (s + t)])

    # take absolute real parts and sort
    rt = np.sort(np.abs(np.real(rt)))
    mn_eig_val = rt[0]

    # solve for eigenvector: (det(D)*F_orig – D*λ) x = 0, set x3 = 1
    P = detD * Forg - D * mn_eig_val
    v12 = -np.linalg.inv(P[0:2, 0:2]) @ P[0:2, 2]
    mn_eig_vec = np.hstack([v12, 1.0])

    return mn_eig_vec, mn_eig_val