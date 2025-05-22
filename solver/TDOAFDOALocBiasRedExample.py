import numpy as np
import matplotlib.pyplot as plt
from TDOAFDOALoc_BiasRed import tdoafdoaloc_biasred
from MinGenEigCmp import min_gen_eig_cmp
from TDOAFDOALocMvgSrcSen import tdoafdoaloc_mvg_src_sen
from TDOAFDOALocMvgSrcSenCRLB import tdoafdoaloc_mvg_src_sen_crlb

def main():
    np.random.seed(17)
    M = 8
    L = 2000
    uo = np.array([2000,2500,3000])
    u_doto = np.array([-20,15,40])

    # true sensor positions & velocities
    so = np.vstack([
        [-150,-200,200,100,-100,350,300,400],
        [150,250,-150,100,-100,200,500,150],
        [-130,-250,300,100,-100,100,200,100]
    ])[:,:M]
    sd = np.vstack([
        [20,-10,-7,15,-20,10,10,-30],
        [0,-15,-10,15,10,20,-20,10],
        [-20,10,20,-10,10,30,10,20]
    ])[:,:M]

    # measurement covariance
    R = (np.eye(M-1)+1)/2
    R = np.block([[R, np.zeros((M-1,M-1))],
                  [np.zeros((M-1,M-1)), R*0.01]])
    chol_R = np.linalg.cholesky(R)

    sigma2_dVec = np.arange(-30,20,5)
    mseuBR = []; mseu = []; crlbu = []
    mseudotBR = []; mseudot = []; crlbudot = []

    for sigma2_d in sigma2_dVec:
        NseStd = 10**(sigma2_d/20)
        Q_alpha = 10**(sigma2_d/10)*R
        CRLB = tdoafdoaloc_mvg_src_sen_crlb(so, sd, uo, u_doto, Q_alpha)
        crlbu.append(np.trace(CRLB[:3,:3]))
        crlbudot.append(np.trace(CRLB[3:,3:]))

        SimMSEu = SimMSEud = SimMSEuBR = SimMSEudBR = 0
        # noise mean removal
        PP = np.mean([chol_R.T @ np.random.randn(2*(M-1)) for _ in range(L)], axis=0)
        for _ in range(L):
            delta_r = (chol_R.T @ np.random.randn(2*(M-1)) - PP)*NseStd
            rd     = delta_r[:M-1] + (np.linalg.norm(uo[:,None]-so, axis=0)[1:] - np.linalg.norm(uo-so[:,[0]]))
            rd_dot = delta_r[M-1:] + ((uo[:,None]-so).T@(u_doto[:,None]-sd).T / np.linalg.norm(uo-so[:,[0]]))[1:]

            u,    u_dot    = tdoafdoaloc_mvg_src_sen(so, sd, rd, rd_dot, Q_alpha)
            uBR,  u_dotBR  = tdoafdoaloc_biasred(so, sd, rd, rd_dot, Q_alpha)

            SimMSEu   += np.linalg.norm(u-uo)**2
            SimMSEud  += np.linalg.norm(u_dot-u_doto)**2
            SimMSEuBR += np.linalg.norm(uBR-uo)**2
            SimMSEudBR+= np.linalg.norm(u_dotBR-u_doto)**2

        mseu.append(10*np.log10(SimMSEuBR/L))
        mseu.append(10*np.log10(SimMSEu/L))
        mseudotBR.append(10*np.log10(SimMSEudBR/L))
        mseudot.append(10*np.log10(SimMSEud/L))

    # plot
    plt.figure()
    plt.plot(sigma2_dVec, mseuBR, 'kx-', label='BiasRed')
    plt.plot(sigma2_dVec, mseu,   'ko-', label='WMLS,Ho-Xu')
    plt.plot(sigma2_dVec, crlbu,  'k-',  label='CRLB')
    plt.xlabel('20 log(σ_r)'); plt.ylabel('10 log(MSE), Position')
    plt.legend(); plt.grid()
    plt.show()

if __name__ == '__main__':
    main()