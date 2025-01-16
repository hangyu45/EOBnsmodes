import numpy as np 
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integ
import scipy.special as special

from numba import njit

from .constants import *

@njit(cache=True)
def get_eob_coeffs(par):
    # global par
    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    S1_MM_z = X1*X1*chi1z
    S2_MM_z = X2*X2*chi2z

    sKerr_MM_z = S1_MM_z + S2_MM_z
    sStar_MM_z = X2/X1 * S1_MM_z + X1/X2 * S2_MM_z

    a_M = sKerr_MM_z
    a_M2 = a_M * a_M

    s1sq = S1_MM_z * S1_MM_z
    s2sq = S2_MM_z * S2_MM_z

    # since we do not include higher order SO and SS terms, only use the v1 coeffs
    c0 = 1.4467
    c1 = -1.7152360250654402
    c2 = -3.246255899738242

    KK = c0 + c1 * eta + c2 * eta * eta
    m1PlusEtaKK = -1. + eta * KK
    k0 = KK * (m1PlusEtaKK - 1.)
    k1 = -2. * (k0 + KK) * m1PlusEtaKK
    k1p2 = k1 * k1
    k1p3 = k1 * k1p2
    k2 = (k1 * (k1 - 4. * m1PlusEtaKK)) / 2. - a_M * a_M * k0 * m1PlusEtaKK * m1PlusEtaKK
    k3 = -k1 * k1 * k1 / 3. + k1 * k2 + k1 * k1 * m1PlusEtaKK \
         - 2. * (k2 - m1PlusEtaKK) * m1PlusEtaKK - a_M * a_M * k1 * m1PlusEtaKK * m1PlusEtaKK
    k4 =( 24. * k1 * k1 * k1 * k1 - 96. * k1 * k1 * k2 + 48. * k2 * k2\
        - 64. * k1 * k1 * k1 * m1PlusEtaKK\
        + 48. * a_M * a_M * (k1 * k1 - 2. * k2) * m1PlusEtaKK * m1PlusEtaKK\
        + 96. * k1 * (k3 + 2. * k2 * m1PlusEtaKK) - m1PlusEtaKK * (192. * k3 + m1PlusEtaKK * (-3008. + 123. * np.pi * np.pi))) / 96.
    k5, k5l = 0., 0.

    eob_coeff = np.array([
        a_M, a_M2, sKerr_MM_z, sStar_MM_z, s1sq, s2sq, 
        KK, 
        k0, k1, k2, k3, k4, k5, k5l
    ])

    return eob_coeff


@njit(cache=True)
def get_h_mode_tide_coeffs(par, M1_deformed=True):
    # global par
    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    if M1_deformed:
        Xc = X2
        lam_20_M5 = lam_20_M5_1
        lam_30_M7 = lam_30_M7_1
    else:
        Xc = X1
        lam_20_M5 = lam_20_M5_2
        lam_30_M7 = lam_30_M7_2

    Xp = 1.-Xc
    Xc2 = Xc * Xc
    Xc3 = Xc2 * Xc

    qq = Xc/Xp

    # 
    n_kap, n_h_mode, n_pn = 7, 3, 2
    h_tide_mult_coeff = np.zeros((n_kap, n_h_mode, n_pn))

    # 22 mode of GW
    
    # kap_22; N
    h_tide_mult_coeff[0, 0, 0] = lam_20_M5 * 3./Xp *  (1. + 0.75 * Xc)
    # kap_20; N
    h_tide_mult_coeff[1, 0, 0] = lam_20_M5 * 3./2. * qq
    # kap_2n2; N
    h_tide_mult_coeff[2, 0, 0] = lam_20_M5 * 9./4. * qq 
    # kap_33; N
    h_tide_mult_coeff[3, 0, 0] = lam_30_M7 * 25./2. * qq
    # kap_31; N
    h_tide_mult_coeff[4, 0, 0] = lam_30_M7 * 15./2. * qq
    # kap_3n1; N
    h_tide_mult_coeff[5, 0, 0] = h_tide_mult_coeff[4, 0, 0]
    # kap_3n3; N
    h_tide_mult_coeff[6, 0, 0] = h_tide_mult_coeff[3, 0, 0]

    # kap_22; 1PN
    h_tide_mult_coeff[0, 0, 1] = lam_20_M5 * 3./28./Xp \
                                * (84. - 51. * Xc - 5. * Xc2 - 13. * Xc3)
    # kap_20; 1PN
    h_tide_mult_coeff[1, 0, 1] = lam_20_M5 / 14. * qq * (75. - 42.*Xc - 13.*Xc2)
    # kap_2n2; 1PN
    h_tide_mult_coeff[2, 0, 1] = lam_20_M5 * 3./28. * qq * (61. - 85.*Xc - 13.*Xc2)


    # 21 mode of GW; 
    # compared to eq. A16 PHYSICAL REVIEW D 85, 123007 (2012),
    # we differ by a minus sign in the tidal terms

    # kap_22
    h_tide_mult_coeff[0, 1, 0] = lam_20_M5 * 0.375 * qq * (-1. + 18.*Xc)
    # kap_20
    h_tide_mult_coeff[1, 1, 0] = lam_20_M5 * 0.375 * qq * (-2. + 12.*Xc)
    # kap_2n2
    h_tide_mult_coeff[2, 1, 0] = lam_20_M5 * 0.375 * qq * (-9. + 18.*Xc)


    # 33 mode of GW

    # kap_22
    h_tide_mult_coeff[0, 2, 0] = lam_20_M5 * 1.125 * qq * (5. + 6.*Xc)
    # kap_20
    h_tide_mult_coeff[1, 2, 0] = lam_20_M5 * 1.125 * qq * (-2. + 4.*Xc)
    # kap_2n2
    h_tide_mult_coeff[2, 2, 0] = lam_20_M5 * 1.125 * qq * (-3. + 6.*Xc)

    return h_tide_mult_coeff






@njit(cache=True)
def get_h_mode_pn_coeffs(par):
    """
    adapted from 
    lalsimulation/lib/LALSimIMRSpinEOBFactorizedWaveform.c
    XLALSimIMREOBCalcSpinFacWaveformCoefficients
    to be consistent with v1
    """
    
    # global par
    eta, X1, X2, \
    chi1z, chi2z,\
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    dM = X1 - X2
    eta2 = eta * eta
    eta3 = eta2 * eta
    
    m1Plus3eta = -1. + 3.*eta
    m1Plus3eta2 = m1Plus3eta * m1Plus3eta
    m1Plus3eta3 = m1Plus3eta2 * m1Plus3eta

    chiS = 0.5*(chi1z + chi2z)
    chiA = 0.5*(chi1z - chi2z)

    a_M = X1 * X1 * chi1z + X2 * X2 * chi2z
    a_M2 = a_M * a_M
    a_M3 = a_M2 * a_M

    ############################
    # rholm
    
    # the first few modes are hard coded as 
    # [22, 21, 33, 32, 44]
    # pn starts at vw
    n_mode, n_pn = 5, 6
    rholm_coeff = np.zeros((n_mode, n_pn))

    # 22 mode vw2
    rholm_coeff[0, 1] = -43. / 42. + (55. * eta) / 84.
    # 22 mode vw3
    rholm_coeff[0, 2] = (-2. * (chiS + chiA * dM - chiS * eta)) / 3.
    # 22 mode vw4
    rholm_coeff[0, 3] = - 20555. / 10584. + 0.5 * a_M2 \
                        - (33025. * eta) / 21168. + (19583. * eta2) / 42336.
    
    kappaS = 0.5 * (Ces1 + Ces2)
    kappaA = 0.5 * (Ces1 - Ces2)
    
    rholm_coeff[0, 3] += chiA*chiA*(0.5*dM*kappaA - kappaS*eta + 0.5*kappaS + eta - 0.5) \
                        + chiA*chiS*(dM*kappaS - dM - 2*kappaA*eta + kappaA) \
                        + chiS*chiS*(0.5*dM*kappaA - kappaS*eta + 0.5*kappaS + eta - 0.5)
    # 22 mode vw5
    rholm_coeff[0, 4] = (-34. * a_M) / 21.

    # 22 mode vw6
    rholm_coeff[0, 5] = 1556919113. / 122245200. + (89. * a_M2) / 252. \
                -(48993925. * eta) / 9779616. - (6292061. * eta2) / 3259872. \
                +(10620745. * eta3) / 39118464. + (41. * eta * np.pi*np.pi) / 192.\
                -428.*(gamma_E + 2. * np.log(2) )/105.

    # 21 mode vw2
    rholm_coeff[1, 1] = -59. / 56 + (23. * eta) / 84. - 9. / 32. * a_M2
    # 21 mode vw3
    rholm_coeff[1, 2] = 1177. / 672. * a_M - 27. / 128. * a_M3
    # 21 mode vw4
    rholm_coeff[1, 3] = -47009. / 56448. - (865. * a_M2) / 1792. - (405. * a_M2 * a_M2) / 2048. \
                        -(10993. * eta) / 14112. + (617. * eta2) / 4704.
    # 21 mode vw5
    rholm_coeff[1, 4] = (-98635. * a_M) / 75264. + (2031. * a_M3) / 7168. - (1701. * a_M2 * a_M3) / 8192.

    # 33 mode vw2
    rholm_coeff[2, 1] = -7. / 6. + (2. * eta) / 3.
    # 33 mode vw4 
    rholm_coeff[2, 3] = -6719. / 3960. + a_M2 / 2. - (1861. * eta) / 990. \
                        +(149. * eta2) / 330.
    # 33 mode vw5
    rholm_coeff[2, 4] = (-4. * a_M) / 3.

    # 32 mode vw
    rholm_coeff[3, 0] = (4. * chiS * eta) / (-3. * m1Plus3eta)
    # 32 mode vw2 
    rholm_coeff[3, 1] = (-4. * a_M2 * eta2) / (9. * m1Plus3eta2) + (328. - 1115. * eta + 320. * eta2) / (270. * m1Plus3eta)
    # 32 mode vw3
    rholm_coeff[3, 2] = (2. * (45. * a_M * m1Plus3eta3 - a_M * eta * (328. - 2099. * eta + 5. * (733. + 20. * a_M2) * eta2 - 960. * eta3))) \
            / (405. * m1Plus3eta3)
    # 32 mode vw4
    rholm_coeff[3, 3] = 2. / 9. * a_M

    # 44 mode vw2
    rholm_coeff[4, 1] = (1614. - 5870. * eta + 2625. * eta2) / (1320. * m1Plus3eta)
    # 44 mode vw3
    rholm_coeff[4, 2] = (chiA * (10. - 39. * eta) * dM + chiS * (10. - 41. * eta + 42. * eta2)) / (15. * m1Plus3eta)
    # 44 mode vw4
    rholm_coeff[4, 3] = a_M2 / 2. + (-511573572. + 2338945704. * eta \
                                     - 313857376. * eta2 - 6733146000. * eta3 + 1252563795. * eta2 * eta2) \
                                    / (317116800. * m1Plus3eta2)

    ############################
    # rholm_log_coeff
    # note we put gamma_E + log(2) + log(m) into rholm_coeff since it doesn't evolve

    # just 22 mode starting at vw6
    n_mode, n_pn = 1, 1
    rholm_log_coeff = np.zeros((n_mode, n_pn))

    # 22 mode vw6
    rholm_log_coeff[0, 0] = -428./105. / 2.

    
    ############################
    # c_{l+epsilon} * flm^S
    # compute it here to avoid divergence in f_lm_spin when delta_m = 0

    c2 = 1.         # (l=2, ep=0, m=0)
    c3 = X2-X1      # (l=3, ep=0, m=1 & 3)
                    # or (l=2, ep=1, m=1)
    c4 = 1.-3.*eta  # (l=4, ep=0, m=4)
                    # (l=3, ep=1, m=2)

    # note c3/dM = -1
    c3_dM = -1.

    # just for 21 and 33 modes
    n_mode, n_pn = 2, 3
    cflmS_coeff = np.zeros((n_mode, n_pn))

    # 21 mode vw; 
    cflmS_coeff[0, 0] = (-3. * (c3 * chiS + c3_dM * chiA )) / (2.)

    # 33 mode vw3
    cflmS_coeff[1, 2] = (c3 * chiS * (-4. + 5. * eta) + c3_dM * chiA * (-4. + 19. * eta)) / (2.)

    ############################
    # deltalm

    # for (22, 21, 33) modes
    # delta_coeff[mode, pn, 0] gives coeff for Hw
    #            [mode, pn, 1] gives coeff for vw starting from vw5
    n_mode, n_pn = 3, 3
    deltalm_coeff = np.zeros((n_mode, n_pn, 2))

    # 22 mode; Hw
    deltalm_coeff[0, 0, 0] = 7./3.
    deltalm_coeff[0, 1, 0] = (428. * np.pi) / 105.
    deltalm_coeff[0, 2, 0] = -2203. / 81. + (1712. * np.pi * np.pi) / 315.

    # 22 mode vw (starting from vw5)
    deltalm_coeff[0, 0, 1] = -24. * eta

    # 21 mode; Hw
    deltalm_coeff[1, 0, 0] = 2./3.
    deltalm_coeff[1, 1, 0] = (107. * np.pi) / 105.

    # 21 mode; vw
    deltalm_coeff[1, 0, 1] = -493. * eta / 42.

    # 33 mode; Hw
    deltalm_coeff[2, 0, 0] = 13. / 10.
    deltalm_coeff[2, 1, 0] = (39. * np.pi) / 7.

    # 33 mode; vw
    deltalm_coeff[2, 0, 1] = -80897. * eta / 2430.

    return rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff
    
    
@njit(cache=True)
def get_nqc_coeffs(par):
    eta = par[0]

    a1 = -4.55919 + 18.761 * eta - 24.226 * eta * eta
    a2 = 37.683 - 201.468 * eta + 324.591 * eta * eta
    a3 = -39.6024 + 228.899 * eta - 387.222 * eta * eta

    a4, a5, b1, b2, b3, b4 = 0., 0., 0., 0., 0., 0.
    nqc_coeff = np.array([a1, a2, a3, a4, a5, 
                          b1, b2, b3, b4])
    return nqc_coeff