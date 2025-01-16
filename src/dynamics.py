"""
    Codes to generate the EOBnsmodes waveform.  
    Copyright (C) 2025,  Hang Yu (hang.yu2@montana.edu)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np 
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integ
import scipy.special as special
from numba import njit

from .constants import *

##############################

@njit(cache=True)
def get_H_eob_mu(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2):
    """
    all the calibrations are dropped
    """

    # unpack var

    r_M, p_r_mu_tort, phi, PI_phi_Mmu = eob_var[:4]
    H_mode_mu_1, S_ns_Mmu_1 = eob_var[4], eob_var[5]
    H_int_lm_mu_1 = eob_var[6:6+n_mode_1]
    ll_1 = ll[:n_mode_1]
    mm_1 = mm[:n_mode_1]
    ll_2 = ll[n_mode_1:]
    mm_2 = mm[n_mode_1:]

    if n_mode_2>0:
        H_mode_mu_2, S_ns_Mmu_2 = eob_var[6+n_mode_1], eob_var[7+n_mode_1]
        H_int_lm_mu_2 = eob_var[8+n_mode_1:]
        
    else:
        H_mode_mu_2, S_ns_Mmu_2, H_int_lm_mu_2 = 0., 0., np.zeros(n_mode_2)

    # note we evolve in the frame co-rotating with the orbit
    # here the canonical momentum conjugate to \phi is the total AM of the system (L_pp + tidal spin)
    # get L_pp by removing the tidal spin
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    # unpack coeffs

    # global par
    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # EOB coeffs
    a_M, a_M2, sKerr_MM_z, sStar_MM_z, s1sq, s2sq, \
    KK, \
    k0, k1, k2, k3, k4, k5, k5l \
        = eob_coeff[:14]

    # powers of r_M
    r_M2 = r_M *r_M
    
    u1 = 1./r_M
    u2 = u1 * u1
    u3 = u2 * u1
    u4 = u3 * u1
    u5 = u4 * u1

    #################
    # tidal part
    H_int_mu_1 = np.sum(H_int_lm_mu_1) 
    H_int_mu_2 = np.sum(H_int_lm_mu_2)

    dH_int_mu_du_1 = np.sum((ll_1 + 1.) * H_int_lm_mu_1 / u1)
    dH_int_mu_du_2 = np.sum((ll_2 + 1.) * H_int_lm_mu_2 / u1)

    z_I_1 = 1. - (2.*X2-eta) * u1 + 5./28. * X1 * (33.*X1 - 7.) * u2
    z_I_2 = 1. - (2.*X1-eta) * u1 + 5./28. * X2 * (33.*X2 - 7.) * u2

    dz_I_du_1 = - (2.*X2-eta) + 5./14. * X1 * (33.*X1 - 7.) * u1
    dz_I_du_2 = - (2.*X1-eta) + 5./14. * X2 * (33.*X2 - 7.) * u1

    z_E_1 = 1. + 1.5 * X1 * u1 + 27./8. * X1 * u2
    z_E_2 = 1. + 1.5 * X2 * u1 + 27./8. * X2 * u2

    udz_C_lm = np.array([-eta, (2.-eta), -eta]) * (u1 + 3.*u2)

    C_int_22_1, C_int_20_1, C_int_2n2_1 = 0., 0., 0.
    for i in range(len(ll_1)):
        if ll_1[i] == 2:
            if mm_1[i] == 2:
                C_int_22_1 = udz_C_lm[0] * H_int_lm_mu_1[i]
            elif mm_1[i] == 0:
                C_int_20_1 = udz_C_lm[1] * H_int_lm_mu_1[i]
            elif mm_1[i] == -2:
                C_int_2n2_1 = udz_C_lm[2] * H_int_lm_mu_1[i]

    C_int_22_2, C_int_20_2, C_int_2n2_2 = 0., 0., 0.
    if n_mode_2>0:
        for i in range(len(ll_2)):
            if ll_2[i] == 2:
                if mm_2[i] == 2:
                    C_int_22_2 = udz_C_lm[0] * H_int_lm_mu_2[i]
                elif mm_2[i] == 0:
                    C_int_20_2 = udz_C_lm[1] * H_int_lm_mu_2[i]
                elif mm_2[i] == -2:
                    C_int_2n2_2 = udz_C_lm[2] * H_int_lm_mu_2[i] 

    H_tSO_eff_mu_1 = ((2 + 1.5*X1/X2) \
                     - (0.625 * eta + (1.125 + 0.75 * eta) * X1/X2) * u1) \
                    * eta * p_phi_Mmu * S_ns_Mmu_1 * u3
    H_tSO_eff_mu_2 = ((2 + 1.5*X2/X1) \
                     - (0.625 * eta + (1.125 + 0.75 * eta) * X2/X1) * u1) \
                    * eta * p_phi_Mmu * S_ns_Mmu_2 * u3

    H_tSS_mu_1 = - u3 * (eta * chi1z + X2 * X2 * chi2z) * S_ns_Mmu_1
    H_tSS_mu_2 = - u3 * (eta * chi2z + X1 * X1 * chi1z) * S_ns_Mmu_2

    #################
    # pp part    
        
    # only consider aligned spin
    # e3_x, e3_y, e3_z = 0., 0., 1.
    # n_v = r_v / r = (1, 0, 0)
    # cth = e3_x * nx + e3_y * ny + e3_z * nz = 0
    cth = 0.

    # in lalsimulation/lib/LALSimIMRSpinEOBHamiltonian.c 
    # also tracked are xi_v = e3_v x n_v = (0, 1, 0); xi2 = xi_v \cdot xi_v = 1.
    # and v_v = n_v x xi_v = (0, 0, 1)
    xi2 = 1. - cth * cth    

    # eq. 4.7 of BB1; made dimless
    w_M2 = r_M2 + a_M2

    # eq. 4.5 of BB1 or Sigma; made dimless
    rho_M2 = r_M2 + a_M2 * cth * cth

    
    # eq. 5.75 of BB1 or \bar{\Delta}_u; dimless
    m1PlusetaKK = -1. + eta * KK
    bulk = 1. / (m1PlusetaKK * m1PlusetaKK) + (2. * u1) / m1PlusetaKK + a_M2 * u2

    # eq. 5.73 of BB1
    logTerms = 1. + eta * k0 + eta * np.log(1. + k1 * u1 + k2 * u2 \
				       + k3 * u3 + k4 * u4 + k5 * u5 + k5l * u5 * np.log(u1))

    # note deltaU is basically A + higher PN
    deltaU_pp = bulk * logTerms

    # tidal contribution
    deltaU = deltaU_pp + 2. * (z_I_1 * H_int_mu_1 + z_I_2 * H_int_mu_2)

    # eq. 5.71 of BB1; made dimless
    deltaT = r_M2 * deltaU

    # ddeltaU/du 
    deltaU_u = 2. * (1. / m1PlusetaKK + a_M2 * u1) * logTerms \
        + bulk * (eta *(k1 + u1 * (2. * k2 + u1 * (3. * k3 + u1 * (4. * k4 \
                                                + 5. * (k5 + k5l * np.log(u1)) * u1))))) \
        / (1. + k1 * u1 + k2 * u2 + k3 * u3 + k4 * u4 + (k5 + k5l *np.log(u1))* u5)


    # tidal contribution
    deltaU_u += 2. * (z_I_1 * dH_int_mu_du_1 + dz_I_du_1 * H_int_mu_1 \
                    + z_I_2 * dH_int_mu_du_2 + dz_I_du_2 * H_int_mu_2 )


    # ddeltaT/dr; made dimless
    deltaT_r = 2. * r_M * deltaU - deltaU_u

    # eq. 5.39 of BB1; made dimless
    Lambda = w_M2 * w_M2 - a_M2 * deltaT * xi2

    # eq. 5.83 of BB1
    invD = 1. + np.log(1. + 6. * eta * u2 + 2. * (26. - 3. * eta) * eta * u3)

    # eq. 5.38 of BB1
    deltaR = deltaT * invD
    sqrtDr = np.sqrt(deltaR)

    # See H_nons below, Eq. 4.34 of Damour et al. PRD 62, 084011 (2000)
    qq = 2. * eta * (4. - 3. * eta)
        
    # eq. 5.40 of BB1; made dimless
    wfd = 2. * a_M * r_M

    # because we already input polar variables
    # no need to further go to cartesian and then change them back to polar as in 
    # lines 1062 - 1085 of LALSimIMRSpinEOBHamiltonian.c
    # we just set ptheta = 0 and restrict ourselves to the x-y plane

    # csi = p_r_tort / p_r = dr / dr^\ast
    # note in LAL, p->data is tortoised while tempP is NOT
    # we use the PP part because it is only a function of r; 
    # the tidal contribution to deltaU ~ A involves other canonical variables than r
    csi = deltaU_pp * np.sqrt(invD) * r_M2 / w_M2
    p_r_mu = p_r_mu_tort / csi

    # sqrt(A) = sqrt[1/(-g^tt)]; 
    # eq. 5.36a of BB1; note the tidal contribution has been added through deltaU and then to deltaT
    sqrt_pot = np.sqrt((rho_M2 * deltaT)/Lambda)

    # now compute  the sqrt(1 + p_phi^2/r^2/mu^2 + ...) term    
    # gam^ff ~ (M/r)^2
    gam_ff = rho_M2 / (Lambda * xi2) 
    # gam^rr ~ 1
    gam_rr = deltaR/rho_M2
    gam_pr2= gam_rr * p_r_mu * p_r_mu

    """
    This part simply uses p_phi = (PI_phi - S_ns) in E_mu2_LO, which is exact from canonical transformation
    use this form as we do not separate linear in p_phi piece in Q in H_LS and H_SS
    """
    E_mu2_LO = 1. + gam_ff * p_phi_Mmu * p_phi_Mmu + gam_pr2

    # add tidal contributions
    E_mu2_LO += 2. * (\
        z_E_1 * H_mode_mu_1 + z_E_2 * H_mode_mu_2 \
              + C_int_22_1 + C_int_20_1 + C_int_2n2_1 + C_int_22_2 + C_int_20_2 + C_int_2n2_2
    )

    # add the pr^4 contribution
    E_mu2 = E_mu2_LO\
            + p_r_mu_tort**4. * qq * u2 
    
    sqrt_E = np.sqrt(E_mu2)    
    
    H_nons_mu = sqrt_pot * sqrt_E + wfd / Lambda * p_phi_Mmu

    # eqs. 5.30-5.33 of BB1; mp for metric potentials
    # note B_mp and mu_mp are actually the scaled version in 5.48-5.51
    B_mp = np.sqrt(deltaT)
    w_mp = wfd / Lambda
    nu_mp = 0.5 * np.log(deltaT * rho_M2 / Lambda)
    mu_mp = 0.5 * np.log(rho_M2)

    # dLambda/dr; dimless
    Lambda_r = 4. * r_M * w_M2 - a_M2 * deltaT_r * xi2

    # dwfd/dr; dimless
    wfd_r = 2. * a_M 

    # eqs. 5.47a-5.47d of BB1; BR is actually the scaled version in 5.49
    BR = (-2. * deltaT + np.sqrt(deltaR) * deltaT_r) / (2. * np.sqrt(deltaR * deltaT))
    wr = (-Lambda_r * wfd + Lambda * wfd_r) / (Lambda * Lambda)
    nur = r_M / rho_M2 \
        + (w_M2 * (-4. * r_M * deltaT + w_M2 * deltaT_r)) / (2. * deltaT * Lambda)
    mur = r_M / rho_M2 - 1. / np.sqrt(deltaR)

    # eqs. Eqs. 5.47f - 5.47h of BB1; all zero because cth = 0
    wcos = 0.
    nucos = 0.
    mucos = 0.

    # eq. 5.52 of BB1
    # note Q is just E_mu2_LO
    sqrtQ = np.sqrt(E_mu2_LO)
    Qm1 = E_mu2_LO - 1.

    # 2.5 PN
    # Eq. 5.68 of BB1; dimless by M^2
    deltaSStar_z = (eta/12.) * ( u1 * (14. * sStar_MM_z - 8. * sKerr_MM_z) \
                               + Qm1 * (3. * sKerr_MM_z + 4. * sStar_MM_z)\
                               - gam_pr2 * (36. * sKerr_MM_z + 30. * sStar_MM_z) )
    
    
    # drop the 3.5 PN terms as we only keep the tidal spin to the lowest order
    s_MM_z = sStar_MM_z + deltaSStar_z


    # the spin Hamiltonian Hs should check eqs. 6.2-6.4 of BB1
    # if the linear momenta are norm'ed by \mu, then S^\ast should be norm'ed by M^2
    # and the \mu/M factor in g_SS^eff should be dropped to give H_s/mu. 

    # Eq. 6.3 of BB1
    H_SO_mu = np.exp(2. * nu_mp - mu_mp) / ( B_mp * B_mp * (1. + sqrtQ) * sqrtQ)\
        * ( ( np.exp(mu_mp+nu_mp) - B_mp ) * (1. + sqrtQ) \
           + sqrtDr * ( - sqrtQ * ( BR - 2. * B_mp * nur ) \
                       + (B_mp * nur - BR) )
          ) \
        * p_phi_Mmu * s_MM_z

    # Eq. 6.4 of BB1 less \mu/M
    g_SS_eff = w_mp + 0.5 * B_mp * np.exp( - mu_mp - nu_mp ) * wr * sqrtDr \
        + ( p_phi_Mmu * p_phi_Mmu \
           - np.exp(-2.*(mu_mp + nu_mp)) * B_mp * B_mp * deltaR * p_r_mu * p_r_mu) \
        * np.exp(nu_mp - mu_mp) * wr * sqrtDr / (2. * B_mp * (1. + sqrtQ) * sqrtQ )
    # also adding the last term from eq. 6.1
    H_SS_mu = g_SS_eff * s_MM_z - 0.5 * u3 * s_MM_z * s_MM_z

    # spin-induced NS quad
    H_SS_mu += 0.5 * u3 / eta \
        * ((Ces1 - 1.) * X2/X1 * ( - s1sq) \
         + (Ces2 - 1.) * X1/X2 * ( - s2sq))

    # put things together and add the leading order tidal spin-orbit/spin couplings
    H_eff_mu = H_nons_mu + H_SO_mu + H_SS_mu\
        + H_tSO_eff_mu_1 + H_tSO_eff_mu_2 + H_tSS_mu_1 + H_tSS_mu_2

    # total eob Hamiltonian
    # note we normalize it by mu, whereas in LAL it is in M
    # cf. line 1324 of LALSimIMRSpinEOBHamiltonian.c
    H_eob_mu = 1./eta * np.sqrt( 1. + 2. * eta * (H_eff_mu - 1.) )
    
    return H_eob_mu

@njit(cache=True)
def get_H_eob_mu_full_return(
                 eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2):
    """
    Exact copy of get_H_eob_mu except for this one returns more:
    H_eff_nts_mu,
    deltaU, deltaU_pp, deltaR, csi, 
    z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm,
    dH_tS_dS_1, dH_tS_dS_2
    """

    # unpack var

    r_M, p_r_mu_tort, phi, PI_phi_Mmu = eob_var[:4]
    H_mode_mu_1, S_ns_Mmu_1 = eob_var[4], eob_var[5]
    H_int_lm_mu_1 = eob_var[6:6+n_mode_1]
    ll_1 = ll[:n_mode_1]
    mm_1 = mm[:n_mode_1]
    ll_2 = ll[n_mode_1:]
    mm_2 = mm[n_mode_1:]

    if n_mode_2>0:
        H_mode_mu_2, S_ns_Mmu_2 = eob_var[6+n_mode_1], eob_var[7+n_mode_1]
        H_int_lm_mu_2 = eob_var[8+n_mode_1:]
        
    else:
        H_mode_mu_2, S_ns_Mmu_2, H_int_lm_mu_2 = 0., 0., np.zeros(n_mode_2)

    # note we evolve in the frame co-rotating with the orbit
    # here the canonical momentum conjugate to \phi is the total AM of the system (L_pp + tidal spin)
    # get L_pp by removing the tidal spin
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    # unpack coeffs

    # global par
    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # EOB coeffs
    a_M, a_M2, sKerr_MM_z, sStar_MM_z, s1sq, s2sq, \
    KK, \
    k0, k1, k2, k3, k4, k5, k5l \
        = eob_coeff[:14]

    # powers of r_M
    r_M2 = r_M *r_M
    
    u1 = 1./r_M
    u2 = u1 * u1
    u3 = u2 * u1
    u4 = u3 * u1
    u5 = u4 * u1

    #################
    # tidal part
    H_int_mu_1 = np.sum(H_int_lm_mu_1) 
    H_int_mu_2 = np.sum(H_int_lm_mu_2)

    dH_int_mu_du_1 = np.sum((ll_1 + 1.) * H_int_lm_mu_1 / u1)
    dH_int_mu_du_2 = np.sum((ll_2 + 1.) * H_int_lm_mu_2 / u1)

    z_I_1 = 1. - (2.*X2-eta) * u1 + 5./28. * X1 * (33.*X1 - 7.) * u2
    z_I_2 = 1. - (2.*X1-eta) * u1 + 5./28. * X2 * (33.*X2 - 7.) * u2

    dz_I_du_1 = - (2.*X2-eta) + 5./14. * X1 * (33.*X1 - 7.) * u1
    dz_I_du_2 = - (2.*X1-eta) + 5./14. * X2 * (33.*X2 - 7.) * u1

    z_E_1 = 1. + 1.5 * X1 * u1 + 27./8. * X1 * u2
    z_E_2 = 1. + 1.5 * X2 * u1 + 27./8. * X2 * u2

    udz_C_lm = np.array([-eta, (2.-eta), -eta]) * (u1 + 3.*u2)

    C_int_22_1, C_int_20_1, C_int_2n2_1 = 0., 0., 0.
    for i in range(len(ll_1)):
        if ll_1[i] == 2:
            if mm_1[i] == 2:
                C_int_22_1 = udz_C_lm[0] * H_int_lm_mu_1[i]
            elif mm_1[i] == 0:
                C_int_20_1 = udz_C_lm[1] * H_int_lm_mu_1[i]
            elif mm_1[i] == -2:
                C_int_2n2_1 = udz_C_lm[2] * H_int_lm_mu_1[i]

    C_int_22_2, C_int_20_2, C_int_2n2_2 = 0., 0., 0.
    if n_mode_2>0:
        for i in range(len(ll_2)):
            if ll_2[i] == 2:
                if mm_2[i] == 2:
                    C_int_22_2 = udz_C_lm[0] * H_int_lm_mu_2[i]
                elif mm_2[i] == 0:
                    C_int_20_2 = udz_C_lm[1] * H_int_lm_mu_2[i]
                elif mm_2[i] == -2:
                    C_int_2n2_2 = udz_C_lm[2] * H_int_lm_mu_2[i] 

    H_tSO_eff_mu_1 = ((2 + 1.5*X1/X2) \
                     - (0.625 * eta + (1.125 + 0.75 * eta) * X1/X2) * u1) \
                    * eta * p_phi_Mmu * S_ns_Mmu_1 * u3
    H_tSO_eff_mu_2 = ((2 + 1.5*X2/X1) \
                     - (0.625 * eta + (1.125 + 0.75 * eta) * X2/X1) * u1) \
                    * eta * p_phi_Mmu * S_ns_Mmu_2 * u3

    H_tSS_mu_1 = - u3 * (eta * chi1z + X2 * X2 * chi2z) * S_ns_Mmu_1
    H_tSS_mu_2 = - u3 * (eta * chi2z + X1 * X1 * chi1z) * S_ns_Mmu_2    

    #################
    # pp part    
        
    # only consider aligned spin
    # e3_x, e3_y, e3_z = 0., 0., 1.
    # n_v = r_v / r = (1, 0, 0)
    # cth = e3_x * nx + e3_y * ny + e3_z * nz = 0
    cth = 0.

    # in lalsimulation/lib/LALSimIMRSpinEOBHamiltonian.c 
    # also tracked are xi_v = e3_v x n_v = (0, 1, 0); xi2 = xi_v \cdot xi_v = 1.
    # and v_v = n_v x xi_v = (0, 0, 1)
    xi2 = 1. - cth * cth    

    # eq. 4.7 of BB1; made dimless
    w_M2 = r_M2 + a_M2

    # eq. 4.5 of BB1 or Sigma; made dimless
    rho_M2 = r_M2 + a_M2 * cth * cth

    
    # eq. 5.75 of BB1 or \bar{\Delta}_u; dimless
    m1PlusetaKK = -1. + eta * KK
    bulk = 1. / (m1PlusetaKK * m1PlusetaKK) + (2. * u1) / m1PlusetaKK + a_M2 * u2

    # eq. 5.73 of BB1
    logTerms = 1. + eta * k0 + eta * np.log(1. + k1 * u1 + k2 * u2 \
				       + k3 * u3 + k4 * u4 + k5 * u5 + k5l * u5 * np.log(u1))

    # note deltaU is basically A + higher PN
    deltaU_pp = bulk * logTerms

    # tidal contribution
    deltaU = deltaU_pp + 2. * (z_I_1 * H_int_mu_1 + z_I_2 * H_int_mu_2)

    # eq. 5.71 of BB1; made dimless
    deltaT = r_M2 * deltaU

    # ddeltaU/du 
    deltaU_u = 2. * (1. / m1PlusetaKK + a_M2 * u1) * logTerms \
        + bulk * (eta *(k1 + u1 * (2. * k2 + u1 * (3. * k3 + u1 * (4. * k4 \
                                                + 5. * (k5 + k5l * np.log(u1)) * u1))))) \
        / (1. + k1 * u1 + k2 * u2 + k3 * u3 + k4 * u4 + (k5 + k5l *np.log(u1))* u5)


    # tidal contribution
    deltaU_u += 2. * (z_I_1 * dH_int_mu_du_1 + dz_I_du_1 * H_int_mu_1 \
                    + z_I_2 * dH_int_mu_du_2 + dz_I_du_2 * H_int_mu_2 )


    # ddeltaT/dr; made dimless
    deltaT_r = 2. * r_M * deltaU - deltaU_u

    # eq. 5.39 of BB1; made dimless
    Lambda = w_M2 * w_M2 - a_M2 * deltaT * xi2

    # eq. 5.83 of BB1
    invD = 1. + np.log(1. + 6. * eta * u2 + 2. * (26. - 3. * eta) * eta * u3)

    # eq. 5.38 of BB1
    deltaR = deltaT * invD
    sqrtDr = np.sqrt(deltaR)

    # See H_nons below, Eq. 4.34 of Damour et al. PRD 62, 084011 (2000)
    qq = 2. * eta * (4. - 3. * eta)
        
    # eq. 5.40 of BB1; made dimless
    wfd = 2. * a_M * r_M

    # because we already input polar variables
    # no need to further go to cartesian and then change them back to polar as in 
    # lines 1062 - 1085 of LALSimIMRSpinEOBHamiltonian.c
    # we just set ptheta = 0 and restrict ourselves to the x-y plane

    # csi = p_r_tort / p_r = dr / dr^\ast
    # note in LAL, p->data is tortoised while tempP is NOT
    # we use the PP part because it is only a function of r; 
    # the tidal contribution to deltaU ~ A involves other canonical variables than r
    csi = deltaU_pp * np.sqrt(invD) * r_M2 / w_M2
    p_r_mu = p_r_mu_tort / csi

    # sqrt(A) = sqrt[1/(-g^tt)]; 
    # eq. 5.36a of BB1; note the tidal contribution has been added through deltaU and then to deltaT
    sqrt_pot = np.sqrt((rho_M2 * deltaT)/Lambda)

    # now compute  the sqrt(1 + p_phi^2/r^2/mu^2 + ...) term    
    # gam^ff ~ (M/r)^2
    gam_ff = rho_M2 / (Lambda * xi2) 
    # gam^rr ~ 1
    gam_rr = deltaR/rho_M2
    gam_pr2= gam_rr * p_r_mu * p_r_mu

    """
    This part simply uses p_phi = (PI_phi - S_ns) in E_mu2_LO, which is exact from canonical transformation
    use this form as we do not separate linear in p_phi piece in Q in H_LS and H_SS
    """
    E_mu2_LO = 1. + gam_ff * p_phi_Mmu * p_phi_Mmu + gam_pr2

    # add tidal contributions
    E_mu2_LO += 2. * (\
        z_E_1 * H_mode_mu_1 + z_E_2 * H_mode_mu_2 \
              + C_int_22_1 + C_int_20_1 + C_int_2n2_1 + C_int_22_2 + C_int_20_2 + C_int_2n2_2
    )

    # add the pr^4 contribution
    E_mu2 = E_mu2_LO\
            + p_r_mu_tort**4. * qq * u2    
    
    sqrt_E = np.sqrt(E_mu2)    
    
    H_nons_mu = sqrt_pot * sqrt_E + wfd / Lambda * p_phi_Mmu

    # eqs. 5.30-5.33 of BB1; mp for metric potentials
    # note B_mp and mu_mp are actually the scaled version in 5.48-5.51
    B_mp = np.sqrt(deltaT)
    w_mp = wfd / Lambda
    nu_mp = 0.5 * np.log(deltaT * rho_M2 / Lambda)
    mu_mp = 0.5 * np.log(rho_M2)

    # dLambda/dr; dimless
    Lambda_r = 4. * r_M * w_M2 - a_M2 * deltaT_r * xi2

    # dwfd/dr; dimless
    wfd_r = 2. * a_M 

    # eqs. 5.47a-5.47d of BB1; BR is actually the scaled version in 5.49
    BR = (-2. * deltaT + np.sqrt(deltaR) * deltaT_r) / (2. * np.sqrt(deltaR * deltaT))
    wr = (-Lambda_r * wfd + Lambda * wfd_r) / (Lambda * Lambda)
    nur = r_M / rho_M2 \
        + (w_M2 * (-4. * r_M * deltaT + w_M2 * deltaT_r)) / (2. * deltaT * Lambda)
    mur = r_M / rho_M2 - 1. / np.sqrt(deltaR)

    # eqs. Eqs. 5.47f - 5.47h of BB1; all zero because cth = 0
    wcos = 0.
    nucos = 0.
    mucos = 0.

    # eq. 5.52 of BB1
    # note Q is just E_mu2_LO
    sqrtQ = np.sqrt(E_mu2_LO)
    Qm1 = E_mu2_LO - 1.

    # 2.5 PN
    # Eq. 5.68 of BB1; dimless by M^2
    deltaSStar_z = (eta/12.) * ( u1 * (14. * sStar_MM_z - 8. * sKerr_MM_z) \
                               + Qm1 * (3. * sKerr_MM_z + 4. * sStar_MM_z)\
                               - gam_pr2 * (36. * sKerr_MM_z + 30. * sStar_MM_z) )
    
    
    # drop the 3.5 PN terms as we only keep the tidal spin to the lowest order
    s_MM_z = sStar_MM_z + deltaSStar_z


    # the spin Hamiltonian Hs should check eqs. 6.2-6.4 of BB1
    # if the linear momenta are norm'ed by \mu, then S^\ast should be norm'ed by M^2
    # and the \mu/M factor in g_SS^eff should be dropped to give H_s/mu. 

    # Eq. 6.3 of BB1
    H_SO_mu = np.exp(2. * nu_mp - mu_mp) / ( B_mp * B_mp * (1. + sqrtQ) * sqrtQ)\
        * ( ( np.exp(mu_mp+nu_mp) - B_mp ) * (1. + sqrtQ) \
           + sqrtDr * ( - sqrtQ * ( BR - 2. * B_mp * nur ) \
                       + (B_mp * nur - BR) )
          ) \
        * p_phi_Mmu * s_MM_z

    # Eq. 6.4 of BB1 less \mu/M
    g_SS_eff = w_mp + 0.5 * B_mp * np.exp( - mu_mp - nu_mp ) * wr * sqrtDr \
        + ( p_phi_Mmu * p_phi_Mmu \
           - np.exp(-2.*(mu_mp + nu_mp)) * B_mp * B_mp * deltaR * p_r_mu * p_r_mu) \
        * np.exp(nu_mp - mu_mp) * wr * sqrtDr / (2. * B_mp * (1. + sqrtQ) * sqrtQ )
    # also adding the last term from eq. 6.1
    H_SS_mu = g_SS_eff * s_MM_z - 0.5 * u3 * s_MM_z * s_MM_z

    # spin-induced NS quad
    H_SS_mu += 0.5 * u3 / eta \
        * ((Ces1 - 1.) * X2/X1 * ( - s1sq) \
         + (Ces2 - 1.) * X1/X2 * ( - s2sq))

    # put things together and add the leading order tidal spin-orbit/spin couplings
    H_eff_mu = H_nons_mu + H_SO_mu + H_SS_mu\
        + H_tSO_eff_mu_1 + H_tSO_eff_mu_2 + H_tSS_mu_1 + H_tSS_mu_2

    # total eob Hamiltonian
    # note we normalize it by mu, whereas in LAL it is in M
    # cf. line 1324 of LALSimIMRSpinEOBHamiltonian.c
    H_eob_mu = 1./eta * np.sqrt( 1. + 2. * eta * (H_eff_mu - 1.) )

    """
    This part is computed here but not in get_H_eob_mu
    """
    H_eff_nts_mu = sqrt_pot * sqrt_E

    # note d[p(S)*S] /dS = p + dp/dS * S = p - S
    dH_tSO_dS_1 = ((2 + 1.5*X1/X2) \
                     - (0.625 * eta + (1.125 + 0.75 * eta) * X1/X2) * u1) \
                    * eta * (p_phi_Mmu - S_ns_Mmu_1) * u3
    dH_tSO_dS_2 = ((2 + 1.5*X2/X1) \
                     - (0.625 * eta + (1.125 + 0.75 * eta) * X2/X1) * u1) \
                    * eta * (p_phi_Mmu - S_ns_Mmu_2) * u3
    dH_tSS_dS_1 = - u3 * (eta * chi1z + X2 * X2 * chi2z)
    dH_tSS_dS_2 = - u3 * (eta * chi2z + X1 * X1 * chi1z) 

    dH_tS_dS_1 = dH_tSO_dS_1 + dH_tSS_dS_1
    dH_tS_dS_2 = dH_tSO_dS_2 + dH_tSS_dS_2

    # places where p_phi_Mmu (= PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2) appears outside H_eff_nts_mu
    # also need to remove dp_phi_dS_ns = -1
    dH_SS_dp_phi = 2. * p_phi_Mmu * np.exp(nu_mp - mu_mp) * wr * sqrtDr / (2. * B_mp * (1. + sqrtQ) * sqrtQ ) * s_MM_z
    
    # the wfd / Lambda terms comes from H_nons_mu
    dH_tS_dS_1 -= (wfd/Lambda + H_SO_mu/p_phi_Mmu + dH_SS_dp_phi)
    dH_tS_dS_2 -= (wfd/Lambda + H_SO_mu/p_phi_Mmu + dH_SS_dp_phi)

    return H_eob_mu, \
           H_eff_nts_mu, \
           deltaU, deltaU_pp, deltaR, csi, \
           z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
           dH_tS_dS_1, dH_tS_dS_2
           
@njit(cache=True)
def get_H_int_lm_mu(var, par, 
                 ll, mm,
                 WI_a_scaled, 
                 n_mode_1, n_mode_2):
    """
    Newtonian interaction energy
    """
    r_M = var[0]    
    b_a_r_1 = var[4:4+n_mode_1]
    b_a_r_2 = var[4+2*n_mode_1:4+2*n_mode_1+n_mode_2]

    # global par
    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    ll_1, mm_1 = ll[:n_mode_1], mm[:n_mode_1]
    ll_2, mm_2 = ll[n_mode_1:], mm[n_mode_1:]

    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll_1 + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll_2 + 1. )
    
    # v_a_2 should have a sign (-1)^l
    # the sign is inherited by b_a_2
    # so the interaction is not affected
    v_a_2 *= (-1.)**ll_2

    # H_int_lm and H_int_l,-m will always appear in pairs, 
    # so only need to keep the real part
    H_int_lm_mu_1 = np.zeros(n_mode_1)
    for i in range(n_mode_1):
        for j in range(n_mode_1):
            if ( ll_1[j]==ll_1[i] ) and ( np.abs(mm_1[j]) == np.abs (mm_1[i]) ):
                if mm_1[i]==0:
                    H_int_lm_mu_1[i] += 2. * b_a_r_1[j] * v_a_1[i]
                else:
                    H_int_lm_mu_1[i] += b_a_r_1[j] * v_a_1[i]
    H_int_lm_mu_1 *= - E1_mu

    H_int_lm_mu_2 = np.zeros(n_mode_2)
    for i in range(n_mode_2):
        for j in range(n_mode_2):
            if ( ll_2[j]==ll_2[i] ) and ( np.abs(mm_2[j]) == np.abs(mm_2[i]) ):
                if mm_2[i]==0:
                    H_int_lm_mu_2[i] += 2. * b_a_r_2[j] * v_a_2[i]
                else:
                    H_int_lm_mu_2[i] += b_a_r_2[j] * v_a_2[i]
    H_int_lm_mu_2 *= - E2_mu

    return H_int_lm_mu_1, H_int_lm_mu_2     


##############################    


@njit(cache=True)
def get_kappa_for_h(Momega, Mdrdt_r, 
                    vp_pows,
                    X2, MOmegaS1,
                    ll, mm, Mwa, Mwa0):
    """
    To be consistent with PHYSICAL REVIEW D 85, 123007 (2012)
    we use vp instead of vw here for the PN corrections
    """

    vp, vp2 = vp_pows[:2]

    z_pn = 1. - X2 * (2. + X2)/2. * vp2
    frac_w_FD = X2 * (4. - X2)/2. * vp2

    frac_va_22 = - (3. + X2 * (3. + X2)) * vp2 / 2.
    frac_va_20 = - (-1. + X2 * (3. + X2)) * vp2 / 2.

    kap_22, kap_20, kap_2n2 = 0j, 0j, 0j
    kap_33, kap_31, kap_3n1, kap_3n3 = 0j, 0j, 0j, 0j

    MDel_a = z_pn * (Mwa + mm * MOmegaS1) - mm * Momega * (1. - frac_w_FD)

    # actually what is computed is 1/2 * ba / va
    b_m_p = 0.5 * MDel_a * Mwa0 / (MDel_a**2. + 1j * (1.5 * mm * Momega  + (ll+1) * MDel_a ) * Mdrdt_r)
    b_nm_n = np.conjugate(b_m_p)
    
    for i in range(len(ll)):
        if ll[i]==2:
            if mm[i]==2:
                kap_22 +=  b_m_p[i] * (1. + frac_va_22)
                kap_2n2 += b_nm_n[i] * (1. + frac_va_22)

            elif mm[i]==-2:
                kap_2n2 += b_m_p[i] * (1. + frac_va_22)
                kap_22 += b_nm_n[i] * (1. + frac_va_22)

            elif mm[i]==0:
                kap_20 = (b_m_p[i] + b_nm_n[i]) * (1. + frac_va_20)

        elif ll[i]==3:
            if mm[i] == 3:
                kap_33 +=  b_m_p[i]
                kap_3n3 += b_nm_n[i]
                
            elif mm[i] == 1:
                kap_31 += b_m_p[i]
                kap_3n1 += b_nm_n[i]

            elif mm[i] == -3:
                kap_3n3 += b_m_p[i]
                kap_33 += b_nm_n[i]

            elif mm[i] == -1:
                kap_3n1 += b_m_p[i]
                kap_31 += b_nm_n[i]

    return kap_22, kap_20, kap_2n2, \
           kap_33, kap_31, kap_3n1, kap_3n3

@njit(cache=True)
def get_h_tide_mult(Momega, Mdrdt_r, 
                    vp_pows, H_mode_mu_1, S_ns_Mmu_1,
                    X2, MOmegaS1,
                    ll, mm, Mwa, Mwa0, 
                    h_tide_mult_coeff,
                    h_sign=+1):

    vp, vp2, vp3, vp4, vp5 = vp_pows[:5]
    vp10 = vp5 * vp5
    
    kap_22, kap_20, kap_2n2, \
    kap_33, kap_31, kap_3n1, kap_3n3 \
        = get_kappa_for_h(Momega, Mdrdt_r, 
                    vp_pows,
                    X2, MOmegaS1,
                    ll, mm, Mwa, Mwa0)

    kap_vect = np.array([kap_22, kap_20, kap_2n2, kap_33, kap_31, kap_3n1, kap_3n3])
    # kap_3m starts from vp14 
    kap_vect[3:7] *= vp4

    # inertial frame mode energy
    H_ns_iner_mu_1 = H_mode_mu_1

    # note kap_3m has been scaled by vp4 above
    h_tide_mult_22 = vp10 * np.sum(h_tide_mult_coeff[:, 0, 0] * kap_vect \
                             + vp2 * h_tide_mult_coeff[:, 0, 1] * kap_vect)\
                    + X2 * (2 + X2)/3. * H_ns_iner_mu_1\
                    - 2./3. * X2 * (2. - X2) * vp3 * S_ns_Mmu_1

    h_tide_mult_21 = vp10 * np.sum(h_tide_mult_coeff[:, 1, 0] * kap_vect)\
                    + 1.5 * X2 * vp * S_ns_Mmu_1

    h_tide_mult_33 = vp10 * np.sum(h_tide_mult_coeff[:, 2, 0] * kap_vect)

    # when M1 is deformed, sign = +1
    #      M2 is deformed, sign = -1
    return h_tide_mult_22, h_sign*h_tide_mult_21, h_sign*h_tide_mult_33
    
    

@njit(cache=True)
def get_h_pn_mult(H_eob_mu, p_phi_Mmu, vw_pows, 
                  par, 
                  rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff):
    """
    eq. 17 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.86.024011 
    less the Newtonian part but with c_{l+epsilon}
    In other words, we compute here
    c_{l+epsilon} * hat{S}_eff * Tlm * rholm^l * e^{i delta_lm}
    
    
    Note in LAL, H_real is H_eob_M
    """
    vw, vw2, vw3, vw4, vw5, vw6 = vw_pows[:6]

    eta, X1, X2, \
    chi1z, chi2z = par[:5]

    H_eob_M = eta * H_eob_mu

    # eq. A5
    S_even = (H_eob_M * H_eob_M - 1.) / (2. * eta) + 1.
    S_odd = vw * p_phi_Mmu

    # eq. A6
    kk = np.zeros(4)
    hhatkk = np.zeros(len(kk))
    exp_fact = np.zeros(len(kk), dtype=np.complex128)
    r02_M = 2.42612 # 2 r0/M = 4/sqrt(e)
    for i in range(4):
        mh = i + 1
        kk[i] = mh * vw3
        hhatkk[i] = H_eob_M * kk[i]
        exp_fact[i] = np.exp(np.pi * hhatkk[i] + 2j * hhatkk[i] * np.log(kk[i] * r02_M) )

    T22 = Gamma_lp1_m2js_Gamma_lp1_expanded(hhatkk[1], 2) * exp_fact[1]
    T21 = Gamma_lp1_m2js_Gamma_lp1_expanded(hhatkk[0], 2) * exp_fact[0]
    T33 = Gamma_lp1_m2js_Gamma_lp1_expanded(hhatkk[2], 3) * exp_fact[2]
    T32 = Gamma_lp1_m2js_Gamma_lp1_expanded(hhatkk[1], 3) * exp_fact[1]
    T44 = Gamma_lp1_m2js_Gamma_lp1_expanded(hhatkk[3], 4) * exp_fact[3]

    # 
    Hw = hhatkk[0]
    Hw2 = Hw * Hw
    Hw3 = Hw2 * Hw
    Hw_pows = np.array([Hw, Hw2, Hw3])

    logvw2 = np.log(vw2)
    
    # rholm
    n_mode, n_pn = rholm_coeff.shape
    rholm = np.zeros(n_mode)
    for i in range(n_mode):
        
        # for j in range(n_pn):
        #     rholm[i] = vw * (rholm_coeff[i, n_pn-1-j] + rholm[i])
        
        rholm[i] = np.sum(rholm_coeff[i, :] * vw_pows[:n_pn])
        
    rholm += 1.

    # rholm_log
    n_mode, n_pn = rholm_log_coeff.shape
    rholm_log = np.zeros(n_mode)
    for i in range(n_mode):
        rholm_log[i] = np.sum(rholm_log_coeff[i, :] * vw_pows[:n_pn]) * logvw2 * vw5
    rholm[:n_mode] += rholm_log

    # c_{l+epsilon} * flm^S
    # this form has no divergence
    n_mode, n_pn = cflmS_coeff.shape
    cflmS = np.zeros(n_mode)
    for i in range(n_mode):
        cflmS[i] = np.sum(cflmS_coeff[i, :] * vw_pows[:n_pn])

    # deltalm
    n_mode, n_pn, _ = deltalm_coeff.shape
    deltalm = np.zeros(n_mode)
    for i in range(n_mode):
        deltalm[i] = np.sum(deltalm_coeff[i, :, 0] * Hw_pows[:n_pn]) \
                   + np.sum(deltalm_coeff[i, :, 1] * vw_pows[:n_pn]) * vw4

    # our hN excludes the c_{l+ep} part
    # need to include it here
    c2 = 1.         # (l=2, ep=0, m=0)
    c3 = X2-X1      # (l=3, ep=0, m=1 & 3)
                    # or (l=2, ep=1, m=1)
    c4 = 1.-3.*eta  # (l=4, ep=0, m=4)
                    # (l=3, ep=1, m=2)

    cf22 = c2 * rholm[0] * rholm[0] 
    cf21 = c3 * rholm[1] * rholm[1] + cflmS[0]
    cf33 = c3 * rholm[2] * rholm[2] * rholm[2] + cflmS[1]
    cf32 = c4 * rholm[3] * rholm[3] * rholm[3] 
    cf44 = c4 * (rholm[4] * rholm[4])**2.

    delta22 = deltalm[0]
    delta21 = deltalm[1]
    delta33 = deltalm[2]
    
    h_pn_mult_22 = S_even * T22 * cf22 * np.exp(1j * delta22)
    h_pn_mult_21 = S_odd  * T21 * cf21 * np.exp(1j * delta21)
    h_pn_mult_33 = S_even * T33 * cf33 * np.exp(1j * delta33)
    h_pn_mult_32 = S_odd  * T32 * cf32 
    h_pn_mult_44 = S_even * T44 * cf44

    return h_pn_mult_22, h_pn_mult_21, h_pn_mult_33, h_pn_mult_32, h_pn_mult_44

@njit(cache=True)
def get_h_nqc_mult(orb_var, Momega, nqc_coeff):
    """
    From 1106.1021
    just for the 22 mode
    """
    r_M, p_r_mu_tort = orb_var[:2]
    uu = 1./r_M
    uu_1_2 = np.sqrt(uu)
    uu_3_2 = uu * uu_1_2

    pr2 = p_r_mu_tort * p_r_mu_tort

    romega = r_M * Momega
    romega2 = romega * romega

    pr_romega = p_r_mu_tort / romega

    a1, a2, a3, a4, a5, b1, b2, b3, b4 = nqc_coeff

    # only use a1-a3 & b1, b2 for now
    nqc_22_mag = 1. + pr2 / romega \
        * (a1 + a2 * uu + a3 * uu_3_2)
    nqc_22_ph = b1 * pr_romega + pr2 * pr_romega * b2
    
    nqc_22_mult = nqc_22_mag * np.exp(1j * nqc_22_ph)
    return nqc_22_mult
    


@njit(cache=True)
def get_hDL_M_mode(orb_var, vp, vw, Mdrdt_r,
                   H_eob_mu,
                   H_mode_mu_1, S_ns_Mmu_1, H_mode_mu_2, S_ns_Mmu_2,
                   par, 
                   ll, mm, Mwa, Mwa0, 
                   n_mode_1, n_mode_2, 
                   rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
                   nqc_coeff, 
                   h_tide_mult_coeff_1, h_tide_mult_coeff_2):
    """
    The difference between vp and vw is that the former is computed as
        vp^3 = (\partial H_eob / \partial p_\phi)|_pr=0,
    whereas 
        vw^3 = \partial H_eob / \partial p_\phi.
    I.e., the former assumes pr=0 while the latter doesn't. 
    """
    r_M, p_r_mu_tort, phi, p_phi_Mmu = orb_var
    
    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]   

    vp2 = vp * vp
    vp3 = vp2 * vp
    vp4 = vp3 * vp
    vp5 = vp4 * vp
    vp_pows = np.array([vp, vp2, vp3, vp4, vp5])
    
    vw2 = vw * vw
    vw3 = vw2 * vw
    vw4 = vw3 * vw
    vw5 = vw4 * vw
    vw6 = vw5 * vw
    vw_pows = np.array([vw, vw2, vw3, vw4, vw5, vw6])

    Momega = vw3

    exp_phi = np.exp(-1j * phi) 
    exp_2phi = exp_phi * exp_phi
    exp_3phi = exp_2phi * exp_phi
    exp_4phi = exp_2phi * exp_2phi

    # Newtonian modes less c_{l+ep}
    hp_DL_M_22_pp_N = - 8.*np.sqrt(np.pi/5.) * eta * vp2 * exp_2phi
    hp_DL_M_21_pp_N = (8.j/3.) * np.sqrt(np.pi/5.) * eta * vp3 * exp_phi
    hp_DL_M_33_pp_N = - 3j * np.sqrt(6.*np.pi/7.) * eta * vp3 * exp_3phi
    hp_DL_M_32_pp_N = - 8./3. * np.sqrt(np.pi/7.)* eta * vp4 * exp_2phi
    hp_DL_M_44_pp_N = 64./9 * np.sqrt(np.pi/7.)* eta * vp4 * exp_4phi


    ########
    # tide #
    ########    
    
    h_tide_mult_22_1, h_tide_mult_21_1, h_tide_mult_33_1 \
        = get_h_tide_mult(Momega, Mdrdt_r, 
                    vp_pows, H_mode_mu_1, S_ns_Mmu_1,
                    X2, MOmegaS1,
                    ll[:n_mode_1], mm[:n_mode_1], Mwa[:n_mode_1], Mwa0[:n_mode_1], 
                    h_tide_mult_coeff_1,
                    h_sign=+1)

    # note the companion's tidal contribution should have the 21 & 33 modes' signs flipped
    h_tide_mult_22_2, h_tide_mult_21_2, h_tide_mult_33_2 = 0., 0., 0.
    if n_mode_2>0:
        h_tide_mult_22_2, h_tide_mult_21_2, h_tide_mult_33_2 \
            = get_h_tide_mult(Momega, Mdrdt_r, 
                    vp_pows, H_mode_mu_2, S_ns_Mmu_2,
                    X1, MOmegaS2,
                    ll[n_mode_1:], mm[n_mode_1:], Mwa[n_mode_1:], Mwa0[n_mode_1:], 
                    h_tide_mult_coeff_2,
                    h_sign=-1)
    
    ######
    # PP #
    ######

    
    h_pn_mult_22, h_pn_mult_21, h_pn_mult_33, h_pn_mult_32, h_pn_mult_44\
        = get_h_pn_mult(H_eob_mu, p_phi_Mmu, vw_pows, 
                  par, 
                  rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff)

    # nqc to 22 mode only
    h_nqc_mult_22 = get_h_nqc_mult(orb_var, Momega, nqc_coeff)
    
    #######################
    # put things together #
    #######################

    hDL_M_22 = hp_DL_M_22_pp_N * (h_pn_mult_22 * h_nqc_mult_22 + h_tide_mult_22_1 + h_tide_mult_22_2) 
    hDL_M_21 = hp_DL_M_21_pp_N * (h_pn_mult_21 + h_tide_mult_21_1 + h_tide_mult_21_2) 
    hDL_M_33 = hp_DL_M_33_pp_N * (h_pn_mult_33 + h_tide_mult_33_1 + h_tide_mult_33_2) 
    hDL_M_32 = hp_DL_M_32_pp_N * (h_pn_mult_32 ) 
    hDL_M_44 = hp_DL_M_44_pp_N * (h_pn_mult_44 ) 

    hDL_M_modes = np.array([hDL_M_22, hDL_M_21, hDL_M_33, hDL_M_32, hDL_M_44])
    return hDL_M_modes


##############################


@njit(cache=True)
def pp_tide_var_2_eob_var(pp_tide_var, par, 
                          ll, mm,
                          WI_a_scaled, Mwa, Mwa0, 
                          n_mode_1, n_mode_2
                         ):
    r_M, p_r_mu_tort, phi, PI_phi_Mmu = pp_tide_var[:4]
    b_a_re_im_1 = pp_tide_var[4:4+int(n_mode_1*2)]
    b_a_1 = b_a_re_im_1[:n_mode_1] + 1j * b_a_re_im_1[n_mode_1:]

    b_a_re_im_2 = pp_tide_var[4+int(n_mode_1*2):]
    b_a_2 = b_a_re_im_2[:n_mode_2] + 1j * b_a_re_im_2[n_mode_2:]

    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # tide
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll[:n_mode_1] + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll[n_mode_1:] + 1. )
    v_a_2 *= (-1.)**(ll[n_mode_1:])
    
    # corot frame energy (each mode)
    H_ns_mu_1 = E1_mu * (Mwa/Mwa0)[:n_mode_1] * np.abs(b_a_1)**2.
    # canonical spin
    S_ns_Mmu_1 = H_ns_mu_1 * (mm/ Mwa)[:n_mode_1]
    
    H_ns_mu_1 = np.sum(H_ns_mu_1)
    S_ns_Mmu_1 = np.sum(S_ns_Mmu_1)

    # inertial frame energy
    H_mode_mu_1 = H_ns_mu_1 + MOmegaS1 * S_ns_Mmu_1

    # corot frame energy (each mode)
    H_ns_mu_2 = E2_mu * (Mwa/Mwa0)[n_mode_1:] * np.abs(b_a_2)**2.
    # canonical spin
    S_ns_Mmu_2 = H_ns_mu_2 * (mm/ Mwa)[n_mode_1:]
    
    H_ns_mu_2 = np.sum(H_ns_mu_2)
    S_ns_Mmu_2 = np.sum(S_ns_Mmu_2)

    # inertial frame energy
    H_mode_mu_2 = H_ns_mu_2 + MOmegaS2 * S_ns_Mmu_2

    # pp angular momentum
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    
    H_int_lm_mu_1, H_int_lm_mu_2 = get_H_int_lm_mu(pp_tide_var, par, 
                 ll, mm,
                 WI_a_scaled, 
                 n_mode_1, n_mode_2)

    # also need derivatives of the interaction energy
    dH_int_lm_mu_1 = -(ll+1)[:n_mode_1] * H_int_lm_mu_1/r_M
    dH_int_lm_mu_2 = -(ll+1)[n_mode_1:] * H_int_lm_mu_2/r_M

    if n_mode_2>0:
        eob_var = np.zeros(8 + n_mode_1 + n_mode_2)
        eob_var[6+n_mode_1] = H_mode_mu_2
        eob_var[7+n_mode_1] = S_ns_Mmu_2
        eob_var[8+n_mode_1:] = H_int_lm_mu_2
    else:
        eob_var = np.zeros(6 + n_mode_1)
    
    eob_var[:4] = pp_tide_var[:4]
    eob_var[4] = H_mode_mu_1
    eob_var[5] = S_ns_Mmu_1
    eob_var[6:6+n_mode_1] = H_int_lm_mu_1

    return eob_var


@njit(cache=True)
def get_H_eob_mu_from_pp_tide_var(pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff):
    eob_var = pp_tide_var_2_eob_var(pp_tide_var, par, 
                          ll, mm,
                          WI_a_scaled, Mwa, Mwa0, 
                          n_mode_1, n_mode_2
                         )
    H_eob_mu = get_H_eob_mu(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2)
    return H_eob_mu

##############################

@njit
def evol_sys(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2,
             order=2, dlogvar=1e-4):
    
    r_M, p_r_mu_tort, phi, PI_phi_Mmu = pp_tide_var[:4]
    b_a_re_im_1 = pp_tide_var[4:4+int(n_mode_1*2)]
    b_a_1 = b_a_re_im_1[:n_mode_1] + 1j * b_a_re_im_1[n_mode_1:]

    b_a_re_im_2 = pp_tide_var[4+int(n_mode_1*2):]
    b_a_2 = b_a_re_im_2[:n_mode_2] + 1j * b_a_re_im_2[n_mode_2:]

    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # tide
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll[:n_mode_1] + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll[n_mode_1:] + 1. )
    v_a_2 *= (-1.)**(ll[n_mode_1:])
    
    # corot frame energy (each mode)
    H_ns_mu_1 = E1_mu * (Mwa/Mwa0)[:n_mode_1] * np.abs(b_a_1)**2.
    # canonical spin
    S_ns_Mmu_1 = H_ns_mu_1 * (mm/ Mwa)[:n_mode_1]
    
    H_ns_mu_1 = np.sum(H_ns_mu_1)
    S_ns_Mmu_1 = np.sum(S_ns_Mmu_1)

    # inertial frame energy
    H_mode_mu_1 = H_ns_mu_1 + MOmegaS1 * S_ns_Mmu_1

    # corot frame energy (each mode)
    H_ns_mu_2 = E2_mu * (Mwa/Mwa0)[n_mode_1:] * np.abs(b_a_2)**2.
    # canonical spin
    S_ns_Mmu_2 = H_ns_mu_2 * (mm/ Mwa)[n_mode_1:]
    
    H_ns_mu_2 = np.sum(H_ns_mu_2)
    S_ns_Mmu_2 = np.sum(S_ns_Mmu_2)

    # inertial frame energy
    H_mode_mu_2 = H_ns_mu_2 + MOmegaS2 * S_ns_Mmu_2

    # pp angular momentum
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    
    H_int_lm_mu_1, H_int_lm_mu_2 = get_H_int_lm_mu(pp_tide_var, par, 
                 ll, mm,
                 WI_a_scaled, 
                 n_mode_1, n_mode_2)

    # also need derivatives of the interaction energy
    dH_int_lm_mu_1 = -(ll+1)[:n_mode_1] * H_int_lm_mu_1/r_M
    dH_int_lm_mu_2 = -(ll+1)[n_mode_1:] * H_int_lm_mu_2/r_M


    if n_mode_2>0:
        eob_var = np.zeros(8 + n_mode_1 + n_mode_2)
        eob_var[6+n_mode_1] = H_mode_mu_2
        eob_var[7+n_mode_1] = S_ns_Mmu_2
        eob_var[8+n_mode_1:] = H_int_lm_mu_2
    else:
        eob_var = np.zeros(6 + n_mode_1)
    
    eob_var[:4] = pp_tide_var[:4]
    eob_var[4] = H_mode_mu_1
    eob_var[5] = S_ns_Mmu_1
    eob_var[6:6+n_mode_1] = H_int_lm_mu_1

    # get csi to change pr_tort to pr
    # dr/dr^\ast = pr_ast/pr = csi
    # and H_eob as the source term for the flux
    # and all quantities needed for analytical mode evolution
    H_eob_mu, \
    H_eff_nts_mu, \
    deltaU, deltaU_pp, deltaR, csi, \
    z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
    dH_tS_dS_1, dH_tS_dS_2 \
        = get_H_eob_mu_full_return(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2)

    # some quantities needed for analytical differentiation
    A_H_eff_nts_mu = deltaU / H_eff_nts_mu

    dH_eob_dH_eff = 1./(eta * H_eob_mu)
    dH_eff_dH_mode_1 = z_E_1 * A_H_eff_nts_mu
    dH_eff_dH_mode_2 = z_E_2 * A_H_eff_nts_mu

    dH_eff_nts_dS = - p_phi_Mmu/r_M/r_M * A_H_eff_nts_mu

    dH_eff_dH_int_lm_1 = np.ones(n_mode_1) \
        * z_I_1 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_1):
        if ll_1[i]==2:
            if mm_1[i]==2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[0]
            elif mm_1[i]==0:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[1]
            elif mm_1[i]==-2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[2]


    dH_eff_dH_int_lm_2 = np.ones(n_mode_2) \
        * z_I_2 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_2):
        if ll_2[i]==2:
            if mm_2[i]==2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[0]
            elif mm_2[i]==0:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[1]
            elif mm_2[i]==-2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[2]

    # evolve modes
    db_a_1 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_1 * ( -1j * (Mwa + mm * MOmegaS1)[:n_mode_1] * b_a_1 )\
      + (dH_eff_nts_dS + dH_tS_dS_1) * ( -1j * mm[:n_mode_1] * b_a_1 )\
      + dH_eff_dH_int_lm_1 * (1j * Mwa0[:n_mode_1] * v_a_1)
                             )

    db_a_2 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_2 * ( -1j * (Mwa + mm * MOmegaS2)[n_mode_1:] * b_a_2 )\
      + (dH_eff_nts_dS + dH_tS_dS_2) * ( -1j * mm[n_mode_1:] * b_a_2 )\
      + dH_eff_dH_int_lm_2 * (1j * Mwa0[n_mode_1:] * v_a_2)
                             )


    # numerical differentiation for the orbit
    dvar = (np.abs(eob_var[:4]) + 1e-4) * dlogvar
    dH_dvar = np.zeros(4)
    dpp_var = np.zeros(4)

    for i in range(4):
        if i==2:
            # phi does not show up in H
            continue
        dH_dvar[i] = compute_dH_1(eob_var, par, i, 
                                  order=order,
                                  dvar = dvar[i], 
                                  H_func = get_H_eob_mu, 
                                  H_func_args=(eob_coeff, 
                                               ll, mm, n_mode_1, n_mode_2))

    # pp part
    # eq. 10 of PhysRevD.84.124052
    # dr_M / dt_M; note we use tortoised p_r_tort but regular r
    dpp_var[0] = csi * dH_dvar[1] 
    Mdrdt_r = dpp_var[0] / r_M

    # dp_r_mu_tort / dt_M
    # note we also need to include dH_eob_dH_int * dH_int_dr
    dpp_var[1] = - csi * (dH_dvar[0] + dH_eob_dH_eff \
                              * (  np.sum(dH_eff_dH_int_lm_1 * dH_int_lm_mu_1)\
                                 + np.sum(dH_eff_dH_int_lm_2 * dH_int_lm_mu_2)\
                                )    )

    # dphi / dt_M = Momega
    dpp_var[2] = dH_dvar[3] 
    Momega = dH_dvar[3]

    
    # dissipation due to GW radiation
    vw = np.cbrt(Momega)

    # also need vp
    eob_var_circ = eob_var.copy()
    eob_var_circ[1] = 0.
    dH_dp_phi = compute_dH_1(eob_var_circ, par, 3, 
                             order=order,
                             dvar = dvar[3], 
                             H_func = get_H_eob_mu, 
                             H_func_args=(eob_coeff, 
                                          ll, mm, n_mode_1, n_mode_2))
    vp = Momega * (dH_dp_phi)**(-2./3.)

    orb_var = np.array([r_M, p_r_mu_tort, phi, p_phi_Mmu])

    
    hDL_M_modes = get_hDL_M_mode(orb_var, vp, vw, Mdrdt_r,
                                 H_eob_mu,
                                 H_mode_mu_1, S_ns_Mmu_1, H_mode_mu_2, S_ns_Mmu_2,
                                 par, 
                                 ll, mm, Mwa, Mwa0, 
                                 n_mode_1, n_mode_2, 
                                 rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
                                 nqc_coeff, 
                                 h_tide_mult_coeff_1, h_tide_mult_coeff_2)
    
    # eqs. 10-12 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.84.124052
    msq_4_h = np.array([4., 1., 9., 4., 16.])
    Fphi_Mmu = - Momega/(eta * 8. * np.pi) \
            * np.sum(msq_4_h * np.abs(hDL_M_modes)**2.)

    dpp_var[3] += Fphi_Mmu
    dpp_var[1] += Fphi_Mmu * p_r_mu_tort/PI_phi_Mmu 
    
    dpp_tide_var = np.zeros(len(pp_tide_var))
    dpp_tide_var[:4] = dpp_var
    dpp_tide_var[4:4+n_mode_1] = np.real(db_a_1)
    dpp_tide_var[4+n_mode_1:4+2*n_mode_1] = np.imag(db_a_1)
    if n_mode_2 >0:
        dpp_tide_var[4+2*n_mode_1:4+2*n_mode_1+n_mode_2] = np.real(db_a_2)
        dpp_tide_var[4+2*n_mode_1+n_mode_2:] = np.imag(db_a_2)
    
    return dpp_tide_var


@njit
def evol_sys_no_tspin_in_br(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2,
             order=2, dlogvar=1e-4):
    
    r_M, p_r_mu_tort, phi, PI_phi_Mmu = pp_tide_var[:4]
    b_a_re_im_1 = pp_tide_var[4:4+int(n_mode_1*2)]
    b_a_1 = b_a_re_im_1[:n_mode_1] + 1j * b_a_re_im_1[n_mode_1:]

    b_a_re_im_2 = pp_tide_var[4+int(n_mode_1*2):]
    b_a_2 = b_a_re_im_2[:n_mode_2] + 1j * b_a_re_im_2[n_mode_2:]

    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # tide
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll[:n_mode_1] + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll[n_mode_1:] + 1. )
    v_a_2 *= (-1.)**(ll[n_mode_1:])
    
    # corot frame energy (each mode)
    H_ns_mu_1 = E1_mu * (Mwa/Mwa0)[:n_mode_1] * np.abs(b_a_1)**2.
    # canonical spin
    S_ns_Mmu_1 = H_ns_mu_1 * (mm/ Mwa)[:n_mode_1]
    
    H_ns_mu_1 = np.sum(H_ns_mu_1)
    S_ns_Mmu_1 = np.sum(S_ns_Mmu_1)

    # inertial frame energy
    H_mode_mu_1 = H_ns_mu_1 + MOmegaS1 * S_ns_Mmu_1

    # corot frame energy (each mode)
    H_ns_mu_2 = E2_mu * (Mwa/Mwa0)[n_mode_1:] * np.abs(b_a_2)**2.
    # canonical spin
    S_ns_Mmu_2 = H_ns_mu_2 * (mm/ Mwa)[n_mode_1:]
    
    H_ns_mu_2 = np.sum(H_ns_mu_2)
    S_ns_Mmu_2 = np.sum(S_ns_Mmu_2)

    # inertial frame energy
    H_mode_mu_2 = H_ns_mu_2 + MOmegaS2 * S_ns_Mmu_2

    # pp angular momentum
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    
    H_int_lm_mu_1, H_int_lm_mu_2 = get_H_int_lm_mu(pp_tide_var, par, 
                 ll, mm,
                 WI_a_scaled, 
                 n_mode_1, n_mode_2)

    # also need derivatives of the interaction energy
    dH_int_lm_mu_1 = -(ll+1)[:n_mode_1] * H_int_lm_mu_1/r_M
    dH_int_lm_mu_2 = -(ll+1)[n_mode_1:] * H_int_lm_mu_2/r_M


    if n_mode_2>0:
        eob_var = np.zeros(8 + n_mode_1 + n_mode_2)
        eob_var[6+n_mode_1] = H_mode_mu_2
        eob_var[7+n_mode_1] = S_ns_Mmu_2
        eob_var[8+n_mode_1:] = H_int_lm_mu_2
    else:
        eob_var = np.zeros(6 + n_mode_1)
    
    eob_var[:4] = pp_tide_var[:4]
    eob_var[4] = H_mode_mu_1
    eob_var[5] = S_ns_Mmu_1
    eob_var[6:6+n_mode_1] = H_int_lm_mu_1

    # get csi to change pr_tort to pr
    # dr/dr^\ast = pr_ast/pr = csi
    # and H_eob as the source term for the flux
    # and all quantities needed for analytical mode evolution
    H_eob_mu, \
    H_eff_nts_mu, \
    deltaU, deltaU_pp, deltaR, csi, \
    z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
    dH_tS_dS_1, dH_tS_dS_2 \
        = get_H_eob_mu_full_return(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2)

    # some quantities needed for analytical differentiation
    A_H_eff_nts_mu = deltaU / H_eff_nts_mu

    dH_eob_dH_eff = 1./(eta * H_eob_mu)
    dH_eff_dH_mode_1 = z_E_1 * A_H_eff_nts_mu
    dH_eff_dH_mode_2 = z_E_2 * A_H_eff_nts_mu

    dH_eff_nts_dS = - p_phi_Mmu/r_M/r_M * A_H_eff_nts_mu

    dH_eff_dH_int_lm_1 = np.ones(n_mode_1) \
        * z_I_1 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_1):
        if ll_1[i]==2:
            if mm_1[i]==2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[0]
            elif mm_1[i]==0:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[1]
            elif mm_1[i]==-2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[2]


    dH_eff_dH_int_lm_2 = np.ones(n_mode_2) \
        * z_I_2 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_2):
        if ll_2[i]==2:
            if mm_2[i]==2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[0]
            elif mm_2[i]==0:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[1]
            elif mm_2[i]==-2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[2]

    # evolve modes
    db_a_1 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_1 * ( -1j * (Mwa + mm * MOmegaS1)[:n_mode_1] * b_a_1 )\
      + (dH_eff_nts_dS + dH_tS_dS_1) * ( -1j * mm[:n_mode_1] * b_a_1 )\
      + dH_eff_dH_int_lm_1 * (1j * Mwa0[:n_mode_1] * v_a_1)
                             )

    db_a_2 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_2 * ( -1j * (Mwa + mm * MOmegaS2)[n_mode_1:] * b_a_2 )\
      + (dH_eff_nts_dS + dH_tS_dS_2) * ( -1j * mm[n_mode_1:] * b_a_2 )\
      + dH_eff_dH_int_lm_2 * (1j * Mwa0[n_mode_1:] * v_a_2)
                             )


    # numerical differentiation for the orbit
    dvar = (np.abs(eob_var[:4]) + 1e-4) * dlogvar
    dH_dvar = np.zeros(4)
    dpp_var = np.zeros(4)

    # zero tidal spin when evolving the orbit
    eob_br_var = eob_var.copy()
    eob_br_var[5] = 0.
    if n_mode_2>0:
        eob_br_var[7 + n_mode_1] = 0.

    for i in range(4):
        if i==2:
            # phi does not show up in H
            continue

        # use eob_br_var where S_ns' are zeroed.
        dH_dvar[i] = compute_dH_1(eob_br_var, par, i, 
                                  order=order,
                                  dvar = dvar[i], 
                                  H_func = get_H_eob_mu, 
                                  H_func_args=(eob_coeff, 
                                               ll, mm, n_mode_1, n_mode_2))

    # pp part
    # eq. 10 of PhysRevD.84.124052
    # dr_M / dt_M; note we use tortoised p_r_tort but regular r
    dpp_var[0] = csi * dH_dvar[1] 
    Mdrdt_r = dpp_var[0] / r_M

    # dp_r_mu_tort / dt_M
    # note we also need to include dH_eob_dH_int * dH_int_dr
    dpp_var[1] = - csi * (dH_dvar[0] + dH_eob_dH_eff \
                              * (  np.sum(dH_eff_dH_int_lm_1 * dH_int_lm_mu_1)\
                                 + np.sum(dH_eff_dH_int_lm_2 * dH_int_lm_mu_2)\
                                )    )

    # dphi / dt_M = Momega
    dpp_var[2] = dH_dvar[3] 
    Momega = dH_dvar[3]

    
    # dissipation due to GW radiation
    vw = np.cbrt(Momega)

    # also need vp
    eob_var_circ = eob_var.copy()
    eob_var_circ[1] = 0.
    dH_dp_phi = compute_dH_1(eob_var_circ, par, 3, 
                             order=order,
                             dvar = dvar[3], 
                             H_func = get_H_eob_mu, 
                             H_func_args=(eob_coeff, 
                                          ll, mm, n_mode_1, n_mode_2))
    vp = Momega * (dH_dp_phi)**(-2./3.)

    orb_var = np.array([r_M, p_r_mu_tort, phi, p_phi_Mmu])

    
    hDL_M_modes = get_hDL_M_mode(orb_var, vp, vw, Mdrdt_r,
                                 H_eob_mu,
                                 H_mode_mu_1, S_ns_Mmu_1, H_mode_mu_2, S_ns_Mmu_2,
                                 par, 
                                 ll, mm, Mwa, Mwa0, 
                                 n_mode_1, n_mode_2, 
                                 rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
                                 nqc_coeff, 
                                 h_tide_mult_coeff_1, h_tide_mult_coeff_2)
    
    # eqs. 10-12 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.84.124052
    msq_4_h = np.array([4., 1., 9., 4., 16.])
    Fphi_Mmu = - Momega/(eta * 8. * np.pi) \
            * np.sum(msq_4_h * np.abs(hDL_M_modes)**2.)

    dpp_var[3] += Fphi_Mmu
    dpp_var[1] += Fphi_Mmu * p_r_mu_tort/PI_phi_Mmu
    
    dpp_tide_var = np.zeros(len(pp_tide_var))
    dpp_tide_var[:4] = dpp_var
    dpp_tide_var[4:4+n_mode_1] = np.real(db_a_1)
    dpp_tide_var[4+n_mode_1:4+2*n_mode_1] = np.imag(db_a_1)
    if n_mode_2 >0:
        dpp_tide_var[4+2*n_mode_1:4+2*n_mode_1+n_mode_2] = np.real(db_a_2)
        dpp_tide_var[4+2*n_mode_1+n_mode_2:] = np.imag(db_a_2)
    
    return dpp_tide_var


@njit
def evol_sys_no_torque(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2,
             order=2, dlogvar=1e-4):
    
    r_M, p_r_mu_tort, phi, PI_phi_Mmu = pp_tide_var[:4]
    b_a_re_im_1 = pp_tide_var[4:4+int(n_mode_1*2)]
    b_a_1 = b_a_re_im_1[:n_mode_1] + 1j * b_a_re_im_1[n_mode_1:]

    b_a_re_im_2 = pp_tide_var[4+int(n_mode_1*2):]
    b_a_2 = b_a_re_im_2[:n_mode_2] + 1j * b_a_re_im_2[n_mode_2:]

    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # tide
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll[:n_mode_1] + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll[n_mode_1:] + 1. )
    v_a_2 *= (-1.)**(ll[n_mode_1:])

    # tidal torque
    torque_1 = - 2. * E1_mu * np.sum(mm[:n_mode_1] * v_a_1 * np.imag(b_a_1))
    torque_2 = - 2. * E2_mu * np.sum(mm[n_mode_1:] * v_a_2 * np.imag(b_a_2))    
    
    # corot frame energy (each mode)
    H_ns_mu_1 = E1_mu * (Mwa/Mwa0)[:n_mode_1] * np.abs(b_a_1)**2.
    # canonical spin
    S_ns_Mmu_1 = H_ns_mu_1 * (mm/ Mwa)[:n_mode_1]
    
    H_ns_mu_1 = np.sum(H_ns_mu_1)
    S_ns_Mmu_1 = np.sum(S_ns_Mmu_1)

    # inertial frame energy
    H_mode_mu_1 = H_ns_mu_1 + MOmegaS1 * S_ns_Mmu_1

    # corot frame energy (each mode)
    H_ns_mu_2 = E2_mu * (Mwa/Mwa0)[n_mode_1:] * np.abs(b_a_2)**2.
    # canonical spin
    S_ns_Mmu_2 = H_ns_mu_2 * (mm/ Mwa)[n_mode_1:]
    
    H_ns_mu_2 = np.sum(H_ns_mu_2)
    S_ns_Mmu_2 = np.sum(S_ns_Mmu_2)

    # inertial frame energy
    H_mode_mu_2 = H_ns_mu_2 + MOmegaS2 * S_ns_Mmu_2

    # pp angular momentum
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    
    H_int_lm_mu_1, H_int_lm_mu_2 = get_H_int_lm_mu(pp_tide_var, par, 
                 ll, mm,
                 WI_a_scaled, 
                 n_mode_1, n_mode_2)

    # also need derivatives of the interaction energy
    dH_int_lm_mu_1 = -(ll+1)[:n_mode_1] * H_int_lm_mu_1/r_M
    dH_int_lm_mu_2 = -(ll+1)[n_mode_1:] * H_int_lm_mu_2/r_M


    if n_mode_2>0:
        eob_var = np.zeros(8 + n_mode_1 + n_mode_2)
        eob_var[6+n_mode_1] = H_mode_mu_2
        eob_var[7+n_mode_1] = S_ns_Mmu_2
        eob_var[8+n_mode_1:] = H_int_lm_mu_2
    else:
        eob_var = np.zeros(6 + n_mode_1)
    
    eob_var[:4] = pp_tide_var[:4]
    eob_var[4] = H_mode_mu_1
    eob_var[5] = S_ns_Mmu_1
    eob_var[6:6+n_mode_1] = H_int_lm_mu_1

    # get csi to change pr_tort to pr
    # dr/dr^\ast = pr_ast/pr = csi
    # and H_eob as the source term for the flux
    # and all quantities needed for analytical mode evolution
    H_eob_mu, \
    H_eff_nts_mu, \
    deltaU, deltaU_pp, deltaR, csi, \
    z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
    dH_tS_dS_1, dH_tS_dS_2 \
        = get_H_eob_mu_full_return(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2)

    # some quantities needed for analytical differentiation
    A_H_eff_nts_mu = deltaU / H_eff_nts_mu

    dH_eob_dH_eff = 1./(eta * H_eob_mu)
    dH_eff_dH_mode_1 = z_E_1 * A_H_eff_nts_mu
    dH_eff_dH_mode_2 = z_E_2 * A_H_eff_nts_mu

    dH_eff_nts_dS = - p_phi_Mmu/r_M/r_M * A_H_eff_nts_mu

    dH_eff_dH_int_lm_1 = np.ones(n_mode_1) \
        * z_I_1 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_1):
        if ll_1[i]==2:
            if mm_1[i]==2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[0]
            elif mm_1[i]==0:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[1]
            elif mm_1[i]==-2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[2]


    dH_eff_dH_int_lm_2 = np.ones(n_mode_2) \
        * z_I_2 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_2):
        if ll_2[i]==2:
            if mm_2[i]==2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[0]
            elif mm_2[i]==0:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[1]
            elif mm_2[i]==-2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[2]

    # evolve modes
    db_a_1 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_1 * ( -1j * (Mwa + mm * MOmegaS1)[:n_mode_1] * b_a_1 )\
      + (dH_eff_nts_dS + dH_tS_dS_1) * ( -1j * mm[:n_mode_1] * b_a_1 )\
      + dH_eff_dH_int_lm_1 * (1j * Mwa0[:n_mode_1] * v_a_1)
                             )

    db_a_2 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_2 * ( -1j * (Mwa + mm * MOmegaS2)[n_mode_1:] * b_a_2 )\
      + (dH_eff_nts_dS + dH_tS_dS_2) * ( -1j * mm[n_mode_1:] * b_a_2 )\
      + dH_eff_dH_int_lm_2 * (1j * Mwa0[n_mode_1:] * v_a_2)
                             )


    # numerical differentiation for the orbit
    dvar = (np.abs(eob_var[:4]) + 1e-4) * dlogvar
    dH_dvar = np.zeros(4)
    dpp_var = np.zeros(4)

    for i in range(4):
        if i==2:
            # phi does not show up in H
            continue
        dH_dvar[i] = compute_dH_1(eob_var, par, i, 
                                  order=order,
                                  dvar = dvar[i], 
                                  H_func = get_H_eob_mu, 
                                  H_func_args=(eob_coeff, 
                                               ll, mm, n_mode_1, n_mode_2))

    # pp part
    # eq. 10 of PhysRevD.84.124052
    # dr_M / dt_M; note we use tortoised p_r_tort but regular r
    dpp_var[0] = csi * dH_dvar[1] 
    Mdrdt_r = dpp_var[0] / r_M

    # dp_r_mu_tort / dt_M
    # note we also need to include dH_eob_dH_int * dH_int_dr
    dpp_var[1] = - csi * (dH_dvar[0] + dH_eob_dH_eff \
                              * (  np.sum(dH_eff_dH_int_lm_1 * dH_int_lm_mu_1)\
                                 + np.sum(dH_eff_dH_int_lm_2 * dH_int_lm_mu_2)\
                                )    )

    # dphi / dt_M = Momega
    dpp_var[2] = dH_dvar[3] 
    Momega = dH_dvar[3]

    # dp_phi_Mmu; removing the Newtonian tidal torque
    dpp_var[3] = - torque_1 - torque_2
    
    # dissipation due to GW radiation
    vw = np.cbrt(Momega)

    # also need vp
    eob_var_circ = eob_var.copy()
    eob_var_circ[1] = 0.
    dH_dp_phi = compute_dH_1(eob_var_circ, par, 3, 
                             order=order,
                             dvar = dvar[3], 
                             H_func = get_H_eob_mu, 
                             H_func_args=(eob_coeff, 
                                          ll, mm, n_mode_1, n_mode_2))
    vp = Momega * (dH_dp_phi)**(-2./3.)

    orb_var = np.array([r_M, p_r_mu_tort, phi, p_phi_Mmu])

    
    hDL_M_modes = get_hDL_M_mode(orb_var, vp, vw, Mdrdt_r,
                                 H_eob_mu,
                                 H_mode_mu_1, S_ns_Mmu_1, H_mode_mu_2, S_ns_Mmu_2,
                                 par, 
                                 ll, mm, Mwa, Mwa0, 
                                 n_mode_1, n_mode_2, 
                                 rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
                                 nqc_coeff, 
                                 h_tide_mult_coeff_1, h_tide_mult_coeff_2)
    
    # eqs. 10-12 of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.84.124052
    msq_4_h = np.array([4., 1., 9., 4., 16.])
    Fphi_Mmu = - Momega/(eta * 8. * np.pi) \
            * np.sum(msq_4_h * np.abs(hDL_M_modes)**2.)

    dpp_var[3] += Fphi_Mmu
    dpp_var[1] += Fphi_Mmu * p_r_mu_tort/PI_phi_Mmu
    
    dpp_tide_var = np.zeros(len(pp_tide_var))
    dpp_tide_var[:4] = dpp_var
    dpp_tide_var[4:4+n_mode_1] = np.real(db_a_1)
    dpp_tide_var[4+n_mode_1:4+2*n_mode_1] = np.imag(db_a_1)
    if n_mode_2 >0:
        dpp_tide_var[4+2*n_mode_1:4+2*n_mode_1+n_mode_2] = np.real(db_a_2)
        dpp_tide_var[4+2*n_mode_1+n_mode_2:] = np.imag(db_a_2)
    
    return dpp_tide_var

##############################

@njit
def get_tide_prop_from_pp_tide_var_1_pt(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             order=2, dlogvar=1e-4):

    r_M, p_r_mu_tort, phi, PI_phi_Mmu = pp_tide_var[:4]
    b_a_re_im_1 = pp_tide_var[4:4+int(n_mode_1*2)]
    b_a_1 = b_a_re_im_1[:n_mode_1] + 1j * b_a_re_im_1[n_mode_1:]

    b_a_re_im_2 = pp_tide_var[4+int(n_mode_1*2):]
    b_a_2 = b_a_re_im_2[:n_mode_2] + 1j * b_a_re_im_2[n_mode_2:]

    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # tide
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll[:n_mode_1] + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll[n_mode_1:] + 1. )
    v_a_2 *= (-1.)**(ll[n_mode_1:])
    
    # corot frame energy (each mode)
    H_ns_mu_1 = E1_mu * (Mwa/Mwa0)[:n_mode_1] * np.abs(b_a_1)**2.
    # canonical spin
    S_ns_Mmu_1 = H_ns_mu_1 * (mm/ Mwa)[:n_mode_1]
    
    H_ns_mu_1 = np.sum(H_ns_mu_1)
    S_ns_Mmu_1 = np.sum(S_ns_Mmu_1)

    # inertial frame energy
    H_mode_mu_1 = H_ns_mu_1 + MOmegaS1 * S_ns_Mmu_1

    # corot frame energy (each mode)
    H_ns_mu_2 = E2_mu * (Mwa/Mwa0)[n_mode_1:] * np.abs(b_a_2)**2.
    # canonical spin
    S_ns_Mmu_2 = H_ns_mu_2 * (mm/ Mwa)[n_mode_1:]
    
    H_ns_mu_2 = np.sum(H_ns_mu_2)
    S_ns_Mmu_2 = np.sum(S_ns_Mmu_2)

    # inertial frame energy
    H_mode_mu_2 = H_ns_mu_2 + MOmegaS2 * S_ns_Mmu_2

    # pp angular momentum
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    
    H_int_lm_mu_1, H_int_lm_mu_2 = get_H_int_lm_mu(pp_tide_var, par, 
                 ll, mm,
                 WI_a_scaled, 
                 n_mode_1, n_mode_2)

    # also need derivatives of the interaction energy
    dH_int_lm_mu_1 = -(ll+1)[:n_mode_1] * H_int_lm_mu_1/r_M
    dH_int_lm_mu_2 = -(ll+1)[n_mode_1:] * H_int_lm_mu_2/r_M


    if n_mode_2>0:
        eob_var = np.zeros(8 + n_mode_1 + n_mode_2)
        eob_var[6+n_mode_1] = H_mode_mu_2
        eob_var[7+n_mode_1] = S_ns_Mmu_2
        eob_var[8+n_mode_1:] = H_int_lm_mu_2
    else:
        eob_var = np.zeros(6 + n_mode_1)
    
    eob_var[:4] = pp_tide_var[:4]
    eob_var[4] = H_mode_mu_1
    eob_var[5] = S_ns_Mmu_1
    eob_var[6:6+n_mode_1] = H_int_lm_mu_1

    # get csi to change pr_tort to pr
    # dr/dr^\ast = pr_ast/pr = csi
    # and H_eob as the source term for the flux
    # and all quantities needed for analytical mode evolution
    H_eob_mu, \
    H_eff_nts_mu, \
    deltaU, deltaU_pp, deltaR, csi, \
    z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
    dH_tS_dS_1, dH_tS_dS_2 \
        = get_H_eob_mu_full_return(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2)

    # some quantities needed for analytical differentiation
    A_H_eff_nts_mu = deltaU / H_eff_nts_mu

    dH_eob_dH_eff = 1./(eta * H_eob_mu)
    dH_eff_dH_mode_1 = z_E_1 * A_H_eff_nts_mu
    dH_eff_dH_mode_2 = z_E_2 * A_H_eff_nts_mu

    dH_eff_nts_dS = - p_phi_Mmu/r_M/r_M * A_H_eff_nts_mu

    dH_eff_dH_int_lm_1 = np.ones(n_mode_1) \
        * z_I_1 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_1):
        if ll_1[i]==2:
            if mm_1[i]==2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[0]
            elif mm_1[i]==0:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[1]
            elif mm_1[i]==-2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[2]


    dH_eff_dH_int_lm_2 = np.ones(n_mode_2) \
        * z_I_2 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_2):
        if ll_2[i]==2:
            if mm_2[i]==2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[0]
            elif mm_2[i]==0:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[1]
            elif mm_2[i]==-2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[2]

    # evolve modes
    db_a_1 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_1 * ( -1j * (Mwa + mm * MOmegaS1)[:n_mode_1] * b_a_1 )\
      + (dH_eff_nts_dS + dH_tS_dS_1) * ( -1j * mm[:n_mode_1] * b_a_1 )\
      + dH_eff_dH_int_lm_1 * (1j * Mwa0[:n_mode_1] * v_a_1)
                             )

    db_a_2 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_2 * ( -1j * (Mwa + mm * MOmegaS2)[n_mode_1:] * b_a_2 )\
      + (dH_eff_nts_dS + dH_tS_dS_2) * ( -1j * mm[n_mode_1:] * b_a_2 )\
      + dH_eff_dH_int_lm_2 * (1j * Mwa0[n_mode_1:] * v_a_2)
                             )

    # numerical differentiation for the orbit
    dvar = (np.abs(eob_var[:4]) + 1e-4) * dlogvar
    dH_dvar = np.zeros(4)
    dpp_var = np.zeros(4)

    for i in range(4):
        if i==2:
            # phi does not show up in H
            continue
        dH_dvar[i] = compute_dH_1(eob_var, par, i, 
                                  order=order,
                                  dvar = dvar[i], 
                                  H_func = get_H_eob_mu, 
                                  H_func_args=(eob_coeff, 
                                               ll, mm, n_mode_1, n_mode_2))

    # pp part
    # eq. 10 of PhysRevD.84.124052
    # dr_M / dt_M; note we use tortoised p_r_tort but regular r
    dpp_var[0] = csi * dH_dvar[1] 
    Mdrdt_r = dpp_var[0] / r_M

    # dp_r_mu_tort / dt_M
    # note we also need to include dH_eob_dH_int * dH_int_dr
    dpp_var[1] = - csi * (dH_dvar[0] + dH_eob_dH_eff \
                              * (  np.sum(dH_eff_dH_int_lm_1 * dH_int_lm_mu_1)\
                                 + np.sum(dH_eff_dH_int_lm_2 * dH_int_lm_mu_2)\
                                )    )

    # dphi / dt_M = Momega
    dpp_var[2] = dH_dvar[3] 
    Momega = dH_dvar[3]

    if n_mode_2>0:
        dH_dH_S_ns = np.zeros(6 + n_mode_1 + n_mode_2)

        dH_dH_S_ns[4+n_mode_1] = dH_eff_dH_mode_2
        dH_dH_S_ns[5+n_mode_1] = dH_tS_dS_2
        dH_dH_S_ns[6+n_mode_1:] = dH_eff_dH_int_lm_2

    else:
        dH_dH_S_ns = np.zeros(4 + n_mode_1)

    dH_dH_S_ns[0] = dH_eob_dH_eff
    dH_dH_S_ns[1] = dH_eff_nts_dS
        
    dH_dH_S_ns[2] = dH_eff_dH_mode_1
    dH_dH_S_ns[3] = dH_tS_dS_1
    dH_dH_S_ns[4:4+n_mode_1] = dH_eff_dH_int_lm_1

    return Momega, eob_var[4:], dH_dH_S_ns

@njit
def get_tide_prop_from_pp_tide_var(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             order=2, dlogvar=1e-4):
    n_pt = len(t_M)
    Momega = np.zeros(n_pt)
    
    if n_mode_2>0:
        H_S_tide = np.zeros((4 + n_mode_1 + n_mode_2, n_pt))
        dH_dH_S_ns = np.zeros((6 + n_mode_1 + n_mode_2, n_pt))
        
    else:
        H_S_tide = np.zeros((2 + n_mode_1, n_pt))
        dH_dH_S_ns = np.zeros((4 + n_mode_1, n_pt))
                
    for i in range(n_pt):
        Momega[i], H_S_tide[:, i], dH_dH_S_ns[:, i] \
            =  get_tide_prop_from_pp_tide_var_1_pt(t_M[i], pp_tide_var[:,i], par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             order=order, dlogvar=dlogvar)
    return Momega, H_S_tide, dH_dH_S_ns   
        
        
@njit
def get_hDL_M_from_pp_tide_var_1_pt(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2, 
             order=2, dlogvar=1e-4):
    r_M, p_r_mu_tort, phi, PI_phi_Mmu = pp_tide_var[:4]
    b_a_re_im_1 = pp_tide_var[4:4+int(n_mode_1*2)]
    b_a_1 = b_a_re_im_1[:n_mode_1] + 1j * b_a_re_im_1[n_mode_1:]

    b_a_re_im_2 = pp_tide_var[4+int(n_mode_1*2):]
    b_a_2 = b_a_re_im_2[:n_mode_2] + 1j * b_a_re_im_2[n_mode_2:]

    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # tide
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll[:n_mode_1] + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll[n_mode_1:] + 1. )
    v_a_2 *= (-1.)**(ll[n_mode_1:])
    
    # corot frame energy (each mode)
    H_ns_mu_1 = E1_mu * (Mwa/Mwa0)[:n_mode_1] * np.abs(b_a_1)**2.
    # canonical spin
    S_ns_Mmu_1 = H_ns_mu_1 * (mm/ Mwa)[:n_mode_1]
    
    H_ns_mu_1 = np.sum(H_ns_mu_1)
    S_ns_Mmu_1 = np.sum(S_ns_Mmu_1)

    # inertial frame energy
    H_mode_mu_1 = H_ns_mu_1 + MOmegaS1 * S_ns_Mmu_1

    # corot frame energy (each mode)
    H_ns_mu_2 = E2_mu * (Mwa/Mwa0)[n_mode_1:] * np.abs(b_a_2)**2.
    # canonical spin
    S_ns_Mmu_2 = H_ns_mu_2 * (mm/ Mwa)[n_mode_1:]
    
    H_ns_mu_2 = np.sum(H_ns_mu_2)
    S_ns_Mmu_2 = np.sum(S_ns_Mmu_2)

    # inertial frame energy
    H_mode_mu_2 = H_ns_mu_2 + MOmegaS2 * S_ns_Mmu_2

    # pp angular momentum
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    
    H_int_lm_mu_1, H_int_lm_mu_2 = get_H_int_lm_mu(pp_tide_var, par, 
                 ll, mm,
                 WI_a_scaled, 
                 n_mode_1, n_mode_2)

    # also need derivatives of the interaction energy
    dH_int_lm_mu_1 = -(ll+1)[:n_mode_1] * H_int_lm_mu_1/r_M
    dH_int_lm_mu_2 = -(ll+1)[n_mode_1:] * H_int_lm_mu_2/r_M


    if n_mode_2>0:
        eob_var = np.zeros(8 + n_mode_1 + n_mode_2)
        eob_var[6+n_mode_1] = H_mode_mu_2
        eob_var[7+n_mode_1] = S_ns_Mmu_2
        eob_var[8+n_mode_1:] = H_int_lm_mu_2
    else:
        eob_var = np.zeros(6 + n_mode_1)
    
    eob_var[:4] = pp_tide_var[:4]
    eob_var[4] = H_mode_mu_1
    eob_var[5] = S_ns_Mmu_1
    eob_var[6:6+n_mode_1] = H_int_lm_mu_1

    # get csi to change pr_tort to pr
    # dr/dr^\ast = pr_ast/pr = csi
    # and H_eob as the source term for the flux
    # and all quantities needed for analytical mode evolution
    H_eob_mu, \
    H_eff_nts_mu, \
    deltaU, deltaU_pp, deltaR, csi, \
    z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
    dH_tS_dS_1, dH_tS_dS_2 \
        = get_H_eob_mu_full_return(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2)

    # some quantities needed for analytical differentiation
    A_H_eff_nts_mu = deltaU / H_eff_nts_mu

    dH_eob_dH_eff = 1./(eta * H_eob_mu)
    dH_eff_dH_mode_1 = z_E_1 * A_H_eff_nts_mu
    dH_eff_dH_mode_2 = z_E_2 * A_H_eff_nts_mu

    dH_eff_nts_dS = - p_phi_Mmu/r_M/r_M * A_H_eff_nts_mu

    dH_eff_dH_int_lm_1 = np.ones(n_mode_1) \
        * z_I_1 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_1):
        if ll_1[i]==2:
            if mm_1[i]==2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[0]
            elif mm_1[i]==0:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[1]
            elif mm_1[i]==-2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[2]


    dH_eff_dH_int_lm_2 = np.ones(n_mode_2) \
        * z_I_2 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_2):
        if ll_2[i]==2:
            if mm_2[i]==2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[0]
            elif mm_2[i]==0:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[1]
            elif mm_2[i]==-2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[2]

    # evolve modes
    db_a_1 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_1 * ( -1j * (Mwa + mm * MOmegaS1)[:n_mode_1] * b_a_1 )\
      + (dH_eff_nts_dS + dH_tS_dS_1) * ( -1j * mm[:n_mode_1] * b_a_1 )\
      + dH_eff_dH_int_lm_1 * (1j * Mwa0[:n_mode_1] * v_a_1)
                             )

    db_a_2 = dH_eob_dH_eff * ( \
        dH_eff_dH_mode_2 * ( -1j * (Mwa + mm * MOmegaS1)[n_mode_1:] * b_a_2 )\
      + (dH_eff_nts_dS + dH_tS_dS_2) * ( -1j * mm[n_mode_1:] * b_a_2 )\
      + dH_eff_dH_int_lm_2 * (1j * Mwa0[n_mode_1:] * v_a_2)
                             )


    # numerical differentiation for the orbit
    dvar = (np.abs(eob_var[:4]) + 1e-4) * dlogvar
    dH_dvar = np.zeros(4)
    dpp_var = np.zeros(4)

    for i in range(4):
        if i==2:
            # phi does not show up in H
            continue
        dH_dvar[i] = compute_dH_1(eob_var, par, i, 
                                  order=order,
                                  dvar = dvar[i], 
                                  H_func = get_H_eob_mu, 
                                  H_func_args=(eob_coeff, 
                                               ll, mm, n_mode_1, n_mode_2))

    # pp part
    # eq. 10 of PhysRevD.84.124052
    # dr_M / dt_M; note we use tortoised p_r_tort but regular r
    dpp_var[0] = csi * dH_dvar[1] 
    Mdrdt_r = dpp_var[0] / r_M

    # dp_r_mu_tort / dt_M
    # note we also need to include dH_eob_dH_int * dH_int_dr
    dpp_var[1] = - csi * (dH_dvar[0] + dH_eob_dH_eff \
                              * (  np.sum(dH_eff_dH_int_lm_1 * dH_int_lm_mu_1)\
                                 + np.sum(dH_eff_dH_int_lm_2 * dH_int_lm_mu_2)\
                                )    )

    # dphi / dt_M = Momega
    dpp_var[2] = dH_dvar[3] 
    Momega = dH_dvar[3]

    
    # dissipation due to GW radiation
    vw = np.cbrt(Momega)

    # also need vp
    eob_var_circ = eob_var.copy()
    eob_var_circ[1] = 0.
    dH_dp_phi = compute_dH_1(eob_var_circ, par, 3, 
                             order=order,
                             dvar = dvar[3], 
                             H_func = get_H_eob_mu, 
                             H_func_args=(eob_coeff, 
                                          ll, mm, n_mode_1, n_mode_2))
    vp = Momega * (dH_dp_phi)**(-2./3.)

    orb_var = np.array([r_M, p_r_mu_tort, phi, p_phi_Mmu])

    
    hDL_M_modes = get_hDL_M_mode(orb_var, vp, vw, Mdrdt_r,
                                 H_eob_mu,
                                 H_mode_mu_1, S_ns_Mmu_1, H_mode_mu_2, S_ns_Mmu_2,
                                 par, 
                                 ll, mm, Mwa, Mwa0, 
                                 n_mode_1, n_mode_2, 
                                 rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
                                 nqc_coeff, 
                                 h_tide_mult_coeff_1, h_tide_mult_coeff_2)
    return Momega, vp, hDL_M_modes

@njit
def get_hDL_M_from_pp_tide_var(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2, 
             order=2, dlogvar=1e-3, 
             r_M_taper_on = 6., r_M_taper_off = 5.8):
    n_pt = len(t_M)
    
    Momega = np.zeros(n_pt)
    vp = np.zeros(n_pt)
    hDL_M_modes = np.zeros((5, n_pt), dtype=np.complex128)

    if (r_M_taper_off < np.min(pp_tide_var[0, :])) \
        and (r_M_taper_on > np.min(pp_tide_var[0, :])):
        r_M_taper_off = np.min(pp_tide_var[0, :])
        
    
    for i in range(n_pt):
        r_M = pp_tide_var[0, i]
        if r_M < r_M_taper_off:
            continue
            
        Momega[i], vp[i], hDL_M_modes[:, i]\
            = get_hDL_M_from_pp_tide_var_1_pt(t_M[i], pp_tide_var[:, i], par, 
                     ll, mm,
                     WI_a_scaled, Mwa, Mwa0, 
                     n_mode_1, n_mode_2, 
                     eob_coeff,
                     rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
                     nqc_coeff, 
                     h_tide_mult_coeff_1, h_tide_mult_coeff_2, 
                     order, dlogvar)
        
        
        if r_M<=r_M_taper_on and r_M>=r_M_taper_off:
            taper_ph = np.pi/2. * (r_M - r_M_taper_on)/(r_M_taper_on - r_M_taper_off)
            taper = np.cos(taper_ph)
            hDL_M_modes[:, i] *= taper

            taper_arg = - 2. * ((r_M_taper_on - r_M) / (r_M_taper_on - r_M_taper_off))**2.
            taper = np.exp(taper_arg)
            hDL_M_modes[:, i] *= taper
            
    return Momega, vp, hDL_M_modes



@njit
def get_potentials_from_pp_tide_var_1_pt(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             order=2, dlogvar=1e-4):

    r_M, p_r_mu_tort, phi, PI_phi_Mmu = pp_tide_var[:4]
    b_a_re_im_1 = pp_tide_var[4:4+int(n_mode_1*2)]
    b_a_1 = b_a_re_im_1[:n_mode_1] + 1j * b_a_re_im_1[n_mode_1:]

    b_a_re_im_2 = pp_tide_var[4+int(n_mode_1*2):]
    b_a_2 = b_a_re_im_2[:n_mode_2] + 1j * b_a_re_im_2[n_mode_2:]

    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # tide
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll[:n_mode_1] + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll[n_mode_1:] + 1. )
    v_a_2 *= (-1.)**(ll[n_mode_1:])
    
    # corot frame energy (each mode)
    H_ns_mu_1 = E1_mu * (Mwa/Mwa0)[:n_mode_1] * np.abs(b_a_1)**2.
    # canonical spin
    S_ns_Mmu_1 = H_ns_mu_1 * (mm/ Mwa)[:n_mode_1]
    
    H_ns_mu_1 = np.sum(H_ns_mu_1)
    S_ns_Mmu_1 = np.sum(S_ns_Mmu_1)

    # inertial frame energy
    H_mode_mu_1 = H_ns_mu_1 + MOmegaS1 * S_ns_Mmu_1

    # corot frame energy (each mode)
    H_ns_mu_2 = E2_mu * (Mwa/Mwa0)[n_mode_1:] * np.abs(b_a_2)**2.
    # canonical spin
    S_ns_Mmu_2 = H_ns_mu_2 * (mm/ Mwa)[n_mode_1:]
    
    H_ns_mu_2 = np.sum(H_ns_mu_2)
    S_ns_Mmu_2 = np.sum(S_ns_Mmu_2)

    # inertial frame energy
    H_mode_mu_2 = H_ns_mu_2 + MOmegaS2 * S_ns_Mmu_2

    # pp angular momentum
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2

    
    H_int_lm_mu_1, H_int_lm_mu_2 = get_H_int_lm_mu(pp_tide_var, par, 
                 ll, mm,
                 WI_a_scaled, 
                 n_mode_1, n_mode_2)

    # also need derivatives of the interaction energy
    dH_int_lm_mu_1 = -(ll+1)[:n_mode_1] * H_int_lm_mu_1/r_M
    dH_int_lm_mu_2 = -(ll+1)[n_mode_1:] * H_int_lm_mu_2/r_M


    if n_mode_2>0:
        eob_var = np.zeros(8 + n_mode_1 + n_mode_2)
        eob_var[6+n_mode_1] = H_mode_mu_2
        eob_var[7+n_mode_1] = S_ns_Mmu_2
        eob_var[8+n_mode_1:] = H_int_lm_mu_2
    else:
        eob_var = np.zeros(6 + n_mode_1)
    
    eob_var[:4] = pp_tide_var[:4]
    eob_var[4] = H_mode_mu_1
    eob_var[5] = S_ns_Mmu_1
    eob_var[6:6+n_mode_1] = H_int_lm_mu_1

    # get csi to change pr_tort to pr
    # dr/dr^\ast = pr_ast/pr = csi
    # and H_eob as the source term for the flux
    # and all quantities needed for analytical mode evolution
    H_eob_mu, \
    H_eff_nts_mu, \
    deltaU, deltaU_pp, deltaR, csi, \
    z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
    dH_tS_dS_1, dH_tS_dS_2 \
        = get_H_eob_mu_full_return(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2)

    # some quantities needed for analytical differentiation
    A_H_eff_nts_mu = deltaU / H_eff_nts_mu

    dH_eob_dH_eff = 1./(eta * H_eob_mu)
    dH_eff_dH_mode_1 = z_E_1 * A_H_eff_nts_mu
    dH_eff_dH_mode_2 = z_E_2 * A_H_eff_nts_mu

    dH_eff_nts_dS = - p_phi_Mmu/r_M/r_M * A_H_eff_nts_mu

    dH_eff_dH_int_lm_1 = np.ones(n_mode_1) \
        * z_I_1 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_1):
        if ll_1[i]==2:
            if mm_1[i]==2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[0]
            elif mm_1[i]==0:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[1]
            elif mm_1[i]==-2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[2]


    dH_eff_dH_int_lm_2 = np.ones(n_mode_2) \
        * z_I_2 * ( 1./A_H_eff_nts_mu + (p_r_mu_tort/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_2):
        if ll_2[i]==2:
            if mm_2[i]==2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[0]
            elif mm_2[i]==0:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[1]
            elif mm_2[i]==-2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[2]


    # numerical differentiation for the orbit
    dvar = (np.abs(eob_var[:4]) + 1e-4) * dlogvar
    dH_dvar = np.zeros(4)
    dpp_var = np.zeros(4)

    for i in range(4):
        if i==2:
            # phi does not show up in H
            continue
        dH_dvar[i] = compute_dH_1(eob_var, par, i, 
                                  order=order,
                                  dvar = dvar[i], 
                                  H_func = get_H_eob_mu, 
                                  H_func_args=(eob_coeff, 
                                               ll, mm, n_mode_1, n_mode_2))

    # pp part
    # eq. 10 of PhysRevD.84.124052
    # dr_M / dt_M; note we use tortoised p_r_tort but regular r
    dpp_var[0] = csi * dH_dvar[1] 
    Mdrdt_r = dpp_var[0] / r_M

    # dp_r_mu_tort / dt_M
    # note we also need to include dH_eob_dH_int * dH_int_dr
    dpp_var[1] = - csi * (dH_dvar[0] + dH_eob_dH_eff \
                              * (  np.sum(dH_eff_dH_int_lm_1 * dH_int_lm_mu_1)\
                                 + np.sum(dH_eff_dH_int_lm_2 * dH_int_lm_mu_2)\
                                )    )

    # dphi / dt_M = Momega
    dpp_var[2] = dH_dvar[3] 
    Momega = dH_dvar[3]

    if n_mode_2>0:
        dH_dH_S_ns = np.zeros(6 + n_mode_1 + n_mode_2)

        dH_dH_S_ns[4+n_mode_1] = dH_eff_dH_mode_2
        dH_dH_S_ns[5+n_mode_1] = dH_tS_dS_2
        dH_dH_S_ns[6+n_mode_1:] = dH_eff_dH_int_lm_2

    else:
        dH_dH_S_ns = np.zeros(4 + n_mode_1)

    dH_dH_S_ns[0] = dH_eob_dH_eff
    dH_dH_S_ns[1] = dH_eff_nts_dS
        
    dH_dH_S_ns[2] = dH_eff_dH_mode_1
    dH_dH_S_ns[3] = dH_tS_dS_1
    dH_dH_S_ns[4:4+n_mode_1] = dH_eff_dH_int_lm_1

    dH_dr = dH_dvar[0] + dH_eob_dH_eff \
                              * (  np.sum(dH_eff_dH_int_lm_1 * dH_int_lm_mu_1)\
                                 + np.sum(dH_eff_dH_int_lm_2 * dH_int_lm_mu_2)\
                                ) 

    dH_dpr = dH_dvar[1]
    return Momega, deltaU, deltaU_pp, csi, dH_dr, dH_dpr

@njit
def get_potentials_from_pp_tide_var(t_M, pp_tide_var, par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             order=2, dlogvar=1e-4):
    n_pt = len(t_M)
    Momega = np.zeros(n_pt)
    
    deltaU, deltaU_pp, csi, dH_dr = np.zeros(n_pt), np.zeros(n_pt), np.zeros(n_pt), np.zeros(n_pt)
    dH_dpr = np.zeros(n_pt)            
    
    for i in range(n_pt):
        Momega[i], deltaU[i], deltaU_pp[i], csi[i], dH_dr[i], dH_dpr[i] \
            =  get_potentials_from_pp_tide_var_1_pt(t_M[i], pp_tide_var[:,i], par, 
             ll, mm,
             WI_a_scaled, Mwa, Mwa0, 
             n_mode_1, n_mode_2, 
             eob_coeff,
             order=order, dlogvar=dlogvar)
    return Momega, deltaU, deltaU_pp, csi, dH_dr, dH_dpr
    

##############################

@njit(cache=True)
def find_eq_tide_energies_N(r_M, par, 
                            ll, mm,
                            WI_a_scaled, Mwa, Mwa0, 
                            n_mode_1, n_mode_2):
    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]

    # estimate the tidal energies based on Newtonian physics
    Momega = (1./r_M)**(1.5)
    Mdrdt_r = -64./5. * eta / r_M**4
    
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    Mwa_1, Mwa_2 = Mwa[:n_mode_1], Mwa[n_mode_1:]
    Mwa0_1, Mwa0_2 = Mwa0[:n_mode_1], Mwa0[n_mode_1:]    
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll_1 + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll_2 + 1. )
    v_a_2 *= (-1.)**(ll_2)

    MDel_a_1 = (Mwa_1 + mm_1 * MOmegaS1) - mm_1 * Momega 
    MDel_a_2 = (Mwa_2 + mm_2 * MOmegaS2) - mm_2 * Momega 

    b_a_1 = MDel_a_1 * Mwa0_1 * v_a_1 / (MDel_a_1**2. + 1j * (1.5 * mm_1 * Momega  + (ll_1+1.) * MDel_a_1) * Mdrdt_r)
    b_a_2 = MDel_a_2 * Mwa0_2 * v_a_2 / (MDel_a_2**2. + 1j * (1.5 * mm_2 * Momega  + (ll_2+1.) * MDel_a_2) * Mdrdt_r)

    b_a_r_1 = np.real(b_a_1)
    b_a_r_2 = np.real(b_a_2)
    
    H_int_lm_mu_1 = np.zeros(n_mode_1)
    for i in range(n_mode_1):
        for j in range(n_mode_1):
            if ( ll_1[j]==ll_1[i] ) and ( np.abs(mm_1[j]) ==  np.abs(mm_1[i]) ):
                if mm_1[i]==0:
                    H_int_lm_mu_1[i] += 2. * b_a_r_1[j] * v_a_1[i]
                else:
                    H_int_lm_mu_1[i] += b_a_r_1[j] * v_a_1[i]
    H_int_lm_mu_1 *= - E1_mu

    H_int_lm_mu_2 = np.zeros(n_mode_2)
    for i in range(n_mode_2):
        for j in range(n_mode_2):
            if ( ll_2[j]==ll_2[i] ) and ( np.abs(mm_2[j]) == np.abs(mm_2[i]) ):
                if mm_2[i]==0:
                    H_int_lm_mu_2[i] += 2. * b_a_r_2[j] * v_a_2[i]
                else:
                    H_int_lm_mu_2[i] += b_a_r_2[j] * v_a_2[i]
    H_int_lm_mu_2 *= - E2_mu
    
    # corot frame energy (each mode)
    H_ns_mu_1 = E1_mu * (Mwa_1/Mwa0_1) * np.abs(b_a_1)**2.
    # canonical spin
    S_ns_Mmu_1 = H_ns_mu_1 * (mm_1/ Mwa_1)
    
    H_ns_mu_1 = np.sum(H_ns_mu_1)
    S_ns_Mmu_1 = np.sum(S_ns_Mmu_1)

    # inertial frame energy
    H_mode_mu_1 = H_ns_mu_1 + MOmegaS1 * S_ns_Mmu_1
    
    # corot frame energy (each mode)
    H_ns_mu_2 = E2_mu * (Mwa_2/Mwa0_2) * np.abs(b_a_2)**2.
    # canonical spin
    S_ns_Mmu_2 = H_ns_mu_2 * (mm_2/ Mwa_2)
    
    H_ns_mu_2 = np.sum(H_ns_mu_2)
    S_ns_Mmu_2 = np.sum(S_ns_Mmu_2)

    # inertial frame energy
    H_mode_mu_2 = H_ns_mu_2 + MOmegaS2 * S_ns_Mmu_2

    return b_a_1, b_a_2, \
           H_mode_mu_1, S_ns_Mmu_1, H_int_lm_mu_1, \
           H_mode_mu_2, S_ns_Mmu_2, H_int_lm_mu_2


def get_pp_tide_var_init_num(r_M, par, 
                             ll, mm,
                             WI_a_scaled, Mwa, Mwa0, 
                             n_mode_1, n_mode_2, 
                             eob_coeff,
                             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
                             nqc_coeff, 
                             h_tide_mult_coeff_1, h_tide_mult_coeff_2,
                             order=2, dlogvar=1e-4, 
                             phi=0.):
    eta, X1, X2, \
    chi1z, chi2z, \
    MOmegaS1, MOmegaS2, Ces1, Ces2, \
    E1_mu, E2_mu, \
    lam_20_M5_1, lam_30_M7_1, lam_20_M5_2, lam_30_M7_2 \
        = par[:15]
    
    # first estimate tidal energies based on Newtonian physics
    ll_1, ll_2 = ll[:n_mode_1], ll[n_mode_1:]
    mm_1, mm_2 = mm[:n_mode_1], mm[n_mode_1:]
    Mwa_1, Mwa_2 = Mwa[:n_mode_1], Mwa[n_mode_1:]
    Mwa0_1, Mwa0_2 = Mwa0[:n_mode_1], Mwa0[n_mode_1:]    
    
    v_a_1 = X2/X1 * WI_a_scaled[:n_mode_1] / r_M**( ll_1 + 1. )
    v_a_2 = X1/X2 * WI_a_scaled[n_mode_1:] / r_M**( ll_2 + 1. )
    v_a_2 *= (-1.)**(ll_2)
    
    _, _,\
    H_mode_mu_1, S_ns_Mmu_1, H_int_lm_mu_1, \
    H_mode_mu_2, S_ns_Mmu_2, H_int_lm_mu_2 \
        = find_eq_tide_energies_N(r_M, par, 
                            ll, mm,
                            WI_a_scaled, Mwa, Mwa0, 
                            n_mode_1, n_mode_2)

    # also need derivatives of the interaction energy
    dH_int_lm_mu_1 = -(ll+1)[:n_mode_1] * H_int_lm_mu_1/r_M
    dH_int_lm_mu_2 = -(ll+1)[n_mode_1:] * H_int_lm_mu_2/r_M

    Mdrdt_r = -64./5. * eta / r_M**4
    p_r_mu_est = Mdrdt_r * r_M
    
    eob_var = np.hstack((
                    np.array([r_M, 0., phi, np.sqrt(r_M)]), 
                    H_mode_mu_1, S_ns_Mmu_1, H_int_lm_mu_1, 
                    H_mode_mu_2, S_ns_Mmu_2, H_int_lm_mu_2
                    ))

    dvar = (np.abs(eob_var)[:4] + 1e-4) * dlogvar 
        
    # find PI_phi numerically by solving dHdr=0
    def find_PI_phi_Mmu(PI_phi_Mmu):
        if isinstance(PI_phi_Mmu, (list,np.ndarray)):
            PI_phi_Mmu = PI_phi_Mmu[0]
        _eob_var = eob_var.copy()
        _eob_var[3] = PI_phi_Mmu
        dH_dr = compute_dH_1(_eob_var, par, 0, 
                         order=order,
                         dvar = dvar[0], 
                         H_func = get_H_eob_mu, 
                         H_func_args=(eob_coeff, 
                                      ll, mm, n_mode_1, n_mode_2))
        H_eob_mu, \
        H_eff_nts_mu, \
        deltaU, deltaU_pp, deltaR, csi, \
        z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
        dH_tS_dS_1, dH_tS_dS_2 \
            = get_H_eob_mu_full_return(_eob_var, par, 
                     eob_coeff, 
                     ll, mm, n_mode_1, n_mode_2)
    
        # some quantities needed for analytical differentiation
        A_H_eff_nts_mu = deltaU / H_eff_nts_mu
        dH_eob_dH_eff = 1./(eta * H_eob_mu)
        dH_eff_dH_int_lm_1 = np.ones(n_mode_1) \
            * z_I_1 * ( 1./A_H_eff_nts_mu + (_eob_var[1]/deltaU_pp)**2. * A_H_eff_nts_mu)
    
        for i in range(n_mode_1):
            if ll_1[i]==2:
                if mm_1[i]==2:
                    dH_eff_dH_int_lm_1[i] += udz_C_lm[0]
                elif mm_1[i]==0:
                    dH_eff_dH_int_lm_1[i] += udz_C_lm[1]
                elif mm_1[i]==-2:
                    dH_eff_dH_int_lm_1[i] += udz_C_lm[2]
    
        dH_eff_dH_int_lm_2 = np.ones(n_mode_2) \
            * z_I_2 * ( 1./A_H_eff_nts_mu + (_eob_var[1]/deltaU_pp)**2. * A_H_eff_nts_mu)
    
        for i in range(n_mode_2):
            if ll_2[i]==2:
                if mm_2[i]==2:
                    dH_eff_dH_int_lm_2[i] += udz_C_lm[0]
                elif mm_2[i]==0:
                    dH_eff_dH_int_lm_2[i] += udz_C_lm[1]
                elif mm_2[i]==-2:
                    dH_eff_dH_int_lm_2[i] += udz_C_lm[2]
        dH_dr_tot = dH_dr + dH_eob_dH_eff \
                          * (  np.sum(dH_eff_dH_int_lm_1 * dH_int_lm_mu_1)\
                             + np.sum(dH_eff_dH_int_lm_2 * dH_int_lm_mu_2)\
                            )
        
        # resi = dH_dr_tot - eta * 3. * Mdrdt_r * Mdrdt_r * r_M
        resi = dH_dr_tot
        return resi
        
    PI_phi_Mmu = opt.root_scalar(find_PI_phi_Mmu, x0 = eob_var[3])
    PI_phi_Mmu = PI_phi_Mmu.root
    
    p_phi_Mmu = PI_phi_Mmu - S_ns_Mmu_1 - S_ns_Mmu_2
    
    eob_var[3] = PI_phi_Mmu


    # quantities needed for analytical derivatives
    H_eob_mu, \
    H_eff_nts_mu, \
    deltaU, deltaU_pp, deltaR, csi, \
    z_E_1, z_E_2, z_I_1, z_I_2, udz_C_lm, \
    dH_tS_dS_1, dH_tS_dS_2 \
        = get_H_eob_mu_full_return(eob_var, par, 
                 eob_coeff, 
                 ll, mm, n_mode_1, n_mode_2)

    # p_r_mu_tort = Mdrdt_r * r_M * csi
    p_r_mu_tort = Mdrdt_r * r_M / csi * (deltaU_pp/deltaU)**2. * eta * H_eob_mu * H_eff_nts_mu
    eob_var[1] = p_r_mu_tort

    # recalc mode amplitudes with updated p_phi 
    dH_dvar = np.zeros(4)
    for i in range(4):
        if i==2:
            # phi does not show up in H
            continue            
        dH_dvar[i] = compute_dH_1(eob_var, par, i, 
                                  order=order,
                                  dvar = dvar[i], 
                                  H_func = get_H_eob_mu, 
                                  H_func_args=(eob_coeff, 
                                               ll, mm, n_mode_1, n_mode_2))
    Mdrdt_r = csi * dH_dvar[1] / r_M
    # p_r_mu_tort = Mdrdt_r * r_M / csi * (deltaU_pp/deltaU)**2. * eta * H_eob_mu * H_eff_nts_mu

    Momega = dH_dvar[3]

    # some quantities needed for analytical differentiation
    A_H_eff_nts_mu = deltaU / H_eff_nts_mu

    dH_eob_dH_eff = 1./(eta * H_eob_mu)
    dH_eff_dH_mode_1 = z_E_1 * A_H_eff_nts_mu
    dH_eff_dH_mode_2 = z_E_2 * A_H_eff_nts_mu

    dH_eff_nts_dS = - p_phi_Mmu/r_M/r_M * A_H_eff_nts_mu

    dH_eff_dH_int_lm_1 = np.ones(n_mode_1) \
        * z_I_1 * ( 1./A_H_eff_nts_mu + (eob_var[1]/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_1):
        if ll_1[i]==2:
            if mm_1[i]==2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[0]
            elif mm_1[i]==0:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[1]
            elif mm_1[i]==-2:
                dH_eff_dH_int_lm_1[i] += udz_C_lm[2]

    dH_eff_dH_int_lm_2 = np.ones(n_mode_2) \
        * z_I_2 * ( 1./A_H_eff_nts_mu + (eob_var[1]/deltaU_pp)**2. * A_H_eff_nts_mu)

    for i in range(n_mode_2):
        if ll_2[i]==2:
            if mm_2[i]==2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[0]
            elif mm_2[i]==0:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[1]
            elif mm_2[i]==-2:
                dH_eff_dH_int_lm_2[i] += udz_C_lm[2]

    # equilibrium mode amp using derivatives
    MDel_a_1 = dH_eob_dH_eff * ( dH_eff_dH_mode_1 * (Mwa_1 + mm_1 * MOmegaS1) + mm_1 * (dH_eff_nts_dS + dH_tS_dS_1)  )
    MDel_a_2 = dH_eob_dH_eff * ( dH_eff_dH_mode_2 * (Mwa_2 + mm_2 * MOmegaS2) + mm_2 * (dH_eff_nts_dS + dH_tS_dS_2)  )

    b_a_1 =  MDel_a_1 * Mwa0_1 * v_a_1 * (dH_eob_dH_eff * dH_eff_dH_int_lm_1)\
            / (MDel_a_1**2 + 1j * (1.5 * mm_1 * Momega  + (ll_1+1.) * MDel_a_1) * Mdrdt_r)
    b_a_2 =  MDel_a_2 * Mwa0_2 * v_a_2 * (dH_eob_dH_eff * dH_eff_dH_int_lm_2)\
            / (MDel_a_2**2 + 1j * (1.5 * mm_2 * Momega  + (ll_2+1.) * MDel_a_2) * Mdrdt_r)

    # return the final results as the init condition
    pp_tide_var = np.zeros(4+2*n_mode_1+2*n_mode_2) 
    pp_tide_var[0:4] = np.array([r_M, p_r_mu_tort, phi, PI_phi_Mmu]) 
    pp_tide_var[4:4+n_mode_1] = np.real(b_a_1)
    pp_tide_var[4+n_mode_1:4+2*n_mode_1] = np.imag(b_a_1)
    if n_mode_2>0:
        pp_tide_var[4+2*n_mode_1:4+2*n_mode_1+n_mode_2] = np.real(b_a_2)
        pp_tide_var[4+2*n_mode_1+n_mode_2:] = np.imag(b_a_2)
    return pp_tide_var


##############################

@njit(cache=True)
def compute_dH_1(var, par, idx, 
                 order = 2,
                 dvar = 1e-6,
                 H_func=get_H_eob_mu, 
                 H_func_args=None):
    
    var_u = var.copy()
    var_l = var.copy()
    
    if order==4:        
        var_u[idx] += dvar
        H_u = H_func(var_u, par, *H_func_args)        
        var_u[idx] += dvar
        H_uu = H_func(var_u, par, *H_func_args)
                
        var_l[idx] -= dvar
        H_l = H_func(var_l, par, *H_func_args)        
        var_l[idx] -= dvar
        H_ll = H_func(var_l, par, *H_func_args)
        
        dHdvar = (H_ll/12 - 2*H_l/3 + 2*H_u/3 - H_uu/12)/(dvar)
    else:
        var_u[idx] += dvar
        H_u = H_func(var_u, par, *H_func_args)
        
        var_l[idx] -= dvar
        H_l = H_func(var_l, par, *H_func_args)
        
        dHdvar = (H_u - H_l)/(2*dvar)        
    return dHdvar

@njit(cache=True)
def Gamma_lp1_m2js_Gamma_lp1_expanded(ss, lh):
    ss2 = ss * ss
    ss3 = ss2 * ss
    if lh==2:
        Gam_ratio = 1. - 1.84557j * ss \
            - 2.49293 * ss2 + 2.29998j * ss3
        
    elif lh==3:
        Gam_ratio = 1. - 2.51224j * ss - 3.72331 * ss2

    elif lh==4:
        Gam_ratio = 1. - 3.01224j * ss

    else:
        Gam_ratio = 1.

    return Gam_ratio


def get_W_lm(ll, mm):
    W =  (-1)**((ll+mm)/2.) * np.sqrt(4.*np.pi/(2.*ll+1.) * special.factorial(ll+mm) * special.factorial(ll-mm))\
        /(2.**ll * special.factorial((ll+mm)/2.) * special.factorial((ll-mm)/2.))
    return W