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
from . import dynamics as dyn
from . import coeffs as cfs

######################
# top-level wrappers #
######################

def get_hDL_M_modes_vs_t(**kwargs):
    """
    Main code to compute the GW modes as a function of time. 
    
    Except for that the masses are given in solar masses and dt in seconds, all other quantities should be normalized by powers of mass in geometrical units. 

    For NSBH systems, set lam2_l2_M2_5=lam2_l3_M2_7=0
    
    Inputs (replace %i with 1 or 2 for object 1 or 2 in the system; note all tide-related quantities are normalized by the associated component mass): 
        mass%i: Mass of object i=1, 2 in [solar masses]
        spin%iz: z component of oject i's dimensionless spin
        R%i_M%i: radius normalized by mass
        I%i_M%i_3: moment of inertia normalized by its mass^3
        Ces%i: spin-induced quadrupole (1 for BH) 
        lam%i_l2_M%i_5: adiabatic tidal deformability (l=2) normalized by mass^5
        lam%i_l3_M%i_7: adiabatic tidal deformability (l=3) normalized by mass^7
        M%iwa0%i_l2: l=2 f-mode frequency for a non-rotating NS
        C%i_l2: coefficient governing the shift of the l=2 f-modes in the NS frame, wa = wa0 + ma * C * Omega (eq. 11 of the paper)
        C%i_l3: coefficient governing the shift of the l=3 f-modes in the NS frame, wa = wa0 + ma * C * Omega (eq. 11 of the paper)
        ll%i: list of polar quantum numbers of NS eigenmodes (not GW modes!). Should be the same length as the modes to evolve
        mm%i: list of polar quantum numbers of NS eigenmodes (not GW modes!). Should be the same length as the modes to evolve

        dt: time step size in [seconds]
        r_M_init: initial orbital separation in total mass
        phi_init: initial orbital phase in a frame the line of sight is along the x-axis and the orbital AM is along the z-axis. 
        iota: angle between line of sight and the orbital AM (needed only when computing the GW polarizations hp and hc)

        order: 2 or 4. 
            Order of finite-difference methods when computing the numerical derivatives of the Hamiltonian (only used for the orbital part; mode evolution is analytical)
        dlogvar: fractional step size used for finite-differencing the Hamiltonian (not used for \phi as \partial H / \partial \phi=0)
        atol, rtol: ODE numerical tolerances

    Outputs: 
        tt: 1D time array from 0 (where r_M=r_M_init) till contact (r=R1+R2) in [seconds] separated by dt
        hDL_M_modes: (GW strain in modes) * (luminosity distance) / (total mass)
            size=[n_time, 5]
            first dimension for time points, second for GW (l, m) modes
            Hard code the GW modes to be (2,2), (2, 1), (3, 3), (3, 2), (4, 4)
            The first three GW modes include tidal corrections while the last two for PP only. 
    """

    order = kwargs['order']
    dlogvar = kwargs['dlogvar']
    atol = kwargs['atol']
    rtol = kwargs['rtol']

    dt = kwargs['dt']
    
    # pp part
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    chi1z = kwargs['spin1z']
    chi2z = kwargs['spin2z']

    M1 = M1_Ms * Ms
    M2 = M2_Ms * Ms
    rM1 = G*M1/c**2
    tM1 = rM1/c
    SM1 = G*M1*M1/c
    rM2 = G*M2/c**2
    tM2 = rM2/c
    SM2 = G*M2*M2/c

    Mt = M1 + M2
    rMt = rM1 + rM2
    tMt = tM1 + tM2

    X1 = M1/Mt
    X2 = 1.-X1
    eta = X1 * X2
    mu = eta * Mt
    
    Emu = mu * c * c

    # matter effects of M1
    R1_M1 = kwargs['R1_M1']
    I1_M1_3 = kwargs['I1_M1_3']
    Ces1 = kwargs['Ces1']
    lam1_l2_M1_5 = kwargs['lam1_l2_M1_5']
    lam1_l3_M1_7 = kwargs['lam1_l3_M1_7']
    M1wa01_l2 = kwargs['M1wa01_l2']
    M1wa01_l3 = kwargs['M1wa01_l3']
    
    ll1 = kwargs['ll1']
    mm1 = kwargs['mm1']
    C1_l2 = kwargs['C1_l2']
    C1_l3 = kwargs['C1_l3']

    R1 = R1_M1 * rM1
    E1 = G*M1**2./R1
    I1 = I1_M1_3 * M1 * rM1 * rM1
    S1 = chi1z * SM1
    MOmegaS1 = S1 / I1 * tMt # note freq is normed by total mass
    E1_mu = E1 / Emu

    # compile a list of modes
    n_mode1 = len(ll1)
    # mode freqs of a non-spinning NS normed by total mass
    Mwa01 = np.zeros(n_mode1)
    
    # mode freqs of a spinning NS in the NS corot frame normed by total mass
    Mwa1 = np.zeros(n_mode1)

    # overlap integral; not to be confused with moment of inertia
    k1_l2 = 3./2. * lam1_l2_M1_5 / (R1_M1)**5
    Qa1_l2 = np.sqrt(k1_l2 * 5./4./np.pi)
    k1_l3 = 15./2. * lam1_l3_M1_7 / (R1_M1)**7
    Qa1_l3 = np.sqrt(k1_l3 * 7./4./np.pi)
    
    Ia1 = np.zeros(n_mode1)
    Wlm1 = np.zeros(n_mode1)
    
    for i in range(n_mode1):
        Wlm1[i] = np.real(dyn.get_W_lm(ll=ll1[i], mm=mm1[i]))
        if ll1[i]==2:
            Mwa01[i] = M1wa01_l2 / X1
            Mwa1[i] = M1wa01_l2 / X1 + mm1[i] * C1_l2 * MOmegaS1 
            Ia1[i] = Qa1_l2
            
        elif ll1[i] == 3:
            Mwa01[i] = M1wa01_l3 / X1
            Mwa1[i] = M1wa01_l3 / X1 + mm1[i] * C1_l3 * MOmegaS1
            Ia1[i] = Qa1_l3
            
        else:
            raise ValueError("Sorry, the current version only supports l<=3 for NS modes")

    WIa_scaled1 = Wlm1 * Ia1 * (R1/rMt)**(ll1+1)


    # matter effects of M2
    R2_M2 = kwargs['R2_M2']
    I2_M2_3 = kwargs['I2_M2_3']
    Ces2 = kwargs['Ces2']
    lam2_l2_M2_5 = kwargs['lam2_l2_M2_5']
    lam2_l3_M2_7 = kwargs['lam2_l3_M2_7']
    M2wa02_l2 = kwargs['M2wa02_l2']
    M2wa02_l3 = kwargs['M2wa02_l3']
    
    ll2 = kwargs['ll2']
    mm2 = kwargs['mm2']
    C2_l2 = kwargs['C2_l2']
    C2_l3 = kwargs['C2_l3']

    R2 = R2_M2 * rM2
    E2 = G*M2**2./R2
    I2 = I2_M2_3 * M2 * rM2 * rM2
    S2 = chi2z * SM2
    MOmegaS2 = S2 / I2 * tMt # note freq is normed by total mass
    E2_mu = E2 / Emu

    # compile a list of modes
    n_mode2 = len(ll2)
    # mode freqs of a non-spinning NS normed by total mass
    Mwa02 = np.zeros(n_mode2)
    
    # mode freqs of a spinning NS in the NS corot frame normed by total mass
    Mwa2 = np.zeros(n_mode2)

    # overlap integral; not to be confused with moment of inertia
    k2_l2 = 3./2. * lam2_l2_M2_5 / (R2_M2)**5
    Qa2_l2 = np.sqrt(k2_l2 * 5./4./np.pi)
    k2_l3 = 15./2. * lam2_l3_M2_7 / (R2_M2)**7
    Qa2_l3 = np.sqrt(k2_l3 * 7./4./np.pi)
    
    Ia2 = np.zeros(n_mode2)
    Wlm2 = np.zeros(n_mode2)
    
    for i in range(n_mode2):
        Wlm2[i] = np.real(dyn.get_W_lm(ll=ll2[i], mm=mm2[i]))
        if ll2[i]==2:
            Mwa02[i] = M2wa02_l2 / X2
            Mwa2[i] = M2wa02_l2 / X2 + mm2[i] * C2_l2 * MOmegaS2
            Ia2[i] = Qa2_l2
            
        elif ll2[i] == 3:
            Mwa02[i] = M2wa02_l3 / X2
            Mwa2[i] = M2wa02_l3 / X2 + mm2[i] * C2_l3 * MOmegaS2
            Ia2[i] = Qa2_l3
            
        else:
            raise ValueError("Sorry, the current version only supports l<=3 for NS modes")

    WIa_scaled2 = Wlm2 * Ia2 * (R2/rMt)**(ll2+1)

    # combined
    ll = np.hstack((ll1, ll2))
    mm = np.hstack((mm1, mm2))
    WIa_scaled = np.hstack((WIa_scaled1, WIa_scaled2))
    Mwa0 = np.hstack((Mwa01, Mwa02))
    Mwa = np.hstack((Mwa1, Mwa2))

    lam1_l2_M5 = lam1_l2_M1_5 * X1**5.
    lam1_l3_M7 = lam1_l3_M1_7 * X1**7.
    lam2_l2_M5 = lam2_l2_M2_5 * X2**5.
    lam2_l3_M7 = lam2_l3_M2_7 * X2**7.

    # coefficients
    par = np.array([eta, X1, X2, chi1z, chi2z, 
                MOmegaS1, MOmegaS2, Ces1, Ces2, 
                E1_mu, E2_mu, 
                lam1_l2_M5, lam1_l3_M7, lam2_l2_M5, lam2_l3_M7])

    
    eob_coeff = cfs.get_eob_coeffs(par)
    rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff \
        = cfs.get_h_mode_pn_coeffs(par)
    nqc_coeff = cfs.get_nqc_coeffs(par)
    
    h_tide_mult_coeff_1 = cfs.get_h_mode_tide_coeffs(par, M1_deformed=True)
    h_tide_mult_coeff_2 = cfs.get_h_mode_tide_coeffs(par, M1_deformed=False)

    # initial conditions

    r_M_init = kwargs['r_M_init']
    phi_init = kwargs['phi_init']

    pp_tide_var_init = dyn.get_pp_tide_var_init_num(
        r_M_init, par, 
        ll, mm,
        WIa_scaled, Mwa, Mwa0, 
        n_mode1, n_mode2, 
        eob_coeff,
        rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
        nqc_coeff, 
        h_tide_mult_coeff_1, h_tide_mult_coeff_2,
        order=order, dlogvar=dlogvar, 
        phi=phi_init
    )

    t_tot_M_est = 5./(256. * eta) * r_M_init**4.

    # evolve EOB dynamics
    r_in_R1_R2 = 0.8

    @njit
    def terminator(tt, yy):
        r_M = yy[0]
        p_r_mu_tort = yy[1]
        resi = r_M - r_in_R1_R2*(R1+R2)/rMt

        return resi

    terminator.direction = -1
    terminator.terminal = True

    int_func = lambda tt, yy: dyn.evol_sys(tt, yy, par, 
             ll, mm,
             WIa_scaled, Mwa, Mwa0, 
             n_mode1, n_mode2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2,
             order=order, dlogvar=dlogvar)

    sol = integ.solve_ivp(int_func, 
                      t_span=(0/tMt, 2.*t_tot_M_est/tMt), y0=pp_tide_var_init, rtol=rtol, atol=atol, \
                      events=terminator)
    
    # get waveform in modes
    
    # physical time in s
    tt = sol.t * tMt
    
    Momega, vp, hDL_M_modes \
        = dyn.get_hDL_M_from_pp_tide_var(sol.t, sol.y, par, 
             ll, mm,
             WIa_scaled, Mwa, Mwa0, 
             n_mode1, n_mode2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2, 
             r_M_taper_on=1.1*(R1+R2)/rMt, 
             r_M_taper_off = r_in_R1_R2*(R1+R2)/rMt,
             )


    # output
    t_out = np.arange(tt[0], tt[-1], dt)
    n_out = len(t_out)
    hDL_M_modes_out = np.zeros((5, n_out), dtype=np.complex128)
    for i in range(5):
        _hh = hDL_M_modes[i, :]
        _mag = np.abs(_hh)
        _ang = np.unwrap(np.angle(_hh))
        mag_vs_t = interp.CubicSpline(tt, _mag)
        ang_vs_t = interp.CubicSpline(tt, _ang)

        hDL_M_modes_out[i, :] = mag_vs_t(t_out) * np.exp(1j * ang_vs_t(t_out))

    return t_out, hDL_M_modes_out

def get_hDL_M_pc_vs_t(**kwargs):
    """
    Similar to get_hDL_M_modes_vs_t but returns tt, hp, hc
    iota, angle between the line of sight and the orbital AM, will be necessary for this function
    """
    iota = kwargs['iota']
    
    tt, hDL_M_modes = get_hDL_M_modes_vs_t(**kwargs)
    ll_gw = np.array([2, 2, 3, 3, 4])
    mm_gw = np.array([2, 1, 3, 2, 4])

    n_out = len(tt)
    h_pc = np.zeros(n_out, dtype=np.complex128)
    
    for i in range(5):
        _l, _m = ll_gw[i], mm_gw[i]
        _h_pm = hDL_M_modes[i, :]
        _h_nm = np.conj(_h_pm)
        
        # the phi part is accounted for by phi_init of the orbit
        Y_pm = get_s_Ylm_sn2(iota, 0, _l, _m)
        Y_nm = get_s_Ylm_sn2(iota, 0, _l, -_m)

        h_pc += _h_pm * Y_pm + _h_nm * Y_nm

    hDL_M_p = np.real(h_pc)
    hDL_M_c = - np.imag(h_pc)
    return tt, hDL_M_p, hDL_M_c

def get_hDL_M_modes_no_tspin_vs_t(**kwargs):
    """
    Similar to get_hDL_M_modes_no_vs_t with the same inputs and outputs,
    but the tidal spin is zeroed in back reaction to the orbit (essentially the effective Love number treatment)
    """

    order = kwargs['order']
    dlogvar = kwargs['dlogvar']
    atol = kwargs['atol']
    rtol = kwargs['rtol']

    dt = kwargs['dt']
    
    # pp part
    M1_Ms = kwargs['mass1']
    M2_Ms = kwargs['mass2']
    chi1z = kwargs['spin1z']
    chi2z = kwargs['spin2z']

    M1 = M1_Ms * Ms
    M2 = M2_Ms * Ms
    rM1 = G*M1/c**2
    tM1 = rM1/c
    SM1 = G*M1*M1/c
    rM2 = G*M2/c**2
    tM2 = rM2/c
    SM2 = G*M2*M2/c

    Mt = M1 + M2
    rMt = rM1 + rM2
    tMt = tM1 + tM2

    X1 = M1/Mt
    X2 = 1.-X1
    eta = X1 * X2
    mu = eta * Mt
    
    Emu = mu * c * c

    # matter effects of M1
    R1_M1 = kwargs['R1_M1']
    I1_M1_3 = kwargs['I1_M1_3']
    Ces1 = kwargs['Ces1']
    lam1_l2_M1_5 = kwargs['lam1_l2_M1_5']
    lam1_l3_M1_7 = kwargs['lam1_l3_M1_7']
    M1wa01_l2 = kwargs['M1wa01_l2']
    M1wa01_l3 = kwargs['M1wa01_l3']
    
    ll1 = kwargs['ll1']
    mm1 = kwargs['mm1']
    C1_l2 = kwargs['C1_l2']
    C1_l3 = kwargs['C1_l3']

    R1 = R1_M1 * rM1
    E1 = G*M1**2./R1
    I1 = I1_M1_3 * M1 * rM1 * rM1
    S1 = chi1z * SM1
    MOmegaS1 = S1 / I1 * tMt # note freq is normed by total mass
    E1_mu = E1 / Emu

    # compile a list of modes
    n_mode1 = len(ll1)
    # mode freqs of a non-spinning NS normed by total mass
    Mwa01 = np.zeros(n_mode1)
    
    # mode freqs of a spinning NS in the NS corot frame normed by total mass
    Mwa1 = np.zeros(n_mode1)

    # overlap integral; not to be confused with moment of inertia
    k1_l2 = 3./2. * lam1_l2_M1_5 / (R1_M1)**5
    Qa1_l2 = np.sqrt(k1_l2 * 5./4./np.pi)
    k1_l3 = 15./2. * lam1_l3_M1_7 / (R1_M1)**7
    Qa1_l3 = np.sqrt(k1_l3 * 7./4./np.pi)
    
    Ia1 = np.zeros(n_mode1)
    Wlm1 = np.zeros(n_mode1)
    
    for i in range(n_mode1):
        Wlm1[i] = np.real(dyn.get_W_lm(ll=ll1[i], mm=mm1[i]))
        if ll1[i]==2:
            Mwa01[i] = M1wa01_l2 / X1
            Mwa1[i] = M1wa01_l2 / X1 + mm1[i] * C1_l2 * MOmegaS1 
            Ia1[i] = Qa1_l2
            
        elif ll1[i] == 3:
            Mwa01[i] = M1wa01_l3 / X1
            Mwa1[i] = M1wa01_l3 / X1 + mm1[i] * C1_l3 * MOmegaS1
            Ia1[i] = Qa1_l3
            
        else:
            raise ValueError("Sorry, the current version only supports l<=3 for NS modes")

    WIa_scaled1 = Wlm1 * Ia1 * (R1/rMt)**(ll1+1)


    # matter effects of M2
    R2_M2 = kwargs['R2_M2']
    I2_M2_3 = kwargs['I2_M2_3']
    Ces2 = kwargs['Ces2']
    lam2_l2_M2_5 = kwargs['lam2_l2_M2_5']
    lam2_l3_M2_7 = kwargs['lam2_l3_M2_7']
    M2wa02_l2 = kwargs['M2wa02_l2']
    M2wa02_l3 = kwargs['M2wa02_l3']
    
    ll2 = kwargs['ll2']
    mm2 = kwargs['mm2']
    C2_l2 = kwargs['C2_l2']
    C2_l3 = kwargs['C2_l3']

    R2 = R2_M2 * rM2
    E2 = G*M2**2./R2
    I2 = I2_M2_3 * M2 * rM2 * rM2
    S2 = chi2z * SM2
    MOmegaS2 = S2 / I2 * tMt # note freq is normed by total mass
    E2_mu = E2 / Emu

    # compile a list of modes
    n_mode2 = len(ll2)
    # mode freqs of a non-spinning NS normed by total mass
    Mwa02 = np.zeros(n_mode2)
    
    # mode freqs of a spinning NS in the NS corot frame normed by total mass
    Mwa2 = np.zeros(n_mode2)

    # overlap integral; not to be confused with moment of inertia
    k2_l2 = 3./2. * lam2_l2_M2_5 / (R2_M2)**5
    Qa2_l2 = np.sqrt(k2_l2 * 5./4./np.pi)
    k2_l3 = 15./2. * lam2_l3_M2_7 / (R2_M2)**7
    Qa2_l3 = np.sqrt(k2_l3 * 7./4./np.pi)
    
    Ia2 = np.zeros(n_mode2)
    Wlm2 = np.zeros(n_mode2)
    
    for i in range(n_mode2):
        Wlm2[i] = np.real(dyn.get_W_lm(ll=ll2[i], mm=mm2[i]))
        if ll2[i]==2:
            Mwa02[i] = M2wa02_l2 / X2
            Mwa2[i] = M2wa02_l2 / X2 + mm2[i] * C2_l2 * MOmegaS2
            Ia2[i] = Qa2_l2
            
        elif ll2[i] == 3:
            Mwa02[i] = M2wa02_l3 / X2
            Mwa2[i] = M2wa02_l3 / X2 + mm2[i] * C2_l3 * MOmegaS2
            Ia2[i] = Qa2_l3
            
        else:
            raise ValueError("Sorry, the current version only supports l<=3 for NS modes")

    WIa_scaled2 = Wlm2 * Ia2 * (R2/rMt)**(ll2+1)

    # combined
    ll = np.hstack((ll1, ll2))
    mm = np.hstack((mm1, mm2))
    WIa_scaled = np.hstack((WIa_scaled1, WIa_scaled2))
    Mwa0 = np.hstack((Mwa01, Mwa02))
    Mwa = np.hstack((Mwa1, Mwa2))

    lam1_l2_M5 = lam1_l2_M1_5 * X1**5.
    lam1_l3_M7 = lam1_l3_M1_7 * X1**7.
    lam2_l2_M5 = lam2_l2_M2_5 * X2**5.
    lam2_l3_M7 = lam2_l3_M2_7 * X2**7.

    # coefficients
    par = np.array([eta, X1, X2, chi1z, chi2z, 
                MOmegaS1, MOmegaS2, Ces1, Ces2, 
                E1_mu, E2_mu, 
                lam1_l2_M5, lam1_l3_M7, lam2_l2_M5, lam2_l3_M7])

    
    eob_coeff = cfs.get_eob_coeffs(par)
    rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff \
        = cfs.get_h_mode_pn_coeffs(par)
    nqc_coeff = cfs.get_nqc_coeffs(par)
    
    h_tide_mult_coeff_1 = cfs.get_h_mode_tide_coeffs(par, M1_deformed=True)
    h_tide_mult_coeff_2 = cfs.get_h_mode_tide_coeffs(par, M1_deformed=False)

    # initial conditions

    r_M_init = kwargs['r_M_init']
    phi_init = kwargs['phi_init']

    pp_tide_var_init = dyn.get_pp_tide_var_init_num(
        r_M_init, par, 
        ll, mm,
        WIa_scaled, Mwa, Mwa0, 
        n_mode1, n_mode2, 
        eob_coeff,
        rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
        nqc_coeff, 
        h_tide_mult_coeff_1, h_tide_mult_coeff_2,
        order=order, dlogvar=dlogvar, 
        phi=phi_init
    )

    t_tot_M_est = 5./(256. * eta) * r_M_init**4.

    # evolve EOB dynamics
    r_in_R1_R2 = 0.8

    @njit
    def terminator(tt, yy):
        r_M = yy[0]
        p_r_mu_tort = yy[1]
        resi = r_M - r_in_R1_R2*(R1+R2)/rMt

        return resi

    terminator.direction = -1
    terminator.terminal = True

    int_func = lambda tt, yy: dyn.evol_sys_no_tspin_in_br(tt, yy, par, 
             ll, mm,
             WIa_scaled, Mwa, Mwa0, 
             n_mode1, n_mode2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2,
             order=order, dlogvar=dlogvar)

    sol = integ.solve_ivp(int_func, 
                      t_span=(0/tMt, 2.*t_tot_M_est/tMt), y0=pp_tide_var_init, rtol=rtol, atol=atol, \
                      events=terminator)
    
    # get waveform in modes
    
    # physical time in s
    tt = sol.t * tMt
    
    Momega, vp, hDL_M_modes \
        = dyn.get_hDL_M_from_pp_tide_var(sol.t, sol.y, par, 
             ll, mm,
             WIa_scaled, Mwa, Mwa0, 
             n_mode1, n_mode2, 
             eob_coeff,
             rholm_coeff, rholm_log_coeff, cflmS_coeff, deltalm_coeff, 
             nqc_coeff, 
             h_tide_mult_coeff_1, h_tide_mult_coeff_2, 
             r_M_taper_on=1.1*(R1+R2)/rMt, 
             r_M_taper_off = r_in_R1_R2*(R1+R2)/rMt,
             )


    # output
    t_out = np.arange(tt[0], tt[-1], dt)
    n_out = len(t_out)
    hDL_M_modes_out = np.zeros((5, n_out), dtype=np.complex128)
    for i in range(5):
        _hh = hDL_M_modes[i, :]
        _mag = np.abs(_hh)
        _ang = np.unwrap(np.angle(_hh))
        mag_vs_t = interp.CubicSpline(tt, _mag)
        ang_vs_t = interp.CubicSpline(tt, _ang)

        hDL_M_modes_out[i, :] = mag_vs_t(t_out) * np.exp(1j * ang_vs_t(t_out))

    return t_out, hDL_M_modes_out

def get_hDL_M_pc_no_tspin_vs_t(**kwargs):
    """
    Get GW polarizations when the tidal spin is turned off in orbital back reactions.
    """
    iota = kwargs['iota']
    
    tt, hDL_M_modes = get_hDL_M_modes_no_tspin_vs_t(**kwargs)
    ll_gw = np.array([2, 2, 3, 3, 4])
    mm_gw = np.array([2, 1, 3, 2, 4])

    n_out = len(tt)
    h_pc = np.zeros(n_out, dtype=np.complex128)
    
    for i in range(5):
        _l, _m = ll_gw[i], mm_gw[i]
        _h_pm = hDL_M_modes[i, :]
        _h_nm = np.conj(_h_pm)
        
        # the phi part is accounted for by phi_init of the orbit
        Y_pm = get_s_Ylm_sn2(iota, 0, _l, _m)
        Y_nm = get_s_Ylm_sn2(iota, 0, _l, -_m)

        h_pc += _h_pm * Y_pm + _h_nm * Y_nm

    hDL_M_p = np.real(h_pc)
    hDL_M_c = - np.imag(h_pc)
    return tt, hDL_M_p, hDL_M_c
        

@njit(fastmath=True, cache=True)
def get_s_Ylm_sn2(th_s, phi_s, l, m):
    """
    https://lscsoft.docs.ligo.org/lalsuite/lal/_spherical_harmonics_8c_source.html
    """
    c_th = np.cos(th_s)
    s_th = np.sin(th_s)
    
    if l==2:
        if m==-2:
            YY = 0.15769578262626002 * ( 1.0 - c_th )*( 1.0 - c_th )
        elif m==-1:
            YY = 0.31539156525252005 * s_th *( 1.0 - c_th )
        elif m==0:
            YY = 0.3862742020231896 * s_th * s_th
        elif m==1:
            YY = 0.31539156525252005 * s_th * ( 1.0 + c_th )
        elif m==2:
            YY = 0.15769578262626002 * ( 1.0 + c_th )*( 1.0 + c_th )
            
    elif l==3:
        if m==-3:
            YY = 1.828183197857863 * np.cos(th_s/2.) * pow(np.sin(th_s/2.), 5.0)
        elif m==-2:
            YY = 0.7463526651802308 * (2.0 + 3.0*c_th) * pow(np.sin(th_s/2.), 4.0)
        elif m==-1:
            YY = 0.07375544874083044 * (s_th + 4.0*np.sin(2.0*th_s) - 3.0*np.sin(3.0*th_s))
        elif m==0:
            YY =  1.0219854764332823 * c_th * pow(s_th, 2.0) 
        elif m==1:
            YY = -0.07375544874083044 * (s_th - 4.0*np.sin(2.0*th_s) - 3.0*np.sin(3.0*th_s))
        elif m==2:
            YY = 0.7463526651802308 * pow(np.cos(th_s/2.), 4.0) * (-2.0 + 3.0*c_th)
        elif m==3:
            YY = -1.828183197857863 * pow(np.cos(th_s/2.),5.0) * np.sin(th_s/2.0)
    
    elif l==4:
        if m==-4:
            YY = 4.478115991081385 * pow(np.cos(th_s/2.0),2.0) * pow(np.sin(th_s/2.0),6.0)
        elif m==-3:
            YY = 3.1665061842335644* np.cos(th_s/2.0)*(1.0 + 2.0*c_th) * pow(np.sin(th_s/2.0),5.0)
        elif m==-2:
            YY = (0.42314218766081724*(9.0 + 14.0*c_th + 7.0*np.cos(2.0*th_s)) * pow(np.sin(th_s/2.0),4.0))
        elif m==-1:
            YY = 0.03740083878763432*(3.0*s_th + 2.0*np.sin(2.0*th_s) + 7.0*np.sin(3.0*th_s) - 7.0*np.sin(4.0*th_s))
        elif m==0:
            YY = 0.1672616358893223*(5.0 + 7.0*np.cos(2.0*th_s)) * s_th * s_th
        elif m==1:
            YY = 0.03740083878763432*(3.0*s_th - 2.0*np.sin(2.0*th_s) + 7.0*np.sin(3.0*th_s) + 7.0*np.sin(4.0*th_s))
        elif m==2:
            YY = (0.42314218766081724*pow(np.cos(th_s/2.0),4.0)*(9.0 - 14.0*c_th + 7.0*np.cos(2.0*th_s)))
        elif m==3:
            YY = -3.1665061842335644*pow(np.cos(th_s/2.0),5.0)*(-1.0 + 2.0*c_th)*np.sin(th_s/2.0)
        elif m==4:
            YY = 4.478115991081385 * pow(np.cos(th_s/2.0),6.0)*pow(np.sin(th_s/2.0),2.0)
        
    else:
        YY = 0j
        
    YY *= np.exp(1j*m*phi_s)
        
    return YY
