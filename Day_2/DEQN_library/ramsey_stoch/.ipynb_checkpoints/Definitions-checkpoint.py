import tensorflow as tf
import numpy as np
import Parameters
import PolicyState
import State

# --------------------------------------------------------------------------- #
# Extract parameters
# --------------------------------------------------------------------------- #
# DICE specific parameters
Tstep, Version = Parameters.Tstep, Parameters.Version

# Logarithmic time transformation
vartheta = Parameters.vartheta

# Economic parameters
alpha, delta = Parameters.alpha, Parameters.delta

# Population
L0, Linfty, deltaL = Parameters.L0, Parameters.Linfty, Parameters.deltaL

# TFP shock on productivity
A0hat, gA0hat, deltaA = Parameters.A0hat, Parameters.gA0hat, Parameters.deltaA

# Carbon intensity
sigma0, gSigma0, deltaSigma = Parameters.sigma0, Parameters.gSigma0, \
    Parameters.deltaSigma

# Mitigation
theta2, pback, gback = Parameters.theta2, Parameters.pback, Parameters.gback

# Land emissions
ELand0, deltaLand = Parameters.ELand0, Parameters.deltaLand

# Exogenous forcings
fex0, fex1, Tyears = Parameters.fex0, Parameters.fex1, Parameters.Tyears

# Climate damage function
pi1 = Parameters.pi1

# Carbon mass transitions
phi12_, phi23_, MATeq, MUOeq, MLOeq = Parameters.phi12_, Parameters.phi23_, \
    Parameters.MATeq, Parameters.MUOeq, Parameters.MLOeq

# Temperature exchange
varphi1_, varphi3_, varphi4_ = Parameters.varphi1_, Parameters.varphi3_, \
    Parameters.varphi4_

# Preindustrial carbon concentration in the atmosphere
MATbase = Parameters.MATbase


# --------------------------------------------------------------------------- #
# Real and computational time periods
# --------------------------------------------------------------------------- #
def tau2t(state, policy_state):
    """Scale back from the computational time tau to the real time t."""
    _t = - tf.math.log(1 - State.taux(state)) / vartheta
    return _t


def tau2tauplus(state, policy_state):
    """Update the computational time tau based on the current real time t."""
    _t = tau2t(state, policy_state)  # Current real time
    _tplus = _t + tf.ones_like(_t)  # Real time t + 1

    _tauplus = 1 - tf.math.exp(- vartheta * _tplus)  # Computational time tau+1
    return _tauplus


# --------------------------------------------------------------------------- #
# Exogenous parameters
# --------------------------------------------------------------------------- #
def tfp(state, policy_state):
    """Deterministic TFP shock on the labor-argumented production function."""
    _t = tau2t(state, policy_state)

    _tfp = A0hat * tf.math.exp(gA0hat * (1 - tf.math.exp(-deltaA*_t)) / deltaA)
    return _tfp


def gr_tfp(state, policy_state):
    """Annual growth rate of the deterministic TFP shock [-/year]."""
    _t = tau2t(state, policy_state)

    _gr_tfp = gA0hat * tf.math.exp(-deltaA * _t)
    return _gr_tfp


def lab(state, policy_state):
    """World population [million]."""
    _t = tau2t(state, policy_state)

    _lab = L0 + (Linfty - L0) * (1 - tf.math.exp(-deltaL * _t))
    return _lab


def gr_lab(state, policy_state):
    """Annual growth rate of the world population [-/year]."""
    _t = tau2t(state, policy_state)

    _gr_lab = deltaL / ((Linfty / (Linfty - L0)) * tf.math.exp(deltaL*_t) - 1)
    return _gr_lab


def betat(state, policy_state):
    """Growth adjusted discout factor."""
    # _psix = Parameters.psi0
    # _rhox = Parameters.rho0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
#     _rhox = State.rhox(state)
    _rhox = rhox_trans(state, policy_state)
    _gr_tfp = gr_tfp(state, policy_state)
    _gr_lab = gr_lab(state, policy_state)

    _betat = tf.math.exp(- _rhox + (1 - 1/_psix) * _gr_tfp + _gr_lab)
    return _betat

# def psix(state, policy_state):
#     _rhox = State.rhox(state)
#     _psix = 0.0217/(0.02946667 - _rhox)
#     return _psix


def sigma(state, policy_state):
    """Carbon intensity."""
    _t = tau2t(state, policy_state)
    if Version == '2016':
        _sigma = sigma0 * tf.math.exp(
            gSigma0 / np.log(1 + deltaSigma) * ((1 + deltaSigma)**_t - 1))
    else:
        _sigma = sigma0 * tf.math.exp(
            gSigma0 * (1 - tf.math.exp(-deltaSigma * _t)) / deltaSigma)
    return _sigma


def theta1(state, policy_state):
    """Cost coefficient of carbon mitigation."""
    _t = tau2t(state, policy_state)
    _sigma = sigma(state, policy_state)
    if Version == '2016':
        c2co2 = Parameters.c2co2
        _theta1 = pback * (1000 * c2co2 * _sigma) * \
            tf.math.exp(-gback * _t) / theta2
    else:
        _theta1 = pback * (1000 * _sigma) * (
            1 + tf.math.exp(-gback * _t)) / theta2
    return _theta1


def Eland(state, policy_state):
    """Natural carbon emission."""
    _t = tau2t(state, policy_state)

    _Eland = ELand0 * tf.math.exp(-deltaLand * _t)
    return _Eland


def Fex(state, policy_state):
    """External radiative forcing."""
    _t = tau2t(state, policy_state)
    Year = int(Tyears)

    _Fex = fex0 + (1 / Year) * (fex1 - fex0) * tf.math.minimum(_t, Year)
    return _Fex


def phi12(state, policy_state):
    """Mass of carbon transmission."""
    return phi12_


def phi23(state, policy_state):
    """Mass of carbon transmission."""
    return phi23_


def phi21(state, policy_state):
    """Mass of carbon transmission."""
    # for exact cjl version return should be rounded by 4 digits
    # for 2007 no need to round
    if Version == 'cjl':
        return np.round(MATeq / MUOeq * phi12_, 2)
    else:
        return MATeq / MUOeq * phi12_


def phi32(state, policy_state):
    """Mass of carbon transmission."""
    # for exact cjl version return should be rounded by 4 digits
    # for 2007 no need to round
    if Version == 'cjl':
        return np.round(MUOeq / MLOeq * phi23_, 5)
    else:
        return MUOeq / MLOeq * phi23_


def varphi21(state, policy_state):
    """varphi21.

    For exact cjl version return should be rounded by 4 digits.
    for 2007 no need to round
    """
    if Version == 'cjl':
        return np.round(varphi1_ * varphi3_, 4)
    else:
        return varphi1_ * varphi3_


def varphi4(state, policy_state):
    """varphi4."""
    return varphi4_


def varphi1(state, policy_state):
    """varphi1."""
    return varphi1_


def xi2(state, policy_state):
    """xi2.

    For exact cjl version return should be rounded by 4 digits
    For 2007 no need to round
    """
    # _f2xco2x = State.f2xco2x(state)
    _f2xco2x = Parameters.f2xco20
    _t2xco2x = State.t2xco2x(state)
#     _t2xco2x = Parameters.t2xco20
    if Version == 'cjl':
        return np.round(varphi1_ * _f2xco2x / _t2xco2x, 3)
    else:
        return varphi1_ * _f2xco2x / _t2xco2x


# --------------------------------------------------------------------------- #
# Shcok and value function variables
# --------------------------------------------------------------------------- #
def zeta(state, policy_state):
    """Stochastic shock on production."""
    _zeta = tf.math.exp(State.zeta_logx(state))
    return _zeta


def zeta_tilde(state, policy_state):
    """Stochastic shock on production in a sotationary form."""
    _zeta_tilde = tf.math.exp(State.zetatilde_logx(state))
    return _zeta_tilde


def vnorm_tilde(state, policy_state):
    """Define normalized value function in a statinary fornm."""
    _vnorm = tf.math.exp(PolicyState.vlog_tildey(policy_state))
    return _vnorm

# --------------------------------------------------------------------------- #
# UQ parameters transformations
# --------------------------------------------------------------------------- #

def rhox_trans(state, policy_state):
    """Transforming rho to meaninigful units"""
    _rhox = State.rhox(state)
    _rhox_trans = _rhox
    return _rhox_trans

def pi2x_trans(state, policy_state):
    """Transforming pi2x to meaninigful units"""
    _pi2x = State.pi2x(state)
    _pi2x_trans = _pi2x
    return _pi2x_trans


# --------------------------------------------------------------------------- #
# Economic variables
# --------------------------------------------------------------------------- #


def lambd(state, policy_state):
    """Lagrange multiplier wrt. the budget constraint."""
    # _psix = Parameters.psi0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
    _con_tildey = PolicyState.con_tildey(policy_state)

    _lambd = (1 - 1/_psix) * _con_tildey**(-1/_psix)
    return _lambd


def lambd_mu(state, policy_state):
    """KKT multiplier wrt. the budget constraint."""
    _k_tildex = State.k_tildex(state)
    _nuATy = PolicyState.nuATy(policy_state)
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)
    _sigma = sigma(state, policy_state)
#     _Omega = Omega(state, policy_state)
    _Omega = Omega_Nordhaus(state, policy_state)
    _Theta_prime = Theta_prime(state, policy_state)
    _zeta = zeta(state, policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)
    _lambd = lambd(state, policy_state)

    _lambd_mu = - _lambd * _Theta_prime * _Omega * _zeta_tilde**(1 - alpha) \
        * _k_tildex**alpha - (-_nuATy) * _sigma * _zeta * _tfp * _lab \
        * _zeta_tilde**(-alpha) * _k_tildex**alpha
    return _lambd_mu


def Theta(state, policy_state):
    """Mitigation cost function."""
    _muy = PolicyState.muy(policy_state)
    _theta1 = theta1(state, policy_state)

    _Theta = _theta1 * _muy**theta2
    return _Theta


def Theta_prime(state, policy_state):
    """First derivative of the mitigation cost function wrt. mu."""
    _muy = PolicyState.muy(policy_state)
    _theta1 = theta1(state, policy_state)

    _Theta_prime = _theta1 * theta2 * _muy**(theta2 - 1)
    return _Theta_prime


# def Omega(state, policy_state):
#     """Climate damage function with climate tipping in Weitzman (2012)."""
#     _TAT = State.TATx(state)
#     # _TP = 3.0405  # There is no random walk climate tipping
#     _TP = State.TPx(state)  # Climate tipping state

#     _Omega = 1 / (
#         1 + (1 / 20.46 * _TAT)**2 + (1 / (2 * _TP) * _TAT)**6.754)
#     return _Omega


# def dOmegadTAT(state, policy_state):
#     """First derivative of the climate damage function with climate tipping."""
#     _TAT = State.TATx(state)
#     # _TP = 3.0405  # There is not random walk climate tipping
#     _TP = State.TPx(state)

#     _dOmegadTAT = - (
#         2 * (1 / 20.46 * _TAT) * 1 / 20.46
#         + 6.754 * (1 / (2*_TP) * _TAT)**(6.754-1) * 1 / (2 * _TP)
#     ) / (1 + (1 / 20.46 * _TAT)**2 + (1 / (2 * _TP) * _TAT)**6.754)**2
#     return _dOmegadTAT


def Omega_Nordhaus(state, policy_state):
    """Climate damage function in Nordhaus (2012)."""
    _TAT = State.TATx(state)
#     _pi2x = State.pi2x(state)
    _pi2x = pi2x_trans(state, policy_state)


    _Omega_Nordhaus = 1 / (1 + pi1 * _TAT + _pi2x * _TAT**2)
    return _Omega_Nordhaus


def dOmegadTAT_Nordhaus(state, policy_state):
    """First derivative of the climate damage function in Nordhaus (2012)."""
    _TAT = State.TATx(state)
#     _pi2x = State.pi2x(state)
    _pi2x = pi2x_trans(state, policy_state)

    _dOmegadTAT_Nordhaus = - (pi1 + 2 * _pi2x * _TAT) / (
        1 + pi1 * _TAT + _pi2x * _TAT**2)**2
    return _dOmegadTAT_Nordhaus


def ygross_tilde(state, policy_state):
    """Gross production in a stationary form."""
    _k_tildex = State.k_tildex(state)
    _zeta_tilde = zeta_tilde(state, policy_state)

    _ygross_tilde = _zeta_tilde**(1 - alpha) * _k_tildex**alpha
    return _ygross_tilde


def ynet_tilde(state, policy_state):
    """Net production in a stationary form.

    Climate damage function Omega is included.
    """
#     _ynet_tilde = Omega(state, policy_state) * ygross_tilde(state, policy_state)
    _ynet_tilde = Omega_Nordhaus(state, policy_state) * ygross_tilde(state, policy_state)
    return _ynet_tilde


def Eind(state, policy_state):
    """Industrial CO2 emission [1000 GtC]."""
    _k_tildex = State.k_tildex(state)
    _muy = PolicyState.muy(policy_state)
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)
    _sigma = sigma(state, policy_state)
    _zeta = zeta(state, policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)

    _Eind = (1 - _muy) * _sigma * _zeta * _tfp * _lab * _zeta_tilde**(-alpha) \
        * _k_tildex**alpha
    return _Eind


def carbontax(state, policy_state):
    """Optimal carbon tax [USD/tC] (Cai and Lontzek, 2019).

    Note that the carbon intensity sigma is defined in the unit of 1000 GtC.
    """
    _muy = PolicyState.muy(policy_state)
    _theta1 = theta1(state, policy_state)
    _sigma = sigma(state, policy_state)
    if Version == '2016':
        c2co2 = Parameters.c2co2
        _carbontax = (_theta1 * theta2 * _muy**(theta2 - 1)) / _sigma / c2co2
    else:
        _carbontax = (_theta1 * theta2 * _muy**(theta2 - 1)) / _sigma
    return _carbontax


def scc(state, policy_state):
    """Social cost of carbon [USD/tC].

    Note that MATx is in 1000 GtC, thus 1000 in the numerator is cancelled.
    """
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)
    _dvdk_tilde_psi = dvdk_tilde_psi(state, policy_state)
    _dvdMAT_tilde_psi = dvdMAT_tilde_psi(state, policy_state)
    _zeta = zeta(state, policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)

    if Version == '2016':
        c2co2 = Parameters.c2co2
        _scc = - _dvdMAT_tilde_psi / _dvdk_tilde_psi * _tfp * _lab * \
            (_zeta/_zeta_tilde) / c2co2
    else:
        _scc = - _dvdMAT_tilde_psi / _dvdk_tilde_psi * _tfp * _lab * \
            (_zeta/_zeta_tilde)
    return _scc


def inv2ynet(state, policy_state):
    """Ratio of investment to net output [%]."""
    _inv2ynet = PolicyState.inv_tildey(policy_state) \
        / ynet_tilde(state, policy_state) * 100
    return _inv2ynet


def con2ynet(state, policy_state):
    """Ratio of consumption to net output [%]."""
    _con2ynet = PolicyState.con_tildey(policy_state) \
        / ynet_tilde(state, policy_state) * 100
    return _con2ynet


def Theta2ynet(state, policy_state):
    """Ratio of mitigation expenditure to net output [%]."""
    _Theta2ynet = Theta(state, policy_state) * 100
    return _Theta2ynet


def scc2ynet(state, policy_state):
    """Ratio of the social cost of carbon to gross output [%].

    The ratio is reported as in (scc / 1000) / Ygross in order to have
    the same units used in Golosov et al. (2014), where Ynet = AL * ynet.
    """
    _zeta = zeta(state, policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)

    _scc2ynet = scc(state, policy_state) / 1000 / (
        tfp(state, policy_state) * lab(state, policy_state)
        * (_zeta/_zeta_tilde) * ynet_tilde(state, policy_state)) * 100
    return _scc2ynet


# --------------------------------------------------------------------------- #
# First-derivatives of the value function
# --------------------------------------------------------------------------- #
def dvdk_tilde_psi(state, policy_state):
    """Envelope theorem wrt. kx."""
    # _psix = Parameters.psi0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
    _k_tildex = State.k_tildex(state)
    _muy = PolicyState.muy(policy_state)
    _nuATy = PolicyState.nuATy(policy_state)
    _vlog_tildey = PolicyState.vlog_tildey(policy_state)
    _zeta = zeta(state, policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)
    _lambd = lambd(state, policy_state)
    _Theta = Theta(state, policy_state)
#     _Omega = Omega(state, policy_state)
    _Omega = Omega_Nordhaus(state, policy_state)
    _sigma = sigma(state, policy_state)
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)

    _dvdk_tilde_psi = tf.math.exp(1/_psix * _vlog_tildey) * (
        _lambd * ((1 - _Theta) * _Omega * alpha * _zeta_tilde**(1 - alpha)
                  * _k_tildex**(alpha - 1)
                  + (1 - delta))
        + (-_nuATy) * (1 - _muy) * _sigma * _zeta * _tfp * _lab * alpha
        * _zeta_tilde**(-alpha) * _k_tildex**(alpha - 1)
    )
    return _dvdk_tilde_psi


def dvdMAT_tilde_psi(state, policy_state):
    """Envelope theorem wrt. MATx."""
    # _psix = Parameters.psi0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
    # _f2xco2x = State.f2xco2x(state)
    _f2xco2x = Parameters.f2xco20
    _MATx = State.MATx(state)
    _nuATy = PolicyState.nuATy(policy_state)
    _nuUOy = PolicyState.nuUOy(policy_state)
    _etaATy = PolicyState.etaATy(policy_state)
    _vlog_tildey = PolicyState.vlog_tildey(policy_state)
    _phi12 = phi12(state, policy_state)
    _varphi1 = varphi1(state, policy_state)

    _dvdMAT_tilde_psi = tf.math.exp(1/_psix * _vlog_tildey) * (
        (-_nuATy) * (1 - _phi12) + _nuUOy * _phi12
        + _etaATy * _varphi1 * _f2xco2x / (tf.math.log(2.) * _MATx)
    )
    return _dvdMAT_tilde_psi


def dvdMUO_tilde_psi(state, policy_state):
    """Envelope theorem wrt. MUOx."""
    # _psix = Parameters.psi0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
    _nuATy = PolicyState.nuATy(policy_state)
    _nuUOy = PolicyState.nuUOy(policy_state)
    _nuLOy = PolicyState.nuLOy(policy_state)
    _vlog_tildey = PolicyState.vlog_tildey(policy_state)
    _phi21 = phi21(state, policy_state)
    _phi23 = phi23(state, policy_state)

    _dvdMUO_tilde_psi = tf.math.exp(1/_psix * _vlog_tildey) * (
        (-_nuATy) * _phi21 + _nuUOy * (1 - _phi21 - _phi23) + _nuLOy * _phi23
    )
    return _dvdMUO_tilde_psi


def dvdMLO_tilde_psi(state, policy_state):
    """Envelope theorem wrt. MLOx."""
    # _psix = Parameters.psi0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
    _nuUOy = PolicyState.nuUOy(policy_state)
    _nuLOy = PolicyState.nuLOy(policy_state)
    _vlog_tildey = PolicyState.vlog_tildey(policy_state)
    _phi32 = phi32(state, policy_state)

    _dvdMLO_tilde_psi = tf.math.exp(1/_psix * _vlog_tildey) * (
        _nuUOy * _phi32 + _nuLOy * (1 - _phi32)
    )
    return _dvdMLO_tilde_psi


def dvdTAT_tilde_psi(state, policy_state):
    """Envelope theorem wrt. TATx."""
    # _psix = Parameters.psi0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
    _k_tildex = State.k_tildex(state)
    _etaATy = PolicyState.etaATy(policy_state)
    _etaOCy = PolicyState.etaOCy(policy_state)
    _vlog_tildey = PolicyState.vlog_tildey(policy_state)
    _lambd = lambd(state, policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)
    _Theta = Theta(state, policy_state)
    _dOmegadTAT = dOmegadTAT_Nordhaus(state, policy_state)
#     _dOmegadTAT = dOmegadTAT(state, policy_state)
    _varphi4 = varphi4(state, policy_state)
    _varphi21 = varphi21(state, policy_state)
    _xi2 = xi2(state, policy_state)

    _dvdTAT_tilde_psi = tf.math.exp(1/_psix * _vlog_tildey) * (
        _lambd * (1 - _Theta) * _dOmegadTAT * _zeta_tilde**(1 - alpha)
        * _k_tildex**alpha
        + _etaATy * (1 - _varphi21 - _xi2) + _etaOCy * _varphi4
    )
    return _dvdTAT_tilde_psi


def dvdTOC_tilde_psi(state, policy_state):
    """Envelope theorem wrt. TOCx."""
    # _psix = Parameters.psi0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
    _etaATy = PolicyState.etaATy(policy_state)
    _etaOCy = PolicyState.etaOCy(policy_state)
    _vlog_tildey = PolicyState.vlog_tildey(policy_state)
    _varphi4 = varphi4(state, policy_state)
    _varphi21 = varphi21(state, policy_state)

    _dvdTOC_psi = tf.math.exp(1/_psix * _vlog_tildey) * (
        _etaATy * _varphi21 + _etaOCy * (1 - _varphi4)
    )
    return _dvdTOC_psi


# --------------------------------------------------------------------------- #
# State variables in period t + 1
# --------------------------------------------------------------------------- #
def ktilde_plus(state, policy_state):
    """Capital stock tomorrow in a stationary format."""
    _k_tildex = State.k_tildex(state)
    _inv_tildey = PolicyState.inv_tildey(policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)
    _gr_tfp = gr_tfp(state, policy_state)
    _gr_lab = gr_lab(state, policy_state)

    _ktilde_plus = ((1 - delta) * _k_tildex + _inv_tildey) / (
        tf.math.exp(_gr_tfp + _gr_lab) * _zeta_tilde)
    return _ktilde_plus


def MATplus(state, policy_state):
    """Carbon mass in the atmosphere."""
    _k_tildex = State.k_tildex(state)
    _MATx = State.MATx(state)
    _MUOx = State.MUOx(state)
    _muy = PolicyState.muy(policy_state)
    _tfp = tfp(state, policy_state)
    _lab = lab(state, policy_state)
    _sigma = sigma(state, policy_state)
    _Eland = Eland(state, policy_state)
    _zeta = zeta(state, policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)
    _phi12 = phi12(state, policy_state)
    _phi21 = phi21(state, policy_state)

    _MATplus = (1-_phi12) * _MATx + _phi21 * _MUOx \
        + (1 - _muy) * _sigma * _zeta * _tfp * _lab * _zeta_tilde**(-alpha) \
        * _k_tildex**alpha + _Eland
    return _MATplus


def MUOplus(state, policy_state):
    """Carbon mass in the upper ocean."""
    _MATx = State.MATx(state)
    _MUOx = State.MUOx(state)
    _MLOx = State.MLOx(state)
    _phi12 = phi12(state, policy_state)
    _phi21 = phi21(state, policy_state)
    _phi23 = phi23(state, policy_state)
    _phi32 = phi32(state, policy_state)

    _MUOplus = _phi12 * _MATx + (1 - _phi21 - _phi23) * _MUOx + _phi32 * _MLOx
    return _MUOplus


def MLOplus(state, policy_state):
    """Carbon mass in the lower ocean."""
    _MUOx = State.MUOx(state)
    _MLOx = State.MLOx(state)
    _phi23 = phi23(state, policy_state)
    _phi32 = phi32(state, policy_state)

    _MLOplus = _phi23 * _MUOx + (1 - _phi32) * _MLOx
    return _MLOplus


def TATplus(state, policy_state):
    """Atmosphere temperature change relative to the preindustrial."""
    # _f2xco2x = State.f2xco2x(state)
    _f2xco2x = Parameters.f2xco20
    _TATx = State.TATx(state)
    _TOCx = State.TOCx(state)
    _MATx = State.MATx(state)
    _Fex = Fex(state, policy_state)
    _varphi1 = varphi1(state, policy_state)
    _varphi21 = varphi21(state, policy_state)
    _xi2 = xi2(state, policy_state)

    _TATplus = (1 - _varphi21 - _xi2) * _TATx + _varphi21 * _TOCx \
        + _varphi1 * (_f2xco2x * (tf.math.log(_MATx/MATbase) / tf.math.log(2.))
                      + _Fex)
    return _TATplus


def TOCplus(state, policy_state):
    """Ocean temperature change relative to the preindustrial."""
    _TATx = State.TATx(state)
    _TOCx = State.TOCx(state)
    _varphi4 = varphi4(state, policy_state)

    _TOCplus = _varphi4 * _TATx + (1 - _varphi4) * _TOCx
    return _TOCplus
