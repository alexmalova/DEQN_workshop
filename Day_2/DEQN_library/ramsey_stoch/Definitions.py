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
    _rhox = State.rhox(state)
    _gr_tfp = gr_tfp(state, policy_state)
    _gr_lab = gr_lab(state, policy_state)

    _betat = tf.math.exp(- _rhox + (1 - 1/_psix) * _gr_tfp + _gr_lab)
    return _betat





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


def ygross_tilde(state, policy_state):
    """Gross production in a stationary form."""
    _k_tildex = State.k_tildex(state)
    _zeta_tilde = zeta_tilde(state, policy_state)

    _ygross_tilde = _zeta_tilde**(1 - alpha) * _k_tildex**alpha
    return _ygross_tilde



# --------------------------------------------------------------------------- #
# First-derivatives of the value function
# --------------------------------------------------------------------------- #
def dvdk_tilde_psi(state, policy_state):
    """Envelope theorem wrt. kx."""
    # _psix = Parameters.psi0
    _psix = State.psix(state)
#     _psix = psix(state, policy_state)
    _k_tildex = State.k_tildex(state)
    _vlog_tildey = PolicyState.vlog_tildey(policy_state)
    _zeta = zeta(state, policy_state)
    _zeta_tilde = zeta_tilde(state, policy_state)
    _lambd = lambd(state, policy_state)

    _dvdk_tilde_psi = tf.math.exp(1/_psix * _vlog_tildey) * (
        _lambd * ( alpha * _zeta_tilde**(1 - alpha)
                  * _k_tildex**(alpha - 1)
                  + (1 - delta))
    )
    return _dvdk_tilde_psi




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



