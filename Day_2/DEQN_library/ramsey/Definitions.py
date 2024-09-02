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
alpha, delta, nu = Parameters.alpha, Parameters.delta, Parameters.nu

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
    _psix = Parameters.psi0
    _rhox = Parameters.rho0
    _gr_tfp = gr_tfp(state, policy_state)
    _gr_lab = gr_lab(state, policy_state)

    _betat = tf.math.exp(- _rhox + (1 - 1/_psix) * _gr_tfp + _gr_lab)
    return _betat



# --------------------------------------------------------------------------- #
# Economic variables
# --------------------------------------------------------------------------- #
def lambd(state, policy_state):
    """Lagrange multiplier wrt. the budget constraint."""
    _psix = Parameters.psi0
    _cony = PolicyState.cony(policy_state)

    _lambd = (1 - 1/_psix) * _cony**(-1/_psix)
    return _lambd



def ygross(state, policy_state):
    """Gross production in a stationary form."""
    _kx = State.kx(state)

    _ygross = _kx**alpha
    return _ygross

# --------------------------------------------------------------------------- #
# First-derivatives of the value function
# --------------------------------------------------------------------------- #
def dvdk_psi(state, policy_state):
    """Envelope theorem wrt. kx."""
 
    _kx = State.kx(state)
    _lambd = lambd(state, policy_state)

    
    _dvdk_psi = _lambd * (  alpha * _kx**(alpha - 1) + (1 - delta))
    return _dvdk_psi


# --------------------------------------------------------------------------- #
# State variables in period t + 1
# --------------------------------------------------------------------------- #
def kplus(state, policy_state):
    """Capital stock tomorrow in a stationary format."""
    _kx = State.kx(state)
    _invy = PolicyState.invy(policy_state)
    _gr_tfp = gr_tfp(state, policy_state)
    _gr_lab = gr_lab(state, policy_state)

    _kplus = ((1 - delta) * _kx + _invy) / (
        tf.math.exp(_gr_tfp + _gr_lab))
    return _kplus

