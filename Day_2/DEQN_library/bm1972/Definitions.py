import Parameters
import PolicyState
import State

# --------------------------------------------------------------------------- #
# Extract parameters
# --------------------------------------------------------------------------- #

alpha, beta, delta = Parameters.alpha, Parameters.beta, Parameters.delta


def k_compute_infty(state, policy_state):
    """ Return the stationary point (or steady state) for full depreciation """
    _k_compute_infty = (1 / (beta * alpha))**(1/(alpha - 1))
    return _k_compute_infty


def Kplus_compute_analytic(state, policy_state):
    """ Return the optimal capital stock in the next period  for full depreciation """
    _K_t = State.K_t(state)
    _Kplus_compute_analytic = alpha * beta * _K_t**alpha
    return _Kplus_compute_analytic


def c_compute(state, policy_state):
    """ Return the optimal consumption policy  for full depreciation """
    _K_tplus1 = K_tplus1(state, policy_state)
    _c_compute = _K_tplus1**alpha - _K_tplus1
    return _c_compute

############################################################################

def Y_t(state, policy_state):
    """compute output today"""
    _K_t = State.K_t(state)
    _Y_t = _K_t ** alpha
    return _Y_t

def K_tplus1(state, policy_state):
    """get the implied capital in the next period"""
    _K_t = State.K_t(state)
    _Y_t = Y_t(state, policy_state)
    _s_t = PolicyState.s_t(policy_state)
    
    _K_tplus1 = (1. - delta) * _K_t + _Y_t * _s_t
    return _K_tplus1

def C_t(state, policy_state):
    """get consumption this period"""
    _Y_t = Y_t(state, policy_state)
    _s_t = PolicyState.s_t(policy_state)
    
    _C_t = _Y_t - _Y_t * _s_t
    return _C_t

def R_tplus1(state, policy_state):
    """compute the return on capital in the next period"""
    _K_tplus1 = K_tplus1(state, policy_state)
    
    _R_tplus1 = alpha * _K_tplus1 ** (alpha - 1.)
    return _R_tplus1


        

