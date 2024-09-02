import tensorflow as tf
import math
import itertools
import State
import PolicyState
import Definitions
import Parameters


# --------------------------------------------------------------------------- #
# Extract parameters
# --------------------------------------------------------------------------- #
alpha = Parameters.alpha

# --------------------------------------------------------------------------- #
# Deterministic case
# For the stochastic case, need to comment out below
# --------------------------------------------------------------------------- #
# We keep zetax = zetax0 = 1 and chix = chix0 = 0 over time.
# There is no random shock, i.e., shock_step_random and shock_step_spec_shock =
# 0 over time.

# Probability of a dummy shock
# shock_probs = tf.constant([1.0])  # Dummy probability

# --------------------------------------------------------------------------- #
# Stochastic TFP shock on the production function
# For the deterministic case, need to comment out below
# --------------------------------------------------------------------------- #
mu_tfp = 0  # Mean of the normal distribution
sigma = 1  # Variance of the normal distribution
shocks_zeta = [math.sqrt(2.) * sigma * x3 + mu_tfp for x3 in
               [-1.224744871, 0., +1.224744871]]
probs_zeta = [omega3 / math.sqrt(math.pi) for omega3 in
              [0.2954089751, 1.181635900, 0.2954089751]]
shocks_chi = shocks_zeta
probs_chi = probs_zeta

shock_values = tf.constant(list(itertools.product(shocks_zeta, shocks_chi)))
shock_probs = tf.constant([pr_zeta * pr_chi for pr_zeta, pr_chi in list(
    itertools.product(probs_zeta, probs_chi))])

# if Parameters.expectation_type == 'monomial':
#     shock_values, shock_probs = State.monomial_rule([sigma, sigma])


# --------------------------------------------------------------------------- #
# Populate and integrate uncertianty
# --------------------------------------------------------------------------- #
def augment_state(state):
    """Clip values with relevant min and max.

    We respect 0.1% and 99.9% percentiles, which depend on the choice of r, to
    omit outliers.
    """
    zeta_logx_min, zeta_logx_max = -40.97052, 40.17667  # r = 0.975
    _state = state
    _state = State.update(_state, 'zeta_logx', tf.clip_by_value(
        State.zeta_logx(state), zeta_logx_min, zeta_logx_max))
    return _state


def total_step_random(prev_state, policy_state):
    """State dependant random shock to evaluate the expectation operator."""
    _ar = AR_step(prev_state)
    _shock = shock_step_random(prev_state)
    _policy = policy_step(prev_state, policy_state)

    _total_random = _ar + _shock + _policy

    return _total_random
    # return augment_state(_total_random)


def total_step_spec_shock(prev_state, policy_state, shock_index):
    """State specific shock to run one episode."""
    _ar = AR_step(prev_state)
    _shock = shock_step_spec_shock(prev_state, shock_index)
    _policy = policy_step(prev_state, policy_state)

    _total_spec = _ar + _shock + _policy

    return _total_spec
    # return augment_state(_total_spec)


def AR_step(prev_state):
    """AR(1) shock on zetalogx and chix."""
    _ar_step = tf.zeros_like(prev_state)  # Initialization
    _ar_step = State.update(
        _ar_step, 'zeta_logx',
        State.zeta_logx(prev_state) + State.chix(prev_state) / (1 - alpha))
    _ar_step = State.update(
        _ar_step, 'zetatilde_logx', State.chix(prev_state) / (1 - alpha))
    # Persistency of long-run risk
    _rx = Parameters.r0
    # _rx = State.rx(prev_state)
    _ar_step = State.update(_ar_step, 'chix', _rx * State.chix(prev_state))

    return _ar_step


def shock_step_random(prev_state):
    """Populate uncertainty for simulating the economy."""
    _shock_step = tf.zeros_like(prev_state)  # Initialization

    # Scale of Gaussian innovation
    varrhox, varsigmax = Parameters.varrho0, Parameters.varsigma0
    # varrhox, varsigmax = State.varrhox(prev_state)/100, State.varsigmax(prev_state)/100
    # ----------------------------------------------------------------------- #
    # In the deterministic case, there is no shock, thus need to comment out
    # below
    # ----------------------------------------------------------------------- #
    _random_normals = Parameters.rng.normal([prev_state.shape[0], 2])
    _shock_step = State.update(
        _shock_step, "zeta_logx", varrhox / (1-alpha) * _random_normals[:, 0])
    _shock_step = State.update(
        _shock_step, "zetatilde_logx",
        varrhox / (1 - alpha) * _random_normals[:, 0])
    _shock_step = State.update(
        _shock_step, "chix", varsigmax * _random_normals[:, 1])
    
    
    # ----------------------------------------------------------------------- #
    # Update pseudo states
    # ----------------------------------------------------------------------- #
    # When we train the model, the pseudo states are drawn from the uniform
    # distributions in every episode to avoide over-fitting.
    # When we post process the model, need to comment out below.
    # _random_uniform = Parameters.rng.uniform([prev_state.shape[0], 5])
    # _shock_step = State.update(_shock_step, 'rhox', _random_uniform[:,0] * 
    #                            (Parameters.rho_upper - Parameters.rho_lower) + Parameters.rho_lower)
    # _shock_step = State.update(_shock_step, 'gammax', _random_uniform[:,1] * 
    #                            (Parameters.gamma_upper - Parameters.gamma_lower) + Parameters.gamma_lower)
    # _shock_step = State.update(_shock_step, 'psix',_random_uniform[:,2] * 
    #                            (Parameters.psi_upper - Parameters.psi_lower) + Parameters.psi_lower)
    
    # When we post process the model, we make the pseudo state constant over
    # time.
    # When we train the model, need to comment out below.
    _shock_step = State.update(_shock_step, 'rhox', State.rhox(prev_state))
    _shock_step = State.update(_shock_step, 'gammax', State.gammax(prev_state))
    _shock_step = State.update(_shock_step, 'psix', State.psix(prev_state))

    return _shock_step


def shock_step_spec_shock(prev_state, shock_index):
    """Populate uncertainty to evaluate an expectation operator.

    Shock on zetalogx, chix, TPx and TPflagx.
    """
    _shock_step = tf.zeros_like(prev_state)  # Initialization

    # Scale of Gaussian innovation
    varrhox, varsigmax = Parameters.varrho0, Parameters.varsigma0
    # varrhox, varsigmax = State.varrhox(prev_state)/100, State.varsigmax(prev_state)/100
    # ----------------------------------------------------------------------- #
    # In the deterministic case, there is no shock, thus need to comment out
    # below
    # ----------------------------------------------------------------------- #
    _shock_step = State.update(
        _shock_step, 'zeta_logx', varrhox / (1 - alpha) * tf.repeat(
            shock_values[shock_index, 0], prev_state.shape[0]))
    _shock_step = State.update(
        _shock_step, 'zetatilde_logx', varrhox / (1 - alpha) * tf.repeat(
            shock_values[shock_index, 0], prev_state.shape[0]))
    _shock_step = State.update(
        _shock_step, 'chix', varsigmax * tf.repeat(
            shock_values[shock_index, 1], prev_state.shape[0]))

    # ----------------------------------------------------------------------- #
    # Update pseudostates
    # ----------------------------------------------------------------------- #
    
    _shock_step = State.update(_shock_step, 'rhox', State.rhox(prev_state))
    _shock_step = State.update(_shock_step, 'gammax', State.gammax(prev_state))
    _shock_step = State.update(_shock_step, 'psix', State.psix(prev_state))


    return _shock_step


def policy_step(prev_state, policy_state):
    """Update state variables.

    Capital state is updated by the optimal policy.
    Climate states are updated based on the laws of motion.
    Pseudo states are updated if they are included.
    """
    _policy_step = tf.zeros_like(prev_state)  # Initialization

    # ----------------------------------------------------------------------- #
    # Update capital, climate and time state variables if needed
    # ----------------------------------------------------------------------- #
    _policy_step = State.update(
        _policy_step, 'k_tildex', Definitions.ktilde_plus(
            prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'taux', Definitions.tau2tauplus(prev_state, policy_state)
    )

    return _policy_step
