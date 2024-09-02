import tensorflow as tf
import math
import itertools
import State
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

# Probability of a dummy shock
shock_probs = tf.constant([1.0])  # Dummy probability


def total_step_random(prev_state, policy_state):
    """State dependant random shock to evaluate the expectation operator."""
    _ar = AR_step(prev_state)
    _shock = shock_step_random(prev_state)
    _policy = policy_step(prev_state, policy_state)

    _total_random = _ar + _shock + _policy

    return _total_random



def total_step_spec_shock(prev_state, policy_state, shock_index):
    """State specific shock to run one episode."""
    _ar = AR_step(prev_state)
    _shock = shock_step_spec_shock(prev_state, shock_index)
    _policy = policy_step(prev_state, policy_state)

    _total_spec = _ar + _shock + _policy

    return _total_spec



def AR_step(prev_state):
    """AR(1) shock on zetalogx and chix."""
    _ar_step = tf.zeros_like(prev_state)  # Initialization

    return _ar_step


def shock_step_random(prev_state):
    """Populate uncertainty for simulating the economy."""
    _shock_step = tf.zeros_like(prev_state)  # Initialization

    return _shock_step


def shock_step_spec_shock(prev_state, shock_index):
    """Populate uncertainty to evaluate an expectation operator.

    Shock on zetalogx, chix, TPx and TPflagx.
    """
    _shock_step = tf.zeros_like(prev_state)  # Initialization

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
        _policy_step, 'kx', Definitions.kplus(prev_state, policy_state))
    _policy_step = State.update(
        _policy_step, 'taux', Definitions.tau2tauplus(prev_state, policy_state)
    )

    return _policy_step
