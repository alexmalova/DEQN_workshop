# TF Module containing state variables

import importlib
import math
import sys
import tensorflow as tf
import numpy as np
from Parameters import expectation_pseudo_draws, expectation_type, MODEL_NAME, policy, states, state_bounds_hard


def monomial_rule(scale_list):
    d = len(scale_list)
    shock_probabilities = tf.constant([1 / (2*d) ] * (2*d))
    shock_values = tf.constant(np.concatenate((np.diag(scale_list) * math.sqrt( d / 2.0), np.diag(scale_list) * -math.sqrt( d / 2.0))), dtype=tf.dtypes.float32)
    return (shock_values, shock_probabilities)

Dynamics = importlib.import_module(MODEL_NAME + ".Dynamics")

for i, state in enumerate(states):
    if (state in state_bounds_hard["lower"]) or (state in state_bounds_hard["upper"]):
        setattr(
            sys.modules[__name__],
            state,
            (
                lambda ind: lambda x: tf.clip_by_value(x[:, ind], state_bounds_hard["lower"].get(states[ind], np.NINF), state_bounds_hard["upper"].get(states[ind], np.Inf))
            )(i),
        )
    else:
        setattr(sys.modules[__name__], state, (lambda ind: lambda x: x[:, ind])(i))

    # always add a 'raw' attribute as well - this can be used for penalties
    setattr(sys.modules[__name__], state + "_RAW", (lambda ind: lambda x: x[:, ind])(i))

def E_t_gen(state, policy_state):
    if expectation_type != 'pseudo_random':
        next_states = [Dynamics.total_step_spec_shock(state, policy_state, i) for i in range(len(Dynamics.shock_probs))]
        next_policies = [policy(next_state) for next_state in next_states]

        def E_t(evalFun):
            # calculate conditional expectation
            res = tf.zeros(state.shape[0])
            for i in range(len(Dynamics.shock_probs)):
                res += Dynamics.shock_probs[i] * evalFun(next_states[i], next_policies[i])
            return res
    else:
        next_states = [Dynamics.total_step_random(state, policy_state) for i in range(expectation_pseudo_draws)]
        next_policies = [policy(next_state) for next_state in next_states]

        def E_t(evalFun):
            # calculate conditional expectation
            res = tf.zeros(state.shape[0])
            for i in range(expectation_pseudo_draws):
                res += 1.0 / expectation_pseudo_draws * evalFun(next_states[i], next_policies[i])
            return res
    return E_t

def update(old_states, at, new_vals):
    i = states.index(at)
    return tf.tensor_scatter_nd_update(old_states,[[j,i] for j in range(old_states.shape[0])], new_vals)

def update_dict(old_states, new_vals_dict):
    new_states = old_states
    for s in new_vals_dict:
        new_states = tf.tensor_scatter_nd_update(new_states,[[j,i] for j in range(new_states.shape[0])], new_vals_dict[s])

    return new_states

def GH(state, policy_state):
    mu_tfp = 0  # Mean of the normal distribution
    sigma = 1  # Variance of the normal distribution
    nodes = [math.sqrt(2.) * sigma * x3 + mu_tfp for x3 in
                   [-1.224744871, 0., +1.224744871]]
    weights = [omega3 / math.sqrt(math.pi) for omega3 in
                  [0.2954089751, 1.181635900, 0.2954089751]]

    def GHint(hfunc):
        res = tf.zeros(state.shape[0])
        for i in range(len(weights)):
            res += weights[i] * hfunc(nodes[i])
        return res
    return GHint

def GH_2s(state, policy_state):

    perturb_states = [Dynamics.shock_perturb(state, policy_state, i) for i in range(len(Dynamics.shock_probs))]
    perturb_policies = [policy(perturb_state) for perturb_state in perturb_states]

    def GHint_2s(evalFun):
        # calculate conditional expectation
        res = tf.zeros(state.shape[0])
        for i in range(len(Dynamics.shock_probs)):
            res += Dynamics.shock_probs[i] * evalFun(perturb_policies[i], Dynamics.shock_values[i,:])
        return res
    return GHint_2s  

def GH_1s(state, policy_state):

    perturb_states = [Dynamics.shock_perturb(state, policy_state, i) for i in range(len(Dynamics.shock_probs))]
    perturb_policies = [policy(perturb_state) for perturb_state in perturb_states]

    def GHint_1s(evalFun):
        # calculate conditional expectation
        res = tf.zeros(state.shape[0])
        for i in range(len(Dynamics.shock_probs)):
            res += Dynamics.shock_probs[i] * evalFun(perturb_policies[i], Dynamics.shock_values[i])
        return res
    return GHint_1s 

