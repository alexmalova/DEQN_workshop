#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: post_process_ramsey.py
Author(s): Simon Scheidegger, Takafumi Usui, Aleksandra Friedl
E-mail: u.takafumi@gmail.com
Description:
Post processing using the optimal policy functions.
With this script, we
- [X] compute and plot the dynamics of the exogenous parameters;
- [X] plot one simulated path of the state, policy and defined variables;
- [X] compute and plot the distributions of the state, policy and defined
variables;
- [X] compute an approximation error in the all equilbrium and KKT conditions.
"""
import numpy as np  # using float32 to have a compatibility with tensorflow
import pandas as pd
import shutil
import importlib
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
from matplotlib import rc
import State
import PolicyState
import Definitions
from Graphs import run_episode

tf.get_logger().setLevel('CRITICAL')
pd.set_option('display.max_columns', None)

# --------------------------------------------------------------------------- #
# Plot setting
# --------------------------------------------------------------------------- #
# Get the size of the current terminal
terminal_size_col = shutil.get_terminal_size().columns

# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
# rc('text', usetex=True)
# Font size
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.title_fontsize"] = 14

# Figure size
fsize = (9, 6)
line_args = {'markerfacecolor': 'None', 'color': 'tab:blue', 'marker': None,
             'linestyle': '-'}
distribution_args = {'markerfacecolor': 'None', 'color': 'tab:blue',
                     'marker': '.', 'linestyle': 'None'}
lb_quantiles = [10, 25, 50, 75, 90]

# Error percentiles used to plot distributions
err_percentiles = [0.001, 0.25, 0.50, 0.75, 0.999]

# --------------------------------------------------------------------------- #
# Economic variabples
# --------------------------------------------------------------------------- #
# Exogenous parameters
exparams = ['tfp', 'gr_tfp', 'lab', 'gr_lab', 'betat']
# Defined economic variables
econ_defs = [ 'lambd', 
              'ygross',
              'dvdk_psi']

# --------------------------------------------------------------------------- #
# Simulation periods and batch size
# --------------------------------------------------------------------------- #
begyear = 2015
# begyear = 2005  # Cai and Lontzek

# Simulate the economy for N_simulated episode length
# N_episode_length = Parameters.N_episode_length + 1
# From begyear to 2100
N_episode_length = 2100 - begyear + 1

# Number of simulation batch, it should be arbitrary big enough
N_sim_batch = 100
# N_sim_batch = 10  # For testing

print("-" * terminal_size_col)
print("Simulate the economy for {} years".format(N_episode_length))

# Import equations
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")


# Number of state, policy and defined variables
N_state = len(Parameters.states)  # Number of state variables
N_policy_state = len(Parameters.policy_states)  # Number of policy variables
N_defined = len(econ_defs)  # Number of defined variables


starting_state = tf.reshape(tf.constant([
    Parameters.k0, 
    Parameters.tau0]), shape=(1, N_state))

# Simulate the economy for N_episode_length time periods
simulation_starting_state = tf.tile(tf.expand_dims(
    starting_state, axis=0), [N_episode_length, 1, 1])

# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Simulate the economy for one episode for {} years".format(
    N_episode_length))
# --------------------------------------------------------------------------- #
# Simulate for one state episode
state_1episode = run_episode(simulation_starting_state)
# Simulate for one policy episode
policy_state_1episode = np.empty(
    shape=[N_episode_length, 1, N_policy_state], dtype=np.float32)
for tidx in range(N_episode_length):
    policy_state_val = Parameters.policy(state_1episode[tidx, :, :])
    policy_state_1episode[tidx, :, :] = policy_state_val

state_1episode = tf.reshape(state_1episode, shape=[N_episode_length, N_state])
policy_state_1episode = tf.reshape(
    policy_state_1episode, shape=[N_episode_length, N_policy_state])

# Get simulation time periods
ts = Definitions.tau2t(state_1episode, policy_state_1episode)
ts = begyear + ts  # The original base year in DICE-2016 is 2015
ts_beg, ts_end = int(tf.round(ts[0]).numpy()), int(tf.round(ts[-1]).numpy())

# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Plot the dynamics of the exogenous parameters for {} years".format(
    N_episode_length))
# --------------------------------------------------------------------------- #
for de in exparams:
    fig, ax = plt.subplots(figsize=fsize)
    de_val = getattr(Definitions, de)(state_1episode, policy_state_1episode)
    if de in ['tfp', 'gr_tfp', 'lab', 'gr_lab', 'betat']:
        de_val = de_val

    ax.plot(ts, de_val)
    ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(r'{}'.format(de.replace('_', '\_')))
    plt.savefig(
        Parameters.LOG_DIR + '/exparams_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + de + '.png')
    plt.close()

# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Plot one simulated episode for {} years".format(N_episode_length))
# --------------------------------------------------------------------------- #
for sidx, state in enumerate(Parameters.states):
    fig, ax = plt.subplots(figsize=fsize)
    # State variable
    state_val = getattr(State, state)(state_1episode)
    # Adjust state variables
    if state in ['kx']:
        tfp = Definitions.tfp(state_1episode, policy_state_1episode)
        lab = Definitions.lab(state_1episode, policy_state_1episode)
        state_val = tfp * lab * state_val
    ax.plot(ts, state_val.numpy(), **line_args)
    ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(r'{}'.format(state.replace('_', '\_')))

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + state + '.png')
    plt.close()

for pidx, ps in enumerate(Parameters.policy_states):
    fig, ax = plt.subplots(figsize=fsize)
    # policy variable
    ps_val = getattr(PolicyState, ps)(policy_state_1episode)
    if ps in ['cony', 'invy']:
        tfp = Definitions.tfp(state_1episode, policy_state_1episode)
        lab = Definitions.lab(state_1episode, policy_state_1episode)
        ps_val = tfp * lab * ps_val
    ax.plot(ts, ps_val.numpy(), **line_args)
    ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(r'{}'.format(ps.replace('_', '\_')))

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + ps + '.png')
    plt.close()

for didx, de in enumerate(econ_defs):
    fig, ax = plt.subplots(figsize=fsize)
    # defined economic variable
    de_val = getattr(Definitions, de)(state_1episode, policy_state_1episode)
    if de in ['ygross', 'ynet']:
        tfp = Definitions.tfp(state_1episode, policy_state_1episode)
        lab = Definitions.lab(state_1episode, policy_state_1episode)
        de_val = tfp * lab * de_val
    ax.plot(ts, de_val.numpy(), **line_args)
    ax.set_xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    ax.set_ylabel(r'{}'.format(de.replace('_', '\_')))

    plt.savefig(
        Parameters.LOG_DIR + '/1episode_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + de + '.png')
    plt.close()


# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Simulate the economy for {} years in {} simulation batch".format(
    N_episode_length, N_sim_batch))
# --------------------------------------------------------------------------- #
simulation_starting_state_batch = tf.tile(tf.expand_dims(
    starting_state, axis=0), [N_episode_length, N_sim_batch, 1])

# Simulate the economy for N_sim_batch times to compute the collection of
# state and policy episodes
state_episode_batch = run_episode(simulation_starting_state_batch)

# Policy variables for N_sim_batch times
policy_state_episode_batch = np.empty(
    shape=[N_episode_length, N_sim_batch, N_policy_state], dtype=np.float32)
for tidx in range(N_episode_length):
    policy_state_batch = Parameters.policy(state_episode_batch[tidx, :, :])
    policy_state_episode_batch[tidx, :, :] = policy_state_batch

# Some variables need to be rescaled for plotting
state_episode_batch_scaled = np.empty_like(
    state_episode_batch, dtype=np.float32)
policy_state_episode_batch_scaled = np.empty_like(
    policy_state_episode_batch, dtype=np.float32)
defined_episode_batch_scaled = np.empty(
    shape=[N_episode_length, N_sim_batch, N_defined], dtype=np.float32)

# State variables
for sidx, state in enumerate(Parameters.states):
    # Adjust state variables
    for tidx in range(N_episode_length):
        state_batch = state_episode_batch[tidx, :, :]
        policy_state_batch = policy_state_episode_batch[tidx, :, :]
        state_val = getattr(State, state)(state_batch)
        if state in ['kx']:
            tfp = Definitions.tfp(state_batch, policy_state_batch)
            lab = Definitions.lab(state_batch, policy_state_batch)
            state_val = tfp * lab *  state_val
        elif state in ['MATx', 'MUOx', 'MLOx']:
            # Rescale to GtC
            state_val = 1000. * state_val
        state_episode_batch_scaled[tidx, :, sidx] = state_val

# Policy variables
for pidx, policy in enumerate(Parameters.policy_states):
    # Adjust policy variables
    for tidx in range(N_episode_length):
        state_batch = state_episode_batch[tidx, :, :]
        policy_state_batch = policy_state_episode_batch[tidx, :, :]
        policy_val = getattr(PolicyState, policy)(policy_state_batch)
        if policy in ['cony', 'invy']:
            tfp = Definitions.tfp(state_batch, policy_state_batch)
            lab = Definitions.lab(state_batch, policy_state_batch)
            gr_tfp = Definitions.gr_tfp(state_batch, policy_state_batch)
            gr_lab = Definitions.gr_lab(state_batch, policy_state_batch)
            policy_val = tfp * lab * policy_val
        policy_state_episode_batch_scaled[tidx, :, pidx] = policy_val

# Defined economic variables
for didx, de in enumerate(econ_defs):
    # Adjust defined variables
    for tidx in range(N_episode_length):
        state_batch = state_episode_batch[tidx, :, :]
        policy_state_batch = policy_state_episode_batch[tidx, :, :]
        defined_val = getattr(Definitions, de)(state_batch, policy_state_batch)
        if de in ['ygross']:
            tfp = Definitions.tfp(state_batch, policy_state_batch)
            lab = Definitions.lab(state_batch, policy_state_batch)
            defined_val = defined_val * tfp * lab 
        defined_episode_batch_scaled[tidx, :, didx] = defined_val

# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Plot the distribution of economic variables for {} years".format(
    N_episode_length))
# --------------------------------------------------------------------------- #
# Compute the quantiles of each variable along with the number of simulations
quantile_state = np.percentile(
    state_episode_batch_scaled, q=lb_quantiles, axis=1)
quantile_policy_state = np.percentile(
    policy_state_episode_batch_scaled, q=lb_quantiles, axis=1)
quantile_defined = np.percentile(
    defined_episode_batch_scaled, q=lb_quantiles, axis=1)

# Compute the range of each variable
range_state = np.percentile(state_episode_batch_scaled, q=[1, 99], axis=1)
range_policy_state = np.percentile(
    policy_state_episode_batch_scaled, q=[1, 99], axis=1)
range_defined = np.percentile(
    defined_episode_batch_scaled, q=[1, 99], axis=1)

# Compute the average of each variable
avg_state = np.average(state_episode_batch_scaled, axis=1)
avg_policy_state = np.average(policy_state_episode_batch_scaled, axis=1)
avg_defined = np.average(defined_episode_batch_scaled, axis=1)

# Plot the distribution of state variables
for sidx, state in enumerate(Parameters.states):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_state[0, :, sidx], range_state[1, :, sidx],
        facecolor='tab:gray', alpha=0.1,
        label=r'Range of sample paths (1\% to 99\%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_state[qidx, :, sidx],
                label=r'{}\% quantile'.format(lb_quantiles[qidx]))
    plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(r'{}'.format(state.replace('_', '\_')))
    plt.legend(loc='upper left')
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + state + '.png')
    plt.close()

for pidx, policy in enumerate(Parameters.policy_states):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_policy_state[0, :, pidx], range_policy_state[1, :, pidx],
        facecolor='tab:gray', alpha=0.1,
        label=r'Range of sample paths (1\% to 99\%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_policy_state[qidx, :, pidx],
                label=r'{}\% quantile'.format(lb_quantiles[qidx]))
    plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(r'{}'.format(policy.replace('_', '\_')))
    plt.legend(loc='upper left')
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + policy + '.png')
    plt.close()

for didx, de in enumerate(econ_defs):
    fig, ax = plt.subplots(figsize=fsize)
    ax.fill_between(
        ts, range_defined[0, :, didx], range_defined[1, :, didx],
        facecolor='tab:gray', alpha=0.1,
        label=r'Range of sample paths (1\% to 99\%)')
    for qidx in range(len(lb_quantiles)):
        ax.plot(ts, quantile_defined[qidx, :, didx],
                label=r'{}\% quantile'.format(lb_quantiles[qidx]))
    plt.xlabel('Year')
    ax.set_xlim([ts_beg, ts_end])
    plt.ylabel(r'{}'.format(de.replace('_', '\_')))
    plt.legend(loc='upper left')
    plt.savefig(
        Parameters.LOG_DIR + '/distribution_' + str(ts_beg) + '-' + str(ts_end)
        + '_' + de + '.png')
    plt.close()







# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print(r"Compute the Euler discrepancies for {} years in {} simulation "
      "batch".format(Parameters.N_episode_length, N_sim_batch))
# --------------------------------------------------------------------------- #

N_episode_length_euler = Parameters.N_episode_length
starting_state_batch_euler = tf.tile(starting_state, [N_sim_batch, 1])

#
simulation_starting_state_batch_euler = tf.tile(tf.expand_dims(
    starting_state_batch_euler, axis=0), [N_episode_length_euler, 1, 1])

# Simulate the economy for N_sim_batch times to compute the collection of
# state and policy episodes
state_episode_batch_euler = run_episode(simulation_starting_state_batch_euler)
# Policy variables for N_sim_batch times
policy_state_episode_batch_euler = np.empty(
    shape=[N_episode_length_euler, N_sim_batch, N_policy_state],
    dtype=np.float32)
for tidx in range(N_episode_length_euler):
    policy_state_episode_batch_euler[tidx, :, :] = Parameters.policy(
        state_episode_batch_euler[tidx, :, :])

# Take the absolute numericl value of each element
for tidx in range(N_episode_length_euler):
    state_batch = state_episode_batch_euler[tidx, :, :]
    policy_state_batch = policy_state_episode_batch_euler[tidx, :, :]
    euler_discrepancy_df = pd.DataFrame(
        Equations.equations(state_batch, policy_state_batch)).abs()
    state_episode_df = pd.DataFrame(
        {s: getattr(State, s)(state_batch) for s in Parameters.states})
    policy_episode_df = pd.DataFrame(
        {ps: getattr(PolicyState, ps)(policy_state_batch)
         for ps in Parameters.policy_states})
    defined_episode_df = pd.DataFrame(
        {de: getattr(Definitions, de)(state_batch, policy_state_batch)
         for de in econ_defs})

    # Initialize each dataframe
    if tidx == 0:
        euler_discrepancies_df = euler_discrepancy_df
        state_episodes_df = state_episode_df
        policy_episodes_df = policy_episode_df
        defined_episodes_df = defined_episode_df
    else:
        euler_discrepancies_df = pd.concat([
            euler_discrepancies_df, euler_discrepancy_df], axis=0)
        state_episodes_df = pd.concat([
            state_episodes_df, state_episode_df], axis=0)
        policy_episodes_df = pd.concat([
            policy_episodes_df, policy_episode_df], axis=0)
        defined_episodes_df = pd.concat([
            defined_episodes_df, defined_episode_df], axis=0)

# --------------------------------------------------------------------------- #
# Print the Euler approximation errors
# --------------------------------------------------------------------------- #
print("-" * terminal_size_col)
print("Print the Euler discrepancies")
# import ipdb; ipdb.set_trace()
print(euler_discrepancies_df.describe(
    percentiles=err_percentiles, include='all'))

# Compute the mean and the max Euler errors for all Euler equations
euler_discrepancies_df_melt = pd.melt(euler_discrepancies_df)
# Convert the Euler equation errors in log10
# euler_discrepancies_df_melt_log10 = np.log10(euler_discrepancies_df_melt['value'])
print("-" * terminal_size_col)
print("Print the mean and the max Euler discrepancies")
print(euler_discrepancies_df_melt.describe(percentiles=err_percentiles))

# Save all relevant quantities along the trajectory
euler_discrepancies_df.describe(percentiles=err_percentiles, include='all').to_csv(
    Parameters.LOG_DIR + "/simulated_euler_discrepancies_describe_"
    + str(N_episode_length_euler) + 'years_' + str(N_sim_batch) + "batch.csv",
    index=False, float_format='%.3e')

euler_discrepancies_df_melt.describe(percentiles=err_percentiles).to_csv(
    Parameters.LOG_DIR + "/simulated_euler_discrepancies_describe_melt_"
    + str(N_episode_length_euler) + 'years_' + str(N_sim_batch) + "batch.csv",
    index=False, float_format='%.3e')

print("-" * terminal_size_col)
print("Finished calculating Euler discrepancies")



print("Exit post processing")

# import ipdb; ipdb.set_trace()
