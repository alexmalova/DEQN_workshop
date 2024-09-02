import tensorflow as tf
import Definitions
import PolicyState
import Parameters
import State


def equations(state, policy_state):
    """Define the dictionary of loss functions.

    Loss functions use the optimal and the first-order conditions directly.
    """
    # Import an expectation operator
    E_t = State.E_t_gen(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Parameters
    # ----------------------------------------------------------------------- #
    alpha = Parameters.alpha

    # Exogenously evolved parameters
    gr_tfp = Definitions.gr_tfp(state, policy_state)
    gr_lab = Definitions.gr_lab(state, policy_state)
    betat = Definitions.betat(state, policy_state)

    # ----------------------------------------------------------------------- #
    # State variables
    # ----------------------------------------------------------------------- #
    k_tildex = State.k_tildex(state)

    # ----------------------------------------------------------------------- #
    # Pseudo state variables to propagate parametric uncertainty
    # ----------------------------------------------------------------------- #
    # gammax = Parameters.gamma0
    # psix = Parameters.psi0
    gammax = State.gammax(state)
    # _psix = Parameters.psi0
    psix = State.psix(state)
#     psix = Definitions.psix(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Optimal policy functions in period t
    # ----------------------------------------------------------------------- #
    con_tildey = PolicyState.con_tildey(policy_state)
    inv_tildey = PolicyState.inv_tildey(policy_state)
    vlog_tildey = PolicyState.vlog_tildey(policy_state)

    # ----------------------------------------------------------------------- #
    # Defined economic variables in period t
    # ----------------------------------------------------------------------- #
    zeta_tilde = Definitions.zeta_tilde(state, policy_state)
    lambd = Definitions.lambd(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Loss functions
    # ----------------------------------------------------------------------- #
    loss_dict = {}

    # ----------------------------------------------------------------------- #
    # Optimal condition
    # For the sake of numerical stability, we use 'optimality_relative' as a
    # loss function. We use optimality_EE when we evaluate an approximation
    # error.
    # ----------------------------------------------------------------------- #
    loss_dict['optimality_REE'] = tf.math.exp(
        (1 - 1/psix) * vlog_tildey) / (
            con_tildey**(1 - 1/psix)
            + betat * zeta_tilde**(1 - 1/psix)
            * tf.math.exp((1 - 1/psix) * vlog_tildey) * E_t(
                lambda s, ps: tf.math.exp(
                    (1 - gammax) * (PolicyState.vlog_tildey(ps) - vlog_tildey))
            )**((1 - 1/psix) / (1 - gammax))
        ) - 1


    budget = zeta_tilde**(1-alpha) * k_tildex**alpha - (
        con_tildey + inv_tildey)
 
    loss_dict['foc_lambdy'] = budget


    # ----------------------------------------------------------------------- #
    # First-order conditions
    # Common expected value throughout the FOCs
    # ----------------------------------------------------------------------- #
    beta_hat_E_t_vplus = tf.math.exp(-1/psix * vlog_tildey) * betat \
        * zeta_tilde**(1 - 1/psix) * E_t(
            lambda s, ps: tf.math.exp(
                (1 - gammax) * (PolicyState.vlog_tildey(ps) - vlog_tildey))
    )**((gammax - 1/psix) / (1 - gammax))

    # ----------------------------------------------------------------------- #
    # FOC wrt. kplusy
    # ----------------------------------------------------------------------- #
    E_t_dvplusdkplus = E_t(
        lambda s, ps: tf.math.exp(
            - gammax * (PolicyState.vlog_tildey(ps) - vlog_tildey))
        * Definitions.dvdk_tilde_psi(s, ps)
    )

    loss_dict['foc_kplusy'] = lambd * tf.math.exp(gr_tfp + gr_lab) \
        * zeta_tilde - beta_hat_E_t_vplus * E_t_dvplusdkplus


    return loss_dict
