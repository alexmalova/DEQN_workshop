import Definitions
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
    beta, delta = Parameters.beta, Parameters.delta

    # ----------------------------------------------------------------------- #
    # Defined economic variables in period t
    # ----------------------------------------------------------------------- #

    C_t = Definitions.C_t(state, policy_state)
    R_tplus1 = Definitions.R_tplus1(state, policy_state)
    # Kplus_compute_analytic = Definitions.Kplus_compute_analytic(state, policy_state)
    # K_t = State.K_t(state)
    # K_tplus1 = Definitions.K_tplus1(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Loss functions
    # ----------------------------------------------------------------------- #
    loss_dict = {}

  
    loss_dict['REE']  = 1. - E_t(lambda s, ps: Definitions.C_t(s,ps) / (beta * C_t * (R_tplus1 + 1. - delta)))
    
    # print("Current capital=", K_t.numpy())
    # print("Tomorrow capital=", K_tplus1.numpy())
    # print("Analytic capital=", Kplus_compute_analytic.numpy())


    return loss_dict
