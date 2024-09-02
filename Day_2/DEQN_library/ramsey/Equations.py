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
    kx = State.kx(state)
   

  
    # ----------------------------------------------------------------------- #
    # Optimal policy functions in period t
    # ----------------------------------------------------------------------- #

    cony = PolicyState.cony(policy_state)
    invy = PolicyState.invy(policy_state)

    # ----------------------------------------------------------------------- #
    # Defined economic variables in period t
    # ----------------------------------------------------------------------- #

    lambd = Definitions.lambd(state, policy_state)

    # ----------------------------------------------------------------------- #
    # Loss functions
    # ----------------------------------------------------------------------- #
    loss_dict = {}



    # ----------------------------------------------------------------------- #
    # FOC wrt. lambdy (budget constraint)
    # ----------------------------------------------------------------------- #

    budget =  kx**alpha  - (cony + invy)
  
    loss_dict['foc_lambdy'] = budget


    # ----------------------------------------------------------------------- #
    # FOC wrt. kplusy
    # ----------------------------------------------------------------------- #

    loss_dict['foc_kplusy'] = lambd * tf.math.exp(gr_tfp + gr_lab) \
         - betat * E_t(lambda s, ps: Definitions.dvdk_psi(s, ps))


    return loss_dict
