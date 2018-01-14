from Utils import get_dimension, get_control_dimension
from EM_Config import EM_Config
from Multiple_Sequence_Kalman_Filter import MSKF
from Single_Sequence_Kalman_Filter import SSKF

def expectation_maximization(X, U, config=None, verbose_level=2, print_every=10):
    """
    Finding the 'best' parameters using the Expectation Maximization algorithm

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
            'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
            None for missing measurements
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
            'U[seq_idx][time_step]' = the control of the seq_idx sequence at time time_step
    config: {String: String}
            specified EM config dict
    verbose_level: int
            0: print nothing
            1: print summary at the end of all EM iterations
            2: print summary for every print_every EM iteration
    print_every: int
            print summary for every print_every EM iteration if verbose_level = 2

    Returns
    -------
    parameters: z_0_hat, P_0_hat, A, B, Q, R
        z_0_hat: [dimension] numpy array
            initial mean for every sequence
        P_0_hat: [dimension, dimension] numpy array
            initial covariance for every sequence
        A: [dimension, dimension] numpy array
            transition matrix
        B: [dimension, control_dimension] numpy array
            control matrix
        Q: [dimension, dimension] numpy array
            transition covariance matrix
        R: [dimension, dimension] numpy array
            emission covariance matrix
    ll_history: [num_iterations] array of numpy float
        'll_history[iteration_idx]' = the log likelihood at EM iteration iteration_idx
    """
    # initialize the parameters
    dimension, control_dimension = get_dimension(X), get_control_dimension(U)
    if config is None:
        config = EM_Config(dimension, control_dimension).get_default_config()
    dimension, control_dimension = get_dimension(X), get_control_dimension(U)
    mskf = MSKF(dimension, control_dimension, config['initial_z_0_hat'], config['initial_P_0_hat'],
                config['initial_A'], config['initial_B'], config['initial_Q'], config['initial_R'])
    mskf.pass_in_measurement_and_control(X, U)

    # set the configuration
    mskf.set_em_config(config)

    # EM
    parameters, ll_history = mskf.em(verbose_level=verbose_level, print_every=print_every)

    return parameters, ll_history

def calculate_log_likelihood(X, U, parameters):
    """
    Calculate the log likelihood per data point given the parameters

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
            'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
            None for missing measurements
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
            'U[seq_idx][time_step]' = the control of the seq_idx sequence at time time_step
    parameters: z_0_hat, P_0_hat, A, B, Q, R
        z_0_hat: [dimension] numpy array
            initial mean for every sequence
        P_0_hat: [dimension, dimension] numpy array
            initial covariance for every sequence
        A: [dimension, dimension] numpy array
            transition matrix
        B: [dimension, control_dimension] numpy array
            control matrix
        Q: [dimension, dimension] numpy array
            transition covariance matrix
        R: [dimension, dimension] numpy array
            emission covariance matrix

    Returns
    -------
    mskf.log_likelihood(): numpy float
        the log likelihood per data point
    """
    # initialize the parameters
    z_0_hat, P_0_hat, A, B, Q, R = parameters
    dimension, control_dimension = get_dimension(X), get_control_dimension(U)
    mskf = MSKF(dimension, control_dimension, z_0_hat, P_0_hat, A, B, Q, R)
    mskf.pass_in_measurement_and_control(X, U)
    
    return mskf.log_likelihood()

def infer(X, U, parameters):
    """
    Infer the hidden variables

    Parameters
    ----------
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
            'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
            None for missing measurements
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
            'U[seq_idx][time_step]' = the control of the seq_idx sequence at time time_step
    parameters: z_0_hat, P_0_hat, A, B, Q, R
        z_0_hat: [dimension] numpy array
            initial mean for every sequence
        P_0_hat: [dimension, dimension] numpy array
            initial covariance for every sequence
        A: [dimension, dimension] numpy array
            transition matrix
        B: [dimension, control_dimension] numpy array
            control matrix
        Q: [dimension, dimension] numpy array
            transition covariance matrix
        R: [dimension, dimension] numpy array
            emission covariance matrix

    Returns
    -------
    predicted: {String: array}
        'predicted['mean'][seq_idx][time_step]' = the mean of the prediction
        of the seq_idx sequence at time time_step
        'predicted['covariance'][seq_idx][time_step]' = the covariance of prediction
        of the seq_idx sequence at time time_step
    filtered: {String: array}
        'filtered['mean'][seq_idx][time_step]' = the mean of filtered value
        of the seq_idx sequence at time time_step
        'filtered['covariance'][seq_idx][time_step]' = the covariance of filtered value
        of the seq_idx sequence at time time_step
    smoothed: {String: array}
        'smoothed['mean'][seq_idx][time_step]' = the mean of smoothed value
        of the seq_idx sequence at time time_step
        'smoothed['covariance'][seq_idx][time_step]' = the covariance of smoothed value
        of the seq_idx sequence at time time_step
    """
    # initialize the parameters
    z_0_hat, P_0_hat, A, B, Q, R = parameters
    dimension, control_dimension = get_dimension(X), get_control_dimension(U)
    mskf = MSKF(dimension, control_dimension, z_0_hat, P_0_hat, A, B, Q, R)
    mskf.pass_in_measurement_and_control(X, U)
    
    # estimate
    predicted, filtered, smoothed = mskf.estimate()
    return predicted, filtered, smoothed

def simulate(U, parameters):
    """
    Simulate under the Kalman Filter assumptions given the parameters

    Parameters
    ----------
    U: [seq_count] array of [seq_length] array of ([dimension] numpy array)
            'U[seq_idx][time_step]' = the control of the seq_idx sequence at time time_step
    parameters: z_0_hat, P_0_hat, A, B, Q, R
        z_0_hat: [dimension] numpy array
            initial mean for every sequence
        P_0_hat: [dimension, dimension] numpy array
            initial covariance for every sequence
        A: [dimension, dimension] numpy array
            transition matrix
        B: [dimension, control_dimension] numpy array
            control matrix
        Q: [dimension, dimension] numpy array
            transition covariance matrix
        R: [dimension, dimension] numpy array
            emission covariance matrix

    Returns
    -------
    Z: [seq_count] array of [seq_length] array of ([dimension] numpy array)
            'Z[seq_idx][time_step]' = the true value of the hidden variable of the seq_idx sequence at time time_step
    X: [seq_count] array of [seq_length] array of ([dimension] numpy array/None)
            'X[seq_idx][time_step]' = the measurement of the seq_idx sequence at time time_step
    """
    # initialize the parameters
    z_0_hat, P_0_hat, A, B, Q, R = parameters
    dimension, control_dimension = B.shape
    sskf = SSKF(dimension, control_dimension, z_0_hat, P_0_hat, A, B, Q, R)
    Z, X = [], []
    
    # simulate
    for u in U:
        zs, xs = sskf.simulate(u)
        Z.append(zs)
        X.append(xs)
    return Z, X
