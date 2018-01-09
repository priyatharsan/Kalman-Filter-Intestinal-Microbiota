from Utils import get_dimension, get_control_dimension
from EM_Config import EM_Config
from Multiple_Sequence_Kalman_Filter import MSKF
from Single_Sequence_Kalman_Filter import SSKF

'''
    Expectation Maximization
    
    # Arguments
        X: measurement
        U: controls
        config: the configuration of EM
        verbose_level: (0) print nothing (1) print info at the end of EM (2) print info at every iteration of EM
        print_every: print the log likelihood every print_every iterations
    # Returns
        parameters: the parameters found by the EM algorithm
'''
def expectation_maximization(X, U, config=None, verbose_level=2, print_every=10):
    
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

'''
    Calculate the log likelihood of measurements given measurements, controls and parameters
    
    # Arguments
        X: measurements
        U: controls
        parameters: z_0_hat, P_0_hat, A, B, Q, R
    # Returns
        log_likelihood: the log likelihood of the measurements
'''
def calculate_log_likelihood(X, U, parameters):
    
    # initialize the parameters
    z_0_hat, P_0_hat, A, B, Q, R = parameters
    dimension, control_dimension = get_dimension(X), get_control_dimension(U)
    mskf = MSKF(dimension, control_dimension, z_0_hat, P_0_hat, A, B, Q, R)
    mskf.pass_in_measurement_and_control(X, U)
    
    return mskf.log_likelihood()

'''
    Predict, Filter, Smooth given measurements, controls, and parameters
    
    # Arguments
        X: measurements
        U: controls
        parameters: z_0_hat, P_0_hat, A, B, Q, R
    # Returns
        predicted: predicted values at each time step;
            it is a dictionary with entries 'mean'/'covariance';
            the values of the dictionary is a 2D array (seq_count * seq_length) of numpy array
        filtered: filtered values at each time step
        smoothed: smoothed values (the expectation) at each time step
'''
def infer(X, U, parameters):
    
    # initialize the parameters
    z_0_hat, P_0_hat, A, B, Q, R = parameters
    dimension, control_dimension = get_dimension(X), get_control_dimension(U)
    mskf = MSKF(dimension, control_dimension, z_0_hat, P_0_hat, A, B, Q, R)
    mskf.pass_in_measurement_and_control(X, U)
    
    # estimate
    predicted, filtered, smoothed = mskf.estimate()
    return predicted, filtered, smoothed

'''
    Simulate under the Kalman Filter assumption
    
    # Arguments
        U: controls
        parameters: z_0_hat, P_0_hat, A, B, Q, R
    # Returns
        Z: simulated hidden values
        X: simulated measurements
'''
def simulate(U, parameters):
    
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
