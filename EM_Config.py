import numpy as np

class EM_Config:
    '''
        Options for the Expectation Maximization algorithm configuration
        Refer to the output of print_em_manual function for further details
    '''
    CONFIG_KEYS = ['z_0_hat_option', 'initial_z_0_hat', 'P_0_hat_option', 'initial_P_0_hat',
                   'AB_option', 'initial_A', 'initial_B', 'Q_option', 'initial_Q',
                   'R_option', 'initial_R', 'threshold', 'num_iterations']

    OPTIONS_CHOICES = ['fixed', 'flexible', 'diag', 'scalar']
    
    def __init__(self, dimension, control_dimension, config=None):
        self.dimension = dimension
        self.control_dimension = control_dimension
        config_proposal = self.get_default_config()
        success_flag = True
        
        # check whether all keys are valid
        if config is not None:
            for key in config:
                if key not in EM_Config.CONFIG_KEYS:
                    print('WARNING: Key %s is not one of the config args, it is ignored' % key)
                    success_flag = False
                if config[key] is not None:
                    config_proposal[key] = config[key]
    
        # check whether all avlues are valid
        for key in config_proposal:
            if 'option' in key and config_proposal[key] not in EM_Config.OPTIONS_CHOICES:
                print('WARNING: Key word \"%s\" not understood for key: %s' % (config_proposal[key], key))
                success_flag = False
        
        self.config = config_proposal
        # ask user to refer to the manual for further details
        if not success_flag:
            print('Please use the print_em_manual function to know more')

    def print_em_manual():
        print('------------ EM_Config Manual ------------')
        print('The list of keys for the configuration')
        print(EM_Config.CONFIG_KEYS)
        print()
        print('--- Option explanations ---')
        print('<parameter_name>_options available choices')
        print(EM_Config.OPTIONS_CHOICES)
        print('fixed: fix the parameter during training time')
        print('flexible: no constraint during traning time')
        print('diag: keep the parameter a diagnol matrix, only available for P_0_hat, Q, R')
        print('scalar: keep the parameter a scalar time identity matrix, only available for P_0_hat, Q, R')
        print('--- Option explanations ---')
        print()
        print('initial_<parameter_name> is the initial value of the EM algorithm for <parameter_name>')
        print()
        print('--- Stopping Criteria ---')
        print('threshold: considered converge whenever the improvement of log likelihood is less than threshold')
        print('num_iterations: perform EM algorithm of num_iterations')
        print('stop whenever either criteria is reached')
        print('--- Stopping Criteria ---')
        print()
        print('------------ EM_Config Manual ------------')

    '''
        Print the configuration to screen
    '''
    def print_config(self):
        for key in CONFIG_KEYS:
            print('--- ' + key + ' ---')
            print(CONFIG_KEYS[key])

    '''
        # Return:
            The return configuration dict
    '''
    def get_default_config(self):
        
        config = {}
        
        # default z_0_hat, zeros, flexible
        config['z_0_hat_option'] = 'flexible'
        config['initial_z_0_hat'] = np.zeros(self.dimension)
        
        # default P_0_hat, identity times a small scalar, flexible
        config['P_0_hat_option'] = 'flexible'
        config['initial_P_0_hat'] = 0.1 * np.eye(self.dimension)
        
        # default A, identity, flexible
        config['AB_option'] = 'flexible'
        config['initial_A'] = np.eye(self.dimension)
        config['initial_B'] = np.zeros((self.dimension, self.control_dimension))
        
        # default Q, identity times a small scalar, flexible
        config['Q_option'] = 'flexible'
        config['initial_Q'] = 0.1 * np.eye(self.dimension)
        
        # default R, identity times a small scalar, flexible
        config['R_option'] = 'flexible'
        config['initial_R'] = 0.1 * np.eye(self.dimension)
        
        # default stopping criteria, threshold 1e-5, num_iterations 1000
        # stop whenever either of the two critieria is reached
        config['threshold'] = 1e-5
        config['num_iterations'] = 1000

        return config
    
    '''
        Override the [] operators
    '''
    def __getitem__(self, key):
        if key not in self.config:
            print('WARNING: Key %s is not one of the config args, it is ignored' % key)
        return self.config[key]

    def __setitem__(self, key, value):
        if key not in self.config:
            print('WARNING: Key %s is not one of the config args, it is ignored' % key)
        if 'option' in key and value not in EM_Config.OPTIONS_CHOICES:
            print('WARNING: Key word \"%s\" not understood for key: %s' % (config_proposal[key], key))
        self.config[key] = value

