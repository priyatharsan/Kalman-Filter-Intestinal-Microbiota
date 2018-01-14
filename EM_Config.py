"""
==========================================
Expectation Maximization algorithm options
==========================================

This module implements a wrapper for the dictionary
that specifies the EM algorithm's configuration
"""

import numpy as np


class EM_Config:
    """
        A wrapper class for Expectation Maximization algorithm options dictionary (config)
        Refer to the output of print_em_manual function for further details
    """
    CONFIG_KEYS = ['z_0_hat_option', 'initial_z_0_hat', 'P_0_hat_option', 'initial_P_0_hat',
                   'AB_option', 'initial_A', 'initial_B', 'Q_option', 'initial_Q',
                   'R_option', 'initial_R', 'threshold', 'num_iterations']

    OPTIONS_CHOICES = ['fixed', 'flexible', 'diag', 'scalar']
    
    def __init__(self, dimension, control_dimension, config=None):
        """
        Initializing a EM configuration

        Parameters
        ----------
        dimension: int
            dimension of x (measurement)
        control_dimension: int
            dimension of u (control)
        config: {String: String}
            specified EM config dict
        """
        self.dimension = dimension
        self.control_dimension = control_dimension

        # except the options specified in config argument
        # set all others to default
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
        """
        Print the explanation of the options/configs for the EM algorithm
        """
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

    def print_config(self):
        """
        Print the configuration to screen
        """
        for key in CONFIG_KEYS:
            print('--- ' + key + ' ---')
            print(CONFIG_KEYS[key])

    def get_default_config(self):
        """
        Get the default EM configuration dict

        Returns
        -------
        config: {String: String}
            return configuration dict
        """
        
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
    
    def __getitem__(self, key):
        """
        Override the [] operators, get item

        Parameters
        ----------
        key: String
            key of config dict

        Returns
        -------
        config[key]: String
            value of key in config dict
        """
        if key not in self.config:
            print('WARNING: Key %s is not one of the config args, it is ignored' % key)
        return self.config[key]

    def __setitem__(self, key, value):
        """
        Override the [] operators, set item

        Parameters
        ----------
        key: String
            the key to be set
        value: String
            the value the key corresponds
        """
        if key not in self.config:
            print('WARNING: Key %s is not one of the config args, it is ignored' % key)
        if 'option' in key and value not in EM_Config.OPTIONS_CHOICES:
            print('WARNING: Key word \"%s\" not understood for key: %s' % (config_proposal[key], key))
        self.config[key] = value
