#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (C) Paolo Di Prodi 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__version__ = '0.1.0'
__author__ = 'Paolo Di Prodi <paolo@robomotic.com>'
__all__ = []

import numpy as np
from numpy.random import random

from diffprivlib.mechanisms.base import DPMechanism


"""
This class implements the generalized randomized response.

Papers used as reference:

@inproceedings{Wang2016UsingRR,
  title={Using Randomized Response for Differential Privacy Preserving Data Collection},
  author={Yue Wang and Xintao Wu and Donghui Hu},
  booktitle={EDBT/ICDT Workshops},
  year={2016}
}

"""

import math
import warnings
from typing import Dict, Tuple, Sequence

class GeneralRandomizedBinaryResponse(DPMechanism):
    '''
    Suppose there are n individuals C_1,...,C_n and each individual C_i has a private binary value x_i in {0,1} regarding
    a sensitive binary attribute X. To ensure privacy, each individual C_i sends to the untrusted curator a modified version
    y_i of the x_i. Using the randomized response, the server can collect perturbed data from individuals.

    '''

    def __init__(self,p_00:float=.5,p_01:float=.5,p_10:float=.5,p_11:float=.5)->None:

        super().__init__()
        self._value0 = None
        self._value1 = None


    def __repr__(self):
        output = super().__repr__()
        output += ".set_labels(" + str(self._value0) + ", " + str(self._value1) + ")" \
            if self._value0 is not None else ""

        return output

    def set_p_matrix(self,p_00:float=.5,p_01:float=.5,p_10:float=.5,p_11:float=.5)->None:

        '''
        Constructor where p_uv= P[y_i = u | x_i = v] and u,v in {0,1} denotes the probability that the random output
        is u when the real attribute value x_i for C_i is v; here p_uv in (0,1).
        The sum of probabilities for each colum and row is 1.

        :param p_00: P[y_i = 0 | x_i = 0]
        :param p_01: P[y_i = 0 | x_i = 1]
        :param p_10: P[y_i = 1 | x_i = 0]
        :param p_11: P[y_i = 1 | x_i = 1]
        '''

        # P is a 2x2 design matrix row by row
        self._P = [[p_00,p_01],[p_10,p_11]]

        # check probability bounds
        if not self.check_probability(self._P[0][0]):
            raise ValueError('Probability must be within range [0,1]')
        if not self.check_probability(self._P[0][1]):
            raise ValueError('Probability must be within range [0,1]')
        if not self.check_probability(self._P[1][0]):
            raise ValueError('Probability must be within range [0,1]')
        if not self.check_probability(self._P[1][1]):
            raise ValueError('Probability must be within range [0,1]')

        # check that everything sums to 1
        if (self._P[0][0] + self._P[0][1]) != 1.0:
            raise Exception('Probability must sum to 1')
        if (self._P[1][0] + self._P[1][1]) != 1.0:
            raise Exception('Probability must sum to 1')

        if (self._P[0][0] == self._P[1][1]) == 1.0:
            warnings.warn('This is equivalent to direct questioning')


    def set_labels(self, value0, value1):
        """Sets the binary labels of the mechanism.

        Labels must be unique, non-empty strings.  If non-string labels are required, consider using a
        :class:`.DPTransformer`.

        Parameters
        ----------
        value0 : str
            0th binary label.

        value1 : str
            1st binary label.

        Returns
        -------
        self : class

        """
        if not isinstance(value0, str) or not isinstance(value1, str):
            raise TypeError("Binary labels must be strings. Use a DPTransformer  (e.g. transformers.IntToString) for "
                            "non-string labels")

        if len(value0) * len(value1) == 0:
            raise ValueError("Binary labels must be non-empty strings")

        if value0 == value1:
            raise ValueError("Binary labels must not match")

        self._value0 = value0
        self._value1 = value1
        return self

    def check_inputs(self, value):
        """Checks that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        Parameters
        ----------
        value : str
            The value to be checked.

        Returns
        -------
        True if the mechanism is ready to be used.

        Raises
        ------
        Exception
            If parameters have not been set correctly, or if `value` falls outside the domain of the mechanism.

        """
        super().check_inputs(value)

        if (self._value0 is None) or (self._value1 is None):
            raise ValueError("Binary labels must be set")

        if not isinstance(value, str):
            raise TypeError("Value to be randomised must be a string")

        if value not in [self._value0, self._value1]:
            raise ValueError("Value to be randomised is not in the domain {\"" + self._value0 + "\", \"" + self._value1
                             + "\"}")

        return True

    def set_optimal_utility(self,eps:float)->None:
        '''
        Set optimal design matrix based on epsilon parameter
        :param eps: the epsilon parameter
        :return: None
        '''
        self.set_epsilon(eps)
        p_00 = p_11 = math.exp(eps)/(1+ math.exp(eps))

        if math.isnan(p_11):
            p_11 = 1.0
        if math.isnan(p_00):
            p_00 = 1.0

        p_01 = p_10 = 1/(1+ math.exp(eps))

        if math.isnan(p_01):
            p_01 = 1.0
        if math.isnan(p_10):
            p_10 = 1.0

        # change the P matrix
        self._P = [[p_00,p_01],[p_10,p_11]]

    def check_eps_privacy(self,eps:float,tol:float=0)->bool:
        '''
        Check that the design matrix is eps-differentially private
        :param eps: the epsilon parameter
        :return: true if satisfied
        '''

        if self._P[0][1] == .0:
            p = float("inf")
        else:
            p = self._P[0][0] / self._P[0][1]

        if self._P[1][0] == .0:
            q = float("inf")
        else:
            q = self._P[1][1] / self._P[1][0]

        if max(p,q) <= math.exp(eps)+tol: return True
        else: return False

    def get_P_X_mass(self,pi_0:float=None,pi_1:float=None)->Dict[str, float]:
        '''

        :param pi_0: true proportion of individuals without sensitive attribute
        :param pi_1: true proportion of individuals with sensitive attribute
        :return:
        '''
        mass = {'P(Y=0)':None,'P(Y=1)':None}

        if pi_0 is not None: pi_1 = 1.0 - pi_0
        if pi_1 is not None: pi_0 = 1.0 - pi_1

        mass['P(Y=0)'] = pi_0 * self._P[0][0] + pi_1 * self._P[0][1]
        mass['P(Y=1)'] = pi_1 * self._P[1][1] + pi_0 * self._P[1][0]

        return mass

    def get_unbiased_mean_estimator(self,lambda_0:float=None,lambda_1:float=None)->Dict[str, float]:
        '''
        We use pi_0 (pi_1) to describe the true proportion of value 0 (1)
        We use pihat_0 (pihat_1) to described the unbiased estimator for pi_0 (pi_1)
        We use lambda_0 (lambda_1) to describe the observed proportion of value 0 (1)

        :return: A dictiionary with he unbiased estimators pihat_0, pihat_1
        '''

        _ = {'pihat_0':None,'pihat_1':None}
        if lambda_0 is not None:
            if lambda_0 >=.0 and lambda_0<=1.0:
                _['pihat_0'] = (self._P[0][0] - 1.0 )/ (2* self._P[0][0] -1) + lambda_0/(2* self._P[0][0] -1)
            else: raise Exception('This is not a probability')

        if lambda_1 is not None:
            if lambda_1 >= .0 and lambda_1 <= 1.0:
                _['pihat_1'] = (self._P[0][0] - 1.0 )/ (2* self._P[0][0] -1) + lambda_1/(2* self._P[0][0] -1)
            else: raise Exception('This is not a probability')

        return _

    def get_unbiased_variance_estimator(self):
        pass



    def randomise(self, value):
        """Randomise `value` with the mechanism.

        Parameters
        ----------
        value : str
            The value to be randomised.

        Returns
        -------
        str
            The randomised value.

        """
        self.check_inputs(value)

        indicator = 0 if value == self._value0 else 1

        P_Y_0 = self._P[0][indicator]
        P_Y_1 = self._P[1][indicator]

        if P_Y_1 >= 1.0:
            return self._value1
        elif P_Y_0 >= 1.0:
            return self._value0
        else:
            unif_rv = random()

            if unif_rv <= P_Y_0:
                return self._value0
            if unif_rv <= P_Y_1:
                return self._value1