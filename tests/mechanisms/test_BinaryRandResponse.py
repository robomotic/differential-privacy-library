import numpy as np
from unittest import TestCase
import pytest
from diffprivlib.mechanisms import GeneralRandomizedBinaryResponse
from diffprivlib.utils import global_seed


class TestBinaryRandResponse(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = GeneralRandomizedBinaryResponse()

    def teardown_method(self, method):
        del self.mech

    def test_wrong_matrix(self):
        with pytest.raises(Exception):
            self.mech.set_p_matrix(p_00=1.0, p_01=1.0, p_10=1.0, p_11=1.0)

        with pytest.raises(ValueError):
            self.mech.set_p_matrix(p_00=1.1, p_01=0.9, p_10=1.0, p_11=1.0)

    def test_good_matrix(self):
        self.mech.set_p_matrix(p_00=.9, p_01=.1, p_10=.1, p_11=.9)

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(GeneralRandomizedBinaryResponse, DPMechanism))

    def test_no_labels(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.randomise("1")

    def test_no_epsilon(self):
        self.mech.set_labels("0", "1")
        with self.assertRaises(ValueError):
            self.mech.randomise("1")

    def test_inf_epsilon(self):
        self.mech.set_labels("0", "1").set_optimal_utility(float('inf'))

        for i in range(1000):
            self.assertEqual(self.mech.randomise("1"), "1")
            self.assertEqual(self.mech.randomise("0"), "0")

    def test_eps_privacy(self):
        # this should be maximum privacy since the responses have 0 utility
        self.mech.set_p_matrix(p_00=.5, p_01=.5, p_10=.5, p_11=.5)

        is_true = self.mech.check_eps_privacy(0.0)

        assert is_true == True

        # this should be maximum utility since the responses are equivalent to direct questions
        self.mech.set_p_matrix(p_00=1.0, p_01=.0, p_10=0.0, p_11=1.0)

        is_false = self.mech.check_eps_privacy(100.0)

        assert is_false == False

        # this should correspond to infinite epsilon
        is_big = self.mech.check_eps_privacy(float('inf'))

        assert is_big == True

    def test_optimal_rr(self):

        target_eps = 1.0
        self.mech.set_optimal_utility(target_eps)

        is_true = self.mech.check_eps_privacy(target_eps, tol=5e-16)

        assert is_true == True

    def test_mass_rr(self):
        def seq_floats(start, stop, step=1):
            stop = stop - step;
            number = int(round((stop - start) / float(step)))
            if number > 1:
                return ([start + step * i for i in range(number + 1)])
            elif number == 1:
                return ([start])
            else:
                return ([])

        self.mech.set_p_matrix(p_00=1.0, p_01=.0, p_10=0.0, p_11=1.0)

        # percentage of yes in truth
        for pi_1 in seq_floats(0.0, 1.1, 0.1):
            P_X = self.mech.get_P_X_mass(pi_1=pi_1)

            assert round(P_X['P(Y=0)'], 1) == round(1.0 - pi_1, 1)
            assert round(P_X['P(Y=1)'], 1) == round(pi_1, 1)

        # percentage of no in truth
        for pi_0 in seq_floats(0.0, 1.1, 0.1):
            P_X = self.mech.get_P_X_mass(pi_0=pi_0)

            assert round(P_X['P(Y=0)'], 1) == round(pi_0, 1)
            assert round(P_X['P(Y=1)'], 1) == round(1.0 - pi_0, 1)

    def test_complex_epsilon(self):
        self.mech.set_labels("0", "1")
        with self.assertRaises(TypeError):
            self.mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        self.mech.set_labels("0", "1")
        with self.assertRaises(TypeError):
            self.mech.set_epsilon("Two")

    def test_non_string_labels(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(TypeError):
            self.mech.set_labels(0, 1)


    def test_empty_label(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.set_labels("0", "")

    def test_same_labels(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.set_labels("0", "0")
