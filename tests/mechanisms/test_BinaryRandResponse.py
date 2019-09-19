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
