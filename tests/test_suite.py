import unittest
from tests.analyzers.test_sensitivity_analyzer import TestSensitivityAnalyzer

# Create a Test Suite
def suite():
    test_suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()

    # Add test cases from test_calculations.py
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestSensitivityAnalyzer))

    return test_suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
