import unittest


def suite():
    """Aggregate unittest test cases. Add new TestCase classes here as they land.

    (The legacy sensitivity-analyzer test was removed in the refactor; rewrite it
    against the current API before re-adding.)
    """
    test_suite = unittest.TestSuite()
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
