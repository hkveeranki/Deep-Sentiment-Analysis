"""
Class to hold results after testing a model
"""


class TestResults(object):
    def __init__(self, f1_score, precision, recall):
        """
        Constructor for the class
        :param f1_score: f1 score of the results
        :param precision: precision of the results
        :param recall: recall of the results
        """
        self.f1_score = f1_score
        self.precision = precision
        self.recall = recall
