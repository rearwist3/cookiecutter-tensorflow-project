from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class TemplateSolver(BaseSolve):
    def __init__(self, sess, model, data, config, logger):
        super(TemplateSolver, self).__init__(sess, model, data, config, logger)

    def fit(self):
        """
       implement the logic of training:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        pass

    def predict(self):
        """
       implement the logic of the prediction
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        pass
