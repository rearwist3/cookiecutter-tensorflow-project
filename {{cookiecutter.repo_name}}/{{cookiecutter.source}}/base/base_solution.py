import tensorflow as tf


class BaseSolution:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def fit(self):
        """
        implement the logic of training:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def predict(self):
        """
        implement the logic of the prediction
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
