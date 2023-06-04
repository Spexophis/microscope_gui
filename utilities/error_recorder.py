import logging


class ErrorRecorder:

    def __init__(self, log_file):
        self.error_log = self.setup_logger(log_file+'.log')

    @staticmethod
    def setup_logger(log_file):
        logging.basicConfig(filename=log_file, level=logging.ERROR)
        logger = logging.getLogger('shared_logger')
        return logger
