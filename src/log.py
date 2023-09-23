import os
import logging
import time


class Logger:
    def __init__(self, my_log_name, mode='a'):
        # (1) create a logger
        self.logger = logging.getLogger(my_log_name)
        self.logger.setLevel(logging.INFO)
        # (2) create a handler which is used to write logs
        log_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
        # (get working dir)/logs/
        log_path = os.getcwd() + '/{}/'.format(my_log_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = log_path + log_time + '.logs'
        # log_file = log_path + my_log_name + '.logs'
        log_file_handler = logging.FileHandler(log_file, mode=mode)
        log_file_handler.setLevel(logging.DEBUG)
        # (3) set the print format of handler
        log_format = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        log_file_handler.setFormatter(log_format)
        # (4) add logger to handler
        self.logger.addHandler(log_file_handler)
        log_stream_handler = logging.StreamHandler()
        log_stream_handler.setLevel(logging.INFO)
        log_stream_handler.setFormatter(log_format)
        self.logger.addHandler(log_stream_handler)
