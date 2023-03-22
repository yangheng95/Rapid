import logging
import os
import sys
import time

import termcolor

today = time.strftime("%Y%m%d %H%M%S", time.localtime(time.time()))


def get_logger(log_path, log_name="", log_type="training_log"):
    if not log_path:
        log_dir = os.path.join(log_path, "logs")
    else:
        log_dir = os.path.join(".", "logs")

    full_path = os.path.join(log_dir, log_name + "_" + today)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    log_path = os.path.join(full_path, "{}.log".format(log_type))
    logger = logging.getLogger(log_name)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = formatter
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)

    return logger
