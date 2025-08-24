import logging
from logging import config
import os
import yaml


class LogHelper:
    # src\config\log_config.yaml
    def __init__(self, log_config_path='./src/config/log_config.yaml', default_log_level=logging.INFO):
        self.log_config_path = log_config_path
        self.default_log_level = default_log_level
        self.init_log()

    def init_log(self):
        if os.path.exists(self.log_config_path):
            with open(self.log_config_path, 'rt') as f:
                yaml_config = yaml.safe_load(f)
            config.dictConfig(yaml_config)
        else:
            logging.basicConfig(level=self.default_log_level,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)
