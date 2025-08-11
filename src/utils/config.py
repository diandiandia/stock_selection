import yaml


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            self.log_config = self.config['log']
            self.data_config = self.config['data']
            self.stock_config = self.config['stock']
            self.indicator_config = self.config['indicator']
            self.strategy_config = self.config['strategy']
            self.analysis_config = self.config['analysis']  
            self.visualization_config = self.config['visualization']
            self.visualization_config = self.config['visualization']    
            self.visualization_config = self.config['visualization']
