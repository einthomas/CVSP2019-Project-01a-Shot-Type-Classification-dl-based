import yaml


class Config:
    def __init__(self):
        self.config = {}

    def loadConfig(self, path):
        with open(path) as configFile:
            self.config = yaml.full_load(configFile)


configHandler = Config()

# def getConfigRelativePath(key):
#    return os.path.join(config['workingDirectory'], config[key])
