import os
import yaml

executablePath = os.path.dirname(os.path.realpath(__file__))

config = {}
with open(os.path.join(executablePath, '..', 'config.yaml')) as configFile:
    config = yaml.full_load(configFile)

def getConfigRelativePath(key):
    return os.path.join(config['workingDirectory'], config[key])
