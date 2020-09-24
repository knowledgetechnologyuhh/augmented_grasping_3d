import configparser


def read_config_file(config_path):
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
    except:
        pass
    return config


def write_config_file(config, config_path):
    with open(config_path, 'w') as config_file:
        config.write(config_file)