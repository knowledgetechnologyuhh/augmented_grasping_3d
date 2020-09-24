import configparser
import jsonpickle

def read_config_file(config_path):
    if config_path.endswith('.ini'):
        config = load_ini(config_path)
        if config is None:
            config = load_json(config_path)
    elif config_path.endswith('.json'):
        config = load_json(config_path)
    else:
        # format does not match the object, but will be read nevertheless
        config = load_json(config_path)
    return config

def write_config_file(config, config_path):
    if isinstance(config, configparser.ConfigParser) and config_path.endswith('.ini'):
        store_ini(config, config_path)
    elif not isinstance(config, configparser.ConfigParser) and config_path.endswith('.json'):
        store_json(config, config_path)
    else:
        # format does not match the object, but will be saved nevertheless
        store_json(config, config_path)

def load_json(json_file):
    # convert args to dict
    with open(json_file, 'r') as fobj:
        json_obj = fobj.read()
        obj = jsonpickle.decode(json_obj)
    return obj


def store_json(store_object, json_file):
    # convert args to dict
    with open(json_file, 'w') as fobj:
        json_obj = jsonpickle.encode(store_object)
        fobj.write(json_obj)


def load_ini(ini_file):
    config = configparser.ConfigParser()
    try:
        config.read(ini_file)
    except:
        return None
    return config


def store_ini(config, ini_file):
    with open(ini_file, 'w') as config_file:
        config.write(config_file)