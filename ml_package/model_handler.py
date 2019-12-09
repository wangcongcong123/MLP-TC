import pickle
import os
from ml_package.ml_models import *
import json
import hashlib
import copy

def save_model(model, configs):
    """
    save a model locally if it is trained and not found locally
    :param model: the model to be saved
    :param configs:
    :return:
    """
    hashcode = get_hash_code(copy.deepcopy(configs))
    model_path = model.get_model_name() + hashcode + ".pkl"
    if os.path.exists(model_path):
        logging.info("the model exists locally already")
    else:
        logging.info("save model...")
        with open(model_path, "wb") as handle:
            pickle.dump(model, handle)
        logging.info("model saved successfully to " + model.get_model_name() + ".pkl")

def load_model(model_path):
    """
    load a model from model path
    :param model_path:
    :return:
    """
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    return model

def get_hash_code(configs):
    """
    get hash code based on configs (without the model attribute in)
    :param configs:
    :return:
    """
    del configs["model"]
    return hashlib.md5(json.dumps(configs).encode('utf-8')).hexdigest()

def get_model(configs):
    """
    initialize a model by the "model" attribute in configs or load from a local file directly if it exists
    :param configs:
    :return:
    """
    hashcode = get_hash_code(copy.deepcopy(configs))
    model_path = type(configs["model"]).__name__ + hashcode + ".pkl"
    if os.path.exists(model_path):
        logging.info("load a model locally...")
        model = load_model(model_path)
    else:
        logging.info("the model is not found locally, so initialize the model...")
        model = MLModel(configs)
    return model
