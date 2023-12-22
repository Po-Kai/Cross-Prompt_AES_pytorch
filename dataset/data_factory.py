import pickle

import pandas as pd

from .reader import (
    get_readability_features,
    get_linguistic_features,
    get_normalized_features,
    read_data
)


class ASAPDataFactory(object):

    def __init__(self, data_configs):
        readability_features = get_readability_features(data_configs["readability_path"])
        linguistic_features = get_linguistic_features(data_configs["features_path"])
        normalized_linguistic_features = get_normalized_features(linguistic_features)
        prompt_data = pd.read_csv(data_configs["prompt_path"])

        self.readability_features = readability_features
        self.normalized_linguistic_features = normalized_linguistic_features
        self.prompt_data = prompt_data
        self.data_configs = data_configs
        self.caches = {}

    def create_data(self, subset="train"):
        if subset not in self.caches:
            with open(self.data_configs[f"{subset}_path"], "rb") as train_file:
                essays_list = pickle.load(train_file)
            data = read_data(
                essays_list, 
                self.readability_features, 
                self.normalized_linguistic_features, 
                self.prompt_data,
                preprocessing=self.data_configs["preprocessing"],
                normalize_score=self.data_configs["normalize_score"]
            )
            self.caches[subset] = data

        return self.caches[subset]