import pickle
import re

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

from utils.general_utils import (
    get_scaled_down_scores,
    get_score_vector_positions,
)


def text_preprocess(text):
    text = text.strip()
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"-?\d+(\.\d+)?", "[NUM]", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace(u'"', u"")
    text = text.replace("�", "")
    text = re.sub(r"@([A-Z]+)[0-9]+", r"[\1]", text)  # @CAP1 -> [CAP]
    
    if "..." in text:
        text = re.sub(r"\.{3,}(\s+\.{3,})*", "...", text)
    if "??" in text:
        text = re.sub(r"\?{2,}(\s+\?{2,})*", "?", text)
    if "!!" in text:
        text = re.sub(r"\!{2,}(\s+\!{2,})*", "!", text)
    return text


def get_readability_features(readability_path):
    with open(readability_path, "rb") as fp:
        readability_features = pickle.load(fp)
    return readability_features


def get_linguistic_features(linguistic_features_path):
    features_df = pd.read_csv(linguistic_features_path)
    return features_df


def get_normalized_features(features_df):
    column_names_not_to_normalize = ["item_id", "prompt_id", "score"]
    column_names_to_normalize = list(features_df.columns.values)
    for col in column_names_not_to_normalize:
        column_names_to_normalize.remove(col)
    final_columns = ["item_id"] + column_names_to_normalize
    normalized_features_df = None
    for prompt_ in range(1, 9):
        is_prompt_id = features_df["prompt_id"] == prompt_
        prompt_id_df = features_df[is_prompt_id]
        x = prompt_id_df[column_names_to_normalize].values
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_pd1 = min_max_scaler.fit_transform(x)
        df_temp = pd.DataFrame(
            normalized_pd1, 
            columns=column_names_to_normalize, 
            index=prompt_id_df.index
        )
        prompt_id_df = prompt_id_df.copy()
        prompt_id_df[column_names_to_normalize] = df_temp
        final_df = prompt_id_df[final_columns]
        if normalized_features_df is not None:
            normalized_features_df = pd.concat(
                [normalized_features_df, final_df], 
                ignore_index=True
            )
        else:
            normalized_features_df = final_df
            
    return normalized_features_df


def read_data(
    essay_list, readability_features, normalized_features_df, 
    prompt_data, preprocessing=False, normalize_score=False
):
    out_data = {
        "prompt_id": [],
        "essay_id": [],
        "prompt_text": [],
        "essay_text": [],
        "essay_readability": [],
        "essay_features": [],
        "scores": [],
        
    }
    
    for essay in tqdm(essay_list, desc="Loading the data"):
        essay_id = int(essay["essay_id"])
        essay_set = int(essay["prompt_id"])
        content = essay["content_text"]

        out_data["prompt_id"].append(essay_set)
        out_data["essay_id"].append(essay_id)
        
        # perparing essay text
        out_data["essay_text"].append(
            text_preprocess(content) if preprocessing else content
        )

        # perparing prompt text
        prompt = prompt_data.loc[prompt_data["prompt_id"] == 1, "prompt"].tolist()[0]
        out_data["prompt_text"].append(
            text_preprocess(prompt) if preprocessing else prompt
        )
        
        # perparing score datas
        scores_and_positions = get_score_vector_positions()
        y_vector = [-1] * len(scores_and_positions)
        for score in scores_and_positions.keys():
            if score in essay.keys():
                y_vector[scores_and_positions[score]] = int(essay[score])
        out_data["scores"].append(y_vector)

        # perparing essay readability feature
        item_index = np.where(readability_features[:, :1] == essay_id)
        item_row_index = item_index[0][0]
        item_features = readability_features[item_row_index][1:]
        out_data["essay_readability"].append(item_features)
        feats_df = normalized_features_df[normalized_features_df.loc[:, "item_id"] == essay_id]

        # perparing essay features(hand craft feature)
        feats_list = feats_df.values.tolist()[0][1:]
        out_data["essay_features"].append(feats_list)
        
    if normalize_score:
        # normalize score
        out_data["norm_scores"] = get_scaled_down_scores(
            out_data["scores"], 
            out_data["prompt_id"]
        )
    
    return out_data