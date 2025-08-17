# Util file

import numpy as np
from enum import unique
import os
import sys
import cv2
import sys
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from sklearn.utils import shuffle
import json
import pickle
import tensorflow.keras as K
import tensorflow as tf
import tifffile


def normalize8(I):
    # print(I.dtype)
    # print(type(I))
    if isinstance(I, tf.Tensor):
        I = I.numpy()
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn) / mx) * 255
    return I.astype(np.uint8)


def Binarize_labels(label):
    label = np.array(label)
    mlb = LabelBinarizer()
    new_labels = mlb.fit_transform(label)
    return new_labels, mlb.classes_, mlb


def split_data_val_test(df):
    # Here the validation data is a two image with all correspondings z planes patches for each class
    # Here the validation data is a two image with all correspondings z planes patches for each class
    # Validation images:  FoF2_231005_fluorescent.nucleus  and FoF3001_220523_brightfield
    # Testing images: FoF2002004_221018_brightfield and FoF3_231005_fluorescent.nucleus
    print(df.shape)
    val_df = df(
        df["image"].str.startswith("FoF2_231005_fluorescent.nucleus")
        or df["image"].str.startswith("FoF3001_220523_brightfield")
    )
    print(val_df.shape)


def load_cellCycleData(path2Data):
    list_of_names = []
    list_of_image_ids = []
    list_of_labels = []
    df = pd.DataFrame()
    for FoF in os.listdir(path2Data):
        if FoF.startswith("._") or not os.path.isdir(
            os.path.join(path2Data, FoF)
        ):
            continue
        for CellCycle in os.listdir(
            os.path.join(path2Data, FoF)
        ):
            if not CellCycle.startswith("cellCycle"):
                continue
            for ImageName in os.listdir(os.path.join(path2Data,FoF,CellCycle,"images_padded")):
                imagePath = os.path.join(FoF, CellCycle,"images_padded",ImageName)
                image_label = int(ImageName.split("cellCycle")[1].split("_")[0])
                image_id = FoF +'_'+ ImageName.split('_')[3]
                list_of_names.append(imagePath)
                list_of_image_ids.append(image_id)
                list_of_labels.append(image_label - 1)
    df["image"] = list_of_names
    df['image_id'] = list_of_image_ids
    df["label"] = list_of_labels
    return df


# Load the keras models
def load_keras_model(filePath):
    if filePath.endswith(".keras") and not filePath.startswith("._"):
        model = K.models.load_model(filePath)
        return model

    model_names = [
        file
        for file in os.listdir(filePath)
        if file.endswith(".keras") and not file.startswith("._")
    ]
    if len(model_names) > 1:
        print(
            "There should be a single model to use for fine-tuning, please specify the model name in -path2Model argument"
        )
        sys.exit()
    else:
        model = K.models.load_model(os.path.join(filePath, model_names[0]))
        return model


def split_dataframe(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Splits a DataFrame into train, validation, and test sets, ensuring no overlap of image_id.
    Removes image_id rows that do not appear at least 10 times in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing an 'image_id' column.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Three DataFrames (train_df, val_df, test_df).
    """
    if 'image_id' not in df.columns:
        raise ValueError("The input DataFrame must contain an 'image_id' column.")

    # Filter out image_ids that appear less than 10 times
    image_id_counts = df['image_id'].value_counts()
    valid_image_ids = image_id_counts[image_id_counts >= 10].index
    df = df[df['image_id'].isin(valid_image_ids)]

    # Shuffle the DataFrame
    df = shuffle(df, random_state=random_state)

    # Get unique image_ids
    unique_image_ids = df['image_id'].unique()

    # Split unique image_ids
    total = len(unique_image_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_ids = unique_image_ids[:train_end]
    val_ids = unique_image_ids[train_end:val_end]
    test_ids = unique_image_ids[val_end:]

    # Create splits
    train_df = df[df['image_id'].isin(train_ids)]
    val_df = df[df['image_id'].isin(val_ids)]
    test_df = df[df['image_id'].isin(test_ids)]

    return train_df, val_df, test_df


def split_data_by_prefix(df):
    """
    Splits a DataFrame into train, validation, and test sets based on file name prefixes.

    Args:
        df (pd.DataFrame): The input DataFrame containing an 'image' column.

    Returns:
        tuple: Three DataFrames (train_df, val_df, test_df).
    """
    # Filter out image_ids that appear less than 10 times
    image_id_counts = df['image_id'].value_counts()
    valid_image_ids = image_id_counts[image_id_counts >= 10].index
    df = df[df['image_id'].isin(valid_image_ids)]
    
    if 'image_id' not in df.columns:
        raise ValueError("The input DataFrame must contain an 'image_id' column.")

    train_df = df[df['image'].str.startswith(('FoF1', 'FoF2'))]
    val_df = df[df['image'].str.startswith('FoF3')]
    test_df = df[df['image'].str.startswith('FoF4')]

    return train_df, val_df, test_df

