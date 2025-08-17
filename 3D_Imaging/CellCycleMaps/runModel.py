from random import shuffle
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.preprocessing import image
import os
import tensorflow.keras as K
import sys
import numpy as np
import pandas as pd
from util import Binarize_labels, load_cellCycleData, load_keras_model, split_dataframe
from sklearn.metrics import accuracy_score
import random as rn
import argparse
from sklearn.model_selection import train_test_split
from DataGenerator import DataGenerator
from custom_resnet import create_resnet, create_resnet_small
from custom_cnn import buildModel
from tensorflow.keras.losses import categorical_crossentropy
from pre_augment import AugmentImages
import matplotlib.pyplot as plt

import tensorflow.keras as k

k.backend.set_image_data_format("channels_last")

"""
Note: Due to this error  GPU MaxPool gradient ops do not yet have a deterministic XLA implementation. Error occurred when finalizing GeneratorDataset iterator: 
FAILED_PRECONDITION: Python interpreter state is not initialized. The process may be terminated. I disabled the deterministic behaviour of tensorflow
"""


def init_seeds(seed=2022):
    np.random.seed(seed)
    rn.seed(seed)
    session_conf = tf.compat.v1.ConfigProto()
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    # session_conf.gpu_options.visible_device_list = gpu
    os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
    os.environ["TF_DETERMINISTIC_OPS"] = "true"
    import tensorflow.keras as k

    k.backend.set_image_data_format("channels_last")
    tf.random.set_seed(seed)
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf
    )
    # tf.compat.v1.keras.backend.set_session(sess)
    return sess


def ResNet(model, input_shape=(64, 64, 3)):
    input = k.layers.Input(input_shape)
    layer = model(input)
    layer = k.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Activation("relu")(layer)
    layer = k.layers.GlobalAveragePooling2D()(layer)
    layer = k.layers.Dense(units=512, activation="relu")(layer)
    layer = k.layers.Dense(units=4, activation="softmax")(layer)
    DenseNet_model = k.models.Model(inputs=input, outputs=layer)
    # print(DenseNet_model.summary())
    DenseNet_model.compile(
        loss=categorical_crossentropy,
        optimizer=k.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return DenseNet_model


def train(
    train_dir,
    val_dir,
    nlabels,
    nepochs,
    expName,
    arch,
    batch_size,
    path2PretrainedModel=None,
    pre_augment=False,
    single_channel=False,
):
    try:
        sess.close()
        k.clear_session()
    except:
        pass
    # sess = init_seeds(seed=2022)

    if not os.path.exists("models"):
        os.makedirs("models")

    df = load_cellCycleData(train_dir)
    if (
        not os.path.exists(os.path.join(train_dir, "train_data.csv"))
        and not os.path.exists(os.path.join(train_dir, "val_data.csv"))
        and not os.path.exists(os.path.join(train_dir, "test_data.csv"))
    ):
        print("Didnt find csv files, splitting the data...")
        train_df, val_df, test_df = split_dataframe(df)
        train_df.to_csv(train_dir + "/train_data.csv", index=False)
        val_df.to_csv(train_dir + "/val_data.csv", index=False)
        test_df.to_csv(train_dir + "/test_data.csv", index=False)
    else:
        train_df = pd.read_csv(train_dir + "/train_data.csv")
        val_df = pd.read_csv(train_dir + "/val_data.csv")
    # Pre-augment option goes here and updating the df_train file
    max_count = train_df['label'].value_counts().max()
    max_count_val = val_df['label'].value_counts().max()
    print('Applying oversampling with max_count is {}'.format(max_count))
    # Oversample
    #train_df = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=max_count, replace=True, random_state=42))
    #train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the oversampled dataset
    #val_df = val_df.groupby("label", group_keys=False).apply(lambda x: x.sample(n=max_count_val, replace=True, random_state=42))
    #val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the oversampled dataset
    counts_train = train_df.groupby("label").size()
    counts_val = val_df.groupby("label").size()
    print(counts_train)
    print(counts_val)
    if pre_augment == True:
        if train_dir.endswith("/"):  # remove the most right '/' in the path to images.
            train_dir = train_dir.rsplit("/", 1)[0]
        if not os.path.exists(train_dir + "-aug"):
            print("Starting pre-augmentation ...")
            Augmentor = AugmentImages(train_dir, train_dir + "-aug")
            Augmentor.iterate_df()
        train_df = load_cellCycleData(train_dir + "-aug")
        min_count = train_df['label'].value_counts().min()
        print('Applying oversampling to augmented data with max_count is {}'.format(min_count))
        #train_df = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min_count, replace=False, random_state=42))
        #train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the oversampled dataset
        counts_train = train_df.groupby("label").size()
        print('train after pre_augment and oversampling size is {}'.format(counts_train))
        training_generator = DataGenerator(train_dir+'-aug',
            train_df, num_classes=4, batch_size=batch_size, augment=False
        )
        val_generator = DataGenerator(train_dir,
            val_df, num_classes=4, batch_size=batch_size, augment=False
        )
    else:
        # val_df = pd.read_csv('./'+train_dir+'/val_data.csv')
        training_generator = DataGenerator(train_dir,
            train_df, num_classes=4, batch_size=batch_size,single_channel=single_channel, augment=True,
        )
        val_generator = DataGenerator(train_dir,
            val_df, num_classes=4, batch_size=batch_size, single_channel = single_channel, augment=False,
        )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("models", expName + "_" + arch + ".keras"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
    )

    # model = ResNet(ResNet_model,(64,64,3))
    if arch == "custom_cnn":
        # hyperparameters
        drop = 0.3  # dropout rate
        hidden_layers = 2  # number of hidden convolutional/max pooling layer sets
        batch = 32  # batch size
        neurons = 28  # number of convolution kernels
        model = buildModel(neurons, drop, hidden_layers, nb_classes=4)
        if path2PretrainedModel is not None:  # fine-tune model path available.
            # load the model from the path.
            print("Loading pre-trained model, please wait...")
            model = load_keras_model(path2PretrainedModel)
    elif arch == "custom_resnet":
        model = create_resnet_small()
        if path2PretrainedModel is not None:  # fine-tune model path available.
            # load the model from the path.
            print("Loading pre-trained model, please wait...")
            model = load_keras_model(path2PretrainedModel)
    model.summary()
    print("Fitting the model ...")
    print(len(training_generator))
    print(len(val_generator))
    model.fit(training_generator,
        validation_data=val_generator,
        batch_size=batch_size,
        steps_per_epoch=int(training_generator.__len__() / batch_size),
        epochs=nepochs,
        callbacks=[model_checkpoint])


def get_test_accuracy_per_label(df):
    df_results_summary = pd.DataFrame()
    accuracy_list = []
    unique_labels = set(df["true_labels"].tolist())
    count_list = []
    for l in unique_labels:
        df_subset = df[df["true_labels"] == l]
        accuracy = (
            accuracy_score(
                df_subset["true_labels"].tolist(), df_subset["pred"].tolist()
            )
            * 100
        )
        print(
            "Accuracy for label {} for {} instance is {}".format(
                l, str(df_subset.shape[0]), accuracy
            )
        )
        accuracy_list.append(accuracy)
        count_list.append(df_subset.shape[0])

    df_results_summary["true_label"] = list(unique_labels)
    df_results_summary["Accuracy"] = accuracy_list
    df_results_summary["Count"] = count_list
    return df_results_summary

def calculate_accuracy_per_fov(df):
    fov_accuracies = df.groupby('FoF').apply(lambda x: accuracy_score(x['true_labels'], x['pred']) * 100).reset_index(name='accuracy')
    fov_accuracies_class_ids = df.groupby(['FoF','Class_id']).apply(lambda x: accuracy_score(x['true_labels'], x['pred']) * 100).reset_index(name='accuracy')
    return fov_accuracies, fov_accuracies_class_ids

def plot_accuracy_per_fov(fov_accuracies, exp):
    plt.figure(figsize=(10, 6))
    plt.boxplot(fov_accuracies)
    plt.title(f'Accuracy per FoF for {exp}')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Field of View')
    plt.savefig(os.path.join("Results", f"{exp}_accuracy_per_fov.png"))
    #plt.show()

def test(train_dir, testCSV, exp, arch, single_channel):
    df = pd.read_csv(testCSV)
    # Use DataGenerator for test data
    test_generator = DataGenerator(train_dir, df, num_classes=4, batch_size=1,shuffle=False,single_channel=single_channel, augment=False)

    model = tf.keras.models.load_model(
        os.path.join("models", exp + "_" + arch + ".keras"),
        custom_objects=None,
        compile=True,
    )

    try:
        bottle_model = k.models.Model(inputs=model.input, outputs=model.layers[-3].output)
        print(f"Bottleneck layer (before dropout) is: {model.layers[-3].name}\n")
        # save bottleneck features to dataframe
        feats = bottle_model.predict(test_generator)
        df_feats = pd.DataFrame(feats)
    except:
        bottle_model = k.models.Model(inputs=model.input, outputs=model.layers[-2].output)
        print(f"Bottleneck layer (before dropout) is: {model.layers[-2].name}\n")
        # save bottleneck features to dataframe
        feats = bottle_model.predict(test_generator)
        df_feats = pd.DataFrame(feats)

    cellCycle = model.predict(test_generator)

    pred = np.argmax(cellCycle, axis=1)
    labels_true = df['label'].values
    df['FoF'] = df['image'].str.split('/').str[0]
    df['Cell_id'] = df['image'].str.split('/').str[-1].str.split('_').str[-1].str.split('.').str[0]
    df['Class_id'] = 'class' + (df['label'] + 1).astype(str)
    df["pred"] = pred
    df["true_labels"] = labels_true
    

    accuracy = accuracy_score(labels_true, pred)
    if not os.path.exists("./Results"):
        os.makedirs("./Results")
    with open(os.path.join("Results", exp + ".txt"), "w") as txt:
        txt.write("accuracy for {} is \n {}\n\n".format(exp, accuracy))
    print("Accuracy is {}".format(accuracy))

    df_results = get_test_accuracy_per_label(df)
    df_results_fof,df_results_fof_class_id = calculate_accuracy_per_fov(df)
    df_results.to_csv(
        os.path.join("Results", exp + "_test.txt"),
        header=True,
        index=None,
        sep=" ",
        mode="w",
    )
    df_results_fof.to_csv(
        os.path.join("Results", exp + "_test_accuracy_per_fof.txt"),
        header=True,
        index=None,
        sep=" ",
        mode="w",
    )

    df_results_fof_class_id.to_csv(
        os.path.join("Results", exp + "_test_accuracy_per_fof_and_class_id.txt"),
        header=True,
        index=None,
        sep=" ",
        mode="w",
    )
    #plotting boxplot per fof accuracy. 
    #plot_accuracy_per_fov(df_results_fof, exp)
    # print bottleneck features to csv with labels
    combined_df = pd.concat([df['FoF'],df['Cell_id'],df["true_labels"], df["pred"], df_feats], axis=1)
    combined_df.to_csv(
        os.path.join("./Results", exp + "_bottleneck_features.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataDir", help="Train directory", required=False)
    parser.add_argument("-valDir", help="Val directory", required=False)
    parser.add_argument("-epochs", help="training epochs", required=False)
    parser.add_argument(
        "-arch",
        help="name of architecture to use, custom_resnet is for the method implement in the paper titled"
        "(An Imbalanced Image Classification Method for the Cell Cycle Phase, whereas the option custom_cnn is for the paper"
        "Robust classification of cell cycle phase and biological feature extraction by image-based deep learning",
        required=True,
    )
    parser.add_argument("-exp", help="experiment name", required=True)
    parser.add_argument("-m", help="mode either train or test", required=True)
    parser.add_argument(
        "-nlabel", help="number of label for the data ex. 1 or 2", required=False
    )
    parser.add_argument(
        "-testCSV",
        help="CSV containing fov relative paths (relative to dataDir) and corresponding ground truth cell cycle labels",
        required=False,
        default="",
    )
    parser.add_argument("-batch_size", help="Batch size", required=False, default=32)
    parser.add_argument(
        "--finetune",
        help="Finetune a pre-trained model for epochs",
        action="store_true",
    )
    parser.add_argument(
        "-path2Model", help="path to model to fine-tune", required=False, default=None
    )
    parser.add_argument(
        "-finetune_epochs",
        help="Number of epochs to fine-tune",
        required=False,
        default=100,
    )
    parser.add_argument(
        "--pre_augment",
        help="Pre-augment images before start training, if False, online-augmentation per batch is done",
        action="store_true",
    )
    parser.add_argument(
        "--single_channel",
        help="If data consist of 3 channels, then use the nucleus only (first channel) for training.",
        action="store_true"
    )
    args = parser.parse_args()
    if args.m == "train":
        if args.finetune and args.path2Model is None:
            print("Please specify the path to pre-trained model for finetuning")
            sys.exit()
        if args.epochs is None:
            print("Please specify argments")
            sys.exit()
        elif args.finetune and args.path2Model is not None:
            print(
                "Started finetuning of {} model from path {}".format(
                    args.arch, args.path2Model
                )
            )
            train(
                args.dataDir,
                args.valDir,
                args.nlabel,
                int(args.finetune_epochs),
                args.exp + "_fine-tuned",
                args.arch,
                int(args.batch_size),
                args.path2Model,
                args.pre_augment,
                args.single_channel,
            )
        elif not args.finetune:
            print("Started training  {} model ".format(args.exp))
            train(
                args.dataDir,
                args.valDir,
                args.nlabel,
                int(args.epochs),
                args.exp,
                args.arch,
                int(args.batch_size),
                None,
                args.pre_augment,
                args.single_channel,
            )
    else:
        if args.testCSV == "":
            print("Error, please enter test CSV")
            sys.exit()
        else:
            print("Started testing {} model".format(args.exp))
            if args.finetune:
                test(
                    args.dataDir, args.testCSV, args.exp + "_fine-tuned", args.arch, args.single_channel
                )  # testing finetuned model
            else:
                test(args.dataDir, args.testCSV, args.exp, args.arch,args.single_channel)  # testing


if __name__ == "__main__":
    main()
