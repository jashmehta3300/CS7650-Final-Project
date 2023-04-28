import os
import sys
import warnings
import logging
import wandb
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)
from data import *
from utils import *
from config import args
from federated import FedTrainer

# set up logging
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# ignore warnings
warnings.filterwarnings("ignore")

def main():

    if not torch.cuda.is_available():
        logging.info("No GPU device available")
        sys.exit(1)
    logging.info(args)
    set_seed(args.seed)

    raw_data = {
        "train": pd.read_csv(os.path.join(args.data, "{}_train.csv".format(args.main_dataset))),
        "valid": pd.read_csv(os.path.join(args.data, "{}_valid.csv".format(args.main_dataset))),
        "test": pd.read_csv(os.path.join(args.data, "{}_test.csv".format(args.main_dataset))),  
    }

    for split in raw_data:
        if args.main_dataset == "senti":
            raw_data[split] = raw_data[split][raw_data[split][args.target_col] != "neutral"]
        else:
            raw_data[split][list(raw_data[split].columns)] = raw_data[split][list(raw_data[split].columns)].applymap(str)
        raw_data[split].reset_index(inplace=True, drop=True)

    categories = list(raw_data["train"][args.target_col].unique())

    # load the tokenizer and model from huggingface hub
    model_ckpt = get_model_ckpt(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=num_classes, label2id=category2id, id2label=id2category
    )

    category2id = {cat: idx for idx, cat in enumerate(categories)}
    id2category = {idx: cat for idx, cat in enumerate(categories)}
    args.class_names = categories

    dataset = {
        "train": create_data_iter(raw_data["train"], category2id, tokenizer, args.target_col),
        "valid": create_data_iter(raw_data["valid"], category2id, tokenizer, args.target_col),
        "test": create_data_iter(raw_data["test"], category2id, tokenizer, args.target_col),
    }

    class_array = np.array([elem["labels"].item() for elem in dataset["train"]])
    num_classes = len(np.unique(class_array))
    args.classes = list(np.unique(class_array))
    args.num_classes = num_classes

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=args.classes, y=class_array
    )
    args.class_weights_array = class_weights
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()

    _label = [elem["labels"].item() for elem in dataset["train"]]

    if args.non_iid:
        _data_dict = non_iid_partition(dataset["train"], _label, args.K, args.K, len(dataset["train"])//args.K, 1)
    else:
        _data_dict = iid_partition(dataset["train"], args.K)

    # log the config to the wandb
    config_dict = dict(
        rounds=args.rounds,
        C=args.C,
        K=args.K,
        E=args.E,
        model=args.model_type,
        algorithm=args.algorithm,
        mu=args.mu,
        client_lr=args.client_lr,
        server_lr=args.server_lr,
        batch_size=args.batch_size,
        percentage=args.percentage,
        class_weights=args.class_weights,
    )

    if args.wandb:
        run = wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_proj_name,
            notes=args.wandb_run_notes,
            config=config_dict,
        )

    fl_trainer = FedTrainer(
        args,
        tokenizer,
        model,
        local_data_idxs=_data_dict,
        train_data=dataset["train"],
        val_data=dataset["valid"],
        test_data=dataset["test"]
    )
    model = fl_trainer.train()

    if args.wandb:
        artifact = wandb.Artifact(args.wandb_run_name, type="model")
        artifact.add_dir(args.save)
        run.log_artifact(artifact)
        run.finish()



if __name__ == "__main__":
    main()
