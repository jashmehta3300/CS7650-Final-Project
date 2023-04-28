import argparse
import os
import time

# command-line arguments
parser = argparse.ArgumentParser("Sentiment Analysis using Federated Learning")

parser.add_argument(
    "--algorithm",
    type=str,
    default="fedprox",
    help="specify which algorithm to use during local updates aggregation (fedavg, fedopt, fedprox)"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="batch size for fine-tuning the model"
)

parser.add_argument(
    "--C",
    type=float,
    default=0.1,
    help="client fraction"
)

parser.add_argument(
    "--class_weights",
    action="store_true",
    default=False,
    help="determine if experiments should use class weights in loss function or not"
)

parser.add_argument(
    "--client_lr",
    type=float,
    default=2e-5,
    help="learning rate for client"
)

parser.add_argument(
    "--data",
    type=str,
    default="data",
    help="location of the data corpus"
)

parser.add_argument(
    "--E",
    "--epochs",
    type=int,
    default=1,
    help="number of training epochs on local dataset for each round"
)

parser.add_argument(
    "--es_patience",
    type=int,
    default=25,
    help="early stopping patience level"
)

parser.add_argument(
    "--K",
    type=int,
    default=50,
    help="number of clients for iid partition"
)

parser.add_argument(
    "--main_dataset",
    default="senti",
    type=str
)

parser.add_argument(
    "--model_type",
    type=str,
    default="distilbert",
    help="specify which model to use (bert, distilbert, mobilebert, tinybert)"
)

parser.add_argument(
    "--mu",
    type=float,
    default=0.01,
    help="proximal term constant"
)

parser.add_argument(
    "--non_iid",
    type=str,
    default=False,
    help="determine if experiments should use class weights in loss function or not"
)

parser.add_argument(
    "--rounds",
    type=int,
    default=3,
    help="number of training rounds"
)

parser.add_argument(
    "--save",
    type=str,
    default="exp",
    help="experiment name"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for deterministic behaviour and reproducibility"
)

parser.add_argument(
    "--server_lr",
    type=float,
    default=0.0,
    help="learning rate for server"
)

parser.add_argument(
    "--target_col",
    default="category",
    type=str
)

parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="use wandb for tracking results"
)

parser.add_argument(
    "--wandb_proj_name",
    type=str,
    default="NLP_Project",
    help="provide a project name"
)

parser.add_argument(
    "--wandb_run_name",
    type=str,
    default="exp",
    help="provide a run name"
)

parser.add_argument(
    "--wandb_run_notes",
    type=str,
    default="",
    help="provide notes for a run if any"
)

args = parser.parse_args()

# proximal term is 0.0 in case of fedavg
args.mu = 0.0 if args.algorithm == "fedavg" else args.mu

# we don't need server optimizer in case of fedprox and fedavg + centralized training
args.server_lr = None if args.algorithm != "fedopt" or args.K == 1 else args.server_lr
args.algorithm = None if args.K == 1 else args.algorithm

# create experiment directory
args.save = f"{args.save}-{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(args.save, exist_ok=True)
print(f"Experiment dir: {args.save}")