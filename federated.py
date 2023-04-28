import copy
import logging
import wandb
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from data import *
from utils import *
from config import args
from client import Client

class FedTrainer:

    def __init__(self, args, tokenizer, model, local_data_idxs, train_data, val_data, test_data):
        self.args = args
        self.num_clients = max(int(args.C * args.K), 1)
        self.tokenizer = tokenizer
        self.model = model
        self.local_data_idxs = local_data_idxs
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self._init_opt()

    def _init_opt(self):
        if self.args.K == 1 or self.args.algorithm != "fedopt":
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.server_lr)

    def _generate_local_epochs(self):
        return np.array([self.args.E] * self.num_clients)

    def train(self):
        best_f1_loss = float("-inf")
        self.args.es_patience
        patience = 0

        logging.info(f"System heterogeneity set to {self.args.percentage}% stragglers.")
        logging.info(f"Picking {self.num_clients} random clients per round.")

        for round_idx in range(1, self.args.rounds + 1):
            w_locals, local_loss = [], []
            client_epoch_list = self._generate_local_epochs()

            client_idxs = np.random.choice(range(self.args.K), self.num_clients, replace=False)

            if self.args.algorithm == "fedavg":
                stragglers_idxs = np.argwhere(client_epoch_list < self.args.E)
                client_epoch_list = np.delete(client_epoch_list, stragglers_idxs)
                client_idxs = np.delete(client_idxs, stragglers_idxs)

            for client_idx, epoch in zip(client_idxs, client_epoch_list):
                client = Client(
                    self.args,
                    epoch,
                    self.train_data,
                    self.local_data_idxs[client_idx],
                    self.tokenizer,
                )
                w, loss = client.train(model=copy.deepcopy(self.model))
                w_locals.append(copy.deepcopy(w))
                local_loss.append(loss)

            # updating the global weights
            w_avg = copy.deepcopy(w_locals[0])
            for k in w_avg.keys():
                for i in range(1, len(w_locals)):
                    w_avg[k] += w_locals[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w_locals))

            if self.optimizer == None:
                self.model.load_state_dict(w_avg)
            else:
                self.optimizer.zero_grad()
                optimizer_state = self.optimizer.state_dict()

                new_model = copy.deepcopy(self.model)
                new_model.load_state_dict(w_avg)
                with torch.no_grad():
                    for parameter, new_parameter in zip(
                        self.model.parameters(), new_model.parameters()
                    ):
                        parameter.grad = parameter.data - new_parameter.data

                model_state_dict = self.model.state_dict()
                new_model_state_dict = new_model.state_dict()

                for k in dict(self.model.named_parameters()).keys():
                    new_model_state_dict[k] = model_state_dict[k]
                self.model.load_state_dict(new_model_state_dict)
                self._init_opt()
                self.optimizer.load_state_dict(optimizer_state)
                self.optimizer.step()
                del new_model   # to avoid GPU OOM issue

            avg_train_loss = sum(local_loss) / len(local_loss)

            # evaluate the model on validation set
            val_metrics, val_loss, conf_matrix = self.eval(stage="valid")
            logging.info(
                f"Round: {round_idx}... \tAverage Train Loss: {round(avg_train_loss, 3)}... \tDev Loss: {round(val_loss, 3)}... "
                f"\tDev Accuracy: {val_metrics['valid/accuracy']}... \tAUC Score: {val_metrics['valid/auc']}... \tPrecision: {val_metrics['valid/precision']}... "
                f"\tRecall: {val_metrics['valid/recall']}... \tF1: {val_metrics['valid/f1_score']}... \tF1-weighted: {val_metrics['valid/f1_score_weighted']}... \tMCC: {val_metrics['valid/mcc']}"
            )

            test_metrics = {
                "test/accuracy": 0.0,
                "test/auc": 0.0,
                "test/precision": 0.0,
                "test/recall": 0.0,
                "test/f1_score": 0.0,
                "test/f1_score_weighted": 0.0, 
                "test/mcc": 0.0,
            }
            test_loss = 0.0

            # early stop if we don't improve till patience level
            if val_metrics["valid/f1_score_weighted"] > best_f1_loss:
                logging.info(
                    f"Dev f1 score improved. Saving model!"
                )
                best_f1_loss = val_metrics["valid/f1_score_weighted"]
                patience = 0
                if torch.cuda.device_count() > 1:
                    self.model.module.save_pretrained(self.args.save)
                else:
                    self.model.save_pretrained(self.args.save)
                self.tokenizer.save_pretrained(self.args.save)
            else:
                patience += 1
                logging.info(f"Early stopping counter {patience} out of {self.args.es_patience}")

                if patience == self.args.es_patience:

                    if torch.cuda.device_count() > 1:
                        self.model.module.from_pretrained(self.args.save)
                    else:
                        self.model.from_pretrained(self.args.save)
                    self.tokenizer.from_pretrained(self.args.save)
                    
                    test_metrics, test_loss = self.eval(stage="test")
                    logging.info(
                        f"FINAL TESTING\n... \tTest Loss: {round(test_loss, 3)}... Test Accuracy: {test_metrics['test/accuracy']}...  "
                        f"\tAUC Score: {test_metrics['test/auc']}... \tPrecision: {test_metrics['test/precision']}... \tRecall: {test_metrics['test/recall']}... "
                        f"\tF1: {test_metrics['test/f1_score']}... \tF1-weighted: {test_metrics['test/f1_score_weighted']}...  \tMCC: {test_metrics['test/mcc']}"
                    )

                    if self.args.wandb:
                        wb_metrics = {
                            "train/loss": avg_train_loss,
                            "valid/loss": val_loss,
                            **val_metrics,
                            "test/loss": test_loss,
                            **test_metrics
                        }
                        wandb.log(wb_metrics, step=round_idx)
                        wandb.run.summary["valid/f1_score"] = val_metrics["valid/f1_score"]

                    break

            # finally evaluate the model on the test set
            if round_idx == self.args.rounds:
                
                if torch.cuda.device_count() > 1:
                    self.model.module.from_pretrained(self.args.save)
                else:
                    self.model.from_pretrained(self.args.save)
                self.tokenizer.from_pretrained(self.args.save)
                
                test_metrics, test_loss, conf_matrix = self.eval(stage="test")
                logging.info(
                    f"FINAL TESTING\n... \tTest Loss: {round(test_loss, 3)}... Test Accuracy: {test_metrics['test/accuracy']}...  "
                    f"\tAUC Score: {test_metrics['test/auc']}... \tPrecision: {test_metrics['test/precision']}... \tRecall: {test_metrics['test/recall']}... "
                    f"\tF1: {test_metrics['test/f1_score']}... \tF1-weighted: {test_metrics['test/f1_score_weighted']}... \tMCC: {test_metrics['test/mcc']}"
                )

            if self.args.wandb:
                wb_metrics = {
                    "train/loss": avg_train_loss,
                    "valid/loss": val_loss,
                    **val_metrics,
                    "test/loss": test_loss,
                    **test_metrics,
                    "conf_mat": conf_matrix
                }
                wandb.log(wb_metrics, step=round_idx)
                wandb.run.summary["valid/f1_score"] = val_metrics["valid/f1_score"]
                wandb.run.summary["valid/f1_score_weighted"] = val_metrics["valid/f1_score_weighted"]

        return self.model

    def eval(self, stage="valid"):
        assert stage in ("valid", "test"), f"stage: {stage} not supported"
        data = self.val_data if stage == "valid" else self.test_data
        dataloader = DataLoader(
            data,
            batch_size=self.args.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.args.class_weights_array).cuda()
        )

        self.model.eval()
        pred_probs, labels = [], []

        batch_loss = []
        for batch in dataloader:
    
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.no_grad():
                output = self.model(**batch)
                logits = output.logits
                loss = criterion(logits, batch["labels"])
                batch_loss.append(loss.item())

            probs = torch.softmax(logits, dim=-1)
            pred_probs.append(probs)
            labels.append(batch["labels"])

        epoch_loss = sum(batch_loss) / len(batch_loss)

        pred_probs = torch.cat(pred_probs)
        preds = torch.argmax(pred_probs, dim=-1)
        labels = torch.cat(labels)

        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        if self.args.num_classes < 3:
            auc = roc_auc_score(
                labels.cpu().numpy(),
                pred_probs.cpu().numpy()[:, 1],
                labels=self.args.classes,
            )
        else:
            auc = roc_auc_score(
                labels.cpu().numpy(),
                pred_probs.cpu().numpy(),
                multi_class="ovo",
                labels=self.args.classes,
            )
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            labels.cpu().numpy(), 
            preds.cpu().numpy(), 
            average=None,
            labels=self.args.classes
        )
        _ , _ , f1_score_weighted, _ = precision_recall_fscore_support(
            labels.cpu().numpy(), 
            preds.cpu().numpy(), 
            average="weighted" if self.args.num_classes > 2 else "macro", 
            labels=self.args.classes
        )
        mcc = matthews_corrcoef(labels.cpu().numpy(), preds.cpu().numpy())

        metrics = {
            f"{stage}/accuracy": 100.0 * accuracy,
            f"{stage}/auc": auc,
            f"{stage}/precision": precision,
            f"{stage}/recall": recall,
            f"{stage}/f1_score": f1_score,
            f"{stage}/f1_score_weighted": f1_score_weighted,
            f"{stage}/mcc": mcc
        }
        return metrics, epoch_loss