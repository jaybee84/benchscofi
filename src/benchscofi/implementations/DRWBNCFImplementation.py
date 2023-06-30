#coding:utf-8

## Concatenated contents of https://github.com/luckymengmeng/DRWBNCF/tree/959f8cf25227729c7609d6f1229a554b1e8e1ed0{/,/src}

import torch
from torch import nn
from torch.nn import functional as F

import logging
import os
import torch
import pandas as pd
import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
import pytorch_lightning as pl

from . import DATA_TYPE_REGISTRY

from collections import namedtuple
from . import DATA_TYPE_REGISTRY
#from .dataloader import Dataset
#from .utils import select_topk

import torch
from torch import nn, optim

from torch.nn import functional as F
from torchvision.ops import focal_loss
import pytorch_lightning as pl
from sklearn import metrics

#from .bgnn import BGNNA, BGCNA
#from .model_help import BaseModel
#from .dataset import PairGraphData
from . import MODEL_REGISTRY

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn import metrics

import os
import sys
import logging
import torch

import os
import sys
import time
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as scio
from pprint import pformat

from src import metric_fn
from src.utils import init_logger, logger
from src.dataloader import CVDataset, DRDataset
from src.model import WBNCF

@torch.no_grad()
def train_test_fn(model, train_loader, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges = [], [], []
    for batch in train_loader:
        model.train_step(batch)
    for batch in val_loader:
        batch = move_data_to_device(batch, device)
        output = model.test_step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = metric_fn.evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    logger.info(f"eval time cost: {eval_end_time_stamp - eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                                 "col": edges[1],
                                 "score": scores,
                                 "label": labels,
                                 })
        logger.info(f"save time cost: {time.time() - eval_end_time_stamp}")
    return scores, labels, edges, metric

@torch.no_grad()
def test_fn(model, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges = [], [], []
    for batch in val_loader:
        batch = move_data_to_device(batch, device)
        output = model.step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = metric_fn.evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    logger.info(f"eval time cost: {eval_end_time_stamp-eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                      "col": edges[1],
                      "score": scores,
                      "label": labels,
                      })
        logger.info(f"save time cost: {time.time()-eval_end_time_stamp}")
    return scores, labels, edges, metric


def train_fn(config, model, train_loader, val_loader):
    checkpoint_callback = ModelCheckpoint(monitor="val/auroc",
                                          mode="max",
                                          save_top_k=1,
                                          verbose=False,
                                          save_last=True)
    lr_callback = pl.callbacks.LearningRateMonitor("epoch")
    trainer = Trainer(max_epochs=config.epochs,
                      default_root_dir=config.log_dir,
                      profiler=config.profiler,
                      fast_dev_run=False,
                      checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_callback],
                      gpus=config.gpus,
                      check_val_every_n_epoch=1
                      )
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    if not hasattr(config, "dirpath"):
        config.dirpath = trainer.checkpoint_callback.dirpath
    # checkpoint and add path
    # checkpoint = torch.load("lightning_logs/version_7/checkpoints/epoch=85.ckpt")
    # trainer.on_load_checkpoint(checkpoint)
    print(model.device)


def train(config, model_cls=WBNCF):
    time_stamp = time.asctime()
    datasets = DRDataset(dataset_name=config.dataset_name, drug_neighbor_num=config.drug_neighbor_num,
                         disease_neighbor_num=config.disease_neighbor_num)
    log_dir = os.path.join(f"{config.comment}", f"{config.split_mode}-{config.n_splits}-fold", f"{config.dataset_name}",
                           f"{model_cls.__name__}", f"{time_stamp}")
    config.log_dir = log_dir
    config.n_drug = datasets.drug_num
    config.n_disease = datasets.disease_num

    config.size_u = datasets.drug_num
    config.size_v = datasets.disease_num

    config.gpus = 1 if torch.cuda.is_available() else 0
    config.pos_weight = datasets.pos_weight
    config.time_stamp = time_stamp
    logger = init_logger(log_dir)
    logger.info(pformat(vars(config)))
    config.dataset_type = config.dataset_dype if config.dataset_type is not None else model_cls.DATASET_TYPE
    cv_spliter = CVDataset(datasets, split_mode=config.split_mode, n_splits=config.n_splits,
                           drug_idx=config.drug_idx, disease_idx=config.disease_idx,
                           train_fill_unknown=config.train_fill_unknown,
                           global_test_all_zero=config.global_test_all_zero, seed=config.seed,
                           dataset_type=config.dataset_type)
    pl.seed_everything(config.seed)
    scores, labels, edges, split_idxs = [], [], [], []
    metrics = {}
    start_time_stamp = time.time()
    for split_id, datamodule in enumerate(cv_spliter):
        # if split_id not in [4, 5]:
        #     continue
        config.split_id = split_id
        split_start_time_stamp = time.time()

        datamodule.prepare_data()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        config.pos_weight = train_loader.dataset.pos_weight
        model = model_cls(**vars(config))
        model = model.cuda() if config.gpus else model

        if split_id==0:
            logger.info(model)
        logger.info(f"begin train fold {split_id}/{len(cv_spliter)}")
        train_fn(config, model, train_loader=train_loader, val_loader=val_loader)
        logger.info(f"end train fold {split_id}/{len(cv_spliter)}")
        save_file_format = os.path.join(config.log_dir,
                                        f"{config.dataset_name}-{config.split_id} fold-{{auroc}}-{{aupr}}.mat")
        score, label, edge, metric = test_fn(model, val_loader, save_file_format)
        # score, label, edge, metric = train_test_fn(model, train_loader, val_loader, save_file_format)
        metrics[f"split_id_{split_id}"] = metric
        scores.append(score)
        labels.append(label)
        edges.append(edge)
        split_idxs.append(np.ones(len(score), dtype=int)*split_id)
        logger.info(f"{split_id}/{len(cv_spliter)} folds: {metric}")
        logger.info(f"{split_id}/{len(cv_spliter)} folds time cost: {time.time()-split_start_time_stamp}")

        if config.debug:
            break
    end_time_stamp = time.time()
    logger.info(f"total time cost:{end_time_stamp-start_time_stamp}")
    with pd.ExcelWriter(os.path.join(log_dir, f"tmp.xlsx")) as f:
        pd.DataFrame(metrics).T.to_excel(f, sheet_name="metrics")
        params = pd.DataFrame({key:str(value) for key, value in vars(config).items()}, index=[str(time.time())])
        params.to_excel(f, sheet_name="params")

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    edges = np.concatenate(edges, axis=1)
    split_idxs = np.concatenate(split_idxs, axis=0)
    final_metric = metric_fn.evaluate(predict=scores, label=labels, is_final=True)
    metrics["final"] = final_metric
    metrics = pd.DataFrame(metrics).T
    metrics.index.name = "split_id"
    metrics["seed"] = config.seed
    logger.info(f"final {config.dataset_name}-{config.split_mode}-{config.n_splits}-fold-auroc:{final_metric['auroc']}-aupr:{final_metric['aupr']}")
    output_file_name = f"final-{config.dataset_name}-{config.split_mode}-{config.n_splits}-auroc:{final_metric['auroc']}-aupr:{final_metric['aupr']}-fold"
    scio.savemat(os.path.join(log_dir, f"{output_file_name}.mat"),
                 {"row": edges[0],
                  "col": edges[1],
                  "score": scores,
                  "label": labels,
                  "split_idx":split_idxs}
                 )
    with pd.ExcelWriter(os.path.join(log_dir, f"{output_file_name}.xlsx")) as f:
        metrics.to_excel(f, sheet_name="metrics")
        params = pd.DataFrame({key:str(value) for key, value in vars(config).items()}, index=[str(time.time())])
        for key, value in final_metric.items():
            params[key] = value
        params["file"] = output_file_name
        params.to_excel(f, sheet_name="params")

    logger.info(f"save final results to r'{os.path.join(log_dir, output_file_name)}.mat'")
    logger.info(f"final results: {final_metric}")



def parse(print_help=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NNCFDR", type=str)
    parser.add_argument("--epochs", default=64, type=int)
    parser.add_argument("--drug_feature_topk", default=20, type=int)  # add prior knowledge
    parser.add_argument("--disease_feature_topk", default=20, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profiler", default=False, type=str)
    parser.add_argument("--comment", default="runs", type=str, help="experiment name")
    parser = DRDataset.add_argparse_args(parser)
    parser = CVDataset.add_argparse_args(parser)
    parser = WBNCF.add_model_specific_args(parser)
    args = parser.parse_args()
    if print_help:
        parser.print_help()
    return args


logger = logging.getLogger("WBNCF")

class NoParsingFilter(logging.Filter):
    def filter(self, record):
        if record.funcName=="summarize" and record.levelno==20:
            return False
        if record.funcName=="_info" and record.funcName=="distributed.py" and record.lineno==20:
            return False
        return True

def init_logger(log_dir):
    lightning_logger = logging.getLogger("pytorch_lightning.core.lightning")
    lightning_logger.addFilter(NoParsingFilter())
    distributed_logger = logging.getLogger("pytorch_lightning.utilities.distributed")
    distributed_logger.addFilter(NoParsingFilter())
    format = '%Y-%m-%d %H-%M-%S'
    fm = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
                           datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fm)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    if len(logger.handlers)==1:
        import time
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            logger.warning(f"error file exist! {log_dir}")
            logger.warning("please init new 'comment' value")
            # exit(0)
        logger.propagate = False
        log_file = os.path.join(log_dir, f"{time.strftime(format, time.localtime())}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fm)
        logger.addHandler(file_handler)
        logger.info(f"terminal cmd: python {' '.join(sys.argv)}")
        logger.info(f"log file: {log_file}")
    else:
        logger.warning("init_logger fail")
    return logger

def select_topk(data, k=-1):
    if k<=0:
        return data
    assert k<=data.shape[1]
    val, col = torch.topk(data ,k=k)
    col = col.reshape(-1)
    row = torch.ones(1, k, dtype=torch.int)*torch.arange(data.shape[0]).view(-1, 1)
    row = row.view(-1).to(device=data.device)
    new_data = torch.zeros_like(data)
    new_data[row, col] = data[row, col]
    # new_data[row, col] = 1.0
    return new_data

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pyre-ignore-all-errors[2,3]
# https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py
from typing import Any, Dict, Iterable, Iterator, Tuple

class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        from tabulate import tabulate
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


class BaseModel(pl.LightningModule):
    DATASET_TYPE: None

    def __init__(self):
        super(BaseModel, self).__init__()

    def select_topk(self, data, k=-1):
        if k is None or k <= 0:
            return data
        assert k <= data.shape[1]
        val, col = torch.topk(data, k=k)
        col = col.reshape(-1)
        row = torch.ones(1, k, dtype=torch.int) * torch.arange(data.shape[0]).view(-1, 1)
        row = row.view(-1).to(device=data.device)
        new_data = torch.zeros_like(data)
        new_data[row, col] = data[row, col]
        return new_data

    def merge_neighbor_feature(self, sims, features, k=5):
        assert sims.shape[0] == features.shape[0] and sims.shape[1] == sims.shape[0]
        if k<0:
            k = sims.shape[1]
        N = features.shape[0]
        value, idx = torch.topk(sims, dim=1, k=k)
        col = idx.reshape(-1)
        features = features[col].view(N, k, -1) * value.view(N, k, 1)
        features = features.sum(dim=1)
        features = features / value.sum(dim=1).view(N, 1)
        return features

    def neighbor_smooth(self, sims, features, replace_rate=0.2):
        merged_u = self.merge_neighbor_feature(sims, features)
        mask = torch.rand(merged_u.shape[0], device=sims.device)
        mask = torch.floor(mask + replace_rate).type(torch.bool)
        new_features = torch.where(mask, merged_u, features)
        return new_features

    def laplacian_matrix(self, S):
        x = torch.sum(S, dim=0)
        y = torch.sum(S, dim=1)
        L = 0.5*(torch.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    def graph_loss_fn(self, x, edge, topk=None, cache_name=None, reduction="mean"):
        if not hasattr(self, f"_{cache_name}") :
            adj = torch.sparse_coo_tensor(*edge).to_dense()
            adj = adj-torch.diag(torch.diag(adj))
            adj = self.select_topk(adj, k=topk)
            la = self.laplacian_matrix(adj)
            if cache_name:
                self.register_buffer(f"_{cache_name}", la)
        else:
            la = getattr(self, f"_{cache_name}")
            assert la.shape==edge[2]

        graph_loss = torch.trace(x.T@la@x)
        graph_loss = graph_loss/(x.shape[0]**2) if reduction=="mean" else graph_loss
        return graph_loss

    def mse_loss_fn(self, predict, label, pos_weight):
        predict = predict.view(-1)
        label = label.view(-1)
        pos_mask = label>0
        loss = F.mse_loss(predict, label, reduction="none")
        loss_pos = loss[pos_mask].mean()
        loss_neg = loss[~pos_mask].mean()
        loss_mse = loss_pos*pos_weight+loss_neg
        return {"loss_mse":loss_mse,
                "loss_mse_pos":loss_pos,
                "loss_mse_neg":loss_neg,
                "loss":loss_mse}

    def bce_loss_fn(self, predict, label, pos_weight):
        predict = predict.view(-1)
        label = label.view(-1)
        weight = pos_weight * label + 1 - label
        loss = F.binary_cross_entropy(input=predict, target=label, weight=weight)
        return {"loss_bce":loss,
                "loss":loss}

    def focal_loss_fn(self, predict, label, alpha, gamma):
        predict = predict.view(-1)
        label = label.view(-1)
        ce_loss = F.binary_cross_entropy(
            predict, label, reduction="none"
        )
        p_t = predict*label+(1-predict)*(1-label)
        loss = ce_loss*((1-p_t)**gamma)
        alpha_t = alpha * label + (1-alpha)*(1-label)
        focal_loss = (alpha_t * loss).mean()
        return {"loss_focal":focal_loss,
                "loss":focal_loss}

    def rank_loss_fn(self, predict, label, margin=0.8, reduction='mean'):
        predict = predict.view(-1)
        label = label.view(-1)
        pos_mask = label > 0
        pos = predict[pos_mask]
        neg = predict[~pos_mask]
        neg_mask = torch.randint(0, neg.shape[0], (pos.shape[0],), device=label.device)
        neg = neg[neg_mask]

        rank_loss = F.margin_ranking_loss(pos, neg, target=torch.ones_like(pos),
                                          margin=margin, reduction=reduction)
        return {"loss_rank":rank_loss,
                "loss":rank_loss}

    def get_epoch_auroc_aupr(self, outputs):
        predict = [output["predict"].detach() for output in outputs]
        label = [output["label"] for output in outputs]
        predict = torch.cat(predict).cpu().view(-1)
        label = torch.cat(label).cpu().view(-1)
        aupr = metrics.average_precision_score(y_true=label, y_score=predict)
        auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
        return auroc, aupr

    def get_epoch_loss(self, outputs):
        loss_keys = [key for key in outputs[0] if key.startswith("loss")]
        loss_info = {key: [output[key].detach().cpu() for output in outputs if not torch.isnan(output[key])] for key in loss_keys}
        loss_info = {key: sum(value)/len(value) for key, value in loss_info.items()}
        return loss_info

    def training_epoch_end(self, outputs):
        stage = "train"
        loss_info = self.get_epoch_loss(outputs)
        auroc, aupr = self.get_epoch_auroc_aupr(outputs)
        # self.log(f"{stage}/loss", loss_info["loss"], prog_bar=True)
        self.log(f"{stage}/auroc", auroc, prog_bar=True)
        self.log(f"{stage}/aupr", aupr, prog_bar=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"{stage}_epoch/{key}", value, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/auroc", auroc, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/aupr", aupr, global_step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        stage = "val"
        loss_info = self.get_epoch_loss(outputs)
        auroc, aupr = self.get_epoch_auroc_aupr(outputs)
        # self.log(f"{stage}/loss", loss_info["loss"], prog_bar=True)
        self.log(f"{stage}/auroc", auroc, prog_bar=True)
        self.log(f"{stage}/aupr", aupr, prog_bar=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"{stage}_epoch/{key}", value, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/auroc", auroc, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/aupr", aupr, global_step=self.current_epoch)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

class NeighborEmbedding(nn.Module):
    def __init__(self, num_embeddings, out_channels=128, dropout=0.5, cached=True, bias=True, lamda=0.8, share=True):
        super(NeighborEmbedding, self).__init__()

        self.shutcut = nn.Linear(in_features=num_embeddings, out_features=out_channels)

        self.bgnn = BGCNA(in_channels=num_embeddings, out_channels=out_channels,
                          cached=cached, bias=bias, lamda=lamda, share=share)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = out_channels

    def forward(self, x, edge, embedding):
        if not hasattr(self, "edge_index"):
            edge_index = torch.sparse_coo_tensor(*edge)
            self.register_buffer("edge_index", edge_index)
        edge_index = self.edge_index

        embedding = self.bgnn(embedding, edge_index=edge_index)

        embedding = self.dropout(embedding)
        x = F.embedding(x, embedding)
        x = F.normalize(x)

        return x


class InteractionEmbedding(nn.Module):
    def __init__(self, n_drug, n_disease, embedding_dim, dropout=0.5):
        super(InteractionEmbedding, self).__init__()
        self.drug_project = nn.Linear(n_drug, embedding_dim, bias=False)
        self.disease_project = nn.Linear(n_disease, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.output_dim = embedding_dim

    def forward(self, association_pairs, drug_embedding, disease_embedding):
        drug_embedding = torch.diag(torch.ones(drug_embedding.shape[0], device=drug_embedding.device))
        disease_embedding = torch.diag(torch.ones(disease_embedding.shape[0], device=disease_embedding.device))

        drug_embedding = self.drug_project(drug_embedding)
        disease_embedding = self.disease_project(disease_embedding)

        drug_embedding = F.embedding(association_pairs[0,:], drug_embedding)
        disease_embedding = F.embedding(association_pairs[1,:], disease_embedding)

        associations = drug_embedding*disease_embedding

        associations = F.normalize(associations)
        associations = self.dropout(associations)
        return associations

class InteractionDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dims=(256, 64), out_channels=1, dropout=0.5):
        super(InteractionDecoder, self).__init__()
        decoder = []
        in_dims = [in_channels]+list(hidden_dims)
        out_dims = hidden_dims
        for in_dim, out_dim in zip(in_dims, out_dims):
            decoder.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            decoder.append(nn.ReLU(inplace=True))
            decoder.append(nn.Dropout(dropout))
        decoder.append(nn.Linear(hidden_dims[-1], out_channels))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


@MODEL_REGISTRY.register()
class WBNCF(BaseModel):
    DATASET_TYPE = "PairGraphDataset"
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WBNCF model config")
        parser.add_argument("--embedding_dim", default=64, type=int, help="编码器关联嵌入特征维度")
        parser.add_argument("--neighbor_embedding_dim", default=32, type=int, help="编码器邻居特征维度")
        parser.add_argument("--hidden_dims", type=int, default=(64, 32), nargs="+", help="解码器每层隐藏单元数")
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--pos_weight", type=float, default=1.0, help="no used, overwrited, use for bce loss")
        parser.add_argument("--alpha", type=float, default=0.5, help="use for focal loss")
        parser.add_argument("--gamma", type=float, default=2.0, help="use for focal loss")
        parser.add_argument("--lamda", type=float, default=0.8, help="weight for bgnn")
        parser.add_argument("--loss_fn", type=str, default="focal", choices=["bce", "focal"])
        parser.add_argument("--separate", default=False, action="store_true")
        return parent_parser

    def __init__(self, n_drug, n_disease, embedding_dim=64, neighbor_embedding_dim=32, hidden_dims=(64, 32),
                 lr=5e-4, dropout=0.5, pos_weight=1.0, alpha=0.5, gamma=2.0, lamda=0.8,
                 loss_fn="focal", separate=False, **config):
        super(WBNCF, self).__init__()
        # lr=0.1
        self.n_drug = n_drug
        self.n_disease = n_disease
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims

        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("gamma", torch.tensor(gamma))
        "rank bce mse focal"
        self.loss_fn_name = loss_fn
        share = not separate
        self.drug_neighbor_encoder = NeighborEmbedding(num_embeddings=n_drug,
                                                       out_channels=neighbor_embedding_dim,
                                                       dropout=dropout, lamda=lamda, share=share)
        self.disease_neighbor_encoder = NeighborEmbedding(num_embeddings=n_disease,
                                                          out_channels=neighbor_embedding_dim,
                                                          dropout=dropout, lamda=lamda, share=share)
        self.interaction_encoder = InteractionEmbedding(n_drug=n_drug, n_disease=n_disease,
                                                        embedding_dim=embedding_dim, dropout=dropout)
        merged_dim = self.disease_neighbor_encoder.output_dim\
                     +self.drug_neighbor_encoder.output_dim\
                     +self.interaction_encoder.output_dim
        self.decoder = InteractionDecoder(in_channels=merged_dim, hidden_dims=hidden_dims, dropout=dropout,
                                          )
        self.config = config
        self.lr = lr
        self.save_hyperparameters()


    def forward(self, interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding):

        drug_neighbor_embedding = self.drug_neighbor_encoder(interaction_pairs[0,:], drug_edge, drug_embedding)
        disease_neighbor_embedding = self.disease_neighbor_encoder(interaction_pairs[1,:], disease_edge, disease_embedding)
        interaction_embedding = self.interaction_encoder(interaction_pairs, drug_embedding, disease_embedding)

        embedding = torch.cat([drug_neighbor_embedding, interaction_embedding, disease_neighbor_embedding], dim=-1)
        score = self.decoder(embedding)
        return score.reshape(-1)


    def loss_fn(self, predict, label, u, v, u_edge, v_edge, reduction="sum"):
        bce_loss = self.bce_loss_fn(predict, label, self.pos_weight)
        focal_loss = self.focal_loss_fn(predict, label, gamma=self.gamma, alpha=self.alpha)
        mse_loss = self.mse_loss_fn(predict, label, self.pos_weight)
        rank_loss = self.rank_loss_fn(predict, label)

        u_graph_loss = self.graph_loss_fn(x=u, edge=u_edge, cache_name="ul",
                                          # topk=5,
                                          topk = self.config["drug_neighbor_num"],
                                          reduction=reduction)
        v_graph_loss = self.graph_loss_fn(x=v, edge=v_edge, cache_name="vl",
                                          # topk=5,
                                          topk = self.config["disease_neighbor_num"],
                                          reduction=reduction)
        graph_loss = u_graph_loss * self.lambda1 + v_graph_loss * self.lambda2


        loss = {}
        loss.update(bce_loss)
        loss.update(focal_loss)
        loss.update(mse_loss)
        loss.update(rank_loss)
        loss["loss_graph"] = graph_loss
        loss["loss_graph_u"] = u_graph_loss
        loss["loss_graph_v"] = v_graph_loss
        loss["loss"] = loss[f"loss_{self.loss_fn_name}"]+graph_loss

        return loss


    def step(self, batch:PairGraphData):
        interaction_pairs = batch.interaction_pair
        label = batch.label
        drug_edge = batch.u_edge
        disease_edge = batch.v_edge
        drug_embedding = batch.u_embedding
        disease_embedding = batch.v_embedding
        u = self.interaction_encoder.drug_project.weight.T
        v = self.interaction_encoder.disease_project.weight.T

        predict = self.forward(interaction_pairs, drug_edge, disease_edge, drug_embedding, disease_embedding)
        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label, u=u, v=v, u_edge=drug_edge, v_edge=disease_edge)
        ans["predict"] = predict
        ans["label"] = label
        return ans


    def training_step(self, batch, batch_idx=None):
        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(lr=self.lr, params=self.parameters(), weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.05*self.lr, max_lr=self.lr,
                                                   gamma=0.95, mode="exp_range", step_size_up=4,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]

    @property
    def lambda1(self):
        max_value = 0.125
        value = self.current_epoch/18.0*max_value
        return torch.tensor(value, device=self.device)

    @property
    def lambda2(self):
        max_value = 0.0625
        value = self.current_epoch / 18.0 * max_value
        return torch.tensor(value, device=self.device)





"""https://github.com/storyandwine/LAGCN
Predicting Drug-Disease Associations through Layer Attention Graph Convolutional Networks
"""

import numpy as np
from sklearn import metrics

def get_metrics(real_score, predict_score):
    import gc
    gc.collect()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]

    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


def evaluate(predict, label, is_final=False):
    if not is_final:
        try:
            res = get_metrics(real_score=label, predict_score=predict)
        except:
            res = [None]*7
    else:
        res = [None]*7
    aupr = metrics.average_precision_score(y_true=label, y_score=predict)
    auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
    result = {"aupr":aupr,
              "auroc":auroc,
              "lagcn_aupr":res[0],
              "lagcn_auc":res[1],
              "lagcn_f1_score":res[2],
              "lagcn_accuracy":res[3],
              "lagcn_recall":res[4],
              "lagcn_specificity":res[5],
              "lagcn_precision":res[6]}
    return result

PairGraphData = namedtuple("PairGraphData", ["u_edge", "v_edge",
                                             "u_embedding", "v_embedding",
                                             "label", "interaction_pair", "valid_mask"])

@DATA_TYPE_REGISTRY.register()
class PairGraphDataset(Dataset):
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", **kwargs):
        fill_unkown = fill_unkown if stage=="train" else False
        super(PairGraphDataset, self).__init__(dataset, mask, fill_unkown=fill_unkown, stage=stage, **kwargs)
        self.interaction_edge = self.interaction_edge
        self.label = self.label.reshape(-1)
        self.valid_mask = self.valid_mask.reshape(-1)
        self.u_edge = self.get_u_edge()
        self.v_edge = self.get_v_edge()
        self.u_embedding = select_topk(self.u_embedding, 20)
        self.v_embedding = select_topk(self.v_embedding, 20)
        # self.u_embedding = torch.sparse_coo_tensor(*self.u_edge).to_dense()
        # self.v_embedding = torch.sparse_coo_tensor(*self.v_edge).to_dense()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        interaction_edge = self.interaction_edge[:, index]
        valid_mask = self.valid_mask[index]
        data = PairGraphData(u_edge=self.u_edge,
                             v_edge=self.v_edge,
                             label=label,
                             valid_mask=valid_mask,
                             interaction_pair=interaction_edge,
                             u_embedding=self.u_embedding,
                             v_embedding=self.v_embedding,
                             )
        return data

class DRDataset():
    def __init__(self, dataset_name="Fdataset", drug_neighbor_num=15, disease_neighbor_num=15):
        assert dataset_name in ["Cdataset", "Fdataset", "DNdataset", "lrssl", "hdvd"]
        self.dataset_name = dataset_name
        if dataset_name=="lrssl":
            old_data = load_DRIMC(name=dataset_name)
        elif dataset_name=="hdvd":
            old_data = load_HDVD()
        else:
            old_data = scio.loadmat(f"dataset/{dataset_name}.mat")

        self.drug_sim = old_data["drug"].astype(np.float)
        self.disease_sim = old_data["disease"].astype(np.float)
        self.drug_name = old_data["Wrname"].reshape(-1)
        self.drug_num = len(self.drug_name)
        self.disease_name = old_data["Wdname"].reshape(-1)
        self.disease_num = len(self.disease_name)
        self.interactions = old_data["didr"].T

        self.drug_edge = self.build_graph(self.drug_sim, drug_neighbor_num)
        self.disease_edge = self.build_graph(self.disease_sim, disease_neighbor_num)
        pos_num = self.interactions.sum()
        neg_num = np.prod(self.interactions.shape) - pos_num
        self.pos_weight = neg_num / pos_num
        print(f"dataset:{dataset_name}, drug:{self.drug_num}, disease:{self.disease_num}, pos weight:{self.pos_weight}")

    def build_graph(self, sim, num_neighbor):
        if num_neighbor>sim.shape[0] or num_neighbor<0:
            num_neighbor = sim.shape[0]
        neighbor = np.argpartition(-sim, kth=num_neighbor, axis=1)[:, :num_neighbor]
        row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
        col_index = neighbor.reshape(-1)
        edge_index = torch.from_numpy(np.array([row_index, col_index]).astype(int))
        values = torch.ones(edge_index.shape[1])
        values = torch.from_numpy(sim[row_index, col_index]).float()*values
        return (edge_index, values, sim.shape)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("dataset config")
        parser.add_argument("--dataset_name", default="Fdataset",
                                   choices=["Cdataset", "Fdataset", "lrssl", "hdvd"])
        parser.add_argument("--drug_neighbor_num", default=25, type=int)
        parser.add_argument("--disease_neighbor_num", default=25, type=int)
        return parent_parser



class Dataset():
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", **kwargs):
        mask = mask.astype(bool)
        self.stage = stage
        self.one_mask = torch.from_numpy(dataset.interactions>0)
        row, col = np.nonzero(mask&dataset.interactions.astype(bool))
        self.valid_row = torch.tensor(np.unique(row))
        self.valid_col = torch.tensor(np.unique(col))
        if not fill_unkown:
            row_idx, col_idx = np.nonzero(mask)
            self.interaction_edge = torch.LongTensor([row_idx, col_idx]).contiguous()
            self.label = torch.from_numpy(dataset.interactions[mask]).float().contiguous()
            self.valid_mask = torch.ones_like(self.label, dtype=torch.bool)
            self.matrix_mask = torch.from_numpy(mask)
        else:
            row_idx, col_idx = torch.meshgrid(torch.arange(mask.shape[0]), torch.arange(mask.shape[1]))
            self.interaction_edge = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)])
            self.label = torch.clone(torch.from_numpy(dataset.interactions)).float()
            self.label[~mask] = 0
            self.valid_mask = torch.from_numpy(mask)
            self.matrix_mask = torch.from_numpy(mask)

        self.drug_edge = dataset.drug_edge
        self.disease_edge = dataset.disease_edge

        self.u_embedding = torch.from_numpy(dataset.drug_sim).float()
        self.v_embedding = torch.from_numpy(dataset.disease_sim).float()

        self.mask = torch.from_numpy(mask)
        pos_num = self.label.sum().item()
        neg_num = np.prod(self.mask.shape) - pos_num
        self.pos_weight = neg_num / pos_num

    def __str__(self):
        return f"{self.__class__.__name__}(shape={self.mask.shape}, interaction_num={len(self.interaction_edge)}, pos_weight={self.pos_weight})"

    @property
    def size_u(self):
        return self.mask.shape[0]

    @property
    def size_v(self):
        return self.mask.shape[1]

    def get_u_edge(self, union_graph=False):
        edge_index, value, size = self.drug_edge
        if union_graph:
            size = (self.size_u+self.size_v, )*2
        return edge_index, value, size

    def get_v_edge(self, union_graph=False):
        edge_index, value, size = self.disease_edge
        if union_graph:
            edge_index = edge_index + torch.tensor(np.array([[self.size_u], [self.size_u]]))
            size = (self.size_u + self.size_v,) * 2
        return edge_index, value, size

    def get_uv_edge(self, union_graph=False):
        train_mask = self.mask if self.stage=="train" else ~self.mask
        train_one_mask = train_mask & self.one_mask
        edge_index = torch.nonzero(train_one_mask).T
        value = torch.ones(edge_index.shape[1])
        size =  (self.size_u, self.size_v)
        if union_graph:
            edge_index = edge_index + torch.tensor([[0], [self.size_u]])
            size = (self.size_u + self.size_v,) * 2
        return edge_index, value, size

    def get_vu_edge(self, union_graph=False):
        edge_index, value, size = self.get_uv_edge(union_graph=union_graph)
        edge_index = reversed(edge_index)
        return edge_index, value, size

    def get_union_edge(self, union_type="u-uv-vu-v"):
        types = union_type.split("-")
        edges = []
        size = (self.size_u+self.size_v, )*2
        for type in types:
            assert type in ["u","v","uv","vu"]
            edge = self.__getattribute__(f"get_{type}_edge")(union_graph=True)
            edges.append(edge)
        edge_index = torch.cat([edge[0] for edge in edges], dim=1)
        value = torch.cat([edge[1] for edge in edges], dim=0)
        return edge_index, value, size

    @staticmethod
    def collate_fn(batch):
        return batch

# batch_size=1024*5 epochs 64 auroc:0.9357351384806536-aupr:0.553179962144391 Cdataset
# batch_size=1024*5 epochs 32 auroc:0.9353403407919936-aupr:0.5319780678531647 Cdataset
# batch_size=1024*10 epochs 128 auroc:0.9205757164088856-aupr:0.5228925211295569
# batch_size=1024*10 epochs 64 auroc:0.9161912302288697-aupr:0.4971947471119751
# batch_size=1024*10 epochs 32 auroc:0.8978203124764995-aupr:0.3906882262477967
# batch_size=1024*5 epochs 64 auroc:0.9214539160124139-aupr:0.5080356373187406
# batch_size=1024*5 epochs 32 auroc:0.9233284816950948-aupr:0.4833606718343333
# batch_size=1024*1 epochs 32 auroc:0.9190991473694987-aupr:0.44253258179201654
class GraphDataIterator(DataLoader):
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", batch_size=1024*5, shuffle=False,
                 dataset_type="FullGraphDataset", **kwargs):
        # assert dataset_type in ["FullGraphDataset", "PairGraphDataset"]
        dataset_cls = DATA_TYPE_REGISTRY.get(dataset_type)
        dataset = dataset_cls(dataset, mask, fill_unkown, stage=stage, **kwargs)
        if len(dataset)<batch_size:
            logging.info(f"dataset size:{len(dataset)}, batch_size:{batch_size} is invalid!")
            batch_size = min(len(dataset), batch_size)
        if shuffle and stage=="train":
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        super(GraphDataIterator, self).__init__(dataset=dataset, batch_size=None, sampler=batch_sampler,
                                            collate_fn=Dataset.collate_fn, **kwargs)


class CVDataset(pl.LightningDataModule):
    """use for cross validation
       split_mode | n_splits |  drug_id   |   disease_id | description
       global     |   1      |   *        |     *        | case study
       global     |  10      |   *        |     *        | 10 fold
       local      |  -1      |   not None |     *        | local leave one for remove drug
       local      |  -1      |   None     |     not None | local leave one for remove disease
       local      |   1      |   int      |     *        | local leave one for remove specific drug
       local      |   1      |   None     |     int      | local leave one for remove specific drug
    """
    def __init__(self, dataset, split_mode="global", n_splits=10,
                 drug_idx=None, disease_idx=None, global_test_all_zero=False,
                 train_fill_unknown=True, seed=666, cached_dir="cached",
                 dataset_type="FullGraphDataset",
                 **kwargs):
        super(CVDataset, self).__init__()
        self.dataset = dataset
        self.split_mode = split_mode
        self.n_splits = n_splits
        self.global_test_all_zero = global_test_all_zero
        self.train_fill_unknown = train_fill_unknown
        self.seed = seed
        self.row_idx = drug_idx
        self.col_idx = disease_idx
        self.dataset_type = dataset_type
        self.save_dir = os.path.join(cached_dir, dataset.dataset_name,
                                     f"{self.split_mode}_{len(self)}_split_{self.row_idx}_{self.col_idx}")
        assert isinstance(n_splits, int) and n_splits>=-1

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("cross validation config")
        parser.add_argument("--split_mode", default="global", choices=["global", "local"])
        parser.add_argument("--n_splits", default=10, type=int)
        parser.add_argument("--drug_idx", default=None, type=int)
        parser.add_argument("--disease_idx", default=None, type=int)
        parser.add_argument("--global_test_all_zero", default=False, action="store_true", help="全局模式每折测试集是否测试所有未验证关联，默认：不测试")
        parser.add_argument("--train_fill_unknown", default=True, action="store_true", help="训练集中是否将测试集关联填0还是丢弃，默认：丢弃")
        parser.add_argument("--dataset_type", default=None, choices=["FullGraphDataset", "PairGraphDataset"])
        parser.add_argument("--seed", default=666, type=int)
        return parent_parser

    def fold_mask_iterator(self, interactions, mode="global", n_splits=10, row_idx=None, col_idx=None, global_test_all_zero=False, seed=666):
        assert mode in ["global", "local"]
        assert n_splits>=-1 and isinstance(n_splits, int)
        if mode=="global":
            if n_splits==1:
                mask = np.ones_like(interactions, dtype="bool")
                yield mask, mask
            else:
                kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                pos_row, pos_col = np.nonzero(interactions)
                neg_row, neg_col = np.nonzero(1 - interactions)
                assert len(pos_row) + len(neg_row) == np.prod(interactions.shape)
                for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                        kfold.split(neg_row)):
                    train_mask = np.zeros_like(interactions, dtype="bool")
                    test_mask = np.zeros_like(interactions, dtype="bool")
                    if global_test_all_zero:
                        test_neg_idx = np.arange(len(neg_row))
                    train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
                    train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
                    test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
                    test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
                    train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
                    test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
                    train_mask[train_edge[0], train_edge[1]] = True
                    test_mask[test_edge[0], test_edge[1]] = True
                    yield train_mask, test_mask
        elif mode=="local":
            if row_idx is not None:
                row_idxs = list(range(interactions.shape[0])) if n_splits==-1 else [row_idx]
                for idx in row_idxs:
                    yield self.get_fold_local_mask(interactions, row_idx=idx)
            elif col_idx is not None:
                col_idxs = list(range(interactions.shape[1])) if n_splits==-1 else [col_idx]
                for idx in col_idxs:
                    yield self.get_fold_local_mask(interactions, col_idx=idx)
        else:
            raise NotImplemented

    def get_fold_local_mask(self, interactions, row_idx=None, col_idx=None):
        train_mask = np.ones_like(interactions, dtype="bool")
        test_mask = np.zeros_like(interactions, dtype="bool")
        if row_idx is not None:
            train_mask[row_idx, :] = False
            test_mask[np.ones(interactions.shape[1], dtype="int")*row_idx,
                      np.arange(interactions.shape[1])] = True
        elif col_idx is not None:
            train_mask[:,col_idx] = False
            test_mask[np.arange(interactions.shape[0]),
                      np.ones(interactions.shape[0], dtype="int") * col_idx] = True
        return train_mask, test_mask

    def prepare_data(self):
        save_dir = self.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        import glob
        if len(glob.glob(os.path.join(save_dir, "split_*.mat")))!=len(self):
            for i, (train_mask, test_mask) in enumerate(self.fold_mask_iterator(interactions=self.dataset.interactions,
                                                                 mode=self.split_mode,
                                                                 n_splits=self.n_splits,
                                                                 global_test_all_zero=self.global_test_all_zero,
                                                                 row_idx=self.row_idx,
                                                                 col_idx=self.col_idx)):
                scio.savemat(os.path.join(save_dir, f"split_{i}.mat"),
                             {"train_mask":train_mask,
                              "test_mask":test_mask},
                             )

        data = scio.loadmat(os.path.join(self.save_dir, f"split_{self.fold_id}.mat"))
        self.train_mask = data["train_mask"]
        self.test_mask = data["test_mask"]

    def train_dataloader(self):
        return GraphDataIterator(self.dataset, self.train_mask, fill_unkown=self.train_fill_unknown,
                                 stage="train", dataset_type=self.dataset_type)

    def val_dataloader(self):
        return GraphDataIterator(self.dataset, self.test_mask, fill_unkown=True,
                                 stage="val", dataset_type=self.dataset_type)

    def __iter__(self):
        for fold_id in range(len(self)):
            self.fold_id = fold_id
            yield self

    def __len__(self):
        if self.split_mode=="global":
            return self.n_splits
        elif self.split_mode=="local":
            if self.n_splits==-1:
                if self.row_idx is not None:
                    return self.dataset.interactions.shape[0]
                elif self.col_idx is not None:
                    return self.dataset.interactions.shape[1]
            else:
                return 1

def load_DRIMC(root_dir="dataset/LRSSL", name="c", reduce=True):
    """ C drug:658, disease:409 association:2520 (False 2353)
        PREDICT drug:593, disease:313 association:1933 (Fdataset)
        LRSSL drug: 763, disease:681, association:3051
    """
    drug_chemical = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_chemical.txt"), sep="\t", index_col=0)
    drug_domain = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_domain.txt"), sep="\t", index_col=0)
    drug_go = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_go.txt"), sep="\t", index_col=0)
    disease_sim = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dg.txt"), sep="\t", index_col=0)
    if reduce:
        drug_sim =  (drug_chemical+drug_domain+drug_go)/3
    else:
        drug_sim = drug_chemical
    drug_disease = pd.read_csv(os.path.join(root_dir, f"{name}_admat_dgc.txt"), sep="\t", index_col=0).T
    if name=="lrssl":
        drug_disease = drug_disease.T
    rr = drug_sim.to_numpy(dtype=np.float32)
    rd = drug_disease.to_numpy(dtype=np.float32)
    dd = disease_sim.to_numpy(dtype=np.float32)
    rname = drug_sim.columns.to_numpy()
    dname = disease_sim.columns.to_numpy()
    return {"drug":rr,
            "disease":dd,
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}


def load_HDVD(root_dir="dataset/hdvd"):
    """drug:219, virus:34, association: 455"""
    dd = pd.read_csv(os.path.join(root_dir, "virussim.csv"), index_col=0).to_numpy(np.float32)
    rd = pd.read_csv(os.path.join(root_dir, "virusdrug.csv"), index_col=0)
    rr = pd.read_csv(os.path.join(root_dir, "drugsim.csv"), index_col=0).to_numpy(np.float32)
    rname = rd.index.to_numpy()
    dname = rd.columns.to_numpy()
    rd = rd.to_numpy(np.float32)
    return {"drug":rr,
            "disease":dd,
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}


def bgnn_pool(xw, adj):
    sum = adj@xw
    sum_squared = sum.square()
    # step2 squared_sum
    squared = xw.square()
    squared_sum = torch.square(adj)@squared
    # step3
    new_embedding = sum_squared - squared_sum
    return new_embedding

def bgnn_a_norm(edge_index, add_self_loop=True):
    adj_t = edge_index.to_dense()
    if add_self_loop:
        adj_all = adj_t+torch.eye(adj_t.shape[0], device=adj_t.device)
    # num_nei = adj_all.sum(dim=-1)
    norm = (adj_all.sum(dim=-1).square()-adj_all.square().sum(dim=-1))
    # norm = num_nei*(num_nei-1)
    norm = norm.pow(-1)
    norm.masked_fill_(torch.isinf(norm), 0.)
    norm = torch.diag(norm)
    norm = norm.to_sparse()
    adj_all = adj_all.to_sparse()
    return adj_all, norm


class BGNNA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        super(BGNNA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self._cache = None
        self.add_self_loops = add_self_loops
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index):
        xw = x@self.weight
        if self.cached:
            if not hasattr(self, "cached_adj") or not hasattr(self, "cached_norm"):
                adj, norm = bgnn_a_norm(edge_index, add_self_loop=self.add_self_loops)
                self.register_buffer("cached_adj", adj)
                self.register_buffer("cached_norm", norm)
            else:
                adj, norm = self.cached_adj, self.cached_norm
        else:
            adj, norm = bgnn_a_norm(edge_index, add_self_loop=self.add_self_loops)
        out = bgnn_pool(xw, adj)
        out = norm@out
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape, device=adj_t.device)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index, None


class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = True, add_self_loops: bool = False,
                 bias: bool = True, **kwargs):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cached = cached
        self.add_self_loops = add_self_loops

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index):
        if self.cached:
            if not hasattr(self, "cached_adj"):
                edge_index, edge_weight = gcn_norm(
                    edge_index, self.add_self_loops)
                self.register_buffer("cached_adj", edge_index)
            edge_index = self.cached_adj
        else:
            edge_index, _ = gcn_norm(edge_index, self.add_self_loops)
        x = torch.matmul(x, self.weight)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = edge_index@x
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)




class BGCNA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, lamda=0.8, share=True, **kwargs):
        super(BGCNA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.bgnn = BGNNA(in_channels=in_channels, out_channels=out_channels, cached=cached,
                          add_self_loops=add_self_loops, bias=bias)
        self.gcn = GCNConv(in_channels=in_channels, out_channels=out_channels, cached=cached,
                           add_self_loops=add_self_loops, bias=bias)

        self.register_buffer("alpha", torch.tensor(lamda))
        self.register_buffer("beta", torch.tensor(1-lamda))
        self.reset_parameters()
        if share:
            self.bgnn.weight = self.gcn.weight

    def reset_parameters(self):
        self.bgnn.reset_parameters()
        self.gcn.reset_parameters()

    def forward(self, x, edge_index):
        x1 = self.gcn(x, edge_index)
        x2 = self.bgnn(x, edge_index)
        x = self.beta*F.relu(x1)+self.alpha*F.relu(x2)
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
