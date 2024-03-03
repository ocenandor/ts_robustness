import json
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from ignite.contrib.handlers import wandb_logger
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import ModelCheckpoint
from ignite.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy, Loss
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from torch import nn
from torch.functional import F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

sys.path.append('../')
from src.datasets import FordDataset
from src.models import TransformerClassification
from src.utils import build_optimizer, duplicate_batch, split_batch, str2torch


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        ### Init data
        train_path = "../data/FordA/FordA_TRAIN.arff"
        train_dataset = FordDataset(train_path, config['data'])
        ### Random samplers from train data
        idx = np.arange(len(train_dataset))
        idx_train, idx_val = train_test_split(idx, train_size=0.8,
                                              stratify=train_dataset.labels, random_state=config['random_state'])
        train_sampler = SubsetRandomSampler(idx_train)
        val_sampler = SubsetRandomSampler(idx_val)
        ### Dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], sampler=train_sampler)
        val_dataloader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], sampler=val_sampler)
        
        ### Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['model']['encoder']['activation'] = str2torch(config['model']['encoder']['activation'])
        config['train']['optimizer'] = str2torch(config['train']['optimizer'])
        model = TransformerClassification(config).to(device)

        ### Initialize optimizer and criterion
        optimizer = build_optimizer(config, model)
        criterion = nn.BCELoss()

        ### Train and validate functions for pytorch ignite
        def train_step(engine, batch):
            model.train()
            optimizer.zero_grad()
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            return loss.item()

        def validation_step(engine, batch):
            model.eval()
            with torch.no_grad():
                x, y = batch[0].to(device), batch[1].to(device)
                y_pred = model(x)
                return y_pred, y
        
        ### Init trainers
        trainer = Engine(train_step)
        train_evaluator = Engine(validation_step)
        val_evaluator = Engine(validation_step)

        ### Attach metrics to the evaluators
        metrics = {
            'accuracy': Accuracy(output_transform=lambda x: (x[0] > 0.5, x[1])),
            'loss': Loss(criterion, output_transform=lambda x: (x[0], x[1].unsqueeze(1)))
        }

        for name, metric in metrics.items():
            metric.attach(train_evaluator, name)
            metric.attach(val_evaluator, name)

        ### Logging
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            train_evaluator.run(train_dataloader)
            metrics = train_evaluator.state.metrics
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}"
                .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))
            wandb.log({"train_accuracy": metrics['accuracy'],
                    "train_loss": metrics['loss']})

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            val_evaluator.run(val_dataloader)
            metrics = val_evaluator.state.metrics
            print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}"
                .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))
            wandb.log({"val_accuracy": metrics['accuracy'],
                    "val_loss": metrics['loss']})

        ### Run the training loop
        trainer.run(train_dataloader, max_epochs=config['train']['n_epoch'])


if __name__ == '__main__':
    # Load config 
    with open('transformer_sweep1.json') as f:
        config = json.load(f)
    # Init wandb sweep
    sweep_id = wandb.sweep(entity='ts-robustness', sweep=config, project="ml-course")
    wandb.agent(sweep_id, train, count=200)