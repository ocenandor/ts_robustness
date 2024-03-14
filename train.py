import argparse
import json

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from torch import nn

import wandb
from src.datasets import make_dataset
from src.models import (CNNClassification, LSTMClassification,
                        TransformerClassification)
from src.utils import build_optimizer, str2torch

MODELS = {
    'cnn': CNNClassification,
    'transformer': TransformerClassification,
    'lstm': LSTMClassification
    }

def train(config=None, wandb_log=True, save_dir=None, dataset_dir=None):
    
    if wandb_log:
        wandb.init(entity='ts-robustness', project='ml-course', config=config)
        config = wandb.config
    torch.manual_seed(config['random_state'])
    torch.backends.cudnn.deterministic = True

    train_dataloader, _, test_dataloader = make_dataset(config, dataset_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['train']['optimizer'] = str2torch(config['train']['optimizer'])
    
    model = MODELS[config['model']['name']] # TODO move all such dicts to one place
    model = model(config).to(device)

    optimizer = build_optimizer(config, model)
    criterion = nn.CrossEntropyLoss()


    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y.long())
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            return y_pred, y


    trainer = Engine(train_step)
    train_evaluator = Engine(validation_step)
    test_evaluator = Engine(validation_step)

    metrics = {
        'accuracy': Accuracy(output_transform=lambda x: (torch.argmax(x[0], dim=1), x[1])),
        'loss': Loss(criterion, output_transform=lambda x: (x[0], x[1].long()))
    }

    for name, metric in metrics.items():
        metric.attach(train_evaluator, name)
        metric.attach(test_evaluator, name)
    if save_dir is None:
        if wandb_log:
            save_dir = wandb.run.dir + '/../saved_models'
        else:
            save_dir = './saved_models'
    print(f'model will be saved in {save_dir}')
    checkpoint_handler = ModelCheckpoint(dirname=save_dir, filename_prefix=config['model']['name'], # TODO make file pattern name
                                        n_saved=1, require_empty=False,
                                        score_function=lambda engine: engine.state.metrics['accuracy'],
                                        score_name="accuracy", global_step_transform=lambda *_: trainer.state.epoch)
    test_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {"model": model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_dataloader)
        metrics = train_evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}"
            .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))
        if wandb_log:
            wandb.log({"train_accuracy": metrics['accuracy'],
                    "train_loss": metrics['loss']})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(trainer):
        test_evaluator.run(test_dataloader)
        metrics = test_evaluator.state.metrics
        print("Test Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.4f}"
            .format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))
        if wandb_log:
            wandb.log({"test_accuracy": metrics['accuracy'],
                "test_loss": metrics['loss']})

    trainer.run(train_dataloader, max_epochs=config['train']['n_epoch'])
    if wandb_log: wandb.finish()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="train json config")
    parser.add_argument("--dir", type=str, default=None, help="dir for saving model weights")
    parser.add_argument("--wandb", type=bool, default=True, help="enable wandb logging, --no-wandb to turn logging off",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--data", type=str, default='data/FordA', help="path to data folder")
    
    args = parser.parse_args()

    with open(args.config) as f:
        config =  json.load(f)

    train(config, args.wandb, args.dir, args.data)
