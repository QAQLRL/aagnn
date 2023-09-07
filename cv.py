import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

from sklearn.model_selection import KFold
import numpy as np

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        return self.model(x)


data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_dataset = MNIST(download=True, root=".", transform=data_transform, train=True)
test_dataset = MNIST(download=True, root=".", transform=data_transform, train=False)

def setup_dataflow(dataset, train_idx, val_idx):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=256, sampler=val_sampler)

    return train_loader, val_loader

def initialize():
    model = Net().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-06)
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

def train_model(train_loader, val_loader):
    max_epochs = 3

    train_results = []
    val_results = []

    model, optimizer, criterion = initialize()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model, metrics={"Accuracy": Accuracy(), "Loss": Loss(criterion)}, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        train_results.append(metrics)
        print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['Accuracy']:.2f} Avg loss: {metrics['Loss']:.2f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        val_results.append(metrics)

    trainer.run(train_loader, max_epochs=max_epochs) 

    return train_results, val_results

dataset = ConcatDataset([train_dataset, test_dataset])

num_folds = 3
splits = KFold(n_splits=num_folds,shuffle=True,random_state=42)

results_per_fold = []

for fold_idx, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

    print('Fold {}'.format(fold_idx + 1))

    train_loader, val_loader = setup_dataflow(dataset, train_idx, val_idx)
    train_results, val_results = train_model(train_loader, val_loader)
    results_per_fold.append([train_results, val_results])

acc_sum = 0
for n_fold in range(len(results_per_fold)):
  current_fold = results_per_fold[n_fold]
  print(f"Validation Results - Fold[{n_fold + 1}] Avg accuracy: {current_fold[1][2]['Accuracy']:.2f} Avg loss: {current_fold[1][2]['Loss']:.2f}")
  acc_sum += current_fold[1][2]['Accuracy']

folds_mean = acc_sum/num_folds
print(f"Model validation average for {num_folds}-folds: {folds_mean :.2f}")
