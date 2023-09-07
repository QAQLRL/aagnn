
from typing import Any, Dict, List, Union
from functools import partial
import torch
from torch import nn
from pandas import DataFrame
import  json,os,time
from jarvis.db.jsonutils import dumpjson
from ignite.contrib.handlers import TensorboardLogger,global_step_from_engine
from ignite.handlers.stores import EpochOutputStore
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
from ignite.engine import create_supervised_trainer,create_supervised_evaluator,Events
from ignite.metrics import Loss, MeanAbsoluteError,Accuracy
from ignite.handlers import EarlyStopping,Checkpoint, DiskSaver, TerminateOnNan 
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.metrics.regression import R2Score
from ignite.engine import Engine
from AttentionAlignn.data import get_train_val_test_loader
from AttentionAlignn.model import BondAngleGraphAttention
from AttentionAlignn.config import TrainingConfig
from AttentionAlignn.graphs import StructureDataset
import json
from ignite.utils import manual_seed

def group_decay(model:torch.nn.Module):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, train_config:TrainingConfig):
    """Set up optimizer for param groups."""
    if  train_config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=train_config.learning_rate, weight_decay=train_config.weight_decay,
        )
    elif train_config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=train_config.learning_rate,
            momentum=0.9,
            weight_decay=train_config.weight_decay,
        )
    return optimizer

def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y

def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y
def train(model: nn.Module,df:Union[DataFrame,StructureDataset], target,train_config:TrainingConfig,save_data_path:str = ''):
    manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dev is {device}")
    # define optimizer, scheduler
    net = model.to(device)

    #dataloader
    train_val_test_loaders = get_train_val_test_loader(df,target, 
                                                    n_train=train_config.n_train,
                                                    classification_threshold=train_config.classification_threshold,
                                                    train_ratio=train_config.train_ratio,
                                                    val_ratio=train_config.val_ratio,
                                                    test_ratio=train_config.test_ratio,
                                                    n_val=train_config.n_val, 
                                                    n_test=train_config.n_test,
                                                    save_data_path=save_data_path,
                                                    pin_memory=train_config.pin_memory,
                                                    workers=train_config.num_workers,
                                                    batch_size=train_config.batch_size,
                                                    use_canonize=train_config.use_canonize,
                                                    cutoff=train_config.cutoff,
                                                    max_neighbors=train_config.max_neighbors
                                                    )
    train_loader = train_val_test_loaders[0]
    val_loader = train_val_test_loaders[1]
    test_loader = train_val_test_loaders[2]
    prepare_batch = train_val_test_loaders[3]
    prepare_batch = partial(prepare_batch, device=device)
    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params,train_config)
    
    #define scheduler
    if train_config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 )
    elif train_config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_config.learning_rate,
            epochs=train_config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif train_config.scheduler == "step":
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer)
    
    def rmse(yhat,y):
        return torch.sqrt(torch.mean((yhat-y)**2))
     # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "rmse": rmse,
        "l1": nn.L1Loss(),
        "smoothl1":nn.SmoothL1Loss(),
        "huber":nn.HuberLoss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
    }
    criterion = criteria[train_config.criterion]
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError(),"r2":R2Score()}
    if train_config.classification_threshold:
        criterion = nn.NLLLoss()
        metrics = {
            "accuracy": Accuracy(
                output_transform=thresholded_output_transform
            ),
            "precision": Precision(
                output_transform=thresholded_output_transform
            ),
            "recall": Recall(output_transform=thresholded_output_transform),
            "rocauc": ROC_AUC(output_transform=activated_output_transform),
            "roccurve": RocCurve(output_transform=activated_output_transform),
            "confmat": ConfusionMatrix(
                output_transform=thresholded_output_transform, num_classes=2
            ),
        }
    # set up training engine and evaluators
   
    trainer = create_supervised_trainer(net,optimizer,criterion, prepare_batch=prepare_batch, device=device, gradient_accumulation_steps=4)
    val_evaluator = create_supervised_evaluator(net,metrics=metrics, prepare_batch=prepare_batch, device=device)
    train_evaluator = create_supervised_evaluator(net,metrics=metrics, prepare_batch=prepare_batch, device=device)
    
    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
     # Register a pruning handler to the evaluator.
    # pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, "loss", trainer)
    # train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, pruning_handler)
    
    #save output
    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        """Print training and validation metrics to console."""
        
        train_evaluator.run(train_loader)
        val_evaluator.run(val_loader)

        tmetrics = train_evaluator.state.metrics 
        vmetrics = val_evaluator.state.metrics
        pbar = ProgressBar()
        if train_config.progress:
            pbar = ProgressBar()
            if not train_config.classification_threshold:
                pbar.log_message(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy(mae): {tmetrics['mae']:.4f} Avg loss: {tmetrics['loss']:.4f} R2: {vmetrics['r2']:.4f}")
                pbar.log_message(f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy(mae): {vmetrics['mae']:.4f} Avg loss: {vmetrics['loss']:.4f} R2: {vmetrics['r2']:.4f}" )
            else:
                pbar.log_message(f"Train ROC AUC: {tmetrics['rocauc']:.4f}")
                pbar.log_message(f"Val ROC AUC: {vmetrics['rocauc']:.4f}")
       

        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()
            if metric == "roccurve":
                tm = [t.cpu().numpy().tolist()for t in tm]
                vm = [t.cpu().numpy().tolist()for t in vm]
                history["train"][metric].extend(tm)
                history["validation"][metric].extend(vm)
            else:
                history["train"][metric].append(tm)
                history["validation"][metric].append(vm)

        dumpjson(
                filename=os.path.join('.', train_config.checkpoint_dir+f"/{target}_history_val.json"),
                data=history["validation"],
            )
        dumpjson(
            filename=os.path.join('.',train_config.checkpoint_dir+f"/{target}_history_train.json"),
            data=history["train"],
        )
    #早停法，没有更多进展时尽早中断训练
    # def default_score_fn(engine):
    #     return -engine.state.metrics['mae']
    
    # es_handler = EarlyStopping(
    #     patience=train_config.n_early_stopping,
    #     score_function=default_score_fn,
    #     trainer=trainer,
    # )

    if train_config.n_early_stopping is not None:

        # early stopping if no improvement (improvement = higher score)
        if  train_config.classification_threshold:
            def es_score(engine):
                """Higher accuracy is better."""
                return engine.state.metrics["accuracy"]

        else:
            def es_score(engine):
                """Lower MAE is better."""
                return -engine.state.metrics["mae"]

        es_handler = EarlyStopping(
            patience=train_config.n_early_stopping,
            score_function=es_score,
            trainer=trainer,
        )
        val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)


    # train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)
    
    #保存模型
    to_save = {
        "model": net,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "trainer": trainer,
    }
    checkpoint_handler = Checkpoint(
        to_save,
        DiskSaver(train_config.checkpoint_dir, create_dir=True, require_empty=False),
        filename_prefix=target,
        score_function=es_score,
        n_saved=1,
        global_step_transform=global_step_from_engine(trainer),
    )
    val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
    
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    # optionally log results to tensorboard
    if train_config.log_tensorboard:
        if not train_config.classification_threshold:
            metric_names=['loss','mae']
        else:
            metric_names=["accuracy","rocauc"]
        
        tb_logger = TensorboardLogger(train_config.log_dir+target)
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", val_evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=metric_names,
                global_step_transform=global_step_from_engine(trainer),
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_time():
        print(f"Epoch {trainer.state.epoch}, Time Taken : {trainer.state.times['EPOCH_COMPLETED']}")


    @trainer.on(Events.COMPLETED)
    def log_total_time():
        print(f"Total Time: {trainer.state.times['COMPLETED']}")
        
   
    
    # train the model!
    trainer.run(train_loader, max_epochs=train_config.epochs)
    tb_logger.close()
    print("check",checkpoint_handler.last_checkpoint)
    
    # return train_evaluator.state.metrics["loss"]
    return history

if __name__ == '__main__':   
    
    
    import pandas as pd
    df = None
    # jarvis dataset
    # df=pd.read_json('/hy-tmp/jdft_3d-8-18-2021.json')  
    # 'formation_energy_peratom',
    # target = ['formation_energy_peratom','optb88vdw_bandgap','optb88vdw_total_energy','ehull']
    # df = df.loc[:,['atoms','formation_energy_peratom','optb88vdw_bandgap','optb88vdw_total_energy','ehull','mbj_bandgap']]
    # ehull = pd.to_numeric(df.loc[:,'ehull'], errors='coerce')
    # df['ehull']=ehull
    # df.dropna(inplace=True)
    
    # mp dataset 
    # 
   
    
    with open('config.json','r',encoding='utf8')as fp:
        config = json.load(fp)
    train_config = TrainingConfig(**config)

    
    #训练模型
    # target = ['formation_energy_peratom','optb88vdw_bandgap','optb88vdw_total_energy','ehull']
    # target = ['optb88vdw_bandgap','mbj_bandgap','slme','spillage','ehull','n-Seebeck','p-Seebeck','n-powerfact', 'p-powerfact']
    target = {'optb88vdw_bandgap':0.01,'mbj_bandgap':0.01,'slme':10,'spillage':0.1,'ehull':0.1,'n-Seebeck':-100,'p-Seebeck':100,'n-powerfact':1000, 'p-powerfact':1000}
    # print('train model is doing')
    for name, thresholded in target.items:
        # 对于分类 不同的目标有不同的阈值
        print("target is: ",name)
        # dataset = "dataset"
        dataset_dir = '/root/autodl-fs/dataset'

        if os.path.exists(f'{name}.pt'):
            data = torch.load(f'{dataset_dir}/graph_{name}.pt')
        else:
            with open(f'{dataset_dir}/jdft_3d-8-18-2021.json', 'r') as f:
                data = f.read()

            # 将字符串 "na" 替换为缺失值（NaN）
            data = data.replace('"na"', 'NaN')

            # 解析 JSON 数据
            df = pd.read_json(data)
            data = df.dropna(subset=name)
        # if train_config.classification_threshold:
        train_config.classification_threshold = thresholded
        train_config.model.classification = True
        net = BondAngleGraphAttention(train_config.model)
        # try:
        train(net,data,name,train_config,dataset_dir)
        print(f"train target: {name} is done\n\n")
        # except Exception as e:
        #     print("出错了：",e)

    # print("trian is end")
    os.system("/usr/bin/shutdown")