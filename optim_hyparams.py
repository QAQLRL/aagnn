
#首先加载数据
import torch
import optuna
import os, json,time


from train import   train
from AttentionAlignn.model import BondAngleGraphAttention
from AttentionAlignn.config import TrainingConfig





#目标函数：
def object(trial:optuna.trial.Trial):
    '''
    优化参数：1. learn rate ,
             2. weight deacy
             3. layers of alignn,gat,gcnii
             4. batch size 
             5. input feature embedding size
             6. attention head numer
    '''
    train_config.epochs = 300
    train_config.criterion = "mse"
    train_config.learning_rate = trial.suggest_loguniform('learning_rate',1e-4,1e-1)
    train_config.weight_decay = trial.suggest_loguniform('weight_decay',1e-7,1e-3)
    train_config.batch_size = trial.suggest_categorical('batch_size',[16,32,64])
    train_config.model.embedding_features = trial.suggest_int('embedding_features',8,128)
    train_config.model.alignn_layers = trial.suggest_int('alignn_layers',1,10)
    train_config.model.gat_layers = trial.suggest_int('gat_layers',1,10)
    train_config.model.gcnii_layers = trial.suggest_int('gcnii_layers',1,20)
    train_config.model.hidden_features = trial.suggest_categorical('hidden_features',[32,64,128])
    train_config.model.heads = trial.suggest_int('attenton_head_number',1,16)

    net = BondAngleGraphAttention(train_config.model)
    
    print("load data")
    df = torch.load('/hy-tmp/jarvis_structure_graph_optb88vdw_bandgap.pt')
     #训练模型
    print('train model is doing')
    t1 = time.time()
    res = train(net,df,'bandgap',train_config,trial)
    t2 = time.time()
    print("trian is end")
    print("Time taken (s):", t2 - t1)
    return res

if __name__ == '__main__':
  
    with open('config.json','r',encoding='utf8')as fp:
        config = json.load(fp)
    train_config = TrainingConfig(**config)
   
    print('optimizing')
    study = optuna.create_study(study_name="hyparams", storage='sqlite:///hyparams2.db',load_if_exists=True)
    study.optimize(object,n_trials=100)
    print(f"best params is: {study.best_params}")
    import os
    os.system('/root/attention-alignn/upload.sh')
    # studynew = optuna.create_study(study_name='hyparams', storage='sqlite:///hyparams.db', load_if_exists=True)

