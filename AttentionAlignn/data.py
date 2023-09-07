#数据转换为图：晶体图，线图，提供dataloader,dataset
from typing import List, Tuple,Union
from torch.utils.data import DataLoader,Dataset
from .graphs import Graph, StructureDataset
from jarvis.core.atoms import Atoms
import os,sys
import csv
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from pandas import DataFrame
import torch
# use pandas progress_apply
tqdm.pandas()

def load_graphs(
    df: DataFrame,
    cutoff: float = 8,
    max_neighbors: int = 12,
    use_canonize: bool = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize = use_canonize,
        )


    graphs = df["atoms"].progress_apply(atoms_to_graph).values
       

    return graphs


#划分数据集，返回索引
def get_index_train_val_test( total_size,n_train=None,n_val=None,n_test=None, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,random_seed=42):
    # indices = list(range(total_size))
    if n_train is None and n_val is None and n_test is None:
        
        n_train = int(train_ratio * total_size)
        n_test = int(test_ratio * total_size)
        n_val = int(val_ratio * total_size)
        
    ids = list(np.arange(total_size))

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    random.seed(random_seed)
    random.shuffle(ids)
    # np.random.shuffle(ids)

    # full train/val test split
    # ids = ids[::-1]
    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test
#数据集分为两部分，一部分是结构文件，另一部分是target
def read_from_file(dir: str = "../predicted_mixed_perovskites",target_file: str ='id_prop.csv') -> DataFrame:
    print(os.getcwd())
    id_prop = os.path.join(dir,target_file)
    with open(id_prop, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        # 文件流迭代，每次读取一行，读出的一行也可以迭代
    dataset = []
   

    #从data读取结构文件到dataset,每个数据是字典格式
    for i in data:
        info = {}
        file_name = i[0]
        file_path = os.path.join(dir, file_name)
        file_ext = os.path.splitext(file_name)[-1]
        if file_ext == ".poscar" or ".vasp":
            atoms = Atoms.from_poscar(file_path)
        elif file_ext == ".cif":
            atoms = Atoms.from_cif(file_path)
        elif file_ext == ".xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_ext == ".pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError(
                "File format not implemented", file_ext
            )
        info["atoms"] = atoms.to_dict()
        info["target"] = float(i[1])
        # info['jid'] = file_name
        dataset.append(info)
    return  DataFrame(dataset)
def load_dataset(df:DataFrame=None,target=None,dir='',target_file='',save_path='',cutoff=8,max_neighbors=12, classification_threshold=None,use_canonize=True) ->StructureDataset:
    '''
    返回StructureDataset    '''

    if df is None:
        df = read_from_file(dir,target_file)
    if classification_threshold:   
        df.loc[df['optb88vdw_bandgap'] <= classification_threshold, 'optb88vdw_bandgap'] = 0
        df.loc[df['optb88vdw_bandgap'] > classification_threshold , 'optb88vdw_bandgap'] = 1
        print("Converting target data into 1 and 0.")

    graphs = load_graphs(df,cutoff=cutoff,max_neighbors=max_neighbors, use_canonize=use_canonize)

    #torch.utils.data.Dataset
    data = StructureDataset(
        df,
        graphs,
        target=target,
        atom_features='cgcnn',
        classification= True if classification_threshold is not None else False,
       
    )
    #保存数据，只有第一次才需要构造图，以后直接从文件读取
    torch.save(data,f'{save_path}/graph_{target}.pt')
    return data

def get_train_val_test_loader(
    df:Union[DataFrame,StructureDataset] = None,
    target:str=None,
    dir: str = "./AttentionAlignn/examples/sample_data",
    target_file: str ='id_prop.csv',
    classification_threshold=None,
    pin_memory = False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_val=None,
    n_test=None,
    batch_size: int = 5,
    workers: int = 0,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_canonize: bool = False
    ) -> Tuple[DataLoader,DataLoader,DataLoader]:
     
    #加载数据集
    if isinstance(df,DataFrame) or df is None:
        data = load_dataset(df,target,dir,target_file,
                            cutoff=cutoff, max_neighbors=max_neighbors,                            
                            classification_threshold =  classification_threshold ,
                            use_canonize=use_canonize)
    # elif isinstance(df,StructureDataset) or isinstance(df,tuple):
    else:
        data = df


    if classification_threshold is not None:
        print(
                "Using ",
                classification_threshold,
                " for classifying ",
                target,
                " data.",
            )

    collate_fn = data.collate_line_graph
    prepare_batch = data.prepare_batch
    id_train, id_val, id_test = get_index_train_val_test(
        total_size=len(data),
        n_train = n_train,
        n_val=n_val,
        n_test=n_test,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
   
    train_data = [data[x] for x in id_train]
    val_data = [data[x] for x in id_val]
    test_data = [data[x] for x in id_test]

    train_loader = DataLoader(train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=workers
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=workers
        )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader, prepare_batch

if __name__ == '__main__':
    # os.chdir(sys.path[0])
    print(os.getcwd())
    tu = get_train_val_test_loader()