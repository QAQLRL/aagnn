# 使用pydantic 待学习
from pydantic.env_settings import BaseSettings
import subprocess
from typing import Optional, Union
import os
from pydantic import root_validator
from pydantic.typing import Literal

class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 10
    gat_layers: int = 1
    gcnii_layers: int = 6
    heads: int = 10
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 82
    hidden_features: int = 64
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    num_classes: int = 2
    

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"
        
class TrainConfig(BaseSettings): 
    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"

try:
    VERSION = (subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip())
except Exception:
    VERSION = "NA"


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}


TARGET_ENUM = Literal[
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
    "gap pbe",
    "e_form",
    "e_hull",
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "e_above_hull",
    "mu_b",
    "bulk modulus",
    "shear modulus",
    "elastic anisotropy",
    "U0",
    "HOMO",
    "LUMO",
    "R2",
    "ZPVE",
    "omega1",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U",
    "H",
    "G",
    "Cv",
    "A",
    "B",
    "C",
    "all",
    "target",
    "max_efg",
    "avg_elec_mass",
    "avg_hole_mass",
    "_oqmd_band_gap",
    "_oqmd_delta_e",
    "_oqmd_stability",
    "edos_up",
    "pdos_elast",
    "bandgap",
    "energy_total",
    "net_magmom",
    "b3lyp_homo",
    "b3lyp_lumo",
    "b3lyp_gap",
    "b3lyp_scharber_pce",
    "b3lyp_scharber_voc",
    "b3lyp_scharber_jsc",
    "log_kd_ki",
    "max_co2_adsp",
    "min_co2_adsp",
    "lcd",
    "pld",
    "void_fraction",
    "surface_area_m2g",
    "surface_area_m2cm3",
    "indir_gap",
    "f_enp",
    "final_energy",
    "energy_per_atom",
]


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION

    # dataset configuration
    dataset: Literal[
        "dft_3d",
        "jdft_3d-8-18-2021",
        "dft_2d",
        "megnet",
        "megnet2",
        "mp_3d_2020",
        "qm9",
        "qm9_dgl",
        "qm9_std_jctc",
        "user_data",
        "oqmd_3d_no_cfid",
        "edos_up",
        "edos_pdos",
        "qmof",
        "qe_tb",
        "hmof",
        "hpov",
        "pdbbind",
        "pdbbind_core",
    ] = "dft_3d"
    target: TARGET_ENUM = "formation_energy_peratom"
    neighbor_strategy  = 'k-nearest'
    # logging configuration

    # training configuration;
    atom_features: str = 'cgcnn'
    random_seed: Optional[int] = 123
    classification_threshold: Optional[float] = None
    # target_range: Optional[List] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    n_train: Optional[int] = None
    train_ratio: Optional[float] = 0.8
    val_ratio: Optional[float] = 0.1
    test_ratio: Optional[float] = 0.1
    checkpoint_dir = 'checkpoint'
    target_multiplication_factor: Optional[float] = None
    epochs: int = 300
    batch_size: int = 64
    weight_decay: float = 0
    learning_rate: float = 1e-2
    filename: str = "sample"
    warmup_steps: int = 2000
    criterion: Literal["mse","rmse", "l1", "poisson", "zig"] = "mse"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "none"] = "onecycle"
    pin_memory: bool = False
    save_dataloader: bool = False
    write_checkpoint: bool = True
    write_predictions: bool = True
    store_outputs: bool = True
    progress: bool = True
    log_tensorboard: bool = False
    standard_scalar_and_pca: bool = False
    use_canonize: bool = True
    num_workers: int = 4
    cutoff: float = 8.0
    max_neighbors: int = 12
    keep_data_order: bool = False
    normalize_graph_level_loss: bool = False
    distributed: bool = False
    n_early_stopping: Optional[int] = None  # typically 50
    output_dir: str = os.path.abspath(".")  # typically 50
    # alignn_layers: int = 4
    # gcn_layers: int =4
    # edge_input_features: int= 80
    # hidden_features: int= 256
    # triplet_input_features: int=40
    # embedding_features: int=64

    # model configuration
    model = ALIGNNConfig(name="alignn")


