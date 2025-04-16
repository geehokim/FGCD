from trainers.build import get_trainer_type


from trainers.base_trainer import Trainer
from trainers.nomp_base_trainer import BaseTrainer
from trainers.local_trainer import LocalTrainer
from trainers.client0_trainer import Client0Trainer
from trainers.EMtrainer_warmuped import EMTrainer
from trainers.centroid_clustering_trainer import CCTrainer
from trainers.prior_update_trainer import PriorUpdateTrainer
# from trainers.base_trainer_whichlearnfaster import Trainer_whichlearnfaster
# from trainers.base_trainer_compare import Trainer_compare
# from trainers.base_trainer_AvgM import Trainer_AvgM
# from trainers.base_trainer_Adam import Trainer_Adam
# from trainers.onlyumap import onlyumap
# from trainers.avgm_recheck import avgm_recheck
# from trainers.avgm_recheck_freqACG import avgm_recheck_freqACG
# from trainers.base_trainer_dyn import Trainer_dyn
# from trainers.base_trainer_mimelite import Trainer_mimelite