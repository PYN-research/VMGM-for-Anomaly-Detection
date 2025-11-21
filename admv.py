'''
Distribution transformation with missing values.
'''
from main_model_table import TabCSDI,Autoencoder
#from networks.main import build_networks
from optim.trainer import Trainer
from data.main import load_dataset

class ADMissingValue(object):
    
    def __init__(self, config, dataset_name, net_name, data_path, optimizer_name: str, lr: float, epochs0: int,epochs: int, input_dim: int, batch_size: int, device: str, results_dir: str, 
                        print=None, in_channels=None, _lambda=None, beta=None, latent_dimension=None, missing_rate=0.0, entropy_reg_coe=None, mu=0, std=1,
                        stop_threshold=None, alpha=None, split=None, mechanism='mcar',num_cond_masks=None):
        self.model0 = Autoencoder(config, device,input_dim)
        self.model = TabCSDI(config,device,input_dim)
        self.num_cond_masks = num_cond_masks
        begin_epoch = 1
        end_epoch0 = begin_epoch + epochs0
        end_epoch = begin_epoch + epochs
  
     
        self.dataset =  load_dataset(dataset_name, data_path, missing_rate=missing_rate, 
                                     split=split, mechanism=mechanism, num_cond_masks=num_cond_masks)

        self.ae_trainer = Trainer(optimizer_name, lr=lr, begin_epoch=begin_epoch, dataset_name=dataset_name, end_epoch0=end_epoch0, end_epoch=end_epoch, batch_size=batch_size, device=device, print=print,
        results_dir=results_dir, _lambda=_lambda, latent_dimension=latent_dimension, beta=beta, entropy_reg_coe=entropy_reg_coe, mu=mu, std=std, missing_rate=missing_rate,
        stop_threshold=stop_threshold, alpha=alpha, input_dim=input_dim)

        
    def train(self):

        results = self.ae_trainer.train(self.dataset, self.model, self.model0, self.num_cond_masks)
        return results
