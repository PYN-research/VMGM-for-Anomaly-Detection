from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import time
import torch
import torch.optim as optim
from tqdm import trange
import os
# self-defined library
class Trainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, begin_epoch: int = 1, end_epoch0: int = 100, end_epoch: int = 100,
                 batch_size: int = 128, device: str = 'cuda', print=None, dataset_name=None, results_dir=None, 
                 _lambda=None, latent_dimension=None, stop_threshold=None, entropy_reg_coe=None, beta=None, mu=0, 
                 std=1, missing_rate=0.0, alpha=None, input_dim=None):

        super().__init__(optimizer_name, lr, end_epoch - begin_epoch, None, batch_size, device)


        self.begin_epoch = begin_epoch
        self.end_epoch0 = end_epoch0
        self.end_epoch = end_epoch
        self._lambda = _lambda
        self.beta = beta
        self.mu = mu
        self.std = std
        self.missing_rate = missing_rate
        self.alpha = alpha
        self.input_dim = input_dim
        self.print = print
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.latent_dimension = latent_dimension
        self.target_distribution_sampling_epoch = 1
        self.test_epoch = 5
        self.results = {
            'auroc': 0.0,
            'auprc': 0.0,
            "Time": None 
        }


    def train(self, dataset: BaseADDataset, model, model0, num_cond_masks):
        
        # Set device for network
        model = model.to(self.device)
        model0 = model0.to(self.device)
        # Set optimizer
        if self.optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
        else:
            raise Exception(f'Unknown optimizer name [{self.optimizer_name}].')
        
        output_path='./pretrained_ae/model.pth'
        train_loader, _ = dataset.loaders(batch_size=self.batch_size)
        # Pretraining
        if os.path.exists(output_path):
            model0.load_state_dict(torch.load(output_path))
        else:
            model0.train()
            step = 1
            start = time.time()
            with trange(self.begin_epoch, self.end_epoch0) as pbar:
                for epoch in pbar:
                    loss_epoch = 0.0
                    n_batches = 0              
                    for data in train_loader:
                        inputs, _, _, obs_masks,_ = data
                        inputs = inputs.to(self.device)
                        obs_masks = obs_masks.to(self.device)
                        optimizer.zero_grad()
                        ae_loss,x_rec,_ = model0(inputs,obs_masks,is_train=1)
                        ae_loss.backward()
                        optimizer.step()
                        step += 1
                        loss_epoch += ae_loss.item()
                        n_batches += 1
                
                    pbar.set_description(
                        f'Loss: {loss_epoch / n_batches:.4f}')    
                torch.save(model0.state_dict(), output_path)
                
        # Training
        model.train()
        model0.eval()
        step = 1
        start = time.time()
        with trange(self.begin_epoch, self.end_epoch) as pbar:
            for epoch in pbar:
                loss_epoch = 0.0
                n_batches = 0              
                for data in train_loader:
                    inputs, _, _, obs_masks,_ = data
                    inputs = inputs.to(self.device)
                    obs_masks = obs_masks.to(self.device)
                    optimizer.zero_grad()
                    _,x_rec,_ = model0(inputs,obs_masks,is_train=0)
                    loss, mts_emb = model(inputs,x_rec,obs_masks,num_cond_masks,is_train=1)
                    loss.backward()
                    optimizer.step()
                    step += 1
                    loss_epoch += loss.item()
                    n_batches += 1
                
                pbar.set_description(
                    f'Loss: {loss_epoch / n_batches:.4f}')
                
                if epoch % self.test_epoch == 0:
                    self.print(f'\nepoch:[{epoch}]#############################')
                    auroc,auprc=self.test(dataset,model,model0,num_cond_masks,epoch)
                    model.train()
                    
                    
        self.results['Time'] = time.time() - start
        self.results['auroc'] = auroc
        self.results['auprc'] = auprc
        print(f'using time: {self.results["Time"]}')
        print('Finished training.')
        
        return self.results


    def test(self, dataset: BaseADDataset, model, model0, num_cond_masks,epoch):
        # Set device for network
        model = model.to(self.device)
        model0 = model0.to(self.device)
        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size)
        self.print('model testing ...')
        
        score = None
        idx_label_score = []
        model.eval()
        model0.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx, obs_masks,complete_inputs = data
                inputs = inputs.to(self.device)
                obs_masks = obs_masks.to(self.device)
                complete_inputs = complete_inputs.to(self.device)
                _,x_rec,x_feat = model0(inputs,obs_masks,is_train=0)
                samples,target_mask,samples_list,cond_samples,mts_emb_test = model.evaluate(inputs,x_rec,obs_masks,num_cond_masks,n_samples=10)
                #Aggregate T steps
                sample_mean = samples_list.mean(dim=0)
                distance = (sample_mean-inputs.unsqueeze(0).unsqueeze(2))*target_mask.unsqueeze(0)
                score = torch.sum(distance**2,dim=0)
                score = score.sum(dim=2)
                score = score.sum(dim=1)
                idx_label_score += list(zip(
                    idx.cpu().data.numpy().tolist(),
                    labels.cpu().data.numpy().tolist(),
                    score.cpu().data.numpy().tolist()
                    ))
        
        _, labels, scores = zip(*idx_label_score)
        # detection performance =========================================
        # AUROC
        auroc = roc_auc_score(labels, scores)
        self.print('Test set AUROC: [{:.2f}%]'.format(100. * auroc))     
        # AUPRC
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = auc(recall, precision)
        self.print('Test set AUPRC: [{:.2f}%]'.format(100. * auprc))
        return auroc, auprc




