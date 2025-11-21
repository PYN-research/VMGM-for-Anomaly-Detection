import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.main import diff_Block
from memae_fc import AutoEncoderFCMem, MemAELoss
class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, input_dim):
        # keep the __init__ the same
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"] 
        self.emb_feature_dim = config["model"]["featureemb"] 

        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"] 

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim 
        if self.is_unconditional == False:
            self.emb_total_dim += 1  
        self.soft_mask_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim,input_dim),
            nn.Sigmoid()
        )
            
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )
        config_model = config["model"]
        config_diff = config["diffusion"]
        config_emb = config["embedding"]
        config_diff["mts_emb_dim"] = config_emb["mts_emb_dim"]
        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_Block(config_diff,config_emb,config_model,self.device)
        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim) 
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  
        side_info = side_info.permute(0, 3, 2, 1)  

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1) 
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, x_rec, observed_mask, num_cond_masks,is_train
    ):
        loss_sum = 0
        # In validation, perform T steps forward and backward.
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, x_rec, observed_mask, num_cond_masks,is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, x_rec, observed_mask, num_cond_masks,is_train, set_t=-1
    ):
        B, K = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  
        masked_data, cond_mask = self.set_input_to_diffmodel(observed_data, observed_mask, num_cond_masks)
        noise = torch.randn_like(masked_data)
        noisy_data = (current_alpha.unsqueeze(2)**0.5) * masked_data + (
            1.0 - current_alpha.unsqueeze(2)
        ) ** 0.5 * noise
        x_rec_miss = (1-observed_mask)*x_rec
        x_cond = cond_mask * observed_data.unsqueeze(1) 
        predicted_noise, mts_emb = self.diffmodel(noisy_data, x_rec_miss, cond_mask, x_cond, t, training=True) 
        target_mask = observed_mask.unsqueeze(1) - cond_mask      
        residual = (noise - predicted_noise) * target_mask
        num_eval = target_mask.sum(dim=0)
        num_eval = num_eval.sum(dim=1)
        #Diffusion_loss
        loss_main = (residual**2).sum(dim=0)
        loss_main = loss_main.sum(dim=1)
        loss_main = loss_main / num_eval
        loss_main = loss_main.sum()
        #Negative Entropy Loss
        #entropy_loss = torch.sum(cond_mask*torch.log(cond_mask+eps.cuda())-(1-cond_mask)*torch.log((1-cond_mask)+eps.cuda()),dim=1)#(B,C)
        #entropy_loss = torch.sum(cond_mask*torch.log(cond_mask+eps.cuda()),dim=1)
        entropy_loss = torch.mean(cond_mask**2,dim=1)
        entropy_loss = entropy_loss.sum()
        alpha = 1.0
        loss = loss_main + alpha*entropy_loss
        return loss, mts_emb
    
    def set_input_to_diffmodel(self, observed_data,observed_mask, num_cond_masks):
        cond_masks = []
        for i in range(num_cond_masks):
            h = self.soft_mask_layer(observed_data)
            cond_mask = gumbel_sigmoid_sampling(h, tau=0.1)
            cond_masks.append(cond_mask)
        cond_masks_all = torch.stack(cond_masks)
        cond_masks_all = cond_masks_all.transpose(1,0)
        masked_data = (observed_mask.unsqueeze(1)-cond_masks_all)*observed_data.unsqueeze(1)
        
        return masked_data, cond_masks_all
    
    def get_randmask(observed_mask):
        observed_mask = torch.from_numpy(observed_mask)
        observed_mask = observed_mask.float()
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)   
        for i in range(len(observed_mask)):
            sample_ratio = 0.5 
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask.numpy()
    

    def impute(self, observed_data, target_mask, x_rec_miss, cond_mask, n_samples):
        B, K= observed_data.shape 
        observed_data = observed_data.unsqueeze(1)
        M = target_mask.shape[1]
        imputed_samples = torch.zeros(B, n_samples, M, K).to(self.device)
        Current_samples_all = torch.zeros(n_samples,self.num_steps, B, M, K).to(self.device)
        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            Current_samples = []
            # perform T steps backward
            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = cond_mask * observed_data
                noisy_target =  target_mask * current_sample  
                predicted, mts_emb_test = self.diffmodel(noisy_target, x_rec_miss, cond_mask, cond_obs, torch.tensor([t]).to(self.device), training=True) 
                
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                current_sample = coeff1 * (current_sample - coeff2 * predicted) 

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
                Current_samples.append(current_sample.detach())
            Current_samples = torch.stack(Current_samples) 
            Current_samples_all[i] = Current_samples.detach()
            imputed_samples[:, i] = current_sample.detach() 
        return imputed_samples, Current_samples_all, mts_emb_test 

    def forward(self,observed_data, x_rec, observed_mask, num_cond_masks, is_train=1):
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, x_rec, observed_mask, num_cond_masks, is_train)

    def evaluate(self, observed_data,x_rec,observed_mask,num_cond_masks,n_samples):
        with torch.no_grad():
            _, cond_mask = self.set_input_to_diffmodel(observed_data, observed_mask, num_cond_masks)
            target_mask = observed_mask.unsqueeze(1) - cond_mask 
            cond_obs = cond_mask * observed_data.unsqueeze(1)
            x_rec_miss = (1-observed_mask)*x_rec
            samples,samples_list,mts_emb_test = self.impute(observed_data, target_mask, x_rec_miss, cond_mask, n_samples)
        return samples, target_mask, samples_list, cond_obs, mts_emb_test


class TabCSDI(CSDI_base):
    def __init__(self, config, device, input_dim=None, target_dim=1):
        super(TabCSDI, self).__init__(target_dim, config, device, input_dim)

def inverse_gumbel_cdf(y,mu,beta):
    return mu - beta * np.log(-np.log(y))

def gumbel_sigmoid_sampling(h,tau=0.5):
    """
    h: (N x K) tensor.
    """
    shape_h = h.shape
    u = torch.rand(shape_h) + 1e-10
    epsilon = torch.log(u)-torch.log(1-u)
    g = torch.log(h + 1e-10) #/(1-h)
    y = (g+epsilon.cuda())/tau
    y = torch.sigmoid(y)
    return y

class Autoencoder(nn.Module):
    """
    Specialized TSDE model for PhysioNet dataset.
    
    Adapts the TSDE_base model for tasks involving PhysioNet data, including imputation and interpolation.
    """
    def __init__(self, config, device, input_dim, sample_feat=False):
        super(Autoencoder,self).__init__()
        in_col_dim = input_dim # number of feature columns in the data
        mem_dim = 80
        shrink_thres = 0.00025
        self.sample_feat=False
        self.device=device
        self.model = AutoEncoderFCMem(in_col_dim=in_col_dim, mem_dim=mem_dim, shrink_thres=shrink_thres)
    
    def forward(self, observed_data, observed_mask, is_train):       
        regularization_parameter = 0.0001
        memae_loss = MemAELoss(regularization_parameter=regularization_parameter)
        if is_train==1:
            x_rec = self.model((observed_data*observed_mask).float())
            mse_loss = memae_loss(prediction=x_rec, ground_truth=observed_data, observed_mask=observed_mask, training=True)
            ae_loss = mse_loss.requires_grad_(True)
        
        if is_train==0:
            observed_mask = observed_mask.to(self.device)
            x_rec = self.model((observed_data*observed_mask).float())
            for param in self.model.parameters():
                param.requires_grad = False
            mse_loss = memae_loss(prediction=x_rec, ground_truth=observed_data, observed_mask=observed_mask, testing=True)     
            ae_loss = mse_loss.detach()
        return ae_loss, x_rec['output'], x_rec['feat']