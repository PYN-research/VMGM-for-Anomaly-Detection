
from .mlp import MVNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base.mtsEmbedding import embedding_MTS
def build_networks(net_name, in_channels=3, mid_dim=128):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mlp')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mlp':
        net = MVNet(input_dim=in_channels, mid_dim=mid_dim)
    
    return net

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    # Weight initialization
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    # t_embedding(t). The embedding dimension is 128 in total for every time step t.
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_Block(nn.Module):
    def __init__(self, config,config_emb,config_model,device):
        super().__init__()
        self.config = config
        self.device = device
        self.channels = config["channels"]
        self.target_dim = config["channels"]
        self.M = config['num_cond_masks']
        self.emb_cat_feature_dim = config_emb['featureemb']*self.M
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"], 
            embedding_dim=config["diffusion_embedding_dim"], 
        )
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_cat_feature_dim
        )
        self.token_emb_dim = config["token_emb_dim"] if config["mixed"] else 1  

        self.input_projection = Conv1d_with_init(1,self.channels,1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, self.token_emb_dim, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.embdmodel = embedding_MTS(config_emb) 
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    mts_emb_dim=config["mts_emb_dim"], 
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"], 
                )
                for _ in range(config["layers"]) 
            ]
        )
    def get_mts_emb(self, diffusion_emb, target_input, x_rec_miss, cond_mask, x_co, training):
        B, M, K = cond_mask.shape
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
            ) 
        cond_embed = self.embdmodel(target_input, x_co, x_rec_miss, feature_embed, diffusion_emb, training)
          
        mts_emb = cond_embed
        return mts_emb.reshape(B,mts_emb.shape[1],M*K) 
    def forward(self, x, x_rec_miss, cond_mask, x_co, diffusion_step, training):
        B, M, K = x.shape        
        x = x.unsqueeze(1)
        x = self.input_projection(x.reshape(B,1,M*K)) 
        x = F.relu(x)               
        x = x.reshape(B,x.shape[1],M,K)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        mts_emb = self.get_mts_emb(diffusion_emb, x, x_rec_miss, cond_mask, x_co, training=True)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, mts_emb, diffusion_emb)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
       
        x = self.output_projection1(x)  
        x = F.relu(x)
        x = self.output_projection2(x) 
        x = x.reshape(B, M, K)
        return x, mts_emb
class ResidualBlock(nn.Module):
    """
    A residual block that processes input data alongside diffusion embeddings and observed MTS part embedding, 
    utilizing a gated mechanism.
    
    Parameters:
    - mts_emb_dim (int): Dimensionality of the embedding of the MTS
    - channels (int): Number of channels for the convolutional layers within the block.
    - diffusion_embedding_dim (int): Dimensionality of the diffusion embeddings.
    
    """
    def __init__(self, mts_emb_dim, channels, diffusion_embedding_dim):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(mts_emb_dim, 2*channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2*channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

    def forward(self, x, mts_emb, diffusion_emb):
        """
        Forward pass of the ResidualBlock.
        
        Parameters:
        - x (Tensor): The projected corrupted input MTS.
        - mts_emb (Tensor): The embedding of the observed part of the MTS.
        - diffusion_emb (Tensor): The projected diffusion embedding tensor.
        
        Returns:
        - Tuple[Tensor, Tensor]: A tuple containing the updated data tensor and a skip connection tensor.
        """
        
        B, C, M, K = x.shape  
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(2).unsqueeze(3)  
        y = x #+ diffusion_emb 
        y = self.mid_projection(y.reshape(B,C,M*K))      
        _, mts_emb_dim, _= mts_emb.shape
        mts_emb = self.cond_projection(mts_emb)  
        y = y + mts_emb  
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        out = (x + residual.reshape(B,C,M,K)) / math.sqrt(2.0)
        return out, skip
