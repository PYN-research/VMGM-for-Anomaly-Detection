import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def get_torch_trans(heads=8, layers=1, channels=64):
    """
    Creates a Transformer encoder module to process MTS timestamps/features as sequences.
    
    Parameters:
    - heads (int): Number of attention heads.
    - layers (int): Number of encoder layers.
    - channels (int): Dimensionality of the model (d_model in Transformer terminology).
    
    Returns:
    - nn.TransformerEncoder: A Transformer encoder object.
    """
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """
    Initializes a 1D convolutional layer with Kaiming normal initialization.
    
    Parameters:
    - in_channels (int): Number of channels in the input signal.
    - out_channels (int): Number of channels produced by the convolution.
    - kernel_size (int): Size of the convolving kernel.
    
    Returns:
    - nn.Conv1d: A 1D convolutional layer with weights initialized.
    """
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class SelfAttention(nn.Module):
    def __init__(self,dim_q,dim_k,dim_v):
        super(SelfAttention,self).__init__()
        self.dim_q=dim_q
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.linear_q=nn.Linear(dim_q,dim_k,bias=False)
        self.linear_k=nn.Linear(dim_q,dim_k,bias=False)
        self.linear_v=nn.Linear(dim_q,dim_v,bias=False)
        self._norm_fact=1/math.sqrt(dim_k)
    
    def forward(self,x):
        batch,n,dim_q=x.shape
        assert dim_q == self.dim_q
        q=self.linear_q(x)
        k=self.linear_k(x)
        v=self.linear_v(x)
        q=q.transpose(1,0)
        k=k.transpose(1,0)
        v=v.transpose(1,0)
        dist=torch.bmm(q,k.transpose(1,2))*self._norm_fact 
        dist=F.softmax(dist,dim=-1)
        att=torch.bmm(dist, v)
        return att.transpose(1,0)
        

class embedding_MTS(nn.Module):
    """
    An embedding module for multivariate time series data, incorporating both temporal and spatial
    encoding via Transformer encoders and convolutional layers, it corresponds to the Embedding block in TSDE architecture.
    
    Parameters:
    - config (dict): A configuration dictionary containing model parameters such as number of channels,
                     embedding dimensions, and number of heads for the Transformer encoders in the embedding block.
    """


    def __init__(self, config):
        super().__init__()
        self.channels = config["channels"]
        self.featureemb = config["featureemb"]
        self.num_steps = config["num_steps"]
        self.diffusion_embedding_dim=config["diffusion_embedding_dim"]
        self.time_embedding_channels = self.channels
        self.emb_size = self.channels + self.channels 
        self.feature_layer = get_torch_trans(heads=config["nheads"], layers=1, channels=self.emb_size) 
        self.input_projection = Conv1d_with_init(1, self.channels, 1) 
        self.xf_projection = Conv1d_with_init(self.emb_size, self.channels, 1)
        self.diffusion_projection = nn.Linear(self.diffusion_embedding_dim, self.time_embedding_channels)
        
    def forward_channel(self, x, base_shape):
        """
        Processes the input data through the temporal Transformer encoder (timestamps are considered as tokens).
        
        Parameters:
        - x (Tensor): The observed part of the MTS combined with timestamps embedding and features embedding as tensor.
        - base_shape (tuple): The base shape of the input tensor before reshaping.
        
        Returns:
        - Tensor: The temporally encoded tensor.
        """
        B, C, K = base_shape 
        x = self.time_layer(x.permute(1, 0, 2)).permute(1, 0, 2)
        return x

    def forward_feature(self, x, base_shape):
        """
        Processes the input data through the spatial Transformer encoder (features are considered as tokens).
        
        Parameters:
        - base_shape (tuple): The base shape of the input tensor before reshaping.
        
        Returns:
        - Tensor: The spatially encoded tensor.
        """
        B, C, M, K = base_shape 
        x = x.reshape(B,C,M,K)
        x = x.permute(0,2,1,3).reshape(B*M,C,K)
        x = self.feature_layer(x.permute(2, 0, 1)).permute(1, 2, 0)
        x = x.reshape(B, M, C, K).permute(0, 2, 1, 3)
        return x
    
    def forward_batch(self, x, base_shape):
        """
        Processes the input data through the Batch Transformer encoder (features are considered as tokens).
        
        Parameters:
        - x (Tensor): The observed part of the MTS combined with timestamps embedding and features embedding as tensor.
        - base_shape (tuple): The base shape of the input tensor before reshaping.
        
        Returns:
        - Tensor: The batch encoded tensor.
        """
        B, C, K, L = base_shape
        if K == 1:
            return x
        x = x.permute(2, 3, 1, 0).reshape(K * L, C, B)
        x = self.batch_layer(x.permute(2, 0, 1))
        x = x.reshape(B, K, L, C).permute(0, 3, 1, 2)
        return x
    def forward(self, x_tar, x, x_rec, feature_embed, diffusion_emb, training):
        """
        The forward pass of the embedding module, processing multivariate time series data with
        temporal and spatial embeddings.
        
        Parameters:
        - x (Tensor): The observed part of the MTS.
        - time_embed (Tensor): The time embeddings tensor.
        - feature_embed (Tensor): The feature embeddings tensor.
        
        Returns:
        - Tuple[Tensor, Tensor, Tensor]: A tuple containing the combined temporal and spatial embeddings generated by the two transformer encoders, 
                                         the processed temporal embedding, and the processed spatial embedding.
        """
        B, M, K = x.shape 
        x = x + x_rec.unsqueeze(1) 
        x = x.unsqueeze(1) 
        x = self.input_projection(x.reshape(B,1,M*K).cuda()) 
        x = F.relu(x)  
        diffusion_emb = self.diffusion_projection(diffusion_emb.cuda())  
        x_t = x 
        x = torch.cat([x_t, feature_embed.unsqueeze(0).expand(B, -1, -1)], dim=1)
        base_shape = B, x.shape[1], M, K 
        
        xf = self.forward_feature(x, base_shape) 
        xf_reshaped = xf.reshape(B, base_shape[1], M*K)
        xf_proj = F.silu(self.xf_projection(xf_reshaped)) 
        xf = xf_proj.reshape(B, self.channels, M, K)
        
        if training == True:
            x = xf  
        if training == False:
            x = xf
        return x