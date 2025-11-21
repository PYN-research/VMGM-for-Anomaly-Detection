import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from utils1 import make_nn, make_cnn, MultivariateNormalDiag
import math
from torch import nn, Tensor
from typing import Optional, Any
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer        
class DiagonalEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(64, 64), **kwargs):
        """ Encoder with factorized Normal posterior over temporal dimension
            Used by disjoint VAE and HI-VAE with Standard Normal prior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
        """
        super(DiagonalEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net, self.mu_layer, self.logvar_layer = make_nn(input_size, (z_size, z_size), hidden_sizes)

    def __call__(self, x):
        output = self.net(x)
        mu = self.mu_layer(output)
        logvar = self.logvar_layer(output)
        return MultivariateNormalDiag(mu, F.softplus(logvar))


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward) #16->35
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model) #35->16

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        print('src2_shape:',src2.shape) # 
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        print('src_shape:',src.shape) # 
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        print('src2_shape:',src2.shape) # 
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        print('src_shape:',src.shape) # 
        return src

def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='learnable', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()
        #Statistics: feat_dim=35,max_len=48,d_model=16,n_heads=4,num_layers=3,
        
        #dim_feedforward=35.
        self.max_len = max_len
        self.d_model = d_model #64
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        #self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        #用了三层的TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.encoder_layer1 = nn.Linear(d_model, d_model)
        self.output_layer_mu = nn.Linear(d_model, d_model)
        
        self.output_layer_sigma = nn.Linear(d_model, feat_dim)
        
        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        #Create padding masks
        max_len=48
        lengths = [x.shape[0] for x in X]  # original sequence length for each time series
        padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        #print('X_shape:',X.shape)(256,274)
        inp = self.project_inp(X) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        #inp = self.pos_enc(inp)  # add positional encoding
        #print('inp_shape:',inp.shape) #(256,64)
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
      #  output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model) src_key_padding_mask=~padding_masks
        output = self.encoder_layer1(inp)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        #print('embedding_shape:',output.shape) #(256,64)
     #   output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        mu = self.output_layer_mu(output)  # (batch_size, seq_length, feat_dim)
        #logvar = self.output_layer_sigma(output)
        #mu = mu.transpose(2,1)
        #logvar = logvar.transpose(2,1)
        return mu #MultivariateNormalDiag(mu, F.softplus(logvar))  


class JointEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(128, 128), window_size=24, transpose=False, **kwargs):
        """ Encoder with 1d-convolutional network and factorized Normal posterior
            Used by joint VAE and HI-VAE with Standard Normal prior or GP-VAE with factorized Normal posterior
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param window_size: kernel size for Conv1D layer
            :param transpose: True for GP prior | False for Standard Normal prior
        """
        super(JointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net, self.mu_layer, self.logvar_layer = make_cnn(
            input_size, (z_size, z_size), hidden_sizes, window_size)
        self.transpose = transpose

    def __call__(self, x):    
        #print('x:',x.shape)#(64,48,35)
        #print('x_range:',(x.min(),x.max()))
        output = self.net(x)
        mu = self.mu_layer(output)
        logvar = self.logvar_layer(output)
        #print('mu_shape:',mu.shape)#64, 48, 35
        #print('logvar_shape:',logvar.shape)#64, 48, 35
        #if self.transpose:
        num_dim = len(x.shape)
        mu = torch.transpose(mu, num_dim-1, num_dim-2)
        logvar = torch.transpose(logvar, num_dim-1, num_dim-2)
        #print('mu_shape:',mu.shape)#[64, 35, 48]
        #print('logvar_shape:',logvar.shape)#[64, 35, 48]
        #return mu, logvar, MultivariateNormalDiag(mu, F.softplus(logvar))
        return MultivariateNormalDiag(mu, F.softplus(logvar))

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
'''class BandedJointEncoder(nn.Module):
    def __init__(self, input_size, z_size, hidden_sizes=(64, 64), window_size=3, data_type=None, **kwargs):
        """ Encoder with 1d-convolutional network and multivariate Normal posterior
            Used by GP-VAE with proposed banded covariance matrix
            :param z_size: latent space dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
            :param window_size: kernel size for Conv1D layer
            :param data_type: needed for some data specific modifications, e.g:
                tf.nn.softplus is a more common and correct choice, however
                tf.nn.sigmoid provides more stable performance on Physionet dataset
        """
        super(BandedJointEncoder, self).__init__()
        self.z_size = int(z_size)
        self.net, self.mu_layer, self.logvar_layer = make_cnn(
            input_size, (z_size, z_size*2), hidden_sizes, window_size)
        self.data_type = data_type

    def __call__(self, x):
        mapped = self.net(x)
        batch_size = mapped.size[0]
        time_length = mapped.size[1]
        # Obtain mean and precision matrix components
        num_dim = len(mapped.shape)
        mu = self.mu_layer(mapped)
        logvar = self.logvar_layer(mapped)
        #perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        mapped_mean = torch.transpose(mu, num_dim - 1, num_dim - 2)
        mapped_covar = torch.transpose(logvar, num_dim - 1, num_dim - 2)
        if self.data_type == 'physionet':
            mapped_covar = F.sigmoid(mapped_covar)
        else:
            mapped_covar = F.softplus(mapped_covar)
        mapped_reshaped = mapped_covar.reshape(batch_size, self.z_size, 2*time_length)
        
        dense_shape = [batch_size, self.z_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.z_size*(2*time_length-1))
        idxs_2 = np.tile(np.repeat(np.arange(self.z_size), (2*time_length-1)), batch_size)
        idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length-1)]), batch_size*self.z_size)
        idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1,time_length)]), batch_size*self.z_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)
        #unfinished!
        # ~10x times faster on CPU then on GPU
        with tf.device('/cpu:0'):
            # Obtain covariance matrix from precision one
            mapped_values = mapped_reshaped[:, :, :-1].reshape(-1)
            prec_sparse = torch.sparse.FloatTensor(
                idxs_all, mapped_values, dense_shape)
            #prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
            #prec_sparse = tf.sparse.reorder(prec_sparse)
            #next
            prec_tril = torch.zeros(prec_sparse.dense_shape, dtype=tf.float32) + prec_sparse
            eye = torch.eye(prec_tril.shape[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            prec_tril = prec_tril + eye
            
            cov_tril = torch.triangular_solve(eye, prec_tril, lower=False)
            cov_tril = torch.where(torch.is_finite(cov_tril), cov_tril, torch.zeros_like(cov_tril))
            #prec_tril = tf.sparse_add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
            #eye = tf.eye(num_rows=prec_tril.shape.as_list()[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            #prec_tril = prec_tril + eye
            #cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
            #cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))
        num_dim = len(cov_tril.shape)
        #perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        #cov_tril_lower = tf.transpose(cov_tril, perm=perm)
        cov_tril_lower = torch.transpose(cov_tril, num_dim - 1, num_dim - 2)
        ##TODO: wtf?
        z_dist = torch.distributions.MultivariateNormal(loc=mapped_mean, scale_tril=cov_tril_lower)
        return z_dist'''