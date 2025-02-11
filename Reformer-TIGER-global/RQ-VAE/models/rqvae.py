import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import torch
import numpy as np
import os
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer

def get_state_dict(state_dict,pre):
    new_state_dict = {}
    for _ in state_dict.keys():
        if pre in _:
            key =_.replace(pre,"")
            new_state_dict[key] = state_dict[_]
    return new_state_dict
class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 init="kmeans",
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 affine_lr=0.0,
                 affine_groups=1.0,
                 replace_freq=0.0,
                 a = 0,
                 new_a = 0,
                 b = 0,
                 b_scale = 1,
                 freq_policy=None,
                 warm_codebook=None,
                 device=None,
                 iso=0,
                 seed=2023
        ):
        super(RQVAE, self).__init__()
        self.seed=seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.affine_lr=affine_lr
        self.affine_groups=affine_groups
        self.replace_freq = replace_freq
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.init = init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.a=a
        self.new_a=new_a
        self.b=b
        self.b_scale=b_scale
        self.freq_policy =freq_policy
        self.device=device
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]

        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        if warm_codebook:
            ckpt = torch.load(warm_codebook, map_location=torch.device('cpu'))
            warm_args = ckpt["args"]
            state_dict = ckpt["state_dict"]
            encoder_state_dict = get_state_dict(state_dict,"encoder.")
            self.encoder.load_state_dict(encoder_state_dict)
            rq_state_dict = get_state_dict(state_dict,"rq.vq_layers.")
            decoder_state_dict = get_state_dict(state_dict,"decoder.")
            self.decoder.load_state_dict(decoder_state_dict)
        else:
            rq_state_dict=None
            warm_args=None

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,warm_args=warm_args,state_dict=rq_state_dict,
                                          init = self.init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          affine_lr=self.affine_lr,
                                          affine_groups=self.affine_groups,
                                          replace_freq = self.replace_freq,
                                          a=self.a,
                                          new_a=self.new_a,
                                          b=self.b,
                                          b_scale=self.b_scale,
                                          freq_policy=self.freq_policy,
                                          device=self.device,
                                          iso=iso
                                          )
    def forward(self, x, use_sk=True,scale=None,p=0):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk,scale=scale,use_freq=True,p=p,bias=None)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False,scale=None,p=0):
        x_e = self.encoder(xs)
        _, _, indices = self.rq.quantize(x_e, use_sk=use_sk,scale=scale,use_freq=False,p=p,bias=None)
        return indices

    def get_indices_emb(self, x, use_sk=False,scale=None,p=0):
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x,use_sk=use_sk,scale=scale,use_freq=True,p=p)
        return indices,x

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon

    def get_in_out_emb(self,x,use_sk=True):
        x = self.encoder(x)
        return self.rq.get_in_out_emb(x,use_sk)