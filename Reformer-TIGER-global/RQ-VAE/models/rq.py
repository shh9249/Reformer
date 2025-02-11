import torch
import torch.nn as nn
import numpy as np
from .vq import VectorQuantizer
from .replace import lru_replacement
from .freq import freq_assign
import torch.nn.functional as F
import copy
def get_state_dict(state_dict,pre):
    new_state_dict = {}
    for _ in state_dict.keys():
        if pre in _:
            key =_.replace(pre,"")
            new_state_dict[key] = state_dict[_]
    return new_state_dict

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_e_list, e_dim, sk_epsilons,state_dict,
                 a,
                 new_a,
                 b,
                 b_scale,
                 init = "kmeans", kmeans_iters = 100, sk_iters=100,
                 replace_freq: int = 0,
                 affine_lr:	float = 0.0,
                 affine_groups: int=1,
                 freq_policy:str=None,
                 device=None,
                 warm_args=None,iso=0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.affine_lr = affine_lr
        self.num_quantizers = len(n_e_list)
        self.init = init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.freq_policy=freq_policy
        self.warm_args=warm_args
        self.device=device
        self.freq = np.sum(a) or np.sum(b)
        emb_list = []
        warm = []
        state_dict_list = []
        if warm_args:
            if isinstance(warm_args.phase,int):
                all_l = int(list(state_dict.keys())[-1][0])
                for i in range(all_l+1):
                    new_state_dict = copy.deepcopy(get_state_dict(state_dict,f"{i}.codebook."))
                    state_dict_list.append(new_state_dict)
                    emb_list.append([_.shape[0] for _ in new_state_dict.values()])
                    warm.append(len(new_state_dict.values()))
                    if i<len(n_e_list):
                        emb_list[i].append(n_e_list[i])

                print(state_dict.keys())
            elif "0" in warm_args.phase:
                print("old codebook")
                for i,_ in enumerate(warm_args.num_emb_list):
                    emb_list.append([_])
                    state_dict_list.append(get_state_dict(state_dict,f"{i}.embedding."))
                    warm.append(len([_]))
                    if i<len(n_e_list):
                        emb_list[i].append(n_e_list[i])
        else:
            if isinstance(n_e_list[0],int):
                emb_list=[[_] for _ in n_e_list]
            else:
                emb_list=[_ for _ in n_e_list]
            state_dict_list = list(range(len(emb_list)))
            warm = np.zeros(len(emb_list))
        self.vq_layers = nn.ModuleList([VectorQuantizer(emb_list[eidx], e_dim,
                                                        init = self.init if eidx==0 else "kmeans",
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters,
                                                        affine_lr = affine_lr if eidx==0 else 0.0,
                                                        affine_groups = affine_groups if eidx==0 else 1.0,
                                                        warm_args=warm_args,
                                                        state_dict=state_dict_list[eidx],
                                                        warm=warm[eidx],
                                                        iso=0 if eidx==0 else iso,
                                                        device=device)
                                        for eidx, sk_epsilon in enumerate(sk_epsilons) ])
        if replace_freq > 0:
            lru_replacement(self, rho=0.01, timeout=replace_freq)
        if self.freq:
            for idx,_ in enumerate(self.vq_layers): 
                if a[idx] or new_a[idx]:
                    freq_assign(_, policy=freq_policy, a=a[idx],new_a=new_a[idx],b=b[idx],b_scale=b_scale[idx])
       
    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)
    
    def get_indices_embedding(self,x,p,indices):
        x_q = 0
        res = x
        loss = 0
        for i,quantizer in enumerate(self.vq_layers):
            device = indices.device
            indice = torch.tensor([_[i] for _ in indices]).to(device)
            emb = quantizer.get_embedding(p,indice)
            loss+= F.mse_loss(emb.detach(), res)
            res = res-emb
            x_q=x_q+emb
        return x_q,loss

    def clear_freq(self):
        if self.freq>0:
            length = torch.tensor(sum([_.weight.shape[0] for _ in self.vq_layers[0].codebook]))
            self._freq=torch.ones(length).type(torch.int64).to(self.device)
            self._dist_scale=torch.ones(length).to(self.device)
            self._bias=torch.zeros(length).to(self.device)

    def assign_freq(self,freq,scale,bias = None):
        if self.freq>0:
            self._freq=freq.to(self.device)
            self._dist_scale=scale.to(self.device)
            if bias:
                self._bias=bias

    def forward(self, x, use_freq,use_sk=True,scale=None,bias=None,p=0):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for i,quantizer in enumerate(self.vq_layers):
            if i == 0 and self.freq>0 :
                x_res, loss, indices = quantizer(residual, use_sk=use_sk,scale=scale,bias=bias,p=p)
            else:
                x_res, loss, indices = quantizer(residual, use_sk=use_sk,p=p)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices

    def quantize(self, x, use_freq,use_sk=True,scale=None,bias=None,p=0):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for i,quantizer in enumerate(self.vq_layers):
            if i == 0 and self.freq>0 and scale is not None:
                x_res, loss, indices = quantizer.quantize_forward(residual, use_sk=use_sk,scale=scale,bias=bias,p=p)
            else:
                x_res, loss, indices = quantizer.quantize_forward(residual, use_sk=use_sk,p=p)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices

    def get_level_embedding(self, x,use_sk=True,level = 0):
        all_losses = []
        all_indices = []
        emb = torch.zeros_like(x)
        x_q = 0
        residual = x
        for i,quantizer in enumerate(self.vq_layers):
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            if i <= level:
                emb+=x_res
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, emb, all_indices

    def get_in_out_emb(self,x,use_sk=False):
        x_q = 0
        residual = x
        with torch.no_grad():
            x_res, loss, indices = self.vq_layers[0](residual, use_sk=use_sk)
        return x,indices,self.vq_layers[0].get_codebook()