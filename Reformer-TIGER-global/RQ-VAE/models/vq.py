import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import kmeans, sinkhorn_algorithm
import numpy as np
from .affine import AffineTransform
def get_state_dict(state_dict,pre):
    new_state_dict = {}
    for _ in state_dict.keys():
        if pre in _:
            key =_.replace(pre,"")
            new_state_dict[key] = state_dict[_]
    return new_state_dict
class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta = 0.8,alpha=0.2, init = "kmeans", kmeans_iters = 10,
                 sk_epsilon=0.01, sk_iters=100,affine_lr=0.0,affine_groups=1,warm_args=None,state_dict=None,warm=None,iso=0,device=None):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.init = init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.alpha = alpha
        self.device=device
        self.iso=iso
        self.warm_args=warm_args
        self.codebook =  nn.ModuleList([nn.Embedding(_, self.e_dim) for _ in n_e])
        self.warm_size=0
        if warm_args:
            
            for vidx in range(warm):
                if list(state_dict.keys())[vidx][0].isdigit():
                    self.codebook[vidx].load_state_dict(get_state_dict(state_dict,f"{int(list(state_dict.keys())[vidx][0])}."))
                    print(state_dict.keys())
                else:
                    self.codebook[vidx].load_state_dict(state_dict)
            self.warm_size = sum([_.weight.shape[0] for _ in self.codebook[:-1]])
            print("warm",self.warm_size)
            if not isinstance(warm_args.phase,int):
                self.p = int(warm_args.phase[-1])
            else:
                self.p=warm_args.phase
            
            if len(self.codebook)>warm:
                if  "kmeans" in init:
                    self.initted = False
                    self.codebook[-1].weight.data.zero_()
                elif "sample" in init:
                    self.initted = False
                    self.codebook[-1].weight.data.zero_()
                else:
                    self.initted = True
                    self.codebook[-1].weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
            else:
                self.initted=True
        else:
            self.p = len(n_e)-1
            if  "kmeans" in init:
                self.initted = False
                self.codebook[-1].weight.data.zero_()
            elif "sample" in init:
                self.initted = False
                self.codebook[-1].weight.data.zero_()
            else:
                self.initted = True
                self.codebook[-1].weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if affine_lr > 0:
            self.affine_transform = AffineTransform(
                                    self.e_dim,
                                    use_running_statistics=False,
                                    lr_scale=affine_lr,
                                    num_groups=affine_groups,
                                    )

    def kmeans_init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e[-1],
            self.kmeans_iters,
        )

        self.codebook[-1].weight.data.copy_(centers)
        self.initted = True


    def sample_init_emb(self, data,cnt_dict,indices):

        c_num = self.codebook[-1].weight.data.shape[0]
        all_c = []
        for _ in list(cnt_dict.keys())[:c_num]:
            c= []
            for i,x in enumerate(data):
                if int(indices[i]) == _:
                    c.append(i)
            all_c.append(c)
        all_emb = []
        for _ in all_c:
            idx =  np.random.choice(_, size=1)
            all_emb.append(data[idx])
        
        all_emb=torch.cat(all_emb,dim=0).to(self.device)
        self.codebook[-1].weight.data.copy_(all_emb)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances


    def quantize(self, codebook, z,use_sk,scale=None,bias=None):
        z_shape = z.shape[:-1]
        z_flat = z.view(z.size(0), -1, z.size(-1))
        latent = z.view(-1, self.e_dim)
        active = codebook.shape[0]
        
        if hasattr(self, 'affine_transform'):
            self.affine_transform.update_running_statistics(z_flat, codebook)
            codebook = self.affine_transform(codebook)
    
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(codebook**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, codebook.t())
        
        if scale is not None:
            d = d*scale[:active]
        
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d,self.sk_epsilon,self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = F.embedding(indices, codebook).view(z.shape)
        return x_q,indices


    def forward(self, x, use_sk=True,scale=None,bias=None,p=0):
        # Flatten input
        if scale is None and hasattr(self, '_freq'):
            scale = self._dist_scale
        if bias is None and hasattr(self, '_freq'):
            bias = self._bias 
        latent = x.view(-1, self.e_dim)

        if  "kmeans" in self.init and self.training and (p>self.p or self.p==0) and not self.initted:
            self.kmeans_init_emb(latent)
        elif "sample" in self.init and self.training and (p>self.p) and not self.initted:
            with torch.no_grad():
                idx =min(len(self.codebook),p+1)
                cb = torch.cat([_.weight for _ in self.codebook[:idx]],dim=0)
                x_q,indices = self.quantize(cb, x,use_sk,scale,bias)
                cnt_dict = {}
                for _ in indices:
                    fnum = int(_.detach().cpu())
                    if fnum not in cnt_dict:
                        cnt_dict[fnum] = 0
                    cnt_dict[fnum] += 1
                for _ in cnt_dict:
                    cnt_dict[_] /=float(1024)
                cnt_dict = dict(sorted(cnt_dict.items(),key=lambda item:item[1],reverse=True))
                self.sample_init_emb(latent,cnt_dict,indices)

        idx =min(len(self.codebook),p+1)
        if not self.iso:
            cb = torch.cat([_.weight for _ in self.codebook[:idx]],dim=0)
        else:
            cb = self.codebook[idx-1].weight
        x_q,indices = self.quantize(cb, x,use_sk,scale,bias)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = self.alpha*codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    def quantize_forward(self, x, use_sk=True,scale=None,bias=None,p=0):
        # Flatten input
        if scale is None and hasattr(self, '_freq'):
            scale = self._dist_scale
        if bias is None and hasattr(self, '_freq'):
            bias = self._bias 
        latent = x.view(-1, self.e_dim)

        if  "kmeans" in self.init and self.training and (p>self.p or self.p==0) and not self.initted:
            self.kmeans_init_emb(latent)
        elif "sample" in self.init and self.training and (p>self.p) and not self.initted:
            with torch.no_grad():
                idx =min(len(self.codebook),p+1)
                cb = torch.cat([_.weight for _ in self.codebook[:idx]],dim=0)
                x_q,indices = self.quantize(cb, x,use_sk,scale,bias)

        idx =min(len(self.codebook),p+1)
        if not self.iso:
            cb = torch.cat([_.weight for _ in self.codebook[:idx]],dim=0)
        else:
            cb = self.codebook[idx-1].weight
        
        x_q,indices = self.quantize(cb, x,use_sk,scale,bias)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = self.alpha*codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices

    def get_embedding(self,p,indices):
        idx =min(len(self.codebook),p+1)
        cb = torch.cat([_.weight for _ in self.codebook[:idx]],dim=0)
        return F.embedding(indices, cb)


