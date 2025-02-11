import logging
import torch.nn.functional as F
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
import seaborn as sns
sns.set()
import matplotlib
matplotlib.pyplot.set_loglevel (level = 'warning')
import matplotlib.pyplot as plt
from utils import ensure_dir,set_color,get_local_time
import os
import copy
class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        
        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        self.png_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        ensure_dir(self.ckpt_dir)
        self.push_freq=args.push_freq
        self.push_start = args.push_start
        self.push_lr = args.push_lr
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.freq_best_collision_ckpt = "freq_best_collision_model.pth"
        self.freq_best_collision_rate = np.inf
        self.p = args.phase
        self.freq=np.sum(args.a) or np.sum(args.b)
        
        self.freq_length = torch.tensor(sum([_.weight.shape[0] for _ in self.model.rq.vq_layers[0].codebook]))
        self._freq=torch.ones(self.freq_length).to(self.device)
        self._dist_scale=torch.ones(self.freq_length).to(self.device)
        self.start_dist_scale=torch.ones(self.freq_length).to(self.device)
        push_optimizer = lambda *args, **kwargs: torch.optim.SGD(*args, **kwargs, lr=self.push_lr, momentum=0.9)
        if self.push_freq:
            self.push_optimizer = push_optimizer(self.model.rq.vq_layers[0].parameters())			

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay


        if learner.lower() == "adam":
            optimizer = optim.Adam(filter(lambda p : p.requires_grad, params), lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(filter(lambda p : p.requires_grad, params), lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                filter(lambda p : p.requires_grad, params), lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                filter(lambda p : p.requires_grad, params), lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
               filter(lambda p : p.requires_grad, params), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(filter(lambda p : p.requires_grad, params), lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")
    def _get_consistency(self,warm_data_loader,idx=4):
        warm_indices = []
        with torch.no_grad():
            for p,iter_data in enumerate(warm_data_loader):
                for batch_idx, data in enumerate(iter_data):
                    data = data.to(self.device)
                    indices = self.model.get_indices(data,p=p,scale=self.start_dist_scale,use_sk=True)
                    warm_indices.append(indices.detach().tolist())
        cnt = 0
        for indices_i,indices_j in zip(self.warm_indices,warm_indices):
            for indice_i,indice_j in zip(indices_i,indices_j):
                stri = ""
                strj = ""
                for i in range(len(indice_i)):
                    stri+=str(indice_i[i])
                    strj+=str(indice_j[i])
                    if i==idx-1:
                        break
                if stri==strj:
                    cnt+=1
        return cnt / sum([len(_.dataset) for _ in warm_data_loader])



    def _warm_up(self,warm_data_loader):
        warm_indices = []
        encoder_emb = []
        with torch.no_grad():
            for p,iter_data in enumerate(warm_data_loader):
                for batch_idx, data in enumerate(iter_data):
                    first_assign = set()
                    all_first = []
                    data = data.to(self.device)
                    out, rq_loss, indices = self.model(data,p=p,scale=self.start_dist_scale)#
                    x_e = self.model.encoder(data)
                    for _ in indices:
                        first_assign.add(int(_[0].detach().cpu()))
                    encoder_emb.append(x_e.detach())
                    warm_indices.append(indices.detach().tolist())
                    for _ in indices:
                        first_assign.add(int(_[0].detach().cpu()))
                    
                if hasattr(self.model.rq.vq_layers[0],"_freq"):
                    print(self.model.rq.vq_layers[0]._freq)
            self._freq = [copy.deepcopy(_._freq.detach().to(self.device)) if hasattr(_, '_freq') else None for _ in self.model.rq.vq_layers]
            self._dist_scale = [copy.deepcopy(_._dist_scale.detach().to(self.device)) if hasattr(_, '_freq') else None for _ in self.model.rq.vq_layers]
        self.warm_indices=warm_indices
        self.encoder_emb = encoder_emb



    def _train_epoch(self, train_data, epoch_idx,warm_data_loader):

        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_warm_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        
        cnt_dict = {}
        test_cnt_dict = {}
        for batch_idx, data in enumerate(iter_data):
            first_assign = set()
            all_first = []
            
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            out, rq_loss, indices = self.model(data,p=self.p)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            if (epoch_idx+1) % 50==0 or epoch_idx==0:
                with torch.no_grad():
                    for _ in indices:
                        fnum = int(_[0].detach().cpu())
                        first_assign.add(fnum)
                        all_first.append(fnum)
                        if fnum not in cnt_dict:
                            cnt_dict[fnum] = 0
                        cnt_dict[fnum] += 1
                    test_indices = self.model.get_indices(data,scale=self.start_dist_scale,p=self.p,use_sk=True)
                    test_all_first = []
                    for _ in test_indices:
                        test_all_first.append(int(_[0].detach().cpu()))
                    cnt = 0
                    for fidx,_ in enumerate(all_first):
                        if _ == test_all_first[fidx]:
                            cnt+=1
                    print("n-penalty consis",cnt/float(len(test_all_first)))
                    all_num = np.sum(list(cnt_dict.values()))
                    for _ in cnt_dict:
                        cnt_dict[_] /=all_num
                    cnt_dict = dict(sorted(cnt_dict.items(),key=lambda item:item[1],reverse=True))
                    print(list(cnt_dict.keys())[:8])
                    print(list(cnt_dict.values())[:8])
            if (epoch_idx + 1) % 1000 == 0 and warm_data_loader:
                print("#####")
                print("warm consis",self._get_consistency(warm_data_loader,idx=1))
                print("warm consis",self._get_consistency(warm_data_loader,idx=2))
                print("warm consis",self._get_consistency(warm_data_loader,idx=3))
                print("warm consis",self._get_consistency(warm_data_loader,idx=4))
                print("#####")
            warm_loss = 0
            warm_loss=torch.tensor(warm_loss).to(self.device)
            loss+=warm_loss
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_warm_loss +=warm_loss.item()
            for vidx,_ in enumerate(self.model.rq.vq_layers):
                if  hasattr(_, '_freq') :
                    _._freq=copy.deepcopy(self._freq[vidx]).to(self.device)
        return total_loss, total_recon_loss,total_warm_loss

    def _push_epoch(self):
        self.model.train()
        code = np.random.randint(self.model.rq.vq_layers[0].embedding.weight.shape[0])
        all_code = list(range(self.model.rq.vq_layers[0].embedding.weight.shape[0]))
        all_code.remove(code)
        code_embedding = self.model.rq.vq_layers[0].embedding.weight[code].detach().unsqueeze(0).unsqueeze(0)
        out = torch.cdist(self.model.rq.vq_layers[0].embedding.weight[all_code].unsqueeze(0),code_embedding,p=2).squeeze()
        out = torch.pow(out,-0.5)
        out.mean().backward()
        self.push_optimizer.step()
        self.push_optimizer.zero_grad()
        out = torch.cdist(self.model.rq.vq_layers[0].embedding.weight[all_code].unsqueeze(0),code_embedding,p=2).squeeze()

    @torch.no_grad()
    def _valid_epoch(self, valid_data,scale=None):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        indices_set = set()
        num_sample = 0
        test_cnt_dict = {}
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data,p=self.p,scale=scale)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)
            for _ in indices:
                fnum = int(_[0])
                if fnum not in test_cnt_dict:
                    test_cnt_dict[fnum] = 0
                test_cnt_dict[fnum] += 1
        collision_rate = (num_sample - len(indices_set))/num_sample
        all_num = np.sum(list(test_cnt_dict.values()))
        for _ in test_cnt_dict:
            test_cnt_dict[_] /=all_num
        test_cnt_dict = dict(sorted(test_cnt_dict.items(),key=lambda item:item[1],reverse=True))

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.freq>0:
            state["freq"] = self._freq
            state["scale"] = self._dist_scale

        torch.save(state, ckpt_path, pickle_protocol=4)
        
        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss,warm_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("warm loss", "blue") + ": %.4f" % warm_loss
        return train_loss_output + "]"


    def fit(self, warm_data_loader,data):

        if warm_data_loader:
            self._warm_up(warm_data_loader)
            print(self._get_consistency(warm_data_loader,idx=4))
        else:
            self._freq = [torch.ones(sum([cb.weight.shape[0] for cb in _.codebook])).to(self.device) if hasattr(_, '_freq') else None for _ in self.model.rq.vq_layers]
            self._dist_scale = [torch.ones(sum([cb.weight.shape[0] for cb in _.codebook])).to(self.device) if hasattr(_, '_freq') else None for _ in self.model.rq.vq_layers]
        print(self._freq)
        print(self._dist_scale)
        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            training_start_time = time()
            train_loss, train_recon_loss,total_warm_loss = self._train_epoch(data, epoch_idx,warm_data_loader)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss,total_warm_loss
            )
            if self.push_freq and epoch_idx>=self.push_start and epoch_idx%self.push_freq==0:
                self._push_epoch()
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss

            if (epoch_idx + 1) % self.eval_step == 0 :
                valid_start_time = time()

                collision_rate = self._valid_epoch(data,scale=self.start_dist_scale)
                if collision_rate < self.freq_best_collision_rate:
                    self.freq_best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                        ckpt_file=self.freq_best_collision_ckpt)

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)


        return self.best_loss, self.best_collision_rate