import numpy as np
import torch
import torch.utils.data as data

class EmbDataset(data.Dataset):

    def __init__(self,data_path,phase = None,dataset = None):

        self.data_path = data_path
        self.embeddings = np.load(data_path+"/%s.emb-llama-td.npy"%(dataset))
        warm = list(np.load(data_path+"/phase%s"%(phase) + "/warm_item.npy", allow_pickle=True).tolist())
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in warm,all_idx))
        self.embeddings = self.embeddings[save,:]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)

class new_EmbDataset(data.Dataset):

    def __init__(self,data_path,phase,dataset):

        self.data_path = data_path
        self.embeddings = np.load(data_path+"/%s.emb-llama-td.npy"%(dataset))
        cold = list(np.load(data_path+"/phase%s"%(phase) + "/cold_item.npy", allow_pickle=True).tolist())
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in cold,all_idx))
        print("cold",len(cold),len(save))
        self.embeddings = self.embeddings[save,:]
        print(self.embeddings.size)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)








class warm_TokenDataset(data.Dataset):

    def __init__(self,data_path,phase,dataset):

        self.data_path = data_path
        self.embeddings = np.load(data_path+"/%s.emb-llama-td.npy"%(dataset))
        warm = list(np.load(data_path+"/phase%s"%(phase) + "/warm_item.npy", allow_pickle=True).tolist())
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in warm,all_idx))
        self.embeddings = self.embeddings[save,:]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)

class cold_TokenDataset(data.Dataset):

    def __init__(self,data_path,phase,dataset):

        self.data_path = data_path
        self.embeddings = np.load(data_path+"/%s.emb-llama-td.npy"%(dataset))
        cold = list(np.load(data_path+"/phase%s"%(phase) + "/cold_item.npy", allow_pickle=True).tolist())
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in cold,all_idx))
        self.embeddings = self.embeddings[save,:]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)

class sp_TokenDataset(data.Dataset):

    def __init__(self,data_path,phase,dataset):

        self.data_path = data_path
        self.embeddings = np.load(data_path+"/%s.emb-llama-td.npy"%(dataset))
        cold = list(np.load(data_path+"/phase%s"%(phase) + "/cold_item.npy", allow_pickle=True).tolist())
        
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in cold,all_idx))
        self.embeddings = self.embeddings[save,:]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)

class all_TokenDataset(data.Dataset):

    def __init__(self,data_path,phase,dataset):

        self.data_path = data_path
        self.embeddings = np.load(data_path+"/%s.emb-llama-td.npy"%(dataset))
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)

