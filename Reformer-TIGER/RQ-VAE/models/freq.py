import torch

import copy

class FreqAssign():
	def __init__(self, policy, a=1,new_a=1,b=0,b_scale=0):
		self.a=a
		self.policy = policy
		self.new_a = new_a
		self.b=b
		self.b_scale=b_scale
		return

	@staticmethod
	def apply(module, policy, a=0,new_a=0,b=0,b_scale=0):
		fn = FreqAssign(policy, a,new_a,b,b_scale)
		device = next(module.parameters()).device
		module.register_forward_hook(fn)
		length = torch.tensor(sum([_.weight.shape[0] for _ in module.codebook]))
		module.register_buffer('_freq',torch.ones(length))
		module._freq = module._freq.to(device)
		module.register_buffer('_dist_scale',torch.ones(length))
		module._dist_scale = module._dist_scale.to(device)
		module.register_buffer('_bias',torch.zeros(length))
		module.register_buffer('_use_freq',torch.zeros(1).type(torch.int64))
		module._bias = module._bias.to(device)
		module._use_freq = torch.tensor(a or new_a)
		return fn

	def __call__(self, module, inputs, outputs):
		if module._use_freq:
			device = next(module.parameters()).device
			all_first = copy.deepcopy(outputs[2])
			for _ in all_first:
				module._freq[_]+=1
			normalized_freq = copy.deepcopy(module._freq)
			normalized_freq = normalized_freq-1
			normalized_freq /= torch.sum(normalized_freq)
			warm_freq = normalized_freq[:module.warm_size]
			cold_freq = normalized_freq[module.warm_size:]
			if self.policy == 'log':
				module._dist_scale = torch.pow((torch.log(normalized_freq)+1) ,self.a).to(device)
			elif self.policy == 'pow':
				warm_dist_scale = (torch.pow(warm_freq, self.a)+1).to(device)
				if self.new_a>0:
					cold_dist_scale = (torch.pow(cold_freq, self.new_a)+1).to(device)
				else:
					cold_dist_scale = torch.ones(len(cold_freq)).to(device)
				module._dist_scale = torch.cat([warm_dist_scale,cold_dist_scale],dim=0)
			elif self.policy == 'exp':
				module._dist_scale = torch.exp(self.a*(normalized_freq)).to(device)
		return outputs



def freq_assign(vq_module, policy="pow",a=0,new_a=0,b=0,b_scale=0):
	FreqAssign.apply(vq_module, policy, a,new_a,b,b_scale)
	return vq_module