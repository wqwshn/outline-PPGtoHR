"""Synthetic-only round3 resource preflight. No project data or labels read."""
import os,time,json,sys,platform,datetime,gc,subprocess
from pathlib import Path
import torch
from torch import nn
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
O=Path(__file__).resolve().parent
previous=json.loads((O/"preflight_receipt.json").read_text(encoding="utf-8"))
class Net(nn.Module):
 def __init__(self,c,arch):
  super().__init__();self.arch=arch
  self.conv=nn.Sequential(nn.Conv1d(c,16,9,stride=2,padding=4),nn.ReLU(),nn.Conv1d(16,24,7,stride=2,padding=3),nn.ReLU(),nn.Conv1d(24,32,5,stride=2,padding=2),nn.ReLU())
  if arch=="CNNLSTM":self.lstm=nn.LSTM(32,32,num_layers=1,batch_first=True)
  self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(32,8))
 def forward(self,x):
  z=self.conv(x)
  if self.arch=="CNNLSTM":z,_=self.lstm(z.transpose(1,2));z=z.mean(1)
  else:z=z.mean(2)
  return self.head(z)
def sync(dev):
 if dev=="cuda":torch.cuda.synchronize()
results=[]
start=time.time()
for dev,arch,c,epochs in [(d,a,c,3 if d=="cuda" else 2) for d in ["cuda","cpu"] for a in ["CNN","CNNLSTM"] for c in ([6,2,8] if d=="cuda" else [8])]:
 torch.manual_seed(20261006)
 # Modest synthetic allocation on CPU; timed epochs include host-to-device batch copies.
 x=torch.randn(1860,c,800);y=torch.randint(0,8,(1860,));xv=torch.randn(512,c,800);yv=torch.randint(0,8,(512,))
 model=Net(c,arch).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 params=sum(p.numel() for p in model.parameters());lossfn=nn.CrossEntropyLoss(reduction="none")
 w=torch.ones(1860)
 # Warm-up, including forward/backward/optimizer, excluded from epoch timings.
 t=time.perf_counter()
 for j in range(3):
  opt.zero_grad(set_to_none=True);loss=(lossfn(model(x[:64].to(dev)),y[:64].to(dev))*w[:64].to(dev)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
 sync(dev);warm=time.perf_counter()-t
 if dev=="cuda":torch.cuda.reset_peak_memory_stats()
 timings=[];valt=[]
 for e in range(epochs):
  model.train();perm=torch.randperm(len(x));sync(dev);t=time.perf_counter()
  for batch in perm.split(64):
   opt.zero_grad(set_to_none=True);loss=(lossfn(model(x[batch].to(dev)),y[batch].to(dev))*w[batch].to(dev)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  sync(dev);timings.append(time.perf_counter()-t)
  model.eval();t=time.perf_counter()
  with torch.no_grad():
   for batch in torch.arange(len(xv)).split(64):lossfn(model(xv[batch].to(dev)),yv[batch].to(dev))
  sync(dev);valt.append(time.perf_counter()-t)
 row=dict(device=str(next(model.parameters()).device),architecture=arch,channels=c,parameters=params,train_windows=1860,validation_windows=512,samples_per_window=800,batch_size=64,epochs_measured=epochs,warmup_seconds=warm,train_epoch_seconds=timings,validation_512_seconds=valt,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if dev=="cuda" else None,peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if dev=="cuda" else None)
 results.append(row);print(json.dumps(row),flush=True)
 del model,opt,x,y,xv,yv;gc.collect()
 if dev=="cuda":torch.cuda.empty_cache()
receipt=dict(status="synthetic_preflight_complete",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,executable=sys.executable,torch=torch.__version__,cuda_build=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory,cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),float_dtype="float32",amp=False,cudnn_benchmark=False,cudnn_deterministic=True,tf32=False,synthetic_data_only=True,real_data_or_labels_read=False,model_weights_saved=False,architecture="Conv1d C->16 k9/s2/p4;16->24 k7/s2/p3;24->32 k5/s2/p2;ReLU each; 800->400->200->100 time points; CNN: temporal mean; CNNLSTM: unidirectional LSTM input32 hidden32 one layer then temporal mean; dropout .1, Linear32->8.",optimizer="AdamW lr .001 weight_decay .0001, CE mean (synthetic weights all one); full precision.",runs=results,total_seconds=time.time()-start)
receipt.update({k:previous[k] for k in ["hardware_readonly_probe","budget_estimate","proposed_not_frozen"]})
receipt["gradient_clip_norm"]=1.0
receipt["previous_unclipped_runs"]=previous["runs"]
(O/"preflight_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print("PREFLIGHT_COMPLETE",receipt["total_seconds"],flush=True)

