"""Uniform completed-epoch windows; initialization excluded, no optimizer/RNG calls."""
from collections import deque
import torch

class EpochWindow:
 def __init__(self,size):
  if not isinstance(size,int) or size<1:raise ValueError("Positive integer window required")
  self.size=size;self._values=deque(maxlen=size);self._last=0
 @property
 def epochs(self):return [epoch for epoch,state in self._values]
 def append(self,epoch,state):
  if epoch!=self._last+1 or not state:raise ValueError("Epoch must advance exactly once")
  if any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):raise ValueError("Nonfinite state")
  if self._values:
   previous=self._values[-1][1]
   if state.keys()!=previous.keys() or any(v.shape!=previous[k].shape or v.dtype!=previous[k].dtype for k,v in state.items()):raise ValueError("State identity changed")
   if any(not v.is_floating_point() and not torch.equal(v,previous[k]) for k,v in state.items()):raise ValueError("Changing integer buffer unsupported")
  self._values.append((epoch,{k:v.detach().clone() for k,v in state.items()}));self._last=epoch
 def average(self):
  if not self._values:raise ValueError("No completed epoch state")
  first=self._values[0][1]
  return {k:torch.stack([s[k] for _,s in self._values]).mean(0) if v.is_floating_point() else v.clone() for k,v in first.items()}
 def snapshot(self):return [(epoch,{k:v.clone() for k,v in state.items()}) for epoch,state in self._values]
