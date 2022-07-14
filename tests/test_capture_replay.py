import torch
import time
import os
import numpy as np
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore

def testCapture():
  hpu = torch.device('hpu')
  steps = torch.ones(3, 3)

  # after capture, steps = 2
  # in replay, steps *= 2 is done for 20 times
  # this is equivalent to power(steps, 21)
  expected_result = torch.pow(steps*2, 21)
  print("expected_result = ", expected_result)
  steps = steps.to(hpu)

  htcore.mark_step()
  steps_new = steps
  print(steps_new)

  g = ht.hpu.HPUGraph()
  s = ht.hpu.Stream()

  with ht.hpu.stream(s):
      g.capture_begin()
      print("Here in capture")
      steps_new = steps*2
      g.capture_end()

  print("Capture done")
  print(steps_new)
  num_w_batches = 10
  for i in range(num_w_batches):
      steps.copy_(steps_new)
      htcore.mark_step()
      g.replay()
      print("replay number: ", i)
      print(steps_new)

  print(steps_new)

  num_w_batches = 20
  for i in range(10, num_w_batches):
      steps.copy_(steps_new)
      htcore.mark_step()
      g.replay()
      print("replay number: ", i)
      print(steps_new)

  print(steps_new)
  assert np.allclose(steps_new.detach().to("cpu"), expected_result, atol=0, rtol=0), f"Data mismatch"

if __name__ == "__main__":
  testCapture()
