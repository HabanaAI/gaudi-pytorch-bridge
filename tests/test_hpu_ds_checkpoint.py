import argparse
import os
import time
import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.debug as htdebug

torch.manual_seed(0)
os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"

class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = torch.nn.Linear(16, 32)
    self.fc2 = torch.nn.Linear(32, 2)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = torch.mean(x, dim=1)
    return x

  def loss(self, x,  y):
    y_pred = self(x)
    loss = torch.nn.functional.nll_loss(y_pred, y)
    return loss

def run_model(args):
  model = Net()
  model = model.to('hpu')
  optim = torch.optim.Adam(model.parameters(), lr=0.01)
  channel_size_list = torch.randint(4, 20, (20, ))

  if args.resume_checkpoint:
    htdebug.load_ds_checkpoint(args.checkpoint_path)

  for c in range(args.start_idx, args.end_idx):
    X = torch.randn((3, channel_size_list[c], 16))
    y = torch.randint(0, 2, (3, ))
    X = X.to('hpu')
    y = y.to('hpu')

    optim.zero_grad()
    loss = model.loss(X, y)
    htcore.mark_step()

    loss.backward()
    htcore.mark_step()

    optim.step()
    htcore.mark_step()
    if args.save_checkpoint:
      htdebug.save_ds_checkpoint(args.checkpoint_path)

def add_op(input_shape):
  input1 = torch.randn(input_shape)
  input2 = torch.randn(input_shape)

  h_input1 = input1.to('hpu')
  h_input2 = input2.to('hpu')
  h_add = torch.add(h_input1, h_input2)
  htcore.mark_step()

def test_add_op(args):
  channel_size_list = [6, 8, 10, 4]

  if args.resume_checkpoint:
    htdebug.load_ds_checkpoint(args.checkpoint_path)

  for c in range(args.start_idx, args.end_idx):
    add_op((4, channel_size_list[c], 3))
    time.sleep(1)
    if args.save_checkpoint:
      htdebug.save_ds_checkpoint(args.checkpoint_path)

def get_args():
  recipe_trace_path = os.getenv("PT_RECIPE_TRACE_PATH", "")

  parser = argparse.ArgumentParser()
  parser.add_argument("--resume_checkpoint", type=int, default=0)
  parser.add_argument("--save_checkpoint", type=int, default=0)
  parser.add_argument("--checkpoint_path", type=str, default="ds_checkpoint.pt")
  parser.add_argument("--start_idx", type=int, default=0)
  parser.add_argument("--end_idx", type=int)

  args = parser.parse_args()
  args.recipe_trace_path = recipe_trace_path
  return args

if __name__ == '__main__':
  args = get_args()
  run_model(args)
  # test_add_op(args)
