from __future__ import print_function
import torch
import hblazy.core.hb_model as hm

py_funcs = dir(hm)
print("pyfuncs=", py_funcs)


class dummyOptimizer:
    def step():
        print("custom optimizer.step()")
        return 42


def dummyClosure(num, string):
    print(string)


in_t = torch.randn(8, 10)
hpu_t = in_t.to("habana")

hm.add_step_closure(dummyClosure, args=(1, "First"))
hm.add_step_closure(dummyClosure, args=(2, "Second"))
hm.add_step_closure(dummyClosure, args=(3, "Third"))

hm.optimizer_step(dummyOptimizer, barrier=True)
