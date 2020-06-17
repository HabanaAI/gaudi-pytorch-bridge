import sys
from .tensor_probe import *
from .tracepoint import *

#TrainMetaData is to set additional configurations/flags on top of those offered by the standard training script.
#example uses include specifying the number of steps to train rather than training a full epoch.
class TrainMetaData():
    def __init__(self, model, device):
        self.current_train_step = 0
        self.current_eval_step = 0
        #use a large value for num_train_steps  by default so that if the num_train_steps is not set,
        #the default behaviour of running training for all the iterations is maintained.
        self.num_train_steps = sys.maxsize
        self.num_eval_steps = sys.maxsize
        self.logging = True #Enable - default
        self.save_checkpt = True
        self.ParamsDump = ModelParamsDump()
        self.hooks = tp_hooks_register(model, device)
        self.tracept = TracePoint()

    def increment_train_step(self):
        self.current_train_step += 1
        self.ParamsDump.increment_curr_iter_idx()
        tp_hooks_increment_iteration_idx(self.hooks)
        return self.current_train_step

    def increment_eval_step(self):
        self.current_eval_step += 1
        return self.current_eval_step

    def set_num_train_steps(self, x):
        self.num_train_steps = x

    def set_num_eval_steps(self, x):
        self.num_eval_steps = x

    def end_train(self):
        if (self.current_train_step == self.num_train_steps):
            return True
        else:
            return False

    def end_eval(self):
        if (self.current_eval_step == self.num_eval_steps):
            return True
        else:
            return False

    def end_train_n_eval(self):
        if (self.end_train() and self.end_eval()):
            return True
        else:
            return False

    def set_logging(self, x):
        self.logging = x

    def is_logging(self):
        return self.logging

    #Enable/disable saving of checkpoint/model
    def set_save_checkpoint_enable(self, enable=True):
        self.save_checkpt = enable

    def is_save_checkpoint(self):
        return self.save_checkpt
