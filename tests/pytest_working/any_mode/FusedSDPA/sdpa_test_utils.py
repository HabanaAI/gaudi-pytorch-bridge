import os

DBG_FLAG_verbose_print = False


def check_dbg_env_var(v):
    env_var_set = False
    if int(os.getenv(v, 0)) == 1:
        env_var_set = True
    return env_var_set


def vb_print(*args, **kwargs):
    if DBG_FLAG_verbose_print:
        print(*args, **kwargs)


def get_dbg_env_var_num(v):
    return int(os.getenv(v, 0))
