import os
import json
import time

class TracePoint(object):
    def __init__(self, out_file='fw_event_trace.json'):
        self.tracepoint_enabled = self.get_flag('TRACE_POINT_ENABLE', 0)
        self.in_progress = False
        if self.tracepoint_enabled:
            try:
                import thread as threading
            except:
                import threading
            self.outfile = open(out_file, 'w')
            self.entry = {'name' : None, 'ph' : 'X', 'pid' : os.getpid(), 'tid': threading.current_thread().ident, 'ts': None}
            self.outfile.write("[\n")

    def start(self, wall_time, name):
        if self.tracepoint_enabled:
            self.entry['ts'] = wall_time * 10**6
            self.entry['ph'] = 'B'
            self.entry['name'] = name
            self.outfile.write(json.dumps(self.entry))
            self.outfile.write(",\n")
            self.outfile.flush()
            self.in_progress = True

    def end(self, wall_time, name):
        if self.tracepoint_enabled:
            self.entry['ts'] = wall_time * 10**6
            self.entry['ph'] = 'E'
            self.entry['name'] = name
            self.outfile.write(json.dumps(self.entry))
            self.outfile.write(",\n")
            self.in_progress = False

    def completion(self, wall_time, name):
        self.entry['ts'] = wall_time * 10**6
        self.entry['ph'] = 'X'
        self.entry['name'] = name
        self.outfile.write(json.dumps(self.entry))
        self.outfile.write("\n]")
        self.outfile.flush()

    def get_flag(self, env_var, default_val):
        flag = default_val
        flag_str = os.environ.get(env_var)
        if flag_str is not None:
            k = int(flag_str)
            flag = 1 if k > 0 else default_val
        return flag

    def __del__(self):
        if self.tracepoint_enabled:
            if self.in_progress:
                self.completion(time.time(), 'Incomplete')
            else:
                self.completion(time.time(), 'Complete')
            self.outfile.close()
