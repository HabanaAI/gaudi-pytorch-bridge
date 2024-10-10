### Usage :
```
python tests/debug_tools/ir_test_generator.py --help
usage: python ir_test_generator [-h] -op OP_NAME [-idx OP_IDX] -m MARKER -ilog INFO_LOG -irlog IR_LOG [-v] [-donnx ONNX_FILE_PATH] [-dtest CPP_FILE_PATH]

optional arguments:
 -h, --help            show this help message and exit
 -op OP_NAME, --op_name OP_NAME
                       op name like conv/relu/dropout etc
 -idx OP_IDX, --op_index OP_IDX
                       search for op at {idx}, -1 / blank for last op occurence
 -m MARKER, --marker_name MARKER
                       marker name from file to search from, example: <mark start : 0>...
 -ilog INFO_LOG, --info_log INFO_LOG
                       HPU_INFO_LOG file path
 -irlog IR_LOG, --ir_log IR_LOG
                       IR Graph log file path
 -v, --verbose         Print Generated file and Adj matrices
 -donnx ONNX_FILE_PATH, --dest_onnx ONNX_FILE_PATH
                       onnx destination file path
 -dtest CPP_FILE_PATH, --dest_test CPP_FILE_PATH
```

### Example Usage :

```
python tests/debug_tools/ir_test_generator.py -ilog /home/pmanvi/test_log_mnist.log -irlog /home/pmanvi/log_dump_example.log -op conv  -m "mark start : 0" -v
```

### Visualization :
```
pip install netron
netron ir_log.onnx (or the path of onnx file generated)
```
