from habana_frameworks.torch import media_pyt_bridge

# create media pipe pytorch proxy object
proxy_device = media_pyt_bridge.CreatePytMediaProxy(0)
proxy_device_ = media_pyt_bridge.CreatePytMediaProxy(0)
# singleton proxy object must create
assert proxy_device == proxy_device_

# access tensor from proxy object
try:
    out_tesnor = media_pyt_bridge.GetOutputTensor(0)
except RuntimeError:
    # access tensor from empty list should thrown error
    assert True
else:
    assert False
