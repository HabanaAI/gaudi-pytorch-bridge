from habana_frameworks.torch import _media_pyt_bridge_C


def CreatePytMediaProxy(device_id):
    return _media_pyt_bridge_C.create_pyt_media_proxy(device_id)


def GetOutputTensor(addr):
    return _media_pyt_bridge_C.get_output_tensor(addr)
