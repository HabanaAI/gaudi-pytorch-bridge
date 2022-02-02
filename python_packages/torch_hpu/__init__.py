def is_available() -> bool:
    try:
        from habana_frameworks.torch.utils.library_loader import load_habana_module # type: ignore[import]
        load_habana_module()
    except:
        return False
    import habana_frameworks.torch.core as htcore # type: ignore[import]
    return htcore.is_available()

def get_device_type() -> int:
    if is_available():
        import habana_frameworks.torch.core as htcore # type: ignore[import]
        return htcore.get_device_type()
    else:
        return -1

def device_count() -> int:
    r"""Returns the number of HPUs available."""
    if is_available():
        import habana_frameworks.torch.core as htcore # type: ignore[import]
        return htcore.get_device_count()
    else:
        return 0

