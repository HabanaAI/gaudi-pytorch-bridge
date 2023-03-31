import pytest
import os

def pytest_addoption(parser):
    parser.addoption("--mode", action="store", default="eager")

def pytest_configure(config):
    pytest.mode = config.getoption("--mode")
    assert pytest.mode in ["eager", "graph", "lazy"]

    if pytest.mode == "eager":
        os.environ['PT_HPU_LAZY_MODE'] = '0'
    elif pytest.mode == "lazy":
        os.environ['PT_HPU_LAZY_MODE'] = '1'
    elif pytest.mode == "graph":
        os.environ['PT_HPU_LAZY_MODE'] = '0'
        os.environ['PT_HPU_DETERMINISTIC_ENABLE'] = '0'


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.mode
    if 'mode' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("mode", [option_value])