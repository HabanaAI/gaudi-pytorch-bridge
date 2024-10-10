###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import datetime

import pytest
from habana_frameworks.torch.utils.event_dispatcher import EventDispatcher, EventId


@pytest.fixture(scope="function")
def evt_disp():
    evt_disp = EventDispatcher.instance()
    yield evt_disp


class CallbackFn:
    def __init__(self, name):
        self._hit_count = 0
        self._name = name
        self._params_log = []

    @property
    def hit_count(self):
        return self._hit_count

    @property
    def params_log(self):
        return self._params_log

    @property
    def callback(self):
        def fn(timestamp, params):
            print(f"Handler fn called {self._name}, timestamp: {timestamp}")
            self._hit_count += 1
            self._params_log.append(params)

        return fn


@pytest.mark.forked
class TestEventDispatcher:
    @pytest.mark.xfail
    def test_simple(self, evt_disp):
        callbacks = [CallbackFn(f"Handler {i}!") for i in range(10)]

        for c in callbacks:
            evt_disp.subscribe(EventId.GRAPH_COMPILATION, c.callback)

        evt_disp.publish(
            EventId.GRAPH_COMPILATION,
            [("duration", 1), ("recipe", "test")],
            datetime.datetime.now(),
        )

        assert all([c.hit_count == 1 for c in callbacks])

    @pytest.mark.xfail
    def test_subscribe_and_partially_unsubscribe(self, evt_disp):
        callbacks = [CallbackFn(f"Handler {i}!") for i in range(10)]
        handles = [evt_disp.subscribe(EventId.GRAPH_COMPILATION, c.callback) for c in callbacks]

        # unsubscribe even callbacks
        for h in handles[::2]:
            evt_disp.unsubscribe(h)

        evt_disp.publish(
            EventId.GRAPH_COMPILATION,
            [("duration", 1), ("recipe", "test")],
            datetime.datetime.now(),
        )

        # check if unsubscribed callbacks weren't called
        assert all([c.hit_count == 0 for c in callbacks[::2]])

        # check if rest callbacks were called once
        assert all([c.hit_count == 1 for c in callbacks[1::2]])

    @pytest.mark.xfail
    def test_subscribe_and_unsubscribe_all(self, evt_disp):
        callbacks = [CallbackFn(f"Handler {i}!") for i in range(10)]
        handles = [evt_disp.subscribe(EventId.GRAPH_COMPILATION, c.callback) for c in callbacks]

        for h in handles:
            evt_disp.unsubscribe(h)

        evt_disp.publish(
            EventId.GRAPH_COMPILATION,
            [("duration", 1), ("recipe", "test")],
            datetime.datetime.now(),
        )

        assert all([c.hit_count == 0 for c in callbacks])

    def test_subscribe_and_publish_different_event(self, evt_disp):
        callback = CallbackFn("Handler!")

        evt_disp.subscribe(EventId.GRAPH_COMPILATION, callback.callback)

        evt_disp.publish(EventId.CUSTOM_EVENT, [], datetime.datetime.now())

        assert callback.hit_count == 0

    def test_publish_in_loop(self, evt_disp):
        callback = CallbackFn("Handler!")
        evt_disp.subscribe(EventId.CUSTOM_EVENT, callback.callback)

        for _ in range(100):
            evt_disp.publish(EventId.CUSTOM_EVENT, [], datetime.datetime.now())

        assert callback.hit_count == 100

    def test_publish_then_unsubscribe_some_and_publish_again(self, evt_disp):
        callbacks = [CallbackFn(f"Handler {i}!") for i in range(10)]
        handles = [evt_disp.subscribe(EventId.CUSTOM_EVENT, c.callback) for c in callbacks]

        evt_disp.publish(EventId.CUSTOM_EVENT, [], datetime.datetime.now())

        # check if all callbacks were called once
        assert all([c.hit_count == 1 for c in callbacks])

        # unsubscribe even callbacks
        for h in handles[::2]:
            evt_disp.unsubscribe(h)

        evt_disp.publish(EventId.CUSTOM_EVENT, [], datetime.datetime.now())

        # check if even callbacks were called once
        assert all([c.hit_count == 1 for c in callbacks[::2]])

        # check if odd callbacks were called twice
        assert all([c.hit_count == 2 for c in callbacks[1::2]])

    @pytest.mark.xfail
    def test_parameters(self, evt_disp):
        callback = CallbackFn("Handler!")
        evt_disp.subscribe(EventId.CUSTOM_EVENT, callback.callback)

        evt_disp.publish(
            EventId.CUSTOM_EVENT,
            [("param1", 1234), ("param2", "test")],
            datetime.datetime.now(),
        )
        assert callback.hit_count == 1
        assert callback.params_log[0] == [("param1", 1234), ("param2", "test")]
