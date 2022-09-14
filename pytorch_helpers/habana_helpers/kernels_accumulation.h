/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#pragma once

#include <ATen/PTThreadPool.h>
#include <string>

namespace habana_lazy {

// returns main accumulation thread pool
at::PTThreadPool& GetAccThreadPool();

// returns thread pool used by accumulation thread pool to release any resources
// moved to it. Its purpose is to avoid deadlock on GIL.
//
// Deadlock in GIL happens when main thread is calling C++ code from Python
// (that acquires GIL by default) and accumulation thread is finishing a
// previous task, which at the end can release Python resources - in this case
// at::Tensors. Main thread is trying to synchronize accumulation thread (due to
// unsupported op) and accumulation thread is trying to acquire GIL to release
// resources. It is avoided by moving any resource from accumulation to cleanup
// thread. Cleanup thread is not synchronized by main thread and can safely wait
// for GIL release.
at::PTThreadPool& GetAccCleanupThreadPool();

// checks if accumulation thread is enabled
bool IsAccThreadEnabled();
// checks if accumulation thread can be used
bool CanUseAccThread();
// synchronizes acc thread pool, if parallel accumulation is enabled
void SyncAccThreadPool();
// synchronizes acc cleanup thread pool, if parallel accumulation is enabled
void SyncCleanupThreadPool();
// synchronizes acc thread pool if the input manual 'op' is not supported
// for parallel accumulation.
//
// Used only for manual ops, not auto-gen.
void SyncManualOpIfNeeded(const std::string& op);

bool IsAccumulationForAutogenSupported(const std::string& op);

} // namespace habana_lazy
