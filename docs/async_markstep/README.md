# **Execution Thread**

**What do we plan to gain?**
--Overlap op accumulation along with execution

**Mechanism:**
--hlexec.Launch is moved to the exec thread part.
--A threadpool of 1 thread is created if PT_HPU_ENABLE_EXECUTION_THREAD=1
--SingleTonExecThreadPool::getInstance will return the instance of launch threadpool.
--As part of StepMarkerBind, Master/Framework thread always collects the live tensor and hands over live tensors to a queue after Postorder.
--Apart from the above function, StageSubmission also will trigger the StepMarker as asynchronous.
--Only the above 2 StepMarker(StepMarkerBind/StageSubmission) will create execution thread. By default, async=False is passed to StepMarker.
--If there are any synchronization points, then HbExecutionContext::JoinPendingLaunchThread() should be called so that the execution threads will join the main thread and the data will be available.
--Ops such as index/non-zero/zero-scalar and ".to(cpu)", ".item()" are synchronization points.

