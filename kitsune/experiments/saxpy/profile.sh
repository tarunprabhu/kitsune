#!/bin/bash
echo "running profiler..."
ncu --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section SchedulerStats \
    --section ComputeWorkloadAnalysis \
    --section WarpStateStats \
    --section LaunchStats \
    --section InstructionStats \
    --section SourceCounters \
    --section Occupancy \
    --section MemoryWorkloadAnalysis \
    --page details \
    --print-summary per-gpu \
    --details-all \
    --log-file ncu.log \
    --export saxpy-kokkos.%i saxpy-kokkos.x86_64


