#!/usr/bin/env python
"""Run JaxFrames benchmarks comparing explicit execution modes."""

import argparse

from tests.benchmarks.test_jit_benchmarks import TestJITBenchmarks
from tests.benchmarks.test_large_scale_benchmarks import run_comprehensive_benchmark


def run_standard_benchmarks(rows):
    """Run explicit ingest / constructor / steady-state benchmarks."""
    print("="*80)
    print(f"STANDARD BENCHMARKS - {rows:,} rows")
    print("="*80)
    print("(Split by ingest, constructor-only, and warmed execution modes)")
    print()
    run_comprehensive_benchmark()


def run_jit_benchmarks(rows):
    """Run JIT-optimized benchmarks."""
    bench = TestJITBenchmarks()
    bench.run_jit_benchmark(rows)


def main():
    parser = argparse.ArgumentParser(description='Run JaxFrames benchmarks')
    parser.add_argument('--rows', type=int, default=1_000_000,
                        help='Number of rows to benchmark (default: 1,000,000)')
    parser.add_argument('--type', choices=['standard', 'jit', 'both'], default='both',
                        help='Type of benchmark to run')
    parser.add_argument('--all-sizes', action='store_true',
                        help='Run benchmarks for all standard sizes (100k, 500k, 1M, 2M)')

    args = parser.parse_args()

    print("="*80)
    print("JAXFRAMES BENCHMARK SUITE")
    print("="*80)
    print()
    print("Running benchmarks to compare JaxFrames with pandas DataFrame")
    print("Note: JAX performance is best on TPUs/GPUs. CPU results shown here.")
    print()

    sizes = [100_000, 500_000, 1_000_000, 2_000_000] if args.all_sizes else [args.rows]

    for num_rows in sizes:
        if args.type in ['standard', 'both']:
            print(f"\n{'='*60}")
            print(f"Standard Benchmarks - {num_rows:,} rows")
            print(f"{'='*60}")
            run_standard_benchmarks(num_rows)

        if args.type in ['jit', 'both']:
            run_jit_benchmarks(num_rows)

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print("""
Key Findings:
1. Host-to-device ingest cost is separate from constructor-only cost
2. Warmed JAX execution should always be measured with block_until_ready
3. Compile-time and steady-state performance should be tracked independently
4. Performance will improve further on TPUs and GPUs

Recommendations:
- Use JaxFrames for numerical computations and ML pipelines
- Leverage JIT compilation for production workloads
- Use vmap for row-wise operations instead of apply
- Consider TPU/GPU deployment for maximum performance
    """)


if __name__ == "__main__":
    main()
