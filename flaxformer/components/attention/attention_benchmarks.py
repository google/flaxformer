# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmarks for attention mechanisms."""

import functools
import itertools
import timeit

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.random
import jax.sharding
from jax.sharding import PartitionSpec as P

from flaxformer.components.attention import dense_attention
from flaxformer.components.attention import memory_efficient_attention

Array = jax.Array


class AttentionBenchmark(parameterized.TestCase):
  """Test harness for attention mechanism benchmarks."""

  def time_and_hbm(self, query, key, value, runner):
    """Returns the runtime and total HBM usage for an attention mechanism."""
    num_trials = 2
    runner = jax.jit(runner)

    # We compile first, to warm up the JIT compile cache.
    compiled = runner.lower(query, key, value).compile()
    duration_in_seconds = timeit.timeit(
        lambda: runner(query, key, value).block_until_ready(), number=num_trials
    )
    memory_analysis = compiled.memory_analysis()
    total_hbm_use_in_bytes = (
        memory_analysis.argument_size_in_bytes
        + memory_analysis.output_size_in_bytes
        - memory_analysis.alias_size_in_bytes
        + memory_analysis.temp_size_in_bytes
    )

    return duration_in_seconds, total_hbm_use_in_bytes

  def show_deltas(
      self,
      query,
      key,
      value,
      baseline_name,
      baseline_runner,
      experiment_runner,
      experiment_name,
      config_name,
  ):
    """Prints runtime and HBM use deltas between two attention mechanisms."""

    baseline_seconds, baseline_bytes = self.time_and_hbm(
        query, key, value, baseline_runner
    )
    experiment_seconds, experiment_bytes = self.time_and_hbm(
        query, key, value, experiment_runner
    )

    print(f"{baseline_name} {config_name} wall time: {baseline_seconds:.2f}s")
    print(
        f"{experiment_name} {config_name} wall time: {experiment_seconds:.2f}s"
    )
    print(
        f"{baseline_name} {config_name} HBM:"
        f" {(baseline_bytes / (1024**3)):.2f}GB"
    )
    print(
        f"{experiment_name} {config_name} HBM use:"
        f" {(experiment_bytes / (1024**3)):.2}GB"
    )

  def test_performance_multihead(self):
    """Benchmarks multi-head attention."""
    batch_size = 8
    num_queries = 2**13
    num_kvs = 2**13
    num_heads = 16
    head_dim = 96
    prng_key = jax.random.PRNGKey(0xFEDE)

    mesh = jax.sharding.Mesh(jax.devices(), ("model",))
    sharding = functools.partial(jax.sharding.NamedSharding, mesh)

    query = jax.device_put(
        jax.random.normal(
            prng_key, (batch_size, num_queries, num_heads, head_dim)
        ),
        sharding(P(None, None, "model", None)),
    )
    key = jax.device_put(
        jax.random.normal(prng_key, (batch_size, num_kvs, num_heads, head_dim)),
        sharding(P(None, None, "model", None)),
    )
    value = jax.device_put(
        jax.random.normal(prng_key, (batch_size, num_kvs, num_heads, head_dim)),
        sharding(P(None, None, "model", None)),
    )

    def run_memory_efficient(query, key, value):
      return memory_efficient_attention.dot_product_attention_multihead(
          query,
          key,
          value,
          float32_logits=True,
          query_chunk_size=1024,
          key_chunk_size=1024,
      )

    def run_baseline(query, key, value):
      return dense_attention.dot_product_attention(
          query,
          key,
          value,
          float32_logits=True,
      )

    self.show_deltas(
        query,
        key,
        value,
        "Baseline",
        run_baseline,
        run_memory_efficient,
        "Memory efficient",
        "multihead",
    )

  def test_performance_multiquery(self):
    """Benchmarks multi-query attention."""
    batch_size = 4
    num_queries = 2**13
    num_kvs = 2**13
    num_heads = 16
    head_dim = 96
    prng_key = jax.random.PRNGKey(0xFEDE)

    mesh = jax.sharding.Mesh(jax.devices(), ("model",))
    sharding = functools.partial(jax.sharding.NamedSharding, mesh)

    query = jax.device_put(
        jax.random.normal(
            prng_key, (batch_size, num_queries, num_heads, head_dim)
        ),
        sharding(P(None, None, "model", None)),
    )
    key = jax.device_put(
        jax.random.normal(prng_key, (batch_size, num_kvs, head_dim)),
        sharding(P(None, None, None)),
    )
    value = jax.device_put(
        jax.random.normal(prng_key, (batch_size, num_kvs, head_dim)),
        sharding(P(None, None, None)),
    )

    def run_memory_efficient(query, key, value):
      return memory_efficient_attention.dot_product_attention_multiquery(
          query,
          key,
          value,
          float32_logits=True,
          query_chunk_size=1024,
          key_chunk_size=1024,
      )

    def run_baseline(query, key, value):
      return dense_attention.dot_product_attention_multiquery(
          query,
          key,
          value,
          float32_logits=True,
      )

    self.show_deltas(
        query,
        key,
        value,
        "Baseline",
        run_baseline,
        run_memory_efficient,
        "Memory efficient",
        "multiquery",
    )

    self.show_deltas(
        query,
        key,
        value,
        "Baseline",
        run_baseline,
        run_memory_efficient,
        "Memory-efficient",
        "multiquery",
    )

  @parameterized.parameters(
      list(
          itertools.product(
              (2**13, 2**14, 2**15, 2**16), (16, 32, 64), (64, 96)
          )
      )
  )
  def test_length_scaling_multiquery(self, input_length, num_heads, head_dim):
    """Benchmarks a range of configurations for peak HBM use and wall time."""
    batch_size = 4
    num_queries = input_length
    num_kvs = input_length
    prng_key = jax.random.PRNGKey(0xFEDE)

    mesh = jax.sharding.Mesh(jax.devices(), ("model",))
    sharding = functools.partial(jax.sharding.NamedSharding, mesh)

    query = jax.device_put(
        jax.random.normal(
            prng_key, (batch_size, num_queries, num_heads, head_dim)
        ),
        sharding(P(None, None, "model", None)),
    )
    key = jax.device_put(
        jax.random.normal(prng_key, (batch_size, num_kvs, head_dim)),
        sharding(P(None, None, None)),
    )
    value = jax.device_put(
        jax.random.normal(prng_key, (batch_size, num_kvs, head_dim)),
        sharding(P(None, None, None)),
    )

    def run_memory_efficient(query, key, value):
      return memory_efficient_attention.dot_product_attention_multiquery(
          query,
          key,
          value,
          float32_logits=True,
          query_chunk_size=1024,
          key_chunk_size=1024,
      )

    run_seconds, run_bytes = self.time_and_hbm(
        query, key, value, run_memory_efficient
    )

    print(f"{input_length=}, {num_heads=}, {head_dim=}")
    print(f"Multiquery wall time: {run_seconds:.2f}s")
    print(f"Multiquery HBM: {(run_bytes / (1024**3)):.2f}GB")


if __name__ == "__main__":
  absltest.main()
