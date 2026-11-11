/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace executorch::extension::llm {

// Serves a .ptn exported with --offgraph_cache on the native ggml engine. Its
// KV cache holds one sequence, so this serves one session at a time, text
// only, and cannot clone; run the worker with one session and no prefix cache.
class MuseGlimmerNativeExecutor final : public batching::Executor {
 public:
  static runtime::Result<MuseGlimmerBackend> create(
      const MuseGlimmerBackendConfig& config);
  ~MuseGlimmerNativeExecutor() override;

  // Compiles the method on the engine thread.
  bool initialize() override;
  size_t preferred_batch_tokens() const override;
  std::optional<batching::SessionId> open_session() override;
  void close_session(batching::SessionId session) override;
  void set_sampling(
      batching::SessionId session,
      const batching::SamplingParams& params,
      std::optional<uint64_t> seed) override;
  bool execute(const batching::BatchInput& batch, batching::BatchOutput& out)
      override;

 private:
  struct Native;
  explicit MuseGlimmerNativeExecutor(std::unique_ptr<Native> native);

  std::unique_ptr<Native> native_;
};

} // namespace executorch::extension::llm
