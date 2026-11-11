/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/native_executor.h>

#include <executorch/backends/native/runtime/OffGraphCache.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/extension/llm/batching/util/sequence_batch_control.h>
#include <executorch/extension/llm/batching/util/session_table.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/backend/options.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace executorch::extension::llm {

namespace {
constexpr char kMethod[] = "forward";
} // namespace

struct MuseGlimmerNativeExecutor::Native {
  std::shared_ptr<const ptn::Package> package;
  std::shared_ptr<const ptn::Program> program;
  std::unique_ptr<ptn::GgmlEngineHost> host;
  std::shared_ptr<cache::Cache> cache;
  std::unique_ptr<cache::InstallGuard> guard;
  // Declared after the cache, so it is destroyed before the cache.
  std::unique_ptr<batching::util::SequenceBatchControl> control;
  std::unique_ptr<batching::util::SessionTable> sessions;
  // Declared after the host and cache, so they outlive it.
  std::unique_ptr<ptn::EngineContext> context;
  std::unique_ptr<ptn::EngineExecutable> executable;
  size_t logits_output = 0;
  int max_step_tokens = 0;
  int32_t vocab_size = 0;
};

MuseGlimmerNativeExecutor::MuseGlimmerNativeExecutor(
    std::unique_ptr<Native> native)
    : native_(std::move(native)) {}

MuseGlimmerNativeExecutor::~MuseGlimmerNativeExecutor() = default;

runtime::Result<MuseGlimmerBackend> MuseGlimmerNativeExecutor::create(
    const MuseGlimmerBackendConfig& config) {
  if (config.max_sessions != 1 || config.use_dflash) {
    ET_LOG(
        Error,
        "Muse Glimmer native: the engine's KV cache holds one sequence; run "
        "with one session, no prefix cache and no DFlash");
    return runtime::Error::NotSupported;
  }
  try {
    auto native = std::make_unique<Native>();
    native->package = std::make_shared<const ptn::Package>(
        ptn::Package::load(config.model_path));
    const ptn::ByteSpan bytes = native->package->program_bytes();
    native->program = std::make_shared<const ptn::Program>(
        ptn::Program::load(bytes.data(), bytes.size()));
    const ptn::Method& method = native->program->get_method(kMethod);
    const auto geometry =
        ptn::offgraph_cache_geometry(method, *native->package);
    if (!geometry) {
      ET_LOG(
          Error,
          "Muse Glimmer native: %s keeps no off-graph KV cache; export it "
          "with --offgraph_cache",
          config.model_path.c_str());
      return runtime::Error::InvalidProgram;
    }
    const std::optional<int64_t> context =
        native->package->metadata_int(kMaxContextLen);
    if (!context || *context <= 0 ||
        *context > std::numeric_limits<int32_t>::max() ||
        config.max_session_tokens < 0 || config.max_session_tokens > *context) {
      ET_LOG(
          Error,
          "Muse Glimmer native: the session token limit must fit the "
          "package's context length");
      return runtime::Error::InvalidArgument;
    }
    const int session_tokens = config.max_session_tokens > 0
        ? config.max_session_tokens
        : static_cast<int>(*context);
    native->max_step_tokens = ptn::max_step_tokens(method);

    for (size_t i = 0; i < method.output_specs.size(); ++i) {
      if (method.output_specs[i].kind == ptn::OutputKind::UserOutput) {
        native->logits_output = i;
        const auto& sizes =
            method.graph.value(method.graph.output_ids[i]).tensor_meta().sizes;
        native->vocab_size =
            sizes.empty() ? 0 : static_cast<int32_t>(sizes.back());
        break;
      }
    }
    if (native->vocab_size <= 200092 ||
        config.bos_id >= static_cast<uint64_t>(native->vocab_size)) {
      ET_LOG(
          Error,
          "Muse Glimmer native: vocabulary %d cannot hold its special tokens",
          native->vocab_size);
      return runtime::Error::InvalidProgram;
    }

    ptn::register_ggml_cache_builders();
    cache::CacheConfig cache_config{
        .capacity = session_tokens,
        .kv_dtype = static_cast<int>(
            config.native_kv_dtype == "fp16" ? ptn::ScalarType::Half
                                             : ptn::ScalarType::Float),
        .max_write = native->max_step_tokens};
    auto built = cache::CacheFactory::global().build(
        ptn::kGgmlCacheBackendId,
        cache::kind::kSingle,
        *geometry,
        cache_config);
    if (!built.ok()) {
      return built.error();
    }
    native->cache = built.get();
    auto* sequence = native->cache->as<cache::SequenceControl>();
    if (sequence == nullptr) {
      return runtime::Error::InvalidType;
    }
    native->control =
        std::make_unique<batching::util::SequenceBatchControl>(*sequence);
    native->sessions = std::make_unique<batching::util::SessionTable>(
        *native->control,
        /*max_sessions=*/1,
        session_tokens,
        native->vocab_size);

    native->guard = std::make_unique<cache::InstallGuard>(native->cache);
    runtime::BackendOptions<1> options;
    const char* key = nullptr;
    if (native->guard->set_option(options) != runtime::Error::Ok ||
        options.get_option(cache::kCacheKeyOption, key) != runtime::Error::Ok) {
      return runtime::Error::Internal;
    }
    native->host = ptn::GgmlEngineHost::create(
        {.device = config.native_device == "cpu" ? ptn::GgmlDevice::Cpu
                                                 : ptn::GgmlDevice::Gpu});
    native->context = native->host->create_context_with_options(
        native->program, native->package, {{cache::kCacheKeyOption, key}});

    auto spec = std::make_shared<const MuseGlimmerPreparationSpec>(
        MuseGlimmerPreparationSpec{
            aten::ScalarType::Float,
            /*hidden_dim=*/0,
            session_tokens,
            native->max_step_tokens,
            native->vocab_size,
            /*has_vision=*/false,
            /*max_soft_tokens=*/0,
            config.image_limits,
            config.bos_id});
    return MuseGlimmerBackend{
        std::unique_ptr<batching::Executor>(
            new MuseGlimmerNativeExecutor(std::move(native))),
        std::move(spec)};
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer native creation failed: %s", error.what());
    return runtime::Error::InvalidProgram;
  }
}

bool MuseGlimmerNativeExecutor::initialize() {
  try {
    native_->executable = native_->context->compile(kMethod);
    return true;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer native compilation failed: %s", error.what());
    return false;
  }
}

size_t MuseGlimmerNativeExecutor::preferred_batch_tokens() const {
  return static_cast<size_t>(native_->max_step_tokens);
}

std::optional<batching::SessionId> MuseGlimmerNativeExecutor::open_session() {
  return native_->sessions->open();
}

void MuseGlimmerNativeExecutor::close_session(batching::SessionId session) {
  native_->sessions->close(session);
}

void MuseGlimmerNativeExecutor::set_sampling(
    batching::SessionId session,
    const batching::SamplingParams& params,
    std::optional<uint64_t> seed) {
  native_->sessions->set_sampling(session, params, seed);
}

bool MuseGlimmerNativeExecutor::execute(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());
  try {
    const auto step = native_->sessions->pack(batch);
    if (!step.ok()) {
      return false;
    }
    ptn::EngineExecutable& executable = *native_->executable;
    const int total = static_cast<int>(step->tokens.size());
    std::vector<float> logits;
    // The method returns only its last token's logits, so a forward also ends
    // at each token whose prediction an input wants.
    for (int off = 0; off < total;) {
      const int limit = std::min(off + native_->max_step_tokens, total);
      int end = limit;
      for (const int row : step->logit_indices) {
        if (row >= off && row < limit) {
          end = std::min(end, row + 1);
        }
      }
      const int n = end - off;
      if (!native_->control->declare_step(std::vector<int32_t>(
              step->seq_ids.begin() + off, step->seq_ids.begin() + end))) {
        ET_LOG(Error, "Muse Glimmer native: the cache refused %d tokens", n);
        return false;
      }
      executable.resize_input(0, {1, n});
      executable.set_input(
          0, step->tokens.data() + off, n, ptn::ScalarType::Long);
      executable.resize_input(1, {n});
      executable.set_input(
          1, step->positions.data() + off, n, ptn::ScalarType::Long);
      executable.execute();

      for (size_t i = 0; i < batch.inputs.size(); ++i) {
        if (step->logit_indices[i] != end - 1) {
          continue;
        }
        const std::vector<int64_t> sizes =
            executable.output_sizes(native_->logits_output);
        int64_t numel = 1;
        for (const int64_t size : sizes) {
          numel *= size;
        }
        logits.resize(static_cast<size_t>(numel));
        executable.get_output(
            native_->logits_output,
            logits.data(),
            logits.size(),
            ptn::ScalarType::Float);
        // The last row is the last token's.
        auto row = make_tensor_ptr(
            {1, native_->vocab_size},
            std::vector<float>(
                logits.end() - native_->vocab_size, logits.end()));
        const batching::SessionId session = batch.inputs[i].sid;
        const auto token = native_->sessions->sample(session, *row, 0);
        if (!token) {
          return false;
        }
        out.outputs[i] = batching::Output{session, {*token}};
      }
      off = end;
    }
    return true;
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer native execution failed: %s", error.what());
    return false;
  }
}

} // namespace executorch::extension::llm
