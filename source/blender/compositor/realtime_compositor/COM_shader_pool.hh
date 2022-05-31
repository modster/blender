/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_map.hh"
#include "BLI_string_ref.hh"

#include "GPU_shader.h"

namespace blender::realtime_compositor {

/* -------------------------------------------------------------------------------------------------
 *  Shader Pool
 *
 * A pool of shaders identified by their info name that can be reused throughout the evaluation of
 * the compositor and are only freed when the shader pool is destroyed. */
class ShaderPool {
 private:
  /* The set of shaders identified by their info name that are currently available in the pool to
   * be acquired. */
  Map<StringRef, GPUShader *> shaders_;

 public:
  ~ShaderPool();

  /* Check if there is an available shader with the given info name in the pool, if such shader
   * exists, return it, otherwise, return a newly created shader and add it to the pool. */
  GPUShader *acquire(const char *info_name);
};

}  // namespace blender::realtime_compositor
