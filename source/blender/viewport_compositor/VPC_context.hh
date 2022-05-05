/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_math_vec_types.hh"
#include "BLI_string_ref.hh"

#include "DNA_scene_types.h"

#include "GPU_texture.h"

#include "VPC_shader_pool.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

/* ------------------------------------------------------------------------------------------------
 * Context
 *
 * An abstract class which is used by operations to access data intrinsic to the compositor engine.
 * The compositor engine should implement the class to provide the necessary functionalities for
 * operations. The class also provides a reference to the texture pool which should be implemented
 * by the compositor engine and provided during construction. Finally, the class have an instance
 * of a shader pool for convenient shader acquisition. */
class Context {
 private:
  /* A texture pool that can be used to allocate textures for the compositor efficiently. */
  TexturePool &texture_pool_;
  /* A shader pool that can be used to create shaders for the compositor efficiently. */
  ShaderPool shader_pool_;

 public:
  Context(TexturePool &texture_pool);

  /* Get the active compositing scene. */
  virtual const Scene *get_scene() const = 0;

  /* Get the dimensions of the viewport. */
  virtual int2 get_viewport_size() = 0;

  /* Get the texture representing the viewport where the result of the compositor should be
   * written. This should be called by output nodes to get their target texture. */
  virtual GPUTexture *get_viewport_texture() = 0;

  /* Get the texture where the given render pass is stored. This should be called by the Render
   * Layer node to populate its outputs. */
  virtual GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) = 0;

  /* Get the name of the view currently being rendered. */
  virtual StringRef get_view_name() = 0;

  /* Get the current frame number of the active scene. */
  int get_frame_number() const;

  /* Get the current time in seconds of the active scene. */
  float get_time() const;

  /* Get a reference to the texture pool of this context. */
  TexturePool &texture_pool();

  /* Get a reference to the shader pool of this context. */
  ShaderPool &shader_pool();
};

}  // namespace blender::viewport_compositor
