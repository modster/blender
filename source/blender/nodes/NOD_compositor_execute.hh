/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#pragma once

#include "BLI_string_ref.hh"

#include "DNA_node_types.h"
#include "DNA_scene_types.h"

#include "GPU_texture.h"

namespace blender::nodes {

/* This abstract class is passed to the execution function of the compositor. The compositor engine
 * should implement it to provide the necessary functionality needed by the currently execuing
 * node. */
class CompositorContext {
 public:
  /* Get the input texture for the input socket with the given identifier. Returns nullptr if the
   * socket is not linked. */
  virtual const GPUTexture *get_input_texture(StringRef identifier) = 0;

  /* Get the output texture for the output socket with the given identifier. */
  virtual const GPUTexture *get_output_texture(StringRef identifier) = 0;

  /* Get the texture representing the viewport where the result of the compositor should be
   * written. This should be called by output nodes to get their target texture. */
  virtual const GPUTexture *get_viewport_texture() = 0;

  /* Get the texture where the given render pass is stored. This should be called by the Render
   * Layer node to populate its outputs. */
  virtual const GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) = 0;

  /* Get the node currently being executed. */
  virtual const bNode &node() = 0;
};

}  // namespace blender::nodes
