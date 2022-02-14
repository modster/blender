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

#include "BLI_hash.hh"
#include "BLI_map.hh"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "DNA_node_types.h"
#include "DNA_scene_types.h"

#include "GPU_texture.h"

#include "NOD_derived_node_tree.hh"

namespace blender::compositor {

/* --------------------------------------------------------------------
 * Texture Pool.
 */

/* A key structure used to identify a texture specification in a texture pool. Defines a hash and
 * an equality operator for use in a hash map. */
class TexturePoolKey {
 public:
  int width;
  int height;
  eGPUTextureFormat format;

  TexturePoolKey(int width, int height, eGPUTextureFormat format);
  TexturePoolKey(const GPUTexture *texture);

  uint64_t hash() const;
};

inline TexturePoolKey::TexturePoolKey(int width, int height, eGPUTextureFormat format)
    : width(width), height(height), format(format)
{
}

inline TexturePoolKey::TexturePoolKey(const GPUTexture *texture)
{
  width = GPU_texture_width(texture);
  height = GPU_texture_height(texture);
  format = GPU_texture_format(texture);
}

inline uint64_t TexturePoolKey::hash() const
{
  return get_default_hash_3(width, height, format);
}

inline bool operator==(const TexturePoolKey &a, const TexturePoolKey &b)
{
  return a.width == b.width && a.height == b.height && a.format == b.format;
}

/* A pool of textures that can be allocated and reused transparently throughout the evaluation of
 * the node tree. The textures can be reference counted and will only be effectively released back
 * into the pool when their reference count reaches one. Concrete derived classes are expected to
 * free the textures once the pool is no longer in use. */
class TexturePool {
 private:
  /* The set of textures in the pool that are available to acquire for each distinct texture
   * specification. */
  Map<TexturePoolKey, Vector<GPUTexture *>> textures_;

 public:
  /* Check if there is an available texture with the given specification in the pool, if such
   * texture exists, return it, otherwise, return a newly allocated texture. The texture can be
   * reference counted by providing the number of users that will be using this texture. The
   * reference count will then be users_count + 1, because the texture pool is itself considered a
   * user. Expect the texture to be uncleared and contains garbage data. */
  GPUTexture *acquire(int width, int height, eGPUTextureFormat format, int users_count = 1);

  /* Put the texture back into the pool, potentially to be acquired later by another user. The
   * texture is only effectively release when its reference count reaches one. Notice that the
   * texture is release when the texture reference count reaches one not zero, because the texture
   * pool is itself considered a user of the texture. Expects the texture to be one that was
   * acquired using the same texture pool. */
  void release(GPUTexture *texture);

 private:
  /* Returns a newly allocated texture with the given specification. This method should be
   * implemented by the compositor engine and should ideally use the DRW texture pool for
   * allocation. */
  virtual GPUTexture *allocate_texture(int width, int height, eGPUTextureFormat format) = 0;
};

/* This abstract class is used by node operations to access data intrinsic to the compositor
 * engine. The compositor engine should implement the class to provide the necessary
 * functionalities for node operations. */
class CompositorContext {
 public:
  /* Get the texture representing the viewport where the result of the compositor should be
   * written. This should be called by output nodes to get their target texture. */
  virtual GPUTexture *get_viewport_texture() = 0;

  /* Get the texture where the given render pass is stored. This should be called by the Render
   * Layer node to populate its outputs. */
  virtual GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) = 0;
};

using namespace nodes::derived_node_tree_types;

class Compiler {
 private:
  /* The derived and reference node trees representing the compositor setup. */
  NodeTreeRefMap tree_ref_map_;
  DerivedNodeTree tree_;
  /* The output node whose result should be computed and drawn. */
  DNode output_node_;
  /* Stores a heuristic estimation of the number of needed intermediate buffers
   * to compute every node and all of its dependencies. */
  Map<DNode, int> needed_buffers_;
  /* An ordered set of nodes defining the schedule of node execution. */
  VectorSet<DNode> node_schedule_;

 public:
  Compiler(bNodeTree *scene_node_tree);

  void compile();

  void dump_schedule();

 private:
  /* Computes the output node whose result should be computed and drawn, then store the result in
   * output_node_. The output node is the node marked as NODE_DO_OUTPUT. If multiple types of
   * output nodes are marked, then the preference will be CMP_NODE_COMPOSITE > CMP_NODE_VIEWER >
   * CMP_NODE_SPLITVIEWER. */
  void compute_output_node();

  /* Computes a heuristic estimation of the number of needed intermediate buffers to compute this
   * node and all of its dependencies. The method recursively computes the needed buffers for all
   * node dependencies and stores them in the needed_buffers_ map. So the root/output node can be
   * provided to compute the needed buffers for all nodes. */
  int compute_needed_buffers(DNode node);

  /* Computes the execution schedule of the nodes and stores the schedule in node_schedule_. This
   * is essentially a post-order depth first traversal of the node tree from the output node to the
   * leaf input nodes, with informed order of traversal of children based on a heuristic estimation
   * of the number of needed_buffers. */
  void compute_schedule(DNode node);
};

}  // namespace blender::compositor
