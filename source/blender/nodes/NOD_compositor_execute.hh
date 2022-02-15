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

#include <cstdint>

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

/* --------------------------------------------------------------------
 * Context.
 */

/* This abstract class is used by node operations to access data intrinsic to the compositor
 * engine. The compositor engine should implement the class to provide the necessary
 * functionalities for node operations. */
class Context {
 public:
  /* Get the texture representing the viewport where the result of the compositor should be
   * written. This should be called by output nodes to get their target texture. */
  virtual GPUTexture *get_viewport_texture() = 0;

  /* Get the texture where the given render pass is stored. This should be called by the Render
   * Layer node to populate its outputs. */
  virtual GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) = 0;
};

/* --------------------------------------------------------------------
 * Result.
 */

/* Possible data types that operations can operate on. They either represent the base type of the
 * result texture or a single value result. */
enum class ResultType : uint8_t {
  Float,
  Vector,
  Color,
};

/* A structure that describes the result of an operation. An operator can have multiple results
 * corresponding to multiple outputs. A result either represents a single value or a texture. */
struct Result {
  /* The base type of the texture or the type of the single value. */
  ResultType type;
  /* If true, the result is a texture, otherwise, the result is a single value. */
  bool is_texture;
  /* A union of the possible data that could be stored in the result. One of those members is
   * active depending on the value of the is_texture and type members. */
  union {
    GPUTexture *texture;
    float value;
    float vector[3];
    float color[4];
  } data;
};

/* --------------------------------------------------------------------
 * Operation.
 */

class Operation {
 protected:
  /* A reference to the compositor context. This member references the same object in all
   * operations but is included in the class for convenience. */
  Context &context_;
  /* A mapping between each output of the operation identified by its name and the computed result
   * for that output. Unused outputs are not included. It is the responsibility of the evaluator to
   * add default results for outputs that are needed and should be computed by the operation prior
   * to invoking any methods, which is done by calling ensure_output. The results structures are
   * uninitialized prior to the invocation of the allocate method, and allocate method is expected
   * to initialize the results structures appropriately. The contents of the results data are
   * uninitialized prior to the invocation of the execute method, and the execute method is
   * expected to compute those data appropriately. */
  Map<StringRef, Result> outputs_;
  /* A mapping between each input of the operation identified by its name and a reference to the
   * computed result for the output that it is connected to. It is the responsibility of the
   * evaluator to populate the inputs prior to invoking any method, which is done by calling
   * populate_input. Inputs that are not linked reference meta-output single value results. */
  Map<StringRef, Result *> inputs_;

 public:
  Operation(Context &context);

  /* This method should return true if this operation can only operate on buffers, otherwise,
   * return false if the operation can be applied pixel-wise. */
  virtual bool is_buffered() const = 0;

  /* This method should allocate all the necessary buffers needed by the operation and initialize
   * the output results. This includes the output textures as well as any temporary intermediate
   * buffers used by the operation. The texture pool provided by the context should be used to any
   * texture allocations. */
  virtual void allocate() = 0;

  /* This method should execute the operation, compute its outputs, and write them to the
   * appropriate result. */
  virtual void execute() = 0;

  /* This method should release any temporary intermediate buffer that was allocated in the
   * allocation method. */
  virtual void release();

  /* Declares that the output identified by the given name is needed and should be computed by the
   * operation. See outputs_ member for more details. */
  void ensure_output(StringRef name);

  /* Get a reference to the output result identified by the given name. Expect the result to be
   * uninitialized when calling from the allocate method and expect the result data to be
   * uninitialized when calling from the execute method. */
  Result &get_result(StringRef name);

  /* Populate the inputs map by mapping an input identified by the given name to a reference to the
   * output result it is connected to. See inputs_ member for more details. */
  void populate_input(StringRef name, Result *result);

  /* Get a reference to the result connected to the input identified by the given name. */
  const Result &get_input(StringRef name) const;
};

/* --------------------------------------------------------------------
 * Node Operation.
 */

using namespace nodes::derived_node_tree_types;

class NodeOperation : public Operation {
 private:
  DNode &node_;

 public:
  NodeOperation(Context &context, DNode &node);

  virtual bool is_buffered() const override;

  /* Returns true if the output identified by the given name is needed and should be computed,
   * otherwise returns false. */
  bool is_output_needed(StringRef name) const;
};

/* --------------------------------------------------------------------
 * Compiler.
 */

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
