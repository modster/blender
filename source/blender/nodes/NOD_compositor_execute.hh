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

namespace blender::viewport_compositor {

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
   * texture exists, return it, otherwise, return a newly allocated texture. Expect the texture to
   * be uncleared and contains garbage data. */
  GPUTexture *acquire(int width, int height, eGPUTextureFormat format);

  /* Shorthand for acquire with GPU_RGBA16F format. */
  GPUTexture *acquire_color(int width, int height);

  /* Shorthand for acquire with GPU_RGB16F format. */
  GPUTexture *acquire_vector(int width, int height);

  /* Shorthand for acquire with GPU_R16F format. */
  GPUTexture *acquire_float(int width, int height);

  /* Decrement the reference count of the texture and put it back into the pool if its reference
   * count reaches one, potentially to be acquired later by another user. Notice that the texture
   * is release when the texture reference count reaches one, not zero, because the texture pool is
   * itself considered a user of the texture. Expects the texture to be one that was acquired using
   * the same texture pool. */
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
 private:
  /* A texture pool that can be used to allocate textures for the compositor efficiently. */
  TexturePool &texture_pool_;

 public:
  Context(TexturePool &texture_pool);
  /* Get the texture representing the viewport where the result of the compositor should be
   * written. This should be called by output nodes to get their target texture. */
  virtual GPUTexture *get_viewport_texture() = 0;

  /* Get the texture where the given render pass is stored. This should be called by the Render
   * Layer node to populate its outputs. */
  virtual GPUTexture *get_pass_texture(int view_layer, eScenePassType pass_type) = 0;

  /* Get a reference to the texture pool of this context. */
  TexturePool &texture_pool();
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

/* A class that describes the result of an operation. An operator can have multiple results
 * corresponding to multiple outputs. A result either represents a single value or a texture. */
class Result {
 public:
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

 public:
  Result(ResultType type, bool is_texture);

  /* If the result is a texture, increment its reference count.  */
  void incremenet_reference_count();

  /* If the result is a texture, release it back into the given texture pool.  */
  void release(TexturePool &texture_pool);
};

/* --------------------------------------------------------------------
 * Operation.
 */

/* The most basic unit of the compositor. The class can be implemented to perform a certain action
 * in the compositor. */
class Operation {
 private:
  /* A reference to the compositor context. This member references the same object in all
   * operations but is included in the class for convenience. */
  Context &context_;
  /* A mapping between each output of the operation identified by its identifier and the computed
   * result for that output. The initialize method is expected to add the needed results. The
   * contents of the results data are uninitialized prior to the invocation of the execute method,
   * and the execute method is expected to compute those data appropriately. */
  Map<StringRef, Result> results_;
  /* A mapping between each input of the operation identified by its identifier and a reference to
   * the computed result for the output that it is connected to. It is the responsibility of the
   * evaluator to map the inputs to their results prior to invoking any method, which is done by
   * calling map_input_to_result. Inputs that are not linked are mapped to the result of a meta
   * operation emitted by the evaluator, such as SingleValueInputOperation. */
  Map<StringRef, Result *> inputs_to_results_map_;

 public:
  Operation(Context &context);

  virtual ~Operation() = default;

  /* This method should return true if this operation can only operate on buffers, otherwise,
   * return false if the operation can be applied pixel-wise. */
  virtual bool is_buffered() const = 0;

  /* This method should allocate all the necessary buffers needed by the operation and initialize
   * the output results. This includes the output textures as well as any temporary intermediate
   * buffers used by the operation. The texture pool provided by the context should be used to any
   * texture allocations. */
  virtual void initialize();

  /* This method should execute the operation, compute its outputs, and write them to the
   * appropriate result. */
  virtual void execute() = 0;

  /* This method should release any temporary intermediate buffers that were allocated in the
   * allocation method. */
  virtual void release();

  /* Add the given result to the results_ map identified by the given output identifier. */
  void add_result(StringRef identifier, Result result);

  /* Get a reference to the output result identified by the given identifier. Expect the result
   * data to be uninitialized when calling from the execute method. */
  Result &get_result(StringRef identifier);

  /* Map the input identified by the given identifier to a reference to the result it is connected
   * to. This also increments the reference count of texture results. See inputs_to_results_map_
   * member for more details. */
  void map_input_to_result(StringRef identifier, Result *result);

  /* Get a reference to the result connected to the input identified by the given identifier. */
  const Result &get_input(StringRef identifier) const;

  /* Release any textures allocated for the results mapped to the inputs of the operation. This is
   * called after the execution of the operation to declare that the results are no longer needed
   * by this operation. */
  void release_inputs();

  /* Returns a reference to the compositor context. */
  Context &context();

  /* Returns a reference to the texture pool of the compositor context. */
  TexturePool &texture_pool();
};

/* --------------------------------------------------------------------
 * Node Operation.
 */

using namespace nodes::derived_node_tree_types;

/* The operation class that nodes should implement and instantiate in the
 * bNodeType.get_compositor_operation, passing the given inputs to the base constructor.  */
class NodeOperation : public Operation {
 private:
  /* The node that this operation represents. */
  DNode node_;

 public:
  NodeOperation(Context &context, DNode node);

  virtual bool is_buffered() const override;

  /* Returns true if the output identified by the given identifier is needed and should be
   * computed, otherwise returns false. */
  bool is_output_needed(StringRef identifier) const;

  const bNode &node() const;
};

/* --------------------------------------------------------------------
 * Meta Operation.
 */

/* A meta operation is an operation that is emitted by the evaluator to make execution easier for
 * node operations by hiding certain implementations details. See any of the derived classes as an
 * example. */
class MetaOperation : public Operation {
 public:
  MetaOperation(Context &context);
};

/* --------------------------------------------------------------------
 * Single Value Input Operation.
 */

/* A meta operation that provide single value results to node inputs that are not linked. The
 * operation merely pass the default value of the input socket to its output result, which can then
 * be mapped to the input by the evaluator. */
class SingleValueInputOperation : public MetaOperation {
 public:
  /* The identifier of the output for this operation. This is constant for all operations. */
  static const StringRef output_identifier;

 private:
  DInputSocket input_;

 public:
  SingleValueInputOperation(Context &context, DInputSocket input);

  virtual bool is_buffered() const override;

  virtual void initialize() override;

  virtual void execute() override;

 private:
  /* Returns the result type corresponding to the type of the member input socket. */
  ResultType get_input_result_type() const;
};

/* --------------------------------------------------------------------
 * Evaluator.
 */

/* The main class of the viewport compositor. The evaluator compiles the compositor node tree into
 * a stream of operations that are then executed to compute the output of the compositor. */
class Evaluator {
 public:
  /* A reference to the compositor context provided by the compositor engine. */
  Context &context;
  /* The derived and reference node trees representing the compositor setup. */
  NodeTreeRefMap tree_ref_map;
  DerivedNodeTree tree;

 private:
  /* A mapping between nodes and instances of their operations. Initialized with default instances
   * of operations by calling create_node_operations(). Typically initialized early on to be used
   * by various methods to query information about node operations. */
  Map<DNode, NodeOperation *> node_operations_;
  /* The compiled operations stream. This contains ordered references to the operations that were
   * compiled and needs to be evaluated. The operations can be node operations or meta-operations
   * that were emitted by the evaluator. */
  Vector<Operation *> operations_stream_;

  /* A type representing a mapping between nodes and heuristic estimations of the number of needed
   * intermediate buffers to compute the nodes and all of their dependencies. */
  using NeededBuffers = Map<DNode, int>;
  /* A type representing the ordered set of nodes defining the schedule of node execution. */
  using NodeSchedule = VectorSet<DNode>;

 public:
  Evaluator(Context &context, bNodeTree *scene_node_tree);

  /* Delete operations in the operations stream. */
  ~Evaluator();

  /* Compile the compositor node tree into an operations stream then execute that stream. */
  void evaluate();

 private:
  /* Computes the output node whose result should be computed and drawn. The output node is the
   * node marked as NODE_DO_OUTPUT. If multiple types of output nodes are marked, then the
   * preference will be CMP_NODE_COMPOSITE > CMP_NODE_VIEWER > CMP_NODE_SPLITVIEWER. */
  DNode compute_output_node() const;

  /* Returns true if the compositor node tree is valid, false otherwise. */
  bool is_valid(DNode output_node);

  /* Default instantiate node operations for all nodes reachable from the given node. The result is
   * stored in node_operations_. The instances are owned by the evaluator and should be deleted in
   * the destructor. */
  void create_node_operations(DNode node);

  /* Computes a heuristic estimation of the number of needed intermediate buffers to compute this
   * node and all of its dependencies. The method recursively computes the needed buffers for all
   * node dependencies and stores them in the given needed_buffers map. So the root/output node can
   * be provided to compute the needed buffers for all nodes. */
  int compute_needed_buffers(DNode node, NeededBuffers &needed_buffers);

  /* Computes the execution schedule of the nodes and stores it in the given node_schedule. This is
   * essentially a post-order depth first traversal of the node tree from the output node to the
   * leaf input nodes, with informed order of traversal of children based on a heuristic estimation
   * of the number of needed_buffers. */
  void compute_schedule(DNode node, NeededBuffers &needed_buffers, NodeSchedule &node_schedule);

  /* Compile the node schedule into the stream of operations that will be executed in order by the
   * evaluator, and store the result in operations_stream_. */
  void compute_operations_stream(NodeSchedule &node_schedule);

  /* Emit and initialize the node operation corresponding to the given node. This may also emit
   * and initialize extra meta operations that are required by the node operation. */
  void emit_node_operation(DNode node);

  /* Maps each of the inputs of the node operation to their results. Essentially calls
   * map_node_input_to_result for every input in the node. */
  void map_node_inputs_to_results(DNode node);

  /* Map the input to its result. If the input is not linked, it will be mapped to a newly emitted
   * and initialized SingleValueInputOperation. */
  void map_node_input_to_result(DInputSocket input);

  /* Map the input to the result of the given node operation output. */
  void map_node_linked_input_to_result(DInputSocket input, DOutputSocket output);

  /* Map the input to the result of a newly emitted and initialized SingleValueInputOperation. */
  void map_node_unlinked_input_to_result(DInputSocket input);

  /* Execute every operation in the stream in order. This essentially calls execute_operation for
   * every operation in order. */
  void execute_operations_stream();

  /* Execute the given operation then call its release method and release its inputs. */
  void execute_operation(Operation *operation);
};

}  // namespace blender::viewport_compositor
