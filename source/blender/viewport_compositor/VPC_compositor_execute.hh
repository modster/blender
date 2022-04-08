/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include <cstdint>
#include <memory>

#include "BLI_map.hh"
#include "BLI_math_vec_types.hh"
#include "BLI_transformation_2d.hh"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "DNA_node_types.h"
#include "DNA_scene_types.h"

#include "GPU_material.h"
#include "GPU_shader.h"
#include "GPU_texture.h"

#include "NOD_derived_node_tree.hh"

#include "VPC_context.hh"
#include "VPC_domain.hh"
#include "VPC_node_operation.hh"
#include "VPC_operation.hh"
#include "VPC_result.hh"
#include "VPC_scheduler.hh"
#include "VPC_texture_pool.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

/* --------------------------------------------------------------------
 * GPU Material Node.
 */

/* A class that represents a node in a GPU material. The GPU node stacks for inputs and outputs are
 * stored and populated during construction. Derived class should implement the compile method to
 * implement the node and link it to the GPU material. The GPU material compiler is expected to
 * initialize the input links of node before invoking the compile method. */
class GPUMaterialNode {
 private:
  /* The node that this operation represents. */
  DNode node_;
  /* The GPU node stacks of the inputs of the node. Those are populated during construction in the
   * populate_inputs method. The links of the inputs are initialized by the GPU material compiler
   * prior to calling the compile method. There is an extra stack at the end to mark the end of the
   * array, as this is what the GPU module functions expect. */
  Vector<GPUNodeStack> inputs_;
  /* The GPU node stacks of the outputs of the node. Those are populated during construction in the
   * populate_outputs method. There is an extra stack at the end to mark the end of the array, as
   * this is what the GPU module functions expect. */
  Vector<GPUNodeStack> outputs_;

 public:
  /* Construct the node by populating both its inputs and outputs. */
  GPUMaterialNode(DNode node);

  virtual ~GPUMaterialNode() = default;

  /* Compile the node by adding the appropriate GPU material graph nodes and linking the
   * appropriate resources. */
  virtual void compile(GPUMaterial *material) = 0;

  /* Returns a contiguous array containing the GPU node stacks of each input. */
  GPUNodeStack *get_inputs_array();

  /* Returns a contiguous array containing the GPU node stacks of each output. */
  GPUNodeStack *get_outputs_array();

 protected:
  /* Returns a reference to the node this operations represents. */
  bNode &node() const;

 private:
  /* Populate the inputs of the node. The input link is set to nullptr and is expected to be
   * initialized by the GPU material compiler before calling the compile method. */
  void populate_inputs();
  /* Populate the outputs of the node. The output link is set to nullptr and is expected to be
   * initialized by the compile method. */
  void populate_outputs();
};

/* --------------------------------------------------------------------
 * GPU Material Operation.
 */

/* A type representing a map that associates the identifier of each input of the operation with the
 * output socket it is linked to. */
using InputIdentifierToOutputSocketMap = Map<StringRef, DOutputSocket>;

/* An operation that compiles a contiguous subset of the node execution schedule into a single
 * GPU shader using the GPU material compiler.
 *
 * Consider the following node graph with a node execution schedule denoted by the number at each
 * node. Suppose that the compiler decided that nodes 2 to 5 are pixel-wise operations that can be
 * computed together in a single GPU shader. Then the compiler can construct a GPU Material
 * Operation from the sub-schedule containing nodes 2 to 5, compiling them into a GPU shader using
 * the GPU material compiler. Links that are internal to the sub-schedule are mapped internally in
 * the GPU material, for instance, the links going from node 2 to node 3. However, links that cross
 * the boundary of the sub-schedule and link to nodes outside of it are handled separately.
 *
 * Any link between an input of a node that is part of the sub-schedule and an output of a node
 * that is not part of the sub-schedule is declared as an input to the operation and GPU material,
 * for instance, the links going from node 1 to node 2. The inputs and outputs involved in such
 * links are recorded in the class to allow the compiler to link the inputs of the operation to
 * their respective output results.
 *
 * Any link between an output of a node that is part of the sub-schedule and an input of a node
 * that is not part of the sub-schedule is declared as an output to the operation and GPU material,
 * for instance, the links going from node 3 to node 6. The inputs and outputs involved in such
 * links are recorded in the class to allow the compiler to link the inputs of the operation to
 * their respective output results.
 *
 * +--------+   +--------+   +--------+    +--------+
 * | Node 1 |---| Node 2 |---| Node 3 |----| Node 6 |
 * +--------+\  +--------+   +--------+  / +--------+
 *            \ +--------+   +--------+ /
 *             \| Node 4 |---| Node 5 |/
 *              +--------+   +--------+
 */
class GPUMaterialOperation : public Operation {
 private:
  /* The execution sub-schedule that will be compiled into this GPU material operation. */
  Schedule sub_schedule_;
  /* The GPU material backing the operation. */
  GPUMaterial *material_;
  /* A map that associates each node in the execution sub-schedule with an instance of its GPU
   * material node. Those instances should be freed when no longer needed. */
  Map<DNode, GPUMaterialNode *> gpu_material_nodes_;
  /* A map that associates the identifier of each input of the operation with the output socket it
   * is linked to. If a node that is part of this GPU material has an input that is linked to an
   * output whose node is not part of this GPU material, then that input is considered to be an
   * input of the compiled GPU material operation. The identifiers of such inputs are then
   * associated with the output sockets they are connected to in this map to allow the compiler to
   * map the inputs to the results of the outputs they are linked to. The compiler can call the
   * get_input_identifier_to_output_socket_map method to get a reference to this map and map the
   * results as needed. */
  InputIdentifierToOutputSocketMap input_identifier_to_output_socket_map_;
  /* A map that associates the output socket that provides the result of an output of the operation
   * with the identifier of that output. If a node that is part of this GPU material has an output
   * that is linked to an input whose node is not part of this GPU material, then that output is
   * considered to be an output of the compiled GPU material operation. Such outputs are mapped to
   * the identifiers of their corresponding operation outputs in this map to allow the compiler to
   * map the results of the operation to the inputs they are linked to. The compiler can call the
   * get_output_identifier_from_output_socket to get the operation output identifier corresponding
   * to the given output socket. */
  Map<DOutputSocket, StringRef> output_socket_to_output_identifier_map_;
  /* A map that associates the output socket of a node that is not part of the GPU material to the
   * GPU node link of the input texture that was created for it. This is used to share the same
   * input texture with all inputs that are linked to the same output socket. */
  Map<DOutputSocket, GPUNodeLink *> output_socket_to_input_link_map_;

 public:
  /* Construct and compile a GPU material from the give execution sub-schedule by calling
   * GPU_material_from_callbacks with the appropriate callbacks. */
  GPUMaterialOperation(Context &context, Schedule &sub_schedule);

  /* Free the GPU material and the GPU material nodes. */
  ~GPUMaterialOperation();

  /* - Allocate the output results.
   * - Bind the shader and any GPU material resources.
   * - Bind the input results.
   * - Bind the output results.
   * - Dispatch the shader. */
  void execute() override;

  /* Get the identifier of the operation output corresponding to the given output socket. See
   * output_socket_to_output_identifier_map_ for more information. */
  StringRef get_output_identifier_from_output_socket(DOutputSocket output);

  /* Get a reference to the input identifier to output socket map of the operation. See
   * input_identifier_to_output_socket_map_ for more information. */
  InputIdentifierToOutputSocketMap &get_input_identifier_to_output_socket_map();

 private:
  /* Bind the uniform buffer of the GPU material as well as any color band textures needed by the
   * GPU material. Other resources like attributes and textures that reference images are not bound
   * because the GPU material is guaranteed not to have any of them. Textures that reference the
   * inputs of the operation and images that reference the outputs of the operation are bound in
   * the bind_inputs and bind_outputs methods respectively. The compiled shader of the material is
   * given as an argument and assumed to be bound. */
  void bind_material_resources(GPUShader *shader);

  /* Bind the input results of the operation to the appropriate textures in the GPU materials. Any
   * texture in the GPU material that does not reference an image or a color band is a textures
   * that references an input of the operation, the input whose identifier is the name of the
   * texture sampler in the GPU material shader. The compiled shader of the material is given as an
   * argument and assumed to be bound. */
  void bind_inputs(GPUShader *shader);

  /* Bind the output results of the operation to the appropriate images in the GPU materials. Every
   * image in the GPU material corresponds to one of the outputs of the operation, the output whose
   * identifier is the name of the image in the GPU material shader. The compiled shader of the
   * material is given as an argument and assumed to be bound. */
  void bind_outputs(GPUShader *shader);

  /* A static callback method of interface GPUMaterialSetupFn that is passed to
   * GPU_material_from_callbacks to setup the GPU material. The thunk parameter will be a pointer
   * to the instance of GPUMaterialOperation that is being compiled. This methods setup the GPU
   * material as a compute one. */
  static void setup_material(void *thunk, GPUMaterial *material);

  /* A static callback method of interface GPUMaterialCompileFn that is passed to
   * GPU_material_from_callbacks to compile the GPU material. The thunk parameter will be a pointer
   * to the instance of GPUMaterialOperation that is being compiled. The method goes over the
   * execution sub-schedule and does the following for each node:
   *
   * - Instantiate a GPUMaterialNode from the node and add it to gpu_material_nodes_.
   * - Link the inputs of the node if needed. The inputs are either linked to other nodes in the
   *   GPU material graph or they are exposed as inputs to the GPU material operation itself if
   *   they are linked to nodes that are not part of the GPU material.
   * - Call the compile method of the GPU material node to actually add and link the GPU material
   *   graph nodes.
   * - If any of the outputs of the node are linked to nodes that are not part of the GPU
   *   material, they are exposed as outputs to the GPU material operation itself. */
  static void compile_material(void *thunk, GPUMaterial *material);

  /* Link the inputs of the node if needed. Unlinked inputs are ignored as they will be linked by
   * the node compile method. If the input is linked to a node that is not part of the GPU
   * material, the input will be exposed as an input to the GPU material operation. While if the
   * input is linked to a node that is part of the GPU material, then it is linked to that node in
   * the GPU material node graph. */
  void link_material_node_inputs(DNode node, GPUMaterial *material);

  /* Given the input of a node that is part of the GPU material which is linked to the given output
   * of a node that is also part of the GPU material, map the output link of the GPU node stack of
   * the output to the input link of the GPU node stack of the input. This essentially establishes
   * the needed links in the GPU material node graph. */
  void map_material_node_input(DInputSocket input, DOutputSocket output);

  /* Given the input of a node that is part of the GPU material which is linked to the given output
   * of a node that is not part of the GPU material, do the following:
   *
   * - If an input was already declared for that same output, no need to do anything and the
   *   following steps are skipped.
   * - Add a new input texture to the GPU material.
   * - Map the output socket to the input texture link that was created for it by adding an
   *   association in output_socket_to_input_link_map_.
   * - Declare a new input for the GPU material operation of an identifier that matches the name of
   *   the texture sampler of the previously added texture in the shader with an appropriate
   *   descriptor that matches that of the given input.
   * - Map the input to the output socket that is linked to by adding a new association in
   *   input_identifier_to_output_socket_map_. */
  void declare_material_input_if_needed(DInputSocket input,
                                        DOutputSocket output,
                                        GPUMaterial *material);

  /* Link the input node stack corresponding to the given input to an input color loader sampling
   * the input texture corresponding to the given output. */
  void link_material_input_loader(DInputSocket input, DOutputSocket output, GPUMaterial *material);

  /* Populate the output results of the GPU material operation for outputs of the given node that
   * are linked to nodes outside of the GPU material. */
  void populate_results_for_material_node(DNode node, GPUMaterial *material);

  /* Given the output of a node that is part of the GPU material which is linked to an input of a
   * node that is not part of the GPU material, do the following:
   *
   * - Add a new output image to the GPU material.
   * - Populate a new output result for the GPU material operation of an identifier that matches
   *   the name of the previously added image in the shader with an appropriate type that matches
   *   that of the given output.
   * - Map the output socket to the identifier of the newly populated result by adding a new
   *   association in output_socket_to_output_identifier_map_.
   * - Link the output node stack corresponding to the given output to an output storer storing in
   *   the newly added output image. */
  void populate_material_result(DOutputSocket output, GPUMaterial *material);

  /* A static callback method of interface GPUCodegenCallbackFn that is passed to
   * GPU_material_from_callbacks to amend the shader create info of the GPU material. The thunk
   * parameter will be a pointer to the instance of GPUMaterialOperation that is being compiled.
   * This method setup the shader create info as a compute shader and sets its generate source
   * based on the GPU material code generator output. */
  static void generate_material(void *thunk,
                                GPUMaterial *material,
                                GPUCodegenOutput *code_generator);
};

/* --------------------------------------------------------------------
 * GPU Material Compile Group
 */

/* A class that represents a sequence of scheduled nodes that can be compiled together into a
 * single GPUMaterialOperation. The compiler keeps a single instance of this class when compiling
 * the node schedule to keep track of nodes that will be compiled together. During complication,
 * when the compiler is going over the node schedule, if it finds a GPU material node, instead of
 * compiling it directly like standard nodes, it adds it to the compiler's instance of this class.
 * And before considering the next node in the schedule for compilation, the compiler first tests
 * if the GPU material compile group is complete by checking if the next node can be added to it.
 * See the is_complete method for more information. If the group was determined to be complete, it
 * is then compiled and the group is reset to start tracking the next potential group. If it was
 * determined to be incomplete, then the next node is a GPU material node and will be added to the
 * group. See the compiler compile method for more information. */
class GPUMaterialCompileGroup {
 private:
  /* The contiguous subset of the execution node schedule that is part of this group. */
  Schedule sub_schedule_;

 public:
  /* Add the given node to the GPU material compile group. */
  void add(DNode node);

  /* Check if the group is complete and should to be compiled by considering the next node. The
   * possible cases are as follows:
   * - If the group has no nodes, then it is considered incomplete.
   * - If the next node is not a GPU material node, then it can't be added to the group and the
   *   group is considered complete.
   * - If the next node has inputs that are linked to nodes that are not part of the group, then it
   *   can't be added to the group and the group is considered complete. That's is because results
   *   from such nodes might have different sizes and transforms, and an attempt to define the
   *   operation domain of the resulting GPU material operation will be ambiguous.
   * - Otherwise, the next node can be added to the group and the group is considered incomplete.
   * See the class description for more information. */
  bool is_complete(DNode next_node);

  /* Reset the compile group by clearing the sub_schedule_ member. This is called after compiling
   * the group to ready it for tracking the next potential group. */
  void reset();

  /* Returns the contiguous subset of the execution node schedule that is part of this group. */
  Schedule &get_sub_schedule();
};

/* --------------------------------------------------------------------
 * Compiler.
 */

/* A type representing the ordered operations that were compiled and needs to be evaluated. */
using OperationsStream = Vector<Operation *>;

/* A class that compiles the compositor node tree into an operations stream that can then be
 * executed. The compiler uses the Scheduler class to schedule the compositor node tree into a node
 * execution schedule and goes over the schedule in order compiling the nodes into operations that
 * are then added to the operations stream. The compiler also maps the inputs of each compiled
 * operation to the output result they are linked to. The compiler can decide to compile a group of
 * nodes together into a single GPU Material Operation, which is done using the class's instance of
 * the GPUMaterialCompileGroup class, see its description for more information. */
class Compiler {
 private:
  /* A reference to the compositor context provided by the compositor engine. */
  Context &context_;
  /* The derived and reference node trees representing the compositor setup. */
  NodeTreeRefMap tree_ref_map_;
  DerivedNodeTree tree_;
  /* The compiled operations stream. This contains ordered pointers to the operations that were
   * compiled and needs to be evaluated. Those should be freed when no longer needed. */
  OperationsStream operations_stream_;
  /* A GPU material compile group used to keep track of the nodes that will be compiled together
   * into a GPUMaterialOperation. See the GPUMaterialCompileGroup class description for more
   * information. */
  GPUMaterialCompileGroup gpu_material_compile_group_;
  /* A map associating each node with the node operation it was compiled into. This is mutually
   * exclusive with gpu_material_operations_, each node is either compiled into a standard node
   * operation and added to this map, or compiled into a GPU material operation and added to
   * gpu_material_operations_. This is used to establish mappings between the operations inputs and
   * the output results linked to them. */
  Map<DNode, NodeOperation *> node_operations_;
  /* A map associating each node with the GPU material operation it was compiled into. It is
   * possible that multiple nodes are associated with the same operation, because the operation is
   * potentially compiled from multiple nodes. This is mutually exclusive with node_operations_,
   * each node is either compiled into a standard node operation and added to node_operations_, or
   * compiled into a GPU material operation and added to node_operations_. This is used to
   * establish mappings between the operations inputs and the output results linked to them. */
  Map<DNode, GPUMaterialOperation *> gpu_material_operations_;

 public:
  Compiler(Context &context, bNodeTree *node_tree);

  /* Free the operations in the computed operations stream. */
  ~Compiler();

  /* Compile the given node tree into an operations stream based on the node schedule computed by
   * the scheduler. */
  void compile();

  /* Get a reference to the compiled operations stream. */
  OperationsStream &operations_stream();

 private:
  /* Compile the given node into a node operation and map each input to the result of the output
   * linked to it. It is assumed that all operations that the resulting node operation depends on
   * have already been compiled, a property which is guaranteed to hold if the compile method was
   * called while going over the node schedule in order. */
  void compile_standard_node(DNode node);

  /* Map each input of the node operation to the result of the output linked to it. Unlinked inputs
   * are left unmapped as they will be mapped internally to internal results in the node operation
   * before execution. */
  void map_node_operation_inputs_to_results(DNode node, NodeOperation *operation);

  /* Compile the current GPU material compile group into a GPU material operation, map each input
   * of the operation to the result of the output linked to it, and finally reset the compile
   * group. It is assumed that the compile group is complete. */
  void compile_gpu_material_group();

  /* Map each input of the GPU material operation to the result of the output linked to it. */
  void map_gpu_material_operation_inputs_to_results(GPUMaterialOperation *operation);

  /* Returns a reference to the result of the operation corresponding to the given output that the
   * given output's node was compiled to. The node of the given output was either compiled into a
   * standard node operation or a GPU material operation. The method will retrieve the
   * appropriate operation, find the result corresponding to the given output, and return a
   * reference to it. */
  Result &get_output_socket_result(DOutputSocket output);
};

/* --------------------------------------------------------------------
 * Evaluator.
 */

/* The main class of the viewport compositor. The evaluator compiles the compositor node tree into
 * a stream of operations that are then executed to compute the output of the compositor. */
class Evaluator {
 private:
  /* The compiler instance used to compile the compositor node tree. */
  Compiler compiler_;

 public:
  Evaluator(Context &context, bNodeTree *node_tree);

  /* Compile the compositor node tree into an operations stream. */
  void compile();

  /* Evaluate the compiled operations stream. */
  void evaluate();
};

}  // namespace blender::viewport_compositor
