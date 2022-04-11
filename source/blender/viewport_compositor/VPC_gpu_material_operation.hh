/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector_set.hh"

#include "GPU_material.h"
#include "GPU_shader.h"

#include "NOD_derived_node_tree.hh"

#include "VPC_context.hh"
#include "VPC_operation.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

/* A type representing a contiguous subset of the node execution. */
using SubSchedule = VectorSet<DNode>;
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
  SubSchedule sub_schedule_;
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
  GPUMaterialOperation(Context &context, SubSchedule &sub_schedule);

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

}  // namespace blender::viewport_compositor
