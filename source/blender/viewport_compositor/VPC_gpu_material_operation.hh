/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <memory>

#include "BLI_map.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector_set.hh"

#include "GPU_material.h"
#include "GPU_shader.h"

#include "NOD_derived_node_tree.hh"

#include "VPC_context.hh"
#include "VPC_operation.hh"
#include "VPC_scheduler.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

/* A type representing a contiguous subset of the node execution schedule. */
using SubSchedule = VectorSet<DNode>;
/* A type representing a map that associates the identifier of each input of the operation with the
 * output socket it is linked to. */
using InputsToLinkedOutputsMap = Map<StringRef, DOutputSocket>;
/* A type representing a map that associates the output socket that provides the result of an
 * output of the operation with the identifier of that output. */
using OutputSocketsToOutputIdentifiersMap = Map<DOutputSocket, StringRef>;

/* ------------------------------------------------------------------------------------------------
 * GPU Material Operation
 *
 * An operation that compiles a contiguous subset of the node execution schedule into a single
 * GPU shader using the GPU material compiler.
 *
 * Consider the following node graph with a node execution schedule denoted by the number of each
 * node. The compiler may decide to compile a subset of the execution schedule into a GPU material
 * operation, in this case, the nodes from 3 to 5 were compiled together into a GPU material. See
 * the discussion in VPC_evaluator.hh for more information on the compilation process. Each of the
 * nodes inside the sub-schedule implements a GPU Material Node which is instantiated, stored in
 * gpu_material_nodes_, and used during compilation. See the discussion in VPC_gpu_material_node.hh
 * for more information. Links that are internal to the GPU material are established between the
 * input and outputs of the GPU material nodes, for instance, the links between nodes 3 <-> 4 and
 * nodes 4 <-> 5. However, links that cross the boundary of the GPU material needs special
 * handling.
 *
 *                                        GPU Material
 *                   +------------------------------------------------------+
 * .------------.    |  .------------.  .------------.      .------------.  |  .------------.
 * |   Node 1   |    |  |   Node 3   |  |   Node 4   |      |   Node 5   |  |  |   Node 6   |
 * |            |----|--|            |--|            |------|            |--|--|            |
 * |            |  .-|--|            |  |            |  .---|            |  |  |            |
 * '------------'  | |  '------------'  '------------'  |   '------------'  |  '------------'
 *                 | +----------------------------------|-------------------+
 * .------------.  |                                    |
 * |   Node 2   |  |                                    |
 * |            |--'------------------------------------'
 * |            |
 * '------------'
 *
 * Links from nodes that are not part of the GPU material to nodes that are part of the GPU
 * material are considered inputs of the operation itself and are declared as such. For instance,
 * the link from node 1 to node 3 is declared as an input to the operation, and the same applies
 * for the links from node 2 to nodes 3 and 5. Note, however, that only one input is declared for
 * each distinct output socket, so both links from node 2 share the same input of the operation.
 * Once an input is declared for a distinct output socket:
 * 1. A material texture is added to the GPU material. This texture will be bound to the result of
 *    the output socket during evaluation.
 * 2. The newly added texture is mapped to the output socket in output_to_material_texture_map_
 *    to share that same texture for all inputs linked to the same output socket.
 * 3. A texture loader GPU material node that samples the newly added material texture is added and
 *    linked to the GPU input stacks of GPU material nodes that the output socket is linked to.
 * 4. The input of the operation is mapped to the output socket to help the compiler in
 *    establishing links between the operations. See the get_inputs_to_linked_outputs_map method.
 *
 * Links from nodes that are part of the GPU material to nodes that are not part of the GPU
 * material are considered outputs of the operation itself and are declared as such. For instance,
 * the link from node 5 to node 6 is declared as an output to the operation. Once an output is
 * declared for an output socket:
 * 1. A material image is added to the GPU material. This image will be bound to the result of
 *    the operation output during evaluation. This is the image where the result of that output
 *    will be written.
 * 2. An image storer GPU material node that stores the output value in the newly added material
 *    image is added and linked to the GPU output stack of the output.
 * 4. The output of the operation is mapped to the output socket to help the compiler in
 *    establishing links between the operations. See the get_inputs_to_linked_outputs_map method.
 *
 * The GPU material is declared as a compute material and its compute source is used to construct a
 * compute shader that is then dispatched during operation evaluation after binding the inputs,
 * outputs, and any necessary resources. */
class GPUMaterialOperation : public Operation {
 private:
  /* The execution sub-schedule that will be compiled into this GPU material operation. */
  SubSchedule sub_schedule_;
  /* The GPU material backing the operation. This is created and compiled during construction and
   * freed during construction. */
  GPUMaterial *material_;
  /* A map that associates each node in the execution sub-schedule with an instance of its GPU
   * material node. */
  Map<DNode, std::unique_ptr<GPUMaterialNode>> gpu_material_nodes_;
  /* A map that associates the identifier of each input of the operation with the output socket it
   * is linked to. See the above discussion for more information. */
  InputsToLinkedOutputsMap inputs_to_linked_outputs_map_;
  /* A map that associates the output socket that provides the result of an output of the operation
   * with the identifier of that output. See the above discussion for more information. */
  OutputSocketsToOutputIdentifiersMap output_sockets_to_output_identifiers_map_;
  /* A map that associates the output socket of a node that is not part of the GPU material to the
   * material texture that was created for it. This is used to share the same material texture with
   * all inputs that are linked to the same output socket. */
  Map<DOutputSocket, GPUMaterialTexture *> output_to_material_texture_map_;

 public:
  /* Construct and compile a GPU material from the give node execution sub-schedule by calling
   * GPU_material_from_callbacks with the appropriate callbacks. */
  GPUMaterialOperation(Context &context, SubSchedule &sub_schedule);

  /* Free the GPU material. */
  ~GPUMaterialOperation();

  /* Allocate the output results, bind the shader and all its needed resources, then dispatch the
   * shader. */
  void execute() override;

  /* Get the identifier of the operation output corresponding to the given output socket. This is
   * called by the compiler to identify the operation output that provides the result for an input
   * by providing the output socket that the input is linked to. See
   * output_sockets_to_output_identifiers_map_ for more information. */
  StringRef get_output_identifier_from_output_socket(DOutputSocket output);

  /* Get a reference to the inputs to linked outputs map of the operation. This is called by the
   * compiler to identify the output that each input of the operation is linked to for correct
   * input mapping. See inputs_to_linked_outputs_map_ for more information. */
  InputsToLinkedOutputsMap &get_inputs_to_linked_outputs_map();

  /* Compute and set the initial reference counts of all the results of the operation. The
   * reference counts of the results are the number of operations that use those results, which is
   * computed as the number of inputs whose node is part of the schedule and is linked to the
   * output corresponding to each result. The node execution schedule is given as an input. */
  void compute_results_reference_counts(const Schedule &schedule);

 private:
  /* Bind the uniform buffer of the GPU material as well as any color band textures needed by the
   * GPU material. Other resources like attributes and textures that reference images are not bound
   * because the GPU material is guaranteed not to have any of them. Textures that reference the
   * inputs of the operation and images that reference the outputs of the operation are bound in
   * the bind_inputs and bind_outputs methods respectively. The compiled shader of the material is
   * given as an argument and assumed to be bound. */
  void bind_material_resources(GPUShader *shader);

  /* Bind the input results of the operation to the appropriate textures in the GPU materials. The
   * material textures stored in output_to_material_texture_map_ have sampler names that match
   * the identifiers of the operation inputs that they correspond to. The compiled shader of the
   * material is given as an argument and assumed to be bound. */
  void bind_inputs(GPUShader *shader);

  /* Bind the output results of the operation to the appropriate images in the GPU materials. Every
   * image in the GPU material corresponds to one of the outputs of the operation, an output whose
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
   * material, the input will be exposed as an input to the GPU material operation and linked to
   * it. While if the input is linked to a node that is part of the GPU material, then it is linked
   * to that node in the GPU material node graph. */
  void link_material_node_inputs(DNode node, GPUMaterial *material);

  /* Given the input of a node that is part of the GPU material which is linked to the given output
   * of a node that is also part of the GPU material, map the output link of the GPU node stack of
   * the output to the input link of the GPU node stack of the input. This essentially establishes
   * the needed links in the GPU material node graph. */
  void map_material_node_input(DInputSocket input, DOutputSocket output);

  /* Given the input of a node that is part of the GPU material which is linked to the given output
   * of a node that is not part of the GPU material, declare a new input to the operation and link
   * it appropriately as detailed in the discussion above. */
  void declare_material_input_if_needed(DInputSocket input,
                                        DOutputSocket output,
                                        GPUMaterial *material);

  /* Link the input node stack corresponding to the given input to an input loader GPU material
   * node sampling the material texture corresponding to the given output. */
  void link_material_input_loader(DInputSocket input, DOutputSocket output, GPUMaterial *material);

  /* Populate the output results of the GPU material operation for outputs of the given node that
   * are linked to nodes outside of the GPU material. */
  void populate_results_for_material_node(DNode node, GPUMaterial *material);

  /* Given the output of a node that is part of the GPU material which is linked to an input of a
   * node that is not part of the GPU material, declare a new output to the operation and link
   * it appropriately as detailed in the discussion above. */
  void populate_material_result(DOutputSocket output, GPUMaterial *material);

  /* A static callback method of interface GPUCodegenCallbackFn that is passed to
   * GPU_material_from_callbacks to create the shader create info of the GPU material. The thunk
   * parameter will be a pointer to the instance of GPUMaterialOperation that is being compiled.
   * This method setup the shader create info as a compute shader and sets its generate source
   * based on the GPU material code generator output. */
  static void generate_material(void *thunk,
                                GPUMaterial *material,
                                GPUCodegenOutput *code_generator);
};

}  // namespace blender::viewport_compositor
