/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include <limits>
#include <memory>
#include <string>

#include "BLI_assert.h"
#include "BLI_hash.hh"
#include "BLI_listbase.h"
#include "BLI_map.hh"
#include "BLI_math_vec_types.hh"
#include "BLI_math_vector.h"
#include "BLI_stack.hh"
#include "BLI_transformation_2d.hh"
#include "BLI_utildefines.h"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "BKE_node.h"

#include "DNA_node_types.h"
#include "DNA_scene_types.h"

#include "GPU_compute.h"
#include "GPU_material.h"
#include "GPU_shader.h"
#include "GPU_texture.h"
#include "GPU_uniform_buffer.h"

#include "gpu_shader_create_info.hh"

#include "IMB_colormanagement.h"

#include "NOD_derived_node_tree.hh"
#include "NOD_node_declaration.hh"

#include "MEM_guardedalloc.h"

#include "VPC_compositor_execute.hh"
#include "VPC_context.hh"
#include "VPC_domain.hh"
#include "VPC_input_descriptor.hh"
#include "VPC_operation.hh"
#include "VPC_result.hh"
#include "VPC_scheduler.hh"
#include "VPC_texture_pool.hh"
#include "VPC_utils.hh"

namespace blender::viewport_compositor {

/* --------------------------------------------------------------------
 * Node Operation.
 */

using namespace nodes::derived_node_tree_types;

static ResultType get_node_socket_result_type(const SocketRef *socket)
{
  switch (socket->bsocket()->type) {
    case SOCK_FLOAT:
      return ResultType::Float;
    case SOCK_VECTOR:
      return ResultType::Vector;
    case SOCK_RGBA:
      return ResultType::Color;
    default:
      BLI_assert_unreachable();
      return ResultType::Float;
  }
}

NodeOperation::NodeOperation(Context &context, DNode node) : Operation(context), node_(node)
{
  /* Populate the output results. */
  for (const OutputSocketRef *output : node->outputs()) {
    const ResultType result_type = get_node_socket_result_type(output);
    const Result result = Result(result_type, texture_pool());
    populate_result(output->identifier(), result);
  }

  /* Populate the input descriptors. */
  for (const InputSocketRef *input : node->inputs()) {
    InputDescriptor input_descriptor;
    input_descriptor.type = get_node_socket_result_type(input);
    const nodes::SocketDeclarationPtr &socket_declaration =
        input->node().declaration()->inputs()[input->index()];
    input_descriptor.domain_priority = socket_declaration->compositor_domain_priority();
    input_descriptor.expects_single_value = socket_declaration->compositor_expects_single_value();
    declare_input_descriptor(input->identifier(), input_descriptor);
  }

  populate_results_for_unlinked_inputs();
}

const bNode &NodeOperation::node() const
{
  return *node_->bnode();
}

bool NodeOperation::is_output_needed(StringRef identifier) const
{
  DOutputSocket output = node_.output_by_identifier(identifier);
  if (output->logically_linked_sockets().is_empty()) {
    return false;
  }
  return true;
}

void NodeOperation::pre_execute()
{
  /* For each unlinked input socket, allocate a single value and set the value to the socket's
   * default value. */
  for (const Map<StringRef, DInputSocket>::Item &item : unlinked_inputs_sockets_.items()) {
    Result &result = get_input(item.key);
    DInputSocket input = item.value;
    result.allocate_single_value();
    switch (result.type()) {
      case ResultType::Float:
        result.set_float_value(input->default_value<bNodeSocketValueFloat>()->value);
        continue;
      case ResultType::Vector:
        result.set_vector_value(float3(input->default_value<bNodeSocketValueVector>()->value));
        continue;
      case ResultType::Color:
        result.set_color_value(float4(input->default_value<bNodeSocketValueRGBA>()->value));
        continue;
    }
  }
}

void NodeOperation::populate_results_for_unlinked_inputs()
{
  for (const InputSocketRef *input_ref : node_->inputs()) {
    const DInputSocket input{node_.context(), input_ref};
    DSocket origin = get_node_input_origin_socket(input);

    /* Input is linked, skip it. If the origin is an input, that means the input is connected to an
     * unlinked input of a group input node, hence why we check if the origin is an output. */
    if (origin->is_output()) {
      continue;
    }

    /* Construct a result of an appropriate type, add it to the results vector, and map the input
     * to it. */
    const ResultType result_type = get_node_socket_result_type(origin.socket_ref());
    unlinked_inputs_results_.append(std::make_unique<Result>(result_type, texture_pool()));
    map_input_to_result(input->identifier(), unlinked_inputs_results_.last().get());

    /* Map the input to the socket to later allocate and initialize its value. */
    const DInputSocket origin_input{origin.context(), &origin->as_input()};
    unlinked_inputs_sockets_.add_new(input->identifier(), origin_input);
  }
}

/* --------------------------------------------------------------------
 * GPU Material Node.
 */

GPUMaterialNode::GPUMaterialNode(DNode node) : node_(node)
{
  populate_inputs();
  populate_outputs();
}

GPUNodeStack *GPUMaterialNode::get_inputs_array()
{
  return inputs_.data();
}

GPUNodeStack *GPUMaterialNode::get_outputs_array()
{
  return outputs_.data();
}

bNode &GPUMaterialNode::node() const
{
  return *node_->bnode();
}

static eGPUType gpu_type_from_socket_type(eNodeSocketDatatype type)
{
  switch (type) {
    case SOCK_FLOAT:
      return GPU_FLOAT;
    case SOCK_VECTOR:
      return GPU_VEC3;
    case SOCK_RGBA:
      return GPU_VEC4;
    default:
      BLI_assert_unreachable();
      return GPU_NONE;
  }
}

static void gpu_stack_vector_from_socket(float *vector, const SocketRef *socket)
{
  switch (socket->bsocket()->type) {
    case SOCK_FLOAT:
      vector[0] = socket->default_value<bNodeSocketValueFloat>()->value;
      return;
    case SOCK_VECTOR:
      copy_v3_v3(vector, socket->default_value<bNodeSocketValueVector>()->value);
      return;
    case SOCK_RGBA:
      copy_v4_v4(vector, socket->default_value<bNodeSocketValueRGBA>()->value);
      return;
    default:
      BLI_assert_unreachable();
  }
}

static void populate_gpu_node_stack(DSocket socket, GPUNodeStack &stack)
{
  /* Make sure this stack is not marked as the end of the stack array. */
  stack.end = false;
  /* This will be initialized later by the GPU material compiler or the compile method. */
  stack.link = nullptr;
  /* Socket type and its corresponding GPU type. */
  stack.sockettype = socket->bsocket()->type;
  stack.type = gpu_type_from_socket_type((eNodeSocketDatatype)socket->bsocket()->type);

  if (socket->is_input()) {
    /* Get the origin socket connected to the input if any. */
    const DInputSocket input{socket.context(), &socket->as_input()};
    DSocket origin = get_node_input_origin_socket(input);
    /* The input is linked if the origin socket is not null and is an output socket. Had it been an
     * input socket, then it is an unlinked input of a group input node. */
    stack.hasinput = origin->is_output();
    /* Get the socket value from the origin if it is an input, because then it would be an unlinked
     * input of a group input node, otherwise, get the value from the socket itself. */
    if (origin->is_input()) {
      gpu_stack_vector_from_socket(stack.vec, origin.socket_ref());
    }
    else {
      gpu_stack_vector_from_socket(stack.vec, socket.socket_ref());
    }
  }
  else {
    stack.hasoutput = socket->is_logically_linked();
    /* Populate the stack vector even for outputs because some nodes store their properties in the
     * default values of their outputs. */
    gpu_stack_vector_from_socket(stack.vec, socket.socket_ref());
  }
}

void GPUMaterialNode::populate_inputs()
{
  /* Reserve a stack for each input in addition to an extra stack at the end to mark the end of the
   * array, as this is what the GPU module functions expect. */
  inputs_.resize(node_->inputs().size() + 1);
  inputs_.last().end = true;

  for (int i = 0; i < node_->inputs().size(); i++) {
    populate_gpu_node_stack(node_.input(i), inputs_[i]);
  }
}

void GPUMaterialNode::populate_outputs()
{
  /* Reserve a stack for each output in addition to an extra stack at the end to mark the end of
   * the array, as this is what the GPU module functions expect. */
  outputs_.resize(node_->outputs().size() + 1);
  outputs_.last().end = true;

  for (int i = 0; i < node_->outputs().size(); i++) {
    populate_gpu_node_stack(node_.output(i), outputs_[i]);
  }
}

/* --------------------------------------------------------------------
 * GPU Material Operation.
 */

GPUMaterialOperation::GPUMaterialOperation(Context &context, Schedule &sub_schedule)
    : Operation(context), sub_schedule_(sub_schedule)
{
  material_ = GPU_material_from_callbacks(
      &setup_material, &compile_material, &generate_material, this);
}

GPUMaterialOperation::~GPUMaterialOperation()
{
  for (const GPUMaterialNode *gpu_material_node : gpu_material_nodes_.values()) {
    delete gpu_material_node;
  }
  GPU_material_free_single(material_);
}

void GPUMaterialOperation::execute()
{
  for (StringRef identifier : output_socket_to_output_identifier_map_.values()) {
    Result &result = get_result(identifier);
    result.allocate_texture(compute_domain());
  }

  GPUShader *shader = GPU_material_get_shader(material_);
  GPU_shader_bind(shader);

  bind_material_resources(shader);
  bind_inputs(shader);
  bind_outputs(shader);

  const int2 size = compute_domain().size;
  GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

  GPU_texture_unbind_all();
  GPU_texture_image_unbind_all();
  GPU_uniformbuf_unbind_all();
  GPU_shader_unbind();
}

StringRef GPUMaterialOperation::get_output_identifier_from_output_socket(DOutputSocket output)
{
  return output_socket_to_output_identifier_map_.lookup(output);
}

InputIdentifierToOutputSocketMap &GPUMaterialOperation::get_input_identifier_to_output_socket_map()
{
  return input_identifier_to_output_socket_map_;
}

void GPUMaterialOperation::bind_material_resources(GPUShader *shader)
{
  /* Bind the uniform buffer of the material if it exists. */
  GPUUniformBuf *ubo = GPU_material_uniform_buffer_get(material_);
  if (ubo) {
    GPU_uniformbuf_bind(ubo, GPU_shader_get_uniform_block_binding(shader, GPU_UBO_BLOCK_NAME));
  }

  /* Bind color band textures needed by the material. */
  ListBase textures = GPU_material_textures(material_);
  LISTBASE_FOREACH (GPUMaterialTexture *, texture, &textures) {
    if (texture->colorband) {
      const int texture_image_unit = GPU_shader_get_texture_binding(shader, texture->sampler_name);
      GPU_texture_bind(*texture->colorband, texture_image_unit);
    }
  }
}

void GPUMaterialOperation::bind_inputs(GPUShader *shader)
{
  ListBase textures = GPU_material_textures(material_);
  LISTBASE_FOREACH (GPUMaterialTexture *, texture, &textures) {
    /* Input textures are those that do not reference an image or a color band texture. */
    if (!texture->colorband && !texture->ima) {
      get_input(texture->sampler_name).bind_as_texture(shader, texture->sampler_name);
    }
  }
}

void GPUMaterialOperation::bind_outputs(GPUShader *shader)
{
  ListBase images = GPU_material_images(material_);
  LISTBASE_FOREACH (GPUMaterialImage *, image, &images) {
    get_result(image->name_in_shader).bind_as_image(shader, image->name_in_shader);
  }
}

void GPUMaterialOperation::setup_material(void *thunk, GPUMaterial *material)
{
  GPU_material_is_compute_set(material, true);
}

void GPUMaterialOperation::compile_material(void *thunk, GPUMaterial *material)
{
  GPUMaterialOperation *operation = static_cast<GPUMaterialOperation *>(thunk);
  for (DNode node : operation->sub_schedule_) {
    /* Instantiate a GPU material node for the node and add it to the gpu_material_nodes_ map. */
    GPUMaterialNode *gpu_node = node->typeinfo()->get_compositor_gpu_material_node(node);
    operation->gpu_material_nodes_.add_new(node, gpu_node);

    /* Link the inputs of the material node if needed. */
    operation->link_material_node_inputs(node, material);

    /* Compile the node itself. */
    gpu_node->compile(material);

    /* Populate the output results for the material node if needed. */
    operation->populate_results_for_material_node(node, material);
  }
}

void GPUMaterialOperation::link_material_node_inputs(DNode node, GPUMaterial *material)
{
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};

    /* Get the origin socket of this input, which will be an output socket if the input is linked
     * to an output. */
    DSocket origin = get_node_input_origin_socket(input);

    /* If the origin socket is an input, that means the input is unlinked. Unlinked inputs will be
     * linked by the node compile method, so skip here. */
    if (origin->is_input()) {
      continue;
    }

    /* Now that we know the origin is an output, construct a derived output from it. */
    const DOutputSocket output{origin.context(), &origin->as_output()};

    /* If the origin node is part of the GPU material, then just map the output stack link to the
     * input stack link. */
    if (sub_schedule_.contains(output.node())) {
      map_material_node_input(input, output);
      continue;
    }

    /* Otherwise, the origin node is not part of the GPU material, so an input to the GPU material
     * operation must be declared. */
    declare_material_input_if_needed(input, output, material);

    link_material_input_loader(input, output, material);
  }
}

void GPUMaterialOperation::map_material_node_input(DInputSocket input, DOutputSocket output)
{
  /* Get the GPU node stack of the output. */
  GPUMaterialNode &output_node = *gpu_material_nodes_.lookup(output.node());
  GPUNodeStack &output_stack = output_node.get_outputs_array()[output->index()];

  /* Get the GPU node stack of the input. */
  GPUMaterialNode &input_node = *gpu_material_nodes_.lookup(input.node());
  GPUNodeStack &input_stack = input_node.get_inputs_array()[input->index()];

  /* Map the output link to the input link. */
  input_stack.link = output_stack.link;
}

static const char *get_load_function_name(DInputSocket input)
{
  switch (input->bsocket()->type) {
    case SOCK_FLOAT:
      return "load_input_float";
    case SOCK_VECTOR:
      return "load_input_vector";
    case SOCK_RGBA:
      return "load_input_color";
    default:
      BLI_assert_unreachable();
      return "";
  }
}

void GPUMaterialOperation::declare_material_input_if_needed(DInputSocket input,
                                                            DOutputSocket output,
                                                            GPUMaterial *material)
{
  /* An input was already declared for that same output, so no need to declare it again. */
  if (output_socket_to_input_link_map_.contains(output)) {
    return;
  }

  /* Add a new input texture to the GPU material. */
  GPUNodeLink *input_texture_link = GPU_texture(material, GPU_SAMPLER_DEFAULT);

  /* Map the output socket to the input texture link that was created for it. */
  output_socket_to_input_link_map_.add(output, input_texture_link);

  /* Construct an input descriptor from the socket declaration. */
  InputDescriptor input_descriptor;
  input_descriptor.type = get_node_socket_result_type(input.socket_ref());
  const nodes::SocketDeclarationPtr &socket_declaration =
      input.node()->declaration()->inputs()[input->index()];
  input_descriptor.domain_priority = socket_declaration->compositor_domain_priority();
  input_descriptor.expects_single_value = socket_declaration->compositor_expects_single_value();

  /* Declare the input descriptor. */
  StringRef identifier = GPU_material_get_link_texture(input_texture_link)->sampler_name;
  declare_input_descriptor(identifier, input_descriptor);

  /* Map the operation input to the output socket it is linked to. */
  input_identifier_to_output_socket_map_.add_new(identifier, output);
}

void GPUMaterialOperation::link_material_input_loader(DInputSocket input,
                                                      DOutputSocket output,
                                                      GPUMaterial *material)
{
  /* Link the input node stack to an input loader sampling the input texture. */
  GPUMaterialNode &node = *gpu_material_nodes_.lookup(input.node());
  GPUNodeStack &stack = node.get_inputs_array()[input->index()];
  const char *load_function_name = get_load_function_name(input);
  GPUNodeLink *input_texture_link = output_socket_to_input_link_map_.lookup(output);
  GPU_link(material, load_function_name, input_texture_link, &stack.link);
}

void GPUMaterialOperation::populate_results_for_material_node(DNode node, GPUMaterial *material)
{
  for (const OutputSocketRef *output_ref : node->outputs()) {
    const DOutputSocket output{node.context(), output_ref};

    /* Go over the target inputs that are linked to this output. If any of the target nodes is not
     * part of the GPU material, then an output result needs to be populated. */
    bool need_to_populate_result = false;
    output.foreach_target_socket(
        [&](DInputSocket target, const DOutputSocket::TargetSocketPathInfo &path_info) {
          /* Target node is not part of the GPU material. */
          if (!sub_schedule_.contains(target.node())) {
            need_to_populate_result = true;
          }
        });

    if (need_to_populate_result) {
      populate_material_result(output, material);
    }
  }
}

static const char *get_store_function_name(ResultType type)
{
  switch (type) {
    case ResultType::Float:
      return "store_output_float";
    case ResultType::Vector:
      return "store_output_vector";
    case ResultType::Color:
      return "store_output_color";
  }

  BLI_assert_unreachable();
  return nullptr;
}

static eGPUTextureFormat texture_format_from_result_type(ResultType type)
{
  switch (type) {
    case ResultType::Float:
      return GPU_R16F;
    case ResultType::Vector:
      return GPU_RGBA16F;
    case ResultType::Color:
      return GPU_RGBA16F;
  }

  BLI_assert_unreachable();
  return GPU_RGBA16F;
}

void GPUMaterialOperation::populate_material_result(DOutputSocket output, GPUMaterial *material)
{
  /* Construct a result of an appropriate type. */
  const ResultType result_type = get_node_socket_result_type(output.socket_ref());
  const Result result = Result(result_type, texture_pool());

  /* Add a new output image to the GPU material. */
  const eGPUTextureFormat format = texture_format_from_result_type(result_type);
  GPUNodeLink *output_image_link = GPU_image_texture(material, format);

  /* Add the result. */
  StringRef identifier = GPU_material_get_link_image(output_image_link)->name_in_shader;
  populate_result(identifier, result);

  /* Map the output socket to the identifier of the result. */
  output_socket_to_output_identifier_map_.add_new(output, identifier);

  /* Link the output node stack to an output storer storing in the newly added output image. */
  GPUMaterialNode &node = *gpu_material_nodes_.lookup(output.node());
  GPUNodeLink *output_link = node.get_outputs_array()[output->index()].link;
  const char *store_function_name = get_store_function_name(result_type);
  GPU_link(material, store_function_name, output_image_link, output_link);
}

void GPUMaterialOperation::generate_material(void *thunk,
                                             GPUMaterial *material,
                                             GPUCodegenOutput *code_generator_output)
{
  gpu::shader::ShaderCreateInfo &info = *reinterpret_cast<gpu::shader::ShaderCreateInfo *>(
      code_generator_output->create_info);

  /* The GPU material adds resources without explicit locations, so make sure it is done by the
   * shader creator. */
  info.auto_resource_location(true);

  info.local_group_size(16, 16);

  /* Add implementation for implicit conversion operations inserted by the code generator. */
  info.typedef_source("gpu_shader_compositor_type_conversion.glsl");

  /* Add the compute source code of the generator in a main function and set it as the generated
   * compute source of the shader create info. */
  std::string source = "void main()\n{\n" + std::string(code_generator_output->compute) + "}\n";
  info.compute_source_generated = source;
}

/* --------------------------------------------------------------------
 * GPU Material Compile Group.
 */

/* A node is a GPU material node if it defines a method to get a GPU material node operation. */
static bool is_gpu_material_node(DNode node)
{
  return node->typeinfo()->get_compositor_gpu_material_node;
}

void GPUMaterialCompileGroup::add(DNode node)
{
  sub_schedule_.add_new(node);
}

bool GPUMaterialCompileGroup::is_complete(DNode next_node)
{
  /* Sub schedule is empty, so the group is not complete. */
  if (sub_schedule_.is_empty()) {
    return false;
  }

  /* If the next node is not a GPU material node, then it can't be added to the group and the group
   * is considered complete. */
  if (!is_gpu_material_node(next_node)) {
    return true;
  }

  /* If the next node has inputs that are linked to nodes that are not part of this group, then it
   * can't be added to the group and the group is considered complete. */
  for (const InputSocketRef *input_ref : next_node->inputs()) {
    const DInputSocket input{next_node.context(), input_ref};
    DSocket origin = get_node_input_origin_socket(input);

    /* If the origin is an output, then the input is linked to the origin node. Check if the origin
     * node is not part of the group. */
    if (origin->is_output() && !sub_schedule_.contains(origin.node())) {
      return true;
    }
  }

  /* The next node can be added to the group, so it is not complete yet. */
  return false;
}

void GPUMaterialCompileGroup::reset()
{
  sub_schedule_.clear();
}

VectorSet<DNode> &GPUMaterialCompileGroup::get_sub_schedule()
{
  return sub_schedule_;
}

/* --------------------------------------------------------------------
 * Compiler.
 */

Compiler::Compiler(Context &context, bNodeTree *node_tree)
    : context_(context), tree_(*node_tree, tree_ref_map_)
{
}

Compiler::~Compiler()
{
  for (const Operation *operation : operations_stream_) {
    delete operation;
  }
}

void Compiler::compile()
{
  const Schedule schedule = compute_schedule(tree_);
  for (const DNode &node : schedule) {
    /* First check if the material compile group is complete, and if it is, compile it. */
    if (gpu_material_compile_group_.is_complete(node)) {
      compile_gpu_material_group();
    }

    /* If the node is a GPU material node, add it to the GPU material compile group, it will be
     * compiled later once the group is complete, see previous statement. */
    if (is_gpu_material_node(node)) {
      gpu_material_compile_group_.add(node);
      continue;
    }

    /* Otherwise, compile the node into a standard node operation. */
    compile_standard_node(node);
  }
}

OperationsStream &Compiler::operations_stream()
{
  return operations_stream_;
}

void Compiler::compile_standard_node(DNode node)
{
  /* Get an instance of the node's compositor operation and add it to both the operations stream
   * and the node operations map. This instance should be freed by the compiler when it is no
   * longer needed. */
  NodeOperation *operation = node->typeinfo()->get_compositor_operation(context_, node);
  operations_stream_.append(operation);
  node_operations_.add_new(node, operation);

  /* Map the inputs of the operation to the results of the outputs they are linked to. */
  map_node_operation_inputs_to_results(node, operation);
}

void Compiler::map_node_operation_inputs_to_results(DNode node, NodeOperation *operation)
{
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};

    /* Get the origin socket of this input, which will be an output socket if the input is linked
     * to an output. */
    DSocket origin = get_node_input_origin_socket(input);

    /* If the origin socket is an input, that means the input is unlinked. Unlinked inputs are
     * mapped internally to internal results, so skip here. */
    if (origin->is_input()) {
      continue;
    }

    /* Now that we know the origin is an output, construct a derived output from it. */
    const DOutputSocket output{origin.context(), &origin->as_output()};

    /* Map the input to the result we got from the output. */
    Result &result = get_output_socket_result(output);
    operation->map_input_to_result(input->identifier(), &result);
  }
}

void Compiler::compile_gpu_material_group()
{
  /* Get the sub schedule that is part of the GPU material group, instantiate a GPU Material
   * Operation from it, and add it to the operations stream. This instance should be freed by the
   * compiler when it is no longer needed. */
  Schedule &sub_schedule = gpu_material_compile_group_.get_sub_schedule();
  GPUMaterialOperation *operation = new GPUMaterialOperation(context_, sub_schedule);
  operations_stream_.append(operation);

  /* Map each of the nodes in the sub schedule to the compiled operation. */
  for (DNode node : sub_schedule) {
    gpu_material_operations_.add_new(node, operation);
  }

  /* Map the inputs of the operation to the results of the outputs they are linked to. */
  map_gpu_material_operation_inputs_to_results(operation);

  /* Reset the compile group to make it ready to track the next potential group. */
  gpu_material_compile_group_.reset();
}

void Compiler::map_gpu_material_operation_inputs_to_results(GPUMaterialOperation *operation)
{
  /* For each input of the operation, retrieve the result of the output linked to it, and map the
   * result to the input. */
  InputIdentifierToOutputSocketMap &map = operation->get_input_identifier_to_output_socket_map();
  for (const InputIdentifierToOutputSocketMap::Item &item : map.items()) {
    /* Map the input to the result we got from the output. */
    Result &result = get_output_socket_result(item.value);
    operation->map_input_to_result(item.key, &result);
  }
}

Result &Compiler::get_output_socket_result(DOutputSocket output)
{
  /* The output belongs to a node that was compiled into a standard node operation, so return a
   * reference to the result from that operation using the output identifier. */
  if (node_operations_.contains(output.node())) {
    NodeOperation *operation = node_operations_.lookup(output.node());
    return operation->get_result(output->identifier());
  }

  /* Otherwise, the output belongs to a node that was compiled into a GPU material operation, so
   * retrieve the internal identifier of that output and return a reference to the result from
   * that operation using the retrieved identifier. */
  GPUMaterialOperation *operation = gpu_material_operations_.lookup(output.node());
  return operation->get_result(operation->get_output_identifier_from_output_socket(output));
}

/* --------------------------------------------------------------------
 * Evaluator.
 */

Evaluator::Evaluator(Context &context, bNodeTree *node_tree) : compiler_(context, node_tree)
{
}

void Evaluator::compile()
{
  compiler_.compile();
}

void Evaluator::evaluate()
{
  for (Operation *operation : compiler_.operations_stream()) {
    operation->evaluate();
  }
}

}  // namespace blender::viewport_compositor
