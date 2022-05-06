/* SPDX-License-Identifier: GPL-2.0-or-later */

#include <memory>

#include "BLI_listbase.h"
#include "BLI_string_ref.hh"

#include "GPU_material.h"
#include "GPU_shader.h"
#include "GPU_texture.h"
#include "GPU_uniform_buffer.h"

#include "gpu_shader_create_info.hh"

#include "NOD_derived_node_tree.hh"
#include "NOD_node_declaration.hh"

#include "VPC_context.hh"
#include "VPC_gpu_material_node.hh"
#include "VPC_gpu_material_operation.hh"
#include "VPC_operation.hh"
#include "VPC_result.hh"
#include "VPC_scheduler.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

GPUMaterialOperation::GPUMaterialOperation(Context &context, SubSchedule &sub_schedule)
    : Operation(context), sub_schedule_(sub_schedule)
{
  material_ = GPU_material_from_callbacks(
      &setup_material, &compile_material, &generate_material, this);
  GPU_material_status_set(material_, GPU_MAT_QUEUED);
  GPU_material_compile(material_);
}

GPUMaterialOperation::~GPUMaterialOperation()
{
  GPU_material_free_single(material_);
}

void GPUMaterialOperation::execute()
{
  /* Allocate all the outputs of the operation on its computed domain. */
  const Domain domain = compute_domain();
  for (StringRef identifier : output_sockets_to_output_identifiers_map_.values()) {
    Result &result = get_result(identifier);
    result.allocate_texture(domain);
  }

  GPUShader *shader = GPU_material_get_shader(material_);
  GPU_shader_bind(shader);

  bind_material_resources(shader);
  bind_inputs(shader);
  bind_outputs(shader);

  compute_dispatch_global(shader, domain.size);

  GPU_texture_unbind_all();
  GPU_texture_image_unbind_all();
  GPU_uniformbuf_unbind_all();
  GPU_shader_unbind();
}

StringRef GPUMaterialOperation::get_output_identifier_from_output_socket(DOutputSocket output)
{
  return output_sockets_to_output_identifiers_map_.lookup(output);
}

InputsToLinkedOutputsMap &GPUMaterialOperation::get_inputs_to_linked_outputs_map()
{
  return inputs_to_linked_outputs_map_;
}

void GPUMaterialOperation::compute_results_reference_counts(const Schedule &schedule)
{
  for (const OutputSocketsToOutputIdentifiersMap::Item &item :
       output_sockets_to_output_identifiers_map_.items()) {

    const int reference_count = number_of_inputs_linked_to_output_conditioned(
        item.key, [&](DInputSocket input) { return schedule.contains(input.node()); });

    get_result(item.value).set_initial_reference_count(reference_count);
  }
}

void GPUMaterialOperation::bind_material_resources(GPUShader *shader)
{
  /* Bind the uniform buffer of the material if it exists. It may not exist if the GPU material has
   * no uniforms. */
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
  for (const GPUMaterialTexture *material_texture : output_to_material_texture_map_.values()) {
    const char *sampler_name = material_texture->sampler_name;
    get_input(sampler_name).bind_as_texture(shader, sampler_name);
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
    operation->gpu_material_nodes_.add_new(node, std::unique_ptr<GPUMaterialNode>(gpu_node));

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

    /* Get the output linked to the input. If it is null, that means the input is unlinked.
     * Unlinked inputs are linked by the node compile method, so skip this here. */
    const DOutputSocket output = get_output_linked_to_input(input);
    if (!output) {
      continue;
    }

    /* If the origin node is part of the GPU material, then just map the output stack link to the
     * input stack link. */
    if (sub_schedule_.contains(output.node())) {
      map_material_node_input(input, output);
      continue;
    }

    /* Otherwise, the origin node is not part of the GPU material, so an input to the GPU material
     * operation must be declared if it wasn't declared for the same output already. */
    declare_material_input_if_needed(input, output, material);

    /* Link the input to an input loader GPU material node sampling the result of the output. */
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
  if (output_to_material_texture_map_.contains(output)) {
    return;
  }

  /* Add a new material texture to the GPU material. */
  GPUMaterialTexture *material_texture = GPU_material_add_texture(material, GPU_SAMPLER_DEFAULT);

  /* Map the output socket to the material texture that was created for it. */
  output_to_material_texture_map_.add(output, material_texture);

  /* Declare the input descriptor using the name of the sampler in the shader as the identifier. */
  StringRef identifier = material_texture->sampler_name;
  const InputDescriptor input_descriptor = input_descriptor_from_input_socket(input.socket_ref());
  declare_input_descriptor(identifier, input_descriptor);

  /* Map the operation input to the output socket it is linked to. */
  inputs_to_linked_outputs_map_.add_new(identifier, output);
}

void GPUMaterialOperation::link_material_input_loader(DInputSocket input,
                                                      DOutputSocket output,
                                                      GPUMaterial *material)
{
  /* Create a link from the material texture that corresponds to the given output. */
  GPUMaterialTexture *material_texture = output_to_material_texture_map_.lookup(output);
  GPUNodeLink *input_texture_link = GPU_image_from_material_texture(material_texture);

  /* Get the node stack of the input. */
  GPUMaterialNode &node = *gpu_material_nodes_.lookup(input.node());
  GPUNodeStack &stack = node.get_inputs_array()[input->index()];

  /* Link the input node stack to an input loader sampling the input texture. */
  const char *load_function_name = get_load_function_name(input);
  GPU_link(material, load_function_name, input_texture_link, &stack.link);
}

void GPUMaterialOperation::populate_results_for_material_node(DNode node, GPUMaterial *material)
{
  for (const OutputSocketRef *output_ref : node->outputs()) {
    const DOutputSocket output{node.context(), output_ref};

    /* If any of the nodes linked to the output are not part of the GPU material, then an output
     * result needs to be populated. */
    const bool need_to_populate_result = is_output_linked_to_node_conditioned(
        output, [&](DNode node) { return !sub_schedule_.contains(node); });

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

  /* Add a new material image to the GPU material. */
  const eGPUTextureFormat format = texture_format_from_result_type(result_type);
  GPUMaterialImage *material_image = GPU_material_add_image_texture(material, format);

  /* Add the result using the name of the image in the shader as the identifier. */
  StringRef identifier = material_image->name_in_shader;
  populate_result(identifier, result);

  /* Map the output socket to the identifier of the result. */
  output_sockets_to_output_identifiers_map_.add_new(output, identifier);

  /* Create a link from the material image that corresponds to the given output. */
  GPUNodeLink *output_image_link = GPU_image_texture_from_material_image(material_image);

  /* Get the node stack of the output. */
  GPUMaterialNode &node = *gpu_material_nodes_.lookup(output.node());
  GPUNodeLink *output_link = node.get_outputs_array()[output->index()].link;

  /* Link the output node stack to an output storer storing in the newly added image. */
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

}  // namespace blender::viewport_compositor
