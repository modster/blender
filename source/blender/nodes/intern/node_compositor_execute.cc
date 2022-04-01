/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include <limits>
#include <string>

#include "BLI_assert.h"
#include "BLI_hash.hh"
#include "BLI_listbase.h"
#include "BLI_map.hh"
#include "BLI_math_vec_types.hh"
#include "BLI_math_vector.h"
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

#include "../../gpu/intern/gpu_shader_create_info.hh"

#include "IMB_colormanagement.h"

#include "NOD_compositor_execute.hh"
#include "NOD_derived_node_tree.hh"
#include "NOD_node_declaration.hh"

#include "MEM_guardedalloc.h"

namespace blender::viewport_compositor {

/* --------------------------------------------------------------------
 * Texture Pool.
 */

TexturePoolKey::TexturePoolKey(int2 size, eGPUTextureFormat format) : size(size), format(format)
{
}

TexturePoolKey::TexturePoolKey(const GPUTexture *texture)
{
  size = int2{GPU_texture_width(texture), GPU_texture_height(texture)};
  format = GPU_texture_format(texture);
}

uint64_t TexturePoolKey::hash() const
{
  return get_default_hash_3(size.x, size.y, format);
}

bool operator==(const TexturePoolKey &a, const TexturePoolKey &b)
{
  return a.size == b.size && a.format == b.format;
}

GPUTexture *TexturePool::acquire(int2 size, eGPUTextureFormat format)
{
  const TexturePoolKey key = TexturePoolKey(size, format);
  Vector<GPUTexture *> &available_textures = textures_.lookup_or_add_default(key);
  if (available_textures.is_empty()) {
    return allocate_texture(size, format);
  }
  return available_textures.pop_last();
}

GPUTexture *TexturePool::acquire_color(int2 size)
{
  return acquire(size, GPU_RGBA16F);
}

/* Vectors are and should be stored in RGBA textures. */
GPUTexture *TexturePool::acquire_vector(int2 size)
{
  return acquire(size, GPU_RGBA16F);
}

GPUTexture *TexturePool::acquire_float(int2 size)
{
  return acquire(size, GPU_R16F);
}

void TexturePool::release(GPUTexture *texture)
{
  textures_.lookup(TexturePoolKey(texture)).append(texture);
}

/* --------------------------------------------------------------------
 * Context.
 */

Context::Context(TexturePool &texture_pool) : texture_pool_(texture_pool)
{
}

TexturePool &Context::texture_pool()
{
  return texture_pool_;
}

/* --------------------------------------------------------------------
 * Domain.
 */

Domain::Domain(int2 size) : size(size), transformation(Transformation2D::identity())
{
}

Domain::Domain(int2 size, Transformation2D transformation)
    : size(size), transformation(transformation)
{
}

void Domain::transform(const Transformation2D &input_transformation)
{
  transformation = input_transformation * transformation;
}

Domain Domain::identity()
{
  return Domain(int2(1), Transformation2D::identity());
}

/* Only compare the size and transformation members, as other members only describe the method of
 * realization on another domain, which is not technically a proprty of the domain. */
bool operator==(const Domain &a, const Domain &b)
{
  return a.size == b.size && a.transformation == b.transformation;
}

bool operator!=(const Domain &a, const Domain &b)
{
  return !(a == b);
}

/* --------------------------------------------------------------------
 * Result.
 */

Result::Result(ResultType type, TexturePool &texture_pool)
    : type_(type), texture_pool_(&texture_pool)
{
}

void Result::allocate_texture(Domain domain)
{
  is_single_value_ = false;
  switch (type_) {
    case ResultType::Float:
      texture_ = texture_pool_->acquire_float(domain.size);
      break;
    case ResultType::Vector:
      texture_ = texture_pool_->acquire_vector(domain.size);
      break;
    case ResultType::Color:
      texture_ = texture_pool_->acquire_color(domain.size);
      break;
  }
  domain_ = domain;
}

void Result::allocate_single_value()
{
  is_single_value_ = true;
  /* Single values are stored in 1x1 textures. */
  const int2 texture_size{1, 1};
  switch (type_) {
    case ResultType::Float:
      texture_ = texture_pool_->acquire_float(texture_size);
      break;
    case ResultType::Vector:
      texture_ = texture_pool_->acquire_vector(texture_size);
      break;
    case ResultType::Color:
      texture_ = texture_pool_->acquire_color(texture_size);
      break;
  }
  domain_ = Domain::identity();
}

void Result::bind_as_texture(GPUShader *shader, const char *texture_name) const
{
  /* Make sure any prior writes to the texture are reflected before reading from it. */
  GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);

  const int texture_image_unit = GPU_shader_get_texture_binding(shader, texture_name);
  GPU_texture_bind(texture_, texture_image_unit);
}

void Result::bind_as_image(GPUShader *shader, const char *image_name) const
{
  const int image_unit = GPU_shader_get_texture_binding(shader, image_name);
  GPU_texture_image_bind(texture_, image_unit);
}

void Result::unbind_as_texture() const
{
  GPU_texture_unbind(texture_);
}

void Result::unbind_as_image() const
{
  GPU_texture_image_unbind(texture_);
}

void Result::pass_through(Result &target)
{
  /* Increment the reference count of the master by the original reference count of the target. */
  increment_reference_count(target.reference_count());
  /* Copy the result to the target and set its master. */
  target = *this;
  target.master_ = this;
}

void Result::transform(const Transformation2D &transformation)
{
  domain_.transform(transformation);
}

void Result::set_realization_interpolation(Interpolation interpolation)
{
  domain_.realization_interpolation = interpolation;
}

float Result::get_float_value() const
{
  return *value_;
}

float3 Result::get_vector_value() const
{
  return float3(value_);
}

float4 Result::get_color_value() const
{
  return float4(value_);
}

float Result::get_float_value_default(float default_value) const
{
  if (is_single_value()) {
    return get_float_value();
  }
  return default_value;
}

float3 Result::get_vector_value_default(const float3 &default_value) const
{
  if (is_single_value()) {
    return get_vector_value();
  }
  return default_value;
}

float4 Result::get_color_value_default(const float4 &default_value) const
{
  if (is_single_value()) {
    return get_color_value();
  }
  return default_value;
}

void Result::set_float_value(float value)
{
  *value_ = value;
  GPU_texture_update(texture_, GPU_DATA_FLOAT, value_);
}

void Result::set_vector_value(const float3 &value)
{
  copy_v3_v3(value_, value);
  GPU_texture_update(texture_, GPU_DATA_FLOAT, value_);
}

void Result::set_color_value(const float4 &value)
{
  copy_v4_v4(value_, value);
  GPU_texture_update(texture_, GPU_DATA_FLOAT, value_);
}

void Result::increment_reference_count(int count)
{
  /* If there is a master result, increment its reference count instead. */
  if (master_) {
    master_->increment_reference_count(count);
    return;
  }

  reference_count_ += count;
}

void Result::release()
{
  /* If there is a master result, release it instead. */
  if (master_) {
    master_->release();
    return;
  }

  /* Decrement the reference count, and if it reaches zero, release the texture back into the
   * texture pool. */
  reference_count_--;
  if (reference_count_ == 0) {
    texture_pool_->release(texture_);
  }
}

ResultType Result::type() const
{
  return type_;
}

bool Result::is_texture() const
{
  return !is_single_value_;
}

bool Result::is_single_value() const
{
  return is_single_value_;
}

GPUTexture *Result::texture() const
{
  return texture_;
}

int Result::reference_count() const
{
  /* If there is a master result, return its reference count instead. */
  if (master_) {
    return master_->reference_count();
  }
  return reference_count_;
}

const Domain &Result::domain() const
{
  return domain_;
}

/* --------------------------------------------------------------------
 * Operation.
 */

Operation::Operation(Context &context) : context_(context)
{
}

Operation::~Operation()
{
  for (const Vector<ProcessorOperation *> &processors : input_processors_.values()) {
    for (ProcessorOperation *processor : processors) {
      delete processor;
    }
  }
}

void Operation::evaluate()
{
  pre_execute();

  evaluate_input_processors();

  execute();

  release_inputs();
}

Result &Operation::get_result(StringRef identifier)
{
  return results_.lookup(identifier);
}

void Operation::map_input_to_result(StringRef identifier, Result *result)
{
  inputs_to_results_map_.add_new(identifier, result);
  result->increment_reference_count();
}

Domain Operation::compute_domain()
{
  /* In case no domain input was found, likely because all inputs are single values, then return an
   * identity domain. */
  Domain operation_domain = Domain::identity();
  int current_domain_priority = std::numeric_limits<int>::max();

  for (StringRef identifier : input_descriptors_.keys()) {
    const Result &result = get_input(identifier);
    const InputDescriptor &descriptor = get_input_descriptor(identifier);

    /* A single value input can't be a domain input. */
    if (result.is_single_value() || descriptor.expects_single_value) {
      continue;
    }

    /* Notice that the lower the domain priority value is, the higher the priority is, hence the
     * less than comparison. */
    if (descriptor.domain_priority < current_domain_priority) {
      operation_domain = result.domain();
      current_domain_priority = descriptor.domain_priority;
    }
  }

  return operation_domain;
}

void Operation::pre_execute()
{
}

void Operation::evaluate_input_processors()
{
  /* First, add all needed processors for each input. */
  for (const StringRef &identifier : inputs_to_results_map_.keys()) {
    add_reduce_to_single_value_input_processor_if_needed(identifier);
    add_implicit_conversion_input_processor_if_needed(identifier);
    add_realize_on_domain_input_processor_if_needed(identifier);
  }

  /* Then, switch the result mapped for each input of the operation to be that of the last
   * processor for that input if any input processor exist for it. */
  for (const StringRef &identifier : inputs_to_results_map_.keys()) {
    Vector<ProcessorOperation *> &processors = input_processors_.lookup_or_add_default(identifier);
    /* No input processors, nothing to do. */
    if (processors.is_empty()) {
      continue;
    }
    /* Replace the currently mapped result with the result of the last input processor. */
    switch_result_mapped_to_input(identifier, &processors.last()->get_result());
  }

  /* Finally, evaluate the input processors in order. */
  for (const Vector<ProcessorOperation *> &processors : input_processors_.values()) {
    for (ProcessorOperation *processor : processors) {
      processor->evaluate();
    }
  }
}

Result &Operation::get_input(StringRef identifier) const
{
  return *inputs_to_results_map_.lookup(identifier);
}

void Operation::switch_result_mapped_to_input(StringRef identifier, Result *result)
{
  get_input(identifier).release();
  inputs_to_results_map_.lookup(identifier) = result;
}

void Operation::populate_result(StringRef identifier, Result result)
{
  results_.add_new(identifier, result);
}

void Operation::declare_input_descriptor(StringRef identifier, InputDescriptor descriptor)
{
  input_descriptors_.add_new(identifier, descriptor);
}

InputDescriptor &Operation::get_input_descriptor(StringRef identifier)
{
  return input_descriptors_.lookup(identifier);
}

Context &Operation::context()
{
  return context_;
}

TexturePool &Operation::texture_pool()
{
  return context_.texture_pool();
}

void Operation::add_reduce_to_single_value_input_processor_if_needed(StringRef identifier)
{
  const Result &result = get_input(identifier);
  /* Input result is already a single value. */
  if (result.is_single_value()) {
    return;
  }

  /* The input is a full sized texture can can't be reduced to a single value. */
  if (result.domain().size != int2(1)) {
    return;
  }

  /* The input is a texture of a single pixel and can be reduced to a single value. */
  ProcessorOperation *processor = new ReduceToSingleValueProcessorOperation(context(),
                                                                            result.type());
  add_input_processor(identifier, processor);
}

void Operation::add_implicit_conversion_input_processor_if_needed(StringRef identifier)
{
  ResultType result_type = get_input(identifier).type();
  ResultType expected_type = input_descriptors_.lookup(identifier).type;

  if (result_type == ResultType::Float && expected_type == ResultType::Vector) {
    add_input_processor(identifier, new ConvertFloatToVectorProcessorOperation(context()));
  }
  else if (result_type == ResultType::Float && expected_type == ResultType::Color) {
    add_input_processor(identifier, new ConvertFloatToColorProcessorOperation(context()));
  }
  else if (result_type == ResultType::Color && expected_type == ResultType::Float) {
    add_input_processor(identifier, new ConvertColorToFloatProcessorOperation(context()));
  }
  else if (result_type == ResultType::Vector && expected_type == ResultType::Float) {
    add_input_processor(identifier, new ConvertVectorToFloatProcessorOperation(context()));
  }
  else if (result_type == ResultType::Vector && expected_type == ResultType::Color) {
    add_input_processor(identifier, new ConvertVectorToColorProcessorOperation(context()));
  }
}

void Operation::add_realize_on_domain_input_processor_if_needed(StringRef identifier)
{
  const InputDescriptor &descriptor = input_descriptors_.lookup(identifier);
  /* This input does not need realization. */
  if (descriptor.skip_realization) {
    return;
  }

  /* The input expects a single value and if no single value is provided, it will be ignored and a
   * default value will be used, so no need to realize it. */
  if (descriptor.expects_single_value) {
    return;
  }

  const Result &result = get_input(identifier);
  /* Input result is a single value and does not need realization. */
  if (result.is_single_value()) {
    return;
  }

  /* Input result only contains a single pixel and will be reduced to a single value result through
   * a ReduceToSingleValueProcessorOperation, so no need to realize it. */
  if (result.domain().size == int2(1)) {
    return;
  }

  /* The input have an identical domain to the operation domain, so no need to realize it. */
  if (result.domain() == compute_domain()) {
    return;
  }

  /* Realization is needed. */
  ProcessorOperation *processor = new RealizeOnDomainProcessorOperation(
      context(), compute_domain(), descriptor.type);
  add_input_processor(identifier, processor);
}

void Operation::add_input_processor(StringRef identifier, ProcessorOperation *processor)
{
  /* Get a reference to the input processors vector for the given input. */
  Vector<ProcessorOperation *> &processors = input_processors_.lookup_or_add_default(identifier);

  /* Get the result that should serve as the input for the processor. This is either the result
   * mapped to the input or the result of the last processor depending on whether this is the first
   * processor or not. */
  Result &result = processors.is_empty() ? get_input(identifier) : processors.last()->get_result();

  /* Map the input result of the processor and add it to the processors vector. No need to map the
   * result of the processor to the operation input as this will be done later in
   * evaluate_input_processors. */
  processor->map_input_to_result(&result);
  processors.append(processor);
}

void Operation::release_inputs()
{
  for (Result *result : inputs_to_results_map_.values()) {
    result->release();
  }
}

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

/* Get the origin socket of the given node input. If the input is not linked, the socket itself is
 * returned. If the input is linked, the socket that is linked to it is returned, which could
 * either be an input or an output. An input socket is returned when the given input is connected
 * to an unlinked input of a group input node. */
static DSocket get_node_input_origin_socket(DInputSocket input)
{
  /* The input is unlinked. Return the socket itself. */
  if (input->logically_linked_sockets().is_empty()) {
    return input;
  }

  /* Only a single origin socket is guaranteed to exist. */
  DSocket socket;
  input.foreach_origin_socket([&](const DSocket origin) { socket = origin; });
  return socket;
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
    const Result result = Result(result_type, texture_pool());
    unlinked_inputs_results_.append(result);
    map_input_to_result(input->identifier(), &unlinked_inputs_results_.last());

    /* Map the input to the socket to later allocate and initialize its value. */
    const DInputSocket origin_input{origin.context(), &origin->as_input()};
    unlinked_inputs_sockets_.add_new(input->identifier(), origin_input);
  }
}

/* --------------------------------------------------------------------
 * Processor Operation.
 */

const StringRef ProcessorOperation::input_identifier = StringRef("Input");
const StringRef ProcessorOperation::output_identifier = StringRef("Output");

Result &ProcessorOperation::get_result()
{
  return Operation::get_result(output_identifier);
}

void ProcessorOperation::map_input_to_result(Result *result)
{
  Operation::map_input_to_result(input_identifier, result);
}

void ProcessorOperation::evaluate_input_processors()
{
}

Result &ProcessorOperation::get_input()
{
  return Operation::get_input(input_identifier);
}

void ProcessorOperation::switch_result_mapped_to_input(Result *result)
{
  Operation::switch_result_mapped_to_input(input_identifier, result);
}

void ProcessorOperation::populate_result(Result result)
{
  Operation::populate_result(output_identifier, result);
}

void ProcessorOperation::declare_input_descriptor(InputDescriptor descriptor)
{
  Operation::declare_input_descriptor(input_identifier, descriptor);
}

InputDescriptor &ProcessorOperation::get_input_descriptor()
{
  return Operation::get_input_descriptor(input_identifier);
}

/* --------------------------------------------------------------------
 *  Reduce To Single Value Processor Operation.
 */

ReduceToSingleValueProcessorOperation::ReduceToSingleValueProcessorOperation(Context &context,
                                                                             ResultType type)
    : ProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = type;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(type, texture_pool()));
}

void ReduceToSingleValueProcessorOperation::execute()
{
  const Result &input = get_input();
  GPU_memory_barrier(GPU_BARRIER_TEXTURE_UPDATE);
  float *pixel = static_cast<float *>(GPU_texture_read(input.texture(), GPU_DATA_FLOAT, 0));

  Result &result = get_result();
  result.allocate_single_value();
  switch (result.type()) {
    case ResultType::Color:
      result.set_color_value(pixel);
      break;
    case ResultType::Vector:
      result.set_vector_value(pixel);
      break;
    case ResultType::Float:
      result.set_float_value(*pixel);
      break;
  }

  MEM_freeN(pixel);
}

/* --------------------------------------------------------------------
 *  Conversion Processor Operation.
 */

const char *ConversionProcessorOperation::shader_input_sampler_name = "input_sampler";
const char *ConversionProcessorOperation::shader_output_image_name = "output_image";

void ConversionProcessorOperation::execute()
{
  Result &result = get_result();
  const Result &input = get_input();

  if (input.is_single_value()) {
    result.allocate_single_value();
    execute_single(input, result);
    return;
  }

  result.allocate_texture(input.domain());

  GPUShader *shader = get_conversion_shader();
  GPU_shader_bind(shader);

  input.bind_as_texture(shader, shader_input_sampler_name);
  result.bind_as_image(shader, shader_output_image_name);

  const int2 size = result.domain().size;
  GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

  input.unbind_as_texture();
  result.unbind_as_image();
  GPU_shader_unbind();
  GPU_shader_free(shader);
}

/* --------------------------------------------------------------------
 *  Convert Float To Vector Processor Operation.
 */

ConvertFloatToVectorProcessorOperation::ConvertFloatToVectorProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Float;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Vector, texture_pool()));
}

void ConvertFloatToVectorProcessorOperation::execute_single(const Result &input, Result &output)
{
  output.set_vector_value(float3(input.get_float_value()));
}

/* Use the shader for color conversion since they are stored in similar textures and the conversion
 * is practically the same. */
GPUShader *ConvertFloatToVectorProcessorOperation::get_conversion_shader() const
{
  return GPU_shader_create_from_info_name("compositor_convert_float_to_color");
}

/* --------------------------------------------------------------------
 *  Convert Float To Color Processor Operation.
 */

ConvertFloatToColorProcessorOperation::ConvertFloatToColorProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Float;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Color, texture_pool()));
}

void ConvertFloatToColorProcessorOperation::execute_single(const Result &input, Result &output)
{
  float4 color = float4(input.get_float_value());
  color[3] = 1.0f;
  output.set_color_value(color);
}

GPUShader *ConvertFloatToColorProcessorOperation::get_conversion_shader() const
{
  return GPU_shader_create_from_info_name("compositor_convert_float_to_color");
}

/* --------------------------------------------------------------------
 *  Convert Color To Float Processor Operation.
 */

ConvertColorToFloatProcessorOperation::ConvertColorToFloatProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Color;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Float, texture_pool()));
}

void ConvertColorToFloatProcessorOperation::execute_single(const Result &input, Result &output)
{
  float4 color = input.get_color_value();
  output.set_float_value((color[0] + color[1] + color[2]) / 3.0f);
}

GPUShader *ConvertColorToFloatProcessorOperation::get_conversion_shader() const
{
  return GPU_shader_create_from_info_name("compositor_convert_color_to_float");
}

/* --------------------------------------------------------------------
 *  Convert Vector To Float Processor Operation.
 */

ConvertVectorToFloatProcessorOperation::ConvertVectorToFloatProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Vector;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Float, texture_pool()));
}

void ConvertVectorToFloatProcessorOperation::execute_single(const Result &input, Result &output)
{
  float3 vector = input.get_vector_value();
  output.set_float_value((vector[0] + vector[1] + vector[2]) / 3.0f);
}

/* Use the shader for color conversion since they are stored in similar textures and the conversion
 * is practically the same. */
GPUShader *ConvertVectorToFloatProcessorOperation::get_conversion_shader() const
{
  return GPU_shader_create_from_info_name("compositor_convert_color_to_float");
}

/* --------------------------------------------------------------------
 *  Convert Vector To Color Processor Operation.
 */

ConvertVectorToColorProcessorOperation::ConvertVectorToColorProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Vector;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(ResultType::Color, texture_pool()));
}

void ConvertVectorToColorProcessorOperation::execute_single(const Result &input, Result &output)
{
  output.set_color_value(float4(input.get_vector_value(), 1.0f));
}

GPUShader *ConvertVectorToColorProcessorOperation::get_conversion_shader() const
{
  return GPU_shader_create_from_info_name("compositor_convert_vector_to_color");
}

/* --------------------------------------------------------------------
 *  Realize On Domain Processor Operation.
 */

RealizeOnDomainProcessorOperation::RealizeOnDomainProcessorOperation(Context &context,
                                                                     Domain domain,
                                                                     ResultType type)
    : ProcessorOperation(context), domain_(domain)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = type;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(type, texture_pool()));
}

void RealizeOnDomainProcessorOperation::execute()
{
  Result &input = get_input();
  Result &result = get_result();

  result.allocate_texture(domain_);

  GPUShader *shader = get_realization_shader();
  GPU_shader_bind(shader);

  /* Transform the input space into the domain space. */
  const Transformation2D local_transformation = input.domain().transformation *
                                                domain_.transformation.inverted();

  /* Set the pivot of the transformation to be the center of the domain.  */
  const float2 pivot = float2(domain_.size) / 2.0f;
  const Transformation2D pivoted_transformation = local_transformation.set_pivot(pivot);

  /* Invert the transformation because the shader transforms the domain coordinates instead of the
   * input image itself and thus expect the inverse. */
  const Transformation2D inverse_transformation = pivoted_transformation.inverted();

  /* Set the inverse of the transform to the shader. */
  GPU_shader_uniform_mat3(shader, "inverse_transformation", inverse_transformation.matrix());

  /* Make out-of-bound texture access return zero. */
  GPU_texture_wrap_mode(input.texture(), false, false);

  /* Set the approperiate sampler interpolation. */
  const bool use_bilinear = input.domain().realization_interpolation != Interpolation::Nearest;
  GPU_texture_filter_mode(input.texture(), use_bilinear);

  input.bind_as_texture(shader, "input_sampler");
  result.bind_as_image(shader, "domain");

  const int2 size = result.domain().size;
  GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

  input.unbind_as_texture();
  result.unbind_as_image();
  GPU_shader_unbind();
  GPU_shader_free(shader);
}

GPUShader *RealizeOnDomainProcessorOperation::get_realization_shader()
{
  switch (get_result().type()) {
    case ResultType::Color:
      return GPU_shader_create_from_info_name("compositor_realize_on_domain_color");
    case ResultType::Vector:
      return GPU_shader_create_from_info_name("compositor_realize_on_domain_vector");
    case ResultType::Float:
      return GPU_shader_create_from_info_name("compositor_realize_on_domain_float");
  }

  BLI_assert_unreachable();
  return nullptr;
}

Domain RealizeOnDomainProcessorOperation::compute_domain()
{
  return domain_;
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

GPUMaterialOperation::GPUMaterialOperation(Context &context, SubSchedule &sub_schedule)
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
 * Scheduler.
 */

Scheduler::Scheduler(bNodeTree *node_tree) : tree_(*node_tree, tree_ref_map_)
{
}

void Scheduler::schedule()
{
  /* Get the output node whose result should be computed and drawn. */
  DNode output_node = compute_output_node();

  /* Compute the number of buffers needed by each node. */
  compute_needed_buffers(output_node);

  /* Compute the execution schedule of the nodes. */
  compute_schedule(output_node);
}

Schedule &Scheduler::get_schedule()
{
  return schedule_;
}

/* The output node is the one marked as NODE_DO_OUTPUT. If multiple types of output nodes are
 * marked, then preference will be CMP_NODE_COMPOSITE > CMP_NODE_VIEWER > CMP_NODE_SPLITVIEWER. */
DNode Scheduler::compute_output_node() const
{
  const NodeTreeRef &root_tree = tree_.root_context().tree();
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeComposite")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree_.root_context(), node);
    }
  }
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeViewer")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree_.root_context(), node);
    }
  }
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeSplitViewer")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree_.root_context(), node);
    }
  }
  return DNode();
}

/* Consider a node that takes n number of buffers as an input from a number of node dependencies,
 * which we shall call the input nodes. The node also computes and outputs m number of buffers.
 * In order for the node to compute its output, a number of intermediate buffers will be needed.
 * Since the node takes n buffers and outputs m buffers, then the number of buffers directly
 * needed by the node is (n + m). But each of the input buffers are computed by a node that, in
 * turn, needs a number of buffers to compute its output. So the total number of buffers needed
 * to compute the output of the node is max(n + m, d) where d is the number of buffers needed by
 * the input node that needs the largest number of buffers. We only consider the input node that
 * needs the largest number of buffers, because those buffers can be reused by any input node
 * that needs a lesser number of buffers.
 *
 * If the node tree was, in fact, a tree, then this would be an accurate computation. However,
 * the node tree is in fact a graph that allows output sharing, so the computation in this case
 * is merely a heuristic estimation that works well in most cases. */
int Scheduler::compute_needed_buffers(DNode node)
{
  /* Compute the number of buffers that the node takes as an input as well as the number of
   * buffers needed to compute the most demanding dependency node. */
  int input_buffers = 0;
  int buffers_needed_by_dependencies = 0;
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};
    /* Only consider inputs that are linked, that is, those that take a buffer. */
    input.foreach_origin_socket([&](const DSocket origin) {
      input_buffers++;
      /* The origin node was already computed before, so skip it. */
      if (needed_buffers_.contains(origin.node())) {
        return;
      }
      /* Recursively compute the number of buffers needed to compute this dependency node. */
      const int buffers_needed_by_origin = compute_needed_buffers(origin.node());
      if (buffers_needed_by_origin > buffers_needed_by_dependencies) {
        buffers_needed_by_dependencies = buffers_needed_by_origin;
      }
    });
  }

  /* Compute the number of buffers that will be computed/output by this node. */
  int output_buffers = 0;
  for (const OutputSocketRef *output : node->outputs()) {
    if (!output->logically_linked_sockets().is_empty()) {
      output_buffers++;
    }
  }

  /* Compute the heuristic estimation of the number of needed intermediate buffers to compute
   * this node and all of its dependencies. */
  const int total_buffers = MAX2(input_buffers + output_buffers, buffers_needed_by_dependencies);
  needed_buffers_.add_new(node, total_buffers);
  return total_buffers;
}

/* There are multiple different possible orders of evaluating a node graph, each of which needs
 * to allocate a number of intermediate buffers to store its intermediate results. It follows
 * that we need to find the evaluation order which uses the least amount of intermediate buffers.
 * For instance, consider a node that takes two input buffers A and B. Each of those buffers is
 * computed through a number of nodes constituting a sub-graph whose root is the node that
 * outputs that buffer. Suppose the number of intermediate buffers needed to compute A and B are
 * N(A) and N(B) respectively and N(A) > N(B). Then evaluating the sub-graph computing A would be
 * a better option than that of B, because had B was computed first, its outputs will need to be
 * stored in extra buffers in addition to the buffers needed by A.
 *
 * This is a heuristic generalization of the Sethi–Ullman algorithm, a generalization that
 * doesn't always guarantee an optimal evaluation order, as the optimal evaluation order is very
 * difficult to compute, however, this method works well in most cases. */
void Scheduler::compute_schedule(DNode node)
{
  /* Compute the nodes directly connected to the node inputs sorted by their needed buffers such
   * that the node with the highest number of needed buffers comes first. */
  Vector<DNode> sorted_origin_nodes;
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};
    input.foreach_origin_socket([&](const DSocket origin) {
      /* The origin node was added before or was already schedule, so skip it. The number of
       * origin nodes is very small, so linear search is okay. */
      if (sorted_origin_nodes.contains(origin.node()) || schedule_.contains(origin.node())) {
        return;
      }
      /* Sort on insertion, the number of origin nodes is very small, so this is okay. */
      int insertion_position = 0;
      for (int i = 0; i < sorted_origin_nodes.size(); i++) {
        if (needed_buffers_.lookup(origin.node()) >
            needed_buffers_.lookup(sorted_origin_nodes[i])) {
          break;
        }
        insertion_position++;
      }
      sorted_origin_nodes.insert(insertion_position, origin.node());
    });
  }

  /* Recursively schedule origin nodes. Nodes with higher number of needed intermediate buffers
   * are scheduled first. */
  for (const DNode &origin_node : sorted_origin_nodes) {
    compute_schedule(origin_node);
  }

  schedule_.add(node);
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
    : context_(context), scheduler_(node_tree)
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
  scheduler_.schedule();

  for (const DNode &node : scheduler_.get_schedule()) {
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
  SubSchedule &sub_schedule = gpu_material_compile_group_.get_sub_schedule();
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
