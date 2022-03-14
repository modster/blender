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

#include "BLI_assert.h"
#include "BLI_hash.hh"
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
#include "GPU_shader.h"
#include "GPU_texture.h"

#include "IMB_colormanagement.h"

#include "NOD_compositor_execute.hh"
#include "NOD_derived_node_tree.hh"
#include "NOD_node_declaration.hh"

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

Domain Domain::identity()
{
  return Domain(int2(1), Transformation2D::identity());
}

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

void Result::allocate_texture(int2 size)
{
  is_texture_ = true;
  switch (type_) {
    case ResultType::Float:
      texture_ = texture_pool_->acquire_float(size);
      return;
    case ResultType::Vector:
      texture_ = texture_pool_->acquire_vector(size);
      return;
    case ResultType::Color:
      texture_ = texture_pool_->acquire_color(size);
      return;
  }
}

void Result::allocate_single_value()
{
  is_texture_ = false;
  /* Single values are stored in 1x1 textures. */
  const int2 texture_size{1, 1};
  switch (type_) {
    case ResultType::Float:
      texture_ = texture_pool_->acquire_float(texture_size);
      return;
    case ResultType::Vector:
      texture_ = texture_pool_->acquire_vector(texture_size);
      return;
    case ResultType::Color:
      texture_ = texture_pool_->acquire_color(texture_size);
      return;
  }
}

void Result::bind_as_texture(GPUShader *shader, const char *texture_name) const
{
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
  transformation_ = transformation * transformation_;
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
  return is_texture_;
}

bool Result::is_single_value() const
{
  return !is_texture_;
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

int2 Result::size() const
{
  return int2{GPU_texture_width(texture_), GPU_texture_height(texture_)};
}

Transformation2D Result::transformation() const
{
  return transformation_;
}

Domain Result::domain() const
{
  return Domain(size(), transformation_);
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

void Operation::pre_execute()
{
}

void Operation::evaluate_input_processors()
{
  /* First add all needed processors for each input. */
  for (const StringRef &identifier : inputs_to_results_map_.keys()) {
    add_implicit_conversion_input_processor_if_needed(identifier);
    add_realize_on_domain_input_processor_if_needed(identifier);
  }

  /* Then evaluate the input processors in order. */
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

  /* Input is a domain input and does not need realization. */
  if (descriptor.is_domain) {
    return;
  }

  const Result result = get_input(identifier);
  /* Input result is a single value and does not need realization. */
  if (result.is_single_value()) {
    return;
  }

  /* Realization is needed. It could be that the input domain is identical to the operation domain
   * and thus realization will not be needed, but this is handled during execution because
   * transformation is only known at execution time, not allocation time. */
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

  /* Set the input result of the processor, add it to the processors vector, and switch the result
   * mapped to the input to the result of the last processor. */
  processor->map_input_to_result(&result);
  processors.append(processor);
  switch_result_mapped_to_input(identifier, &processor->get_result());
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
    input_descriptor.is_domain =
        node->declaration()->inputs()[input->index()]->is_compositor_domain_input();
    declare_input_descriptor(input->identifier(), input_descriptor);
  }

  populate_results_for_unlinked_inputs();
}

/* Node operations are buffered in most cases, but the derived operation can override
   otherwise. */
bool NodeOperation::is_buffered() const
{
  return true;
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

Domain NodeOperation::compute_domain()
{
  /* If any of the inputs is a domain input and not a single value, return the domain of the first
   * one. */
  for (const InputSocketRef *input : node_->inputs()) {
    const Result &result = get_input(input->identifier());
    bool is_domain = get_input_descriptor(input->identifier()).is_domain;
    if (is_domain && result.is_texture()) {
      return result.domain();
    }
  }

  /* No domain inputs or all of them are single values, so return the domain of the first non
   * single input. */
  for (const InputSocketRef *input : node_->inputs()) {
    const Result &result = get_input(input->identifier());
    if (result.is_texture()) {
      return result.domain();
    }
  }

  /* All inputs are single values. Return an identity domain. */
  return Domain::identity();
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

  /* Only a single origin socket is guranteed to exist. */
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
    if (origin && origin->is_output()) {
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

bool ProcessorOperation::is_buffered() const
{
  return true;
}

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

  result.allocate_texture(input.size());

  GPUShader *shader = get_conversion_shader();
  GPU_shader_bind(shader);

  input.bind_as_texture(shader, shader_input_sampler_name);
  result.bind_as_image(shader, shader_output_image_name);

  const int2 size = result.size();
  GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

  GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);

  input.unbind_as_texture();
  result.unbind_as_image();
  GPU_shader_unbind();
  GPU_shader_free(shader);
}

Domain ConversionProcessorOperation::compute_domain()
{
  return get_input().domain();
}

/* --------------------------------------------------------------------
 *  Convert Float To Vector Processor Operation.
 */

ConvertFloatToVectorProcessorOperation::ConvertFloatToVectorProcessorOperation(Context &context)
    : ConversionProcessorOperation(context)
{
  InputDescriptor input_descriptor;
  input_descriptor.type = ResultType::Float;
  input_descriptor.is_domain = true;
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
  input_descriptor.is_domain = true;
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
  input_descriptor.is_domain = true;
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
  input_descriptor.is_domain = true;
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
  input_descriptor.is_domain = true;
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
  input_descriptor.skip_realization = true;
  declare_input_descriptor(input_descriptor);
  populate_result(Result(type, texture_pool()));
}

void RealizeOnDomainProcessorOperation::execute()
{
  Result &input = get_input();
  Result &result = get_result();

  /* The input have the same domain as the operation domain, so just pass the input through. */
  if (input.domain() == domain_) {
    input.pass_through(result);
    return;
  }

  result.allocate_texture(domain_.size);

  GPUShader *shader = get_realization_shader();
  GPU_shader_bind(shader);

  /* Transform the input space into the domain space. */
  const Transformation2D local_transformation = input.transformation() *
                                                domain_.transformation.inverted();

  /* Set the pivot of the transformation to be the center of the domain.  */
  const float2 pivot = float2(result.size()) / 2.0f;
  const Transformation2D pivoted_transformation = local_transformation.set_pivot(pivot);

  /* Invert the transformation because the shader transforms the domain coordinates instead of the
   * input image itself and thus expect the inverse. */
  const Transformation2D inverse_transformation = pivoted_transformation.inverted();

  /* Set the inverse of the transform to the shader. */
  GPU_shader_uniform_mat3(shader, "inverse_transformation", inverse_transformation.matrix());

  /* Make out-of-bound texture access return zero. */
  GPU_texture_wrap_mode(input.texture(), false, false);

  input.bind_as_texture(shader, "input_sampler");
  result.bind_as_image(shader, "domain");

  const int2 size = result.size();
  GPU_compute_dispatch(shader, size.x / 16 + 1, size.y / 16 + 1, 1);

  GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);

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
}

Domain RealizeOnDomainProcessorOperation::compute_domain()
{
  return domain_;
}

/* --------------------------------------------------------------------
 * Evaluator.
 */

Evaluator::Evaluator(Context &context, bNodeTree *scene_node_tree)
    : context(context), tree(*scene_node_tree, tree_ref_map){};

Evaluator::~Evaluator()
{
  for (const Operation *operation : operations_stream_) {
    delete operation;
  }
}

void Evaluator::evaluate()
{
  /* Get the output node whose result should be computed and drawn. */
  DNode output_node = compute_output_node();

  /* Validate the compositor node tree. */
  if (!is_valid(output_node)) {
    return;
  }

  /* Instantiate a node operation for every node reachable from the output. */
  create_node_operations(output_node);

  /* Compute the number of buffers needed by each node. */
  NeededBuffers needed_buffers;
  compute_needed_buffers(output_node, needed_buffers);

  /* Compute the execution schedule of the nodes. */
  NodeSchedule node_schedule;
  compute_schedule(output_node, needed_buffers, node_schedule);

  /* Compute the operations stream. */
  compute_operations_stream(node_schedule);

  /* Evaluate the operations stream. */
  evaluate_operations_stream();
}

/* The output node is the one marked as NODE_DO_OUTPUT. If multiple types of output nodes are
 * marked, then preference will be CMP_NODE_COMPOSITE > CMP_NODE_VIEWER > CMP_NODE_SPLITVIEWER. */
DNode Evaluator::compute_output_node() const
{
  const NodeTreeRef &root_tree = tree.root_context().tree();
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeComposite")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree.root_context(), node);
    }
  }
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeViewer")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree.root_context(), node);
    }
  }
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeSplitViewer")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree.root_context(), node);
    }
  }
  return DNode();
}

bool Evaluator::is_valid(DNode output_node)
{
  /* The node tree needs to have an output. */
  if (!output_node) {
    return false;
  }

  return true;
}

/* Traverse the node tree starting from the given node and instantiate the node operations for all
 * reachable nodes, adding the instances to node_operations_. */
void Evaluator::create_node_operations(DNode node)
{
  const bNodeType *type = node->typeinfo();
  NodeOperation *operation = type->get_compositor_operation(context, node);
  node_operations_.add_new(node, operation);

  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};
    input.foreach_origin_socket([&](const DSocket origin) {
      if (!node_operations_.contains(origin.node())) {
        create_node_operations(origin.node());
      }
    });
  }
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
int Evaluator::compute_needed_buffers(DNode node, NeededBuffers &needed_buffers)
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
      if (needed_buffers.contains(origin.node())) {
        return;
      }
      /* Recursively compute the number of buffers needed to compute this dependency node. */
      const int buffers_needed_by_origin = compute_needed_buffers(origin.node(), needed_buffers);
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
  needed_buffers.add_new(node, total_buffers);
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
void Evaluator::compute_schedule(DNode node,
                                 NeededBuffers &needed_buffers,
                                 NodeSchedule &node_schedule)
{
  /* Compute the nodes directly connected to the node inputs sorted by their needed buffers such
   * that the node with the highest number of needed buffers comes first. */
  Vector<DNode> sorted_origin_nodes;
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};
    input.foreach_origin_socket([&](const DSocket origin) {
      /* The origin node was added before or was already schedule, so skip it. The number of
       * origin nodes is very small, so linear search is okay.
       */
      if (sorted_origin_nodes.contains(origin.node()) || node_schedule.contains(origin.node())) {
        return;
      }
      /* Sort on insertion, the number of origin nodes is very small, so this is okay. */
      int insertion_position = 0;
      for (int i = 0; i < sorted_origin_nodes.size(); i++) {
        if (needed_buffers.lookup(origin.node()) > needed_buffers.lookup(sorted_origin_nodes[i])) {
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
    compute_schedule(origin_node, needed_buffers, node_schedule);
  }

  node_schedule.add_new(node);
}

void Evaluator::compute_operations_stream(NodeSchedule &node_schedule)
{
  for (const DNode &node : node_schedule) {
    /* First add the node operation corresponding to the given node to the operations stream. */
    operations_stream_.append(node_operations_.lookup(node));
    /* Then map the inputs of the operation to the results connected to them. */
    map_node_inputs_to_results(node);
  }
}

void Evaluator::map_node_inputs_to_results(DNode node)
{
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};
    DSocket origin = get_node_input_origin_socket(input);

    /* If the origin socket is an input, that means the input is unlinked. Unlinked inputs are
     * mapped internally to internal results, so skip here. */
    if (origin->is_input()) {
      continue;
    }

    /* Get the result from the operation that contains the output socket. */
    const DOutputSocket output{origin.context(), &origin->as_output()};
    NodeOperation *output_operation = node_operations_.lookup(output.node());
    Result &result = output_operation->get_result(output->identifier());

    /* Map the input to the result we got from the output. */
    NodeOperation *input_operation = node_operations_.lookup(input.node());
    input_operation->map_input_to_result(input->identifier(), &result);
  }
}

void Evaluator::evaluate_operations_stream()
{
  for (Operation *operation : operations_stream_) {
    operation->evaluate();
  }
}

}  // namespace blender::viewport_compositor
