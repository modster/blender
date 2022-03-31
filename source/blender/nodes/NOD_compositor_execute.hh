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

namespace blender::viewport_compositor {

/* --------------------------------------------------------------------
 * Texture Pool.
 */

/* A key structure used to identify a texture specification in a texture pool. Defines a hash and
 * an equality operator for use in a hash map. */
class TexturePoolKey {
 public:
  int2 size;
  eGPUTextureFormat format;

  TexturePoolKey(int2 size, eGPUTextureFormat format);
  TexturePoolKey(const GPUTexture *texture);

  uint64_t hash() const;
};

/* A pool of textures that can be allocated and reused transparently throughout the evaluation of
 * the compositor. This texture pool only pools textures throughout a single evaluation of the
 * compositor and will reset after evaluation without freeing any textures. Cross-evaluation
 * pooling and freeing of unused textures is the responsibility of the back-end texture pool used
 * by the allocate_texture method. In the case of the viewport compositor engine, this would be the
 * global DRWTexturePool of the draw manager. */
class TexturePool {
 private:
  /* The set of textures in the pool that are available to acquire for each distinct texture
   * specification. */
  Map<TexturePoolKey, Vector<GPUTexture *>> textures_;

 public:
  /* Check if there is an available texture with the given specification in the pool, if such
   * texture exists, return it, otherwise, return a newly allocated texture. Expect the texture to
   * be uncleared and contains garbage data. */
  GPUTexture *acquire(int2 size, eGPUTextureFormat format);

  /* Shorthand for acquire with GPU_RGBA16F format. */
  GPUTexture *acquire_color(int2 size);

  /* Shorthand for acquire with GPU_RGBA16F format. Identical to acquire_color because vector
   * textures are and should internally be stored in RGBA textures. */
  GPUTexture *acquire_vector(int2 size);

  /* Shorthand for acquire with GPU_R16F format. */
  GPUTexture *acquire_float(int2 size);

  /* Put the texture back into the pool, potentially to be acquired later by another user. Expects
   * the texture to be one that was acquired using the same texture pool. */
  void release(GPUTexture *texture);

 private:
  /* Returns a newly allocated texture with the given specification. This method should be
   * implemented by the compositor engine and should use a global texture pool that is persistent
   * across evaluations and capable of freeing unused textures. In the case of the viewport
   * compositor engine, this would be the global DRWTexturePool of the draw manager. */
  virtual GPUTexture *allocate_texture(int2 size, eGPUTextureFormat format) = 0;
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

  /* Get the active compositing scene. */
  virtual const Scene *get_scene() = 0;

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
 * Domain.
 */

/* A domain is a rectangular area of a certain size in pixels that is transformed by a certain
 * transformation in pixel space relative to some reference space.
 *
 * Any result computed by an operation resides in a domain. The size of the domain of the result is
 * the size of its texture. The transformation of the domain of the result is typically an identity
 * transformation, indicating that the result is centered in space. But a transformation operation
 * like the rotate, translate, or transform operations will adjust the transformation to make the
 * result reside somewhere different in space. The domain of a single value result is irrelevant
 * and always set to an identity domain.
 *
 * An operation operates in a certain domain called the operation domain, it follows that the
 * operation only cares about the inputs whose domain is inside or at least intersects the
 * operation domain. To abstract away the different domains of the inputs, any input that have a
 * different domain than the operation domain is realized on the operation domain through a
 * RealizeOnDomainProcessorOperation, except inputs whose descriptor sets skip_realization or
 * expects_single_value, see InputDescriptor for more information. The realization process simply
 * projects the input domain on the operation domain, copies the area of input that intersects the
 * operation domain, and fill the rest with zeros. This process is illustrated below. It follows
 * that operations should expect all their inputs to have the same domain and consequently size,
 * except for inputs that explicitly skip realization.
 *
 *                                   Realized Result
 *             +-------------+       +-------------+
 *             |  Operation  |       |             |
 *             |   Domain    |       |    Zeros    |
 *             |             | ----> |             |
 *       +-----------+       |       |-----+       |
 *       |     |  C  |       |       |  C  |       |
 *       |     +-----|-------+       +-----|-------+
 *       | Domain Of |
 *       |   Input   |
 *       +-----------+
 *
 * Each operation can define an arbitrary operation domain, but in most cases, the operation domain
 * is inferred from the inputs. By default, the operation domain is computed as follows. Typically,
 * one input of the operation is said to be the domain input and the operation domain is inferred
 * from it. The domain input is determined to be the non-single value input that have the highest
 * domain priority, a zero value being the highest priority. If all inputs are single values, then
 * the operation domain is irrelevant and an identity domain is set. See
 * NodeOperation::compute_domain.
 *
 * The aforementioned logic for operation domain computation is only a default that works for most
 * cases, but an operation can override the compute_domain method to implement a different logic.
 * For instance, output nodes have an operation domain the same size as the viewport and with an
 * identity transformation, their operation domain doesn't depend on the inputs at all.
 *
 * For instance, a filter operation have two inputs, a factor and a color, the latter of which
 * has a domain priority of 0 and the former has a domain priority of 1. If the color input is not
 * a single value, then the domain of this operation is computed to be the same size and
 * transformation as the color input, because it has the highest priority. And if the factor input
 * have a different size and/or transformation from the computed domain of the operation, it will
 * be projected and realized on it to have the same size as described above. It follows that the
 * color input, will not need to be realized because it already has the same size and
 * transformation as the domain of the operation, because the operation domain is inferred from it.
 * On the other hand, if the color input is a single value input, then the operation domain will be
 * the same as the domain of the factor input, because it has the second highest domain priority.
 * Finally, if both inputs are single value inputs, the operation domain will be an identity and is
 * irrelevant. */
class Domain {
 public:
  /* The size of the domain in pixels. */
  int2 size;
  /* The 2D transformation of the domain defining its translation in pixels, rotation, and scale in
   * 2D space. */
  Transformation2D transformation;

 public:
  /* A size only constructor that sets the transformation to identity. */
  Domain(int2 size);

  Domain(int2 size, Transformation2D transformation);

  /* Transform the domain by the given transformation. This effectively pre-multiply the given
   * transformation by the current transformation of the domain. */
  void transform(const Transformation2D &transformation);

  /* Returns a domain of size 1x1 and an identity transformation. */
  static Domain identity();
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

/* A class that represents an output of an operation. A result reside in a certain domain defined
 * by its size and transformation, see the Domain class for more information. A result either
 * stores a single value or a texture. An operation will output a single value result if that value
 * would have been constant over the whole texture. Single value results are stored in 1x1 textures
 * to make them easily accessible in shaders. But the same value is also stored in the value member
 * of the result for any host-side processing. */
class Result {
 private:
  /* The base type of the texture or the type of the single value. */
  ResultType type_;
  /* If true, the result is a single value, otherwise, the result is a texture. */
  bool is_single_value_;
  /* A GPU texture storing the result data. This will be a 1x1 texture if the result is a single
   * value, the value of which will be identical to that of the value member. See class description
   * for more information. */
  GPUTexture *texture_ = nullptr;
  /* The texture pool used to allocate the texture of the result, this should be initialized during
   * construction. */
  TexturePool *texture_pool_ = nullptr;
  /* The number of users currently referencing and using this result. If this result have a master
   * result, then this reference count is irrelevant and shadowed by the reference count of the
   * master result. */
  int reference_count_ = 0;
  /* If the result is a single value, this member stores the value of the result, the value of
   * which will be identical to that stored in the texture member. While this member stores 4
   * values, only a subset of which could be initialized depending on the type, for instance, a
   * float result will only initialize the first array element and a vector result will only
   * initialize the first three array elements. This member is uninitialized if the result is a
   * texture. */
  float value_[4];
  /* The domain of the result. This only matters if the result was a texture. See the Domain class
   * for more information. */
  Domain domain_ = Domain::identity();
  /* If not nullptr, then this result wraps and uses the texture of another master result. In this
   * case, calls to texture-related methods like increment_reference_count and release should
   * operate on the master result as opposed to this result. This member is typically set upon
   * calling the pass_through method, which sets this result to be the master of a target result.
   * See that method for more information. */
  Result *master_ = nullptr;

 public:
  /* Construct a result of the given type with the given texture pool that will be used to allocate
   * and release the result's texture. */
  Result(ResultType type, TexturePool &texture_pool);

  /* Declare the result to be a texture result, allocate a texture of an appropriate type with
   * the size of the given domain from the result's texture pool, and set the domain of the result
   * to the given domain. */
  void allocate_texture(Domain domain);

  /* Declare the result to be a single value result, allocate a texture of an appropriate
   * type with size 1x1 from the result's texture pool, and set the domain to be an identity
   * domain. See class description for more information. */
  void allocate_single_value();

  /* Bind the texture of the result to the texture image unit with the given name in the currently
   * bound given shader. This also inserts a memory barrier for texture fetches to ensure any prior
   * writes to the texture are reflected before reading from it. */
  void bind_as_texture(GPUShader *shader, const char *texture_name) const;

  /* Bind the texture of the result to the image unit with the given name in the currently bound
   * given shader. */
  void bind_as_image(GPUShader *shader, const char *image_name) const;

  /* Unbind the texture which was previously bound using bind_as_texture. */
  void unbind_as_texture() const;

  /* Unbind the texture which was previously bound using bind_as_image. */
  void unbind_as_image() const;

  /* Pass this result through to a target result. This method makes the target result a copy of
   * this result, essentially having identical values between the two and consequently sharing the
   * underlying texture. Additionally, this result is set to be the master of the target result, by
   * setting the master member of the target. Finally, the reference count of the result is
   * incremented by the reference count of the target result. This is typically called in the
   * allocate method of an operation whose input texture will not change and can be passed to the
   * output directly. It should be noted that such operations can still adjust other properties of
   * the result, like its domain. So for instance, the transform operation passes its input through
   * to its output because it will not change it, however, it may adjusts its domain. */
  void pass_through(Result &target);

  /* Transform the result by the given transformation. This effectively pre-multiply the given
   * transformation by the current transformation of the domain of the result. */
  void transform(const Transformation2D &transformation);

  /* If the result is a single value result of type float, return its float value. Otherwise, an
   * uninitialized value is returned. */
  float get_float_value() const;

  /* If the result is a single value result of type vector, return its vector value. Otherwise, an
   * uninitialized value is returned. */
  float3 get_vector_value() const;

  /* If the result is a single value result of type color, return its color value. Otherwise, an
   * uninitialized value is returned. */
  float4 get_color_value() const;

  /* Same as get_float_value but returns a default value if the result is not a single value. */
  float get_float_value_default(float default_value) const;

  /* Same as get_vector_value but returns a default value if the result is not a single value. */
  float3 get_vector_value_default(const float3 &default_value) const;

  /* Same as get_color_value but returns a default value if the result is not a single value. */
  float4 get_color_value_default(const float4 &default_value) const;

  /* If the result is a single value result of type float, set its float value and upload it to the
   * texture. Otherwise, an undefined behavior is invoked. */
  void set_float_value(float value);

  /* If the result is a single value result of type vector, set its vector value and upload it to
   * the texture. Otherwise, an undefined behavior is invoked. */
  void set_vector_value(const float3 &value);

  /* If the result is a single value result of type color, set its color value and upload it to the
   * texture. Otherwise, an undefined behavior is invoked. */
  void set_color_value(const float4 &value);

  /* Increment the reference count of the result by the given count. This should be called when a
   * user gets a reference to the result to use. If this result have a master result, the reference
   * count of the master result is incremented instead. */
  void increment_reference_count(int count = 1);

  /* Decrement the reference count of the result and release the result texture back into the
   * texture pool if the reference count reaches zero. This should be called when a user that
   * previously referenced and incremented the reference count of the result no longer needs it. If
   * this result have a master result, the master result is released instead. */
  void release();

  /* Returns the type of the result. */
  ResultType type() const;

  /* Returns true if the result is a texture and false of it is a single value. */
  bool is_texture() const;

  /* Returns true if the result is a single value and false of it is a texture. */
  bool is_single_value() const;

  /* Returns the allocated GPU texture of the result. */
  GPUTexture *texture() const;

  /* Returns the reference count of the result. If this result have a master result, then the
   * reference count of the master result is returned instead. */
  int reference_count() const;

  /* Returns the domain of the result. See the Domain class. */
  const Domain &domain() const;
};

/* --------------------------------------------------------------------
 * Input Descriptor.
 */

/* A class that describes an input of an operation. */
class InputDescriptor {
 public:
  /* The type of input. This may be different that the type of result that the operation will
   * receive for the input, in which case, an implicit conversion input processor operation will
   * be added to convert it to the required type. */
  ResultType type;
  /* If true, then the input does not need to be realized on the domain of the operation before its
   * execution. See the Domain class for more information. */
  bool skip_realization = false;
  /* The priority of the input for determining the operation domain. The non-single value input
   * with the highest priority will be used to infer the operation domain, the highest priority
   * being zero. See the Domain class for more information. */
  int domain_priority = 0;
  /* If true, the input expects a single value, and if a non-single value is provided, a default
   * single value will be used instead, see the get_*_value_default methods in the Result
   * class. It follows that this also imply skip_realization, because we don't need to realize a
   * result that will be discarded anyways. If false, the input can work with both single and
   * non-single values. */
  bool expects_single_value = false;
};

/* --------------------------------------------------------------------
 * Operation.
 */

/* Forward declare processor operation because it is used in the operation definition.  */
class ProcessorOperation;

/* The most basic unit of the compositor. The class can be implemented to perform a certain action
 * in the compositor. */
class Operation {
 private:
  /* A reference to the compositor context. This member references the same object in all
   * operations but is included in the class for convenience. */
  Context &context_;
  /* A mapping between each output of the operation identified by its identifier and the computed
   * result for that output. A result for each output of an appropriate type should be constructed
   * and added to the map during operation construction. The results should be allocated and their
   * contents should be computed in the execute method. */
  Map<StringRef, Result> results_;
  /* A mapping between each input of the operation identified by its identifier and a reference to
   * the computed result providing its data. The mapped result can be one that was computed by
   * another operation or one that was internally computed in the operation as part of an internal
   * preprocessing step like implicit conversion. It is the responsibility of the evaluator to map
   * the inputs to their linked results prior to invoking any method, which is done by calling
   * map_input_to_result. It is the responsibility of the operation to map the inputs that are not
   * linked to the result of an internal single value result computed by the operation during
   * operation construction. */
  Map<StringRef, Result *> inputs_to_results_map_;
  /* A mapping between each input of the operation identified by its identifier and an ordered list
   * of input processor operations to be applied on that input. */
  Map<StringRef, Vector<ProcessorOperation *>> input_processors_;
  /* A mapping between each input of the operation identified by its identifier and its input
   * descriptor. This should be populated during operation construction. */
  Map<StringRef, InputDescriptor> input_descriptors_;

 public:
  Operation(Context &context);

  virtual ~Operation();

  /* Evaluate the operation as follows:
   * 1. Run any pre-execute computations.
   * 2. Add an evaluate any input processors.
   * 3. Invoking the execute method of the operation.
   * 4. Releasing the results mapped to the inputs. */
  void evaluate();

  /* Get a reference to the output result identified by the given identifier. */
  Result &get_result(StringRef identifier);

  /* Map the input identified by the given identifier to the result providing its data. This also
   * increments the reference count of the result. See inputs_to_results_map_ for more details.
   * This should be called by the evaluator to establish links between different operations. */
  void map_input_to_result(StringRef identifier, Result *result);

 protected:
  /* Compute the operation domain of this operation. By default, this implements a default logic
   * that infers the operation domain from the inputs, which may be overridden for a different
   * logic. See the Domain class for the inference logic and more information. */
  virtual Domain compute_domain();

  /* This method is called before the execute method and can be overridden by a derived class to do
   * any necessary internal computations before the operation is executed. For instance, this is
   * overridden by node operations to compute results for unlinked sockets. */
  virtual void pre_execute();

  /* First, all the necessary input processors for each input. Then update the result mapped to
   * each input to be that of the last processor for that input if any input processors exist for
   * it. This is done now in a separate step after all processors were added because the operation
   * might use the original mapped results to determine what processors needs to be added. Finally,
   * evaluate all input processors in order. This is called before executing the operation to
   * prepare its inputs but after the pre_execute method was called. The class defines a default
   * implementation, but derived class can override the method to have a different
   * implementation, extend the implementation, or remove it. */
  virtual void evaluate_input_processors();

  /* This method should allocate the operation results, execute the operation, and compute the
   * output results. */
  virtual void execute() = 0;

  /* Get a reference to the result connected to the input identified by the given identifier. */
  Result &get_input(StringRef identifier) const;

  /* Switch the result mapped to the input identified by the given identifier with the given
   * result. This will involve releasing the original result, but it is assumed that the result
   * will be mapped to something else. */
  void switch_result_mapped_to_input(StringRef identifier, Result *result);

  /* Add the given result to the results_ map identified by the given output identifier. This
   * should be called during operation construction for every output. The provided result shouldn't
   * be allocated or initialized, this will happen later during execution. */
  void populate_result(StringRef identifier, Result result);

  /* Declare the descriptor of the input identified by the given identifier to be the given
   * descriptor. Adds the given descriptor to the input_descriptors_ map identified by the given
   * input identifier. This should be called during operation constructor for every input. */
  void declare_input_descriptor(StringRef identifier, InputDescriptor descriptor);

  /* Get a reference to the descriptor of the input identified by the given identified. */
  InputDescriptor &get_input_descriptor(StringRef identified);

  /* Returns a reference to the compositor context. */
  Context &context();

  /* Returns a reference to the texture pool of the compositor context. */
  TexturePool &texture_pool();

 private:
  /* Add a reduce to single value input processor for the input identified by the given identifier
   * if needed. */
  void add_reduce_to_single_value_input_processor_if_needed(StringRef identifier);

  /* Add an implicit conversion input processor for the input identified by the given identifier if
   * needed. */
  void add_implicit_conversion_input_processor_if_needed(StringRef identifier);

  /* Add a realize on domain input processor for the input identified by the given identifier if
   * needed. See the Domain class for more information. */
  void add_realize_on_domain_input_processor_if_needed(StringRef identifier);

  /* Add the given input processor operation to the list of input processors for the input
   * identified by the given identifier. This will also involve mapping the input of the processor
   * to be the result of the last input processor or the result mapped to the input if no previous
   * processors exists. The result of the last input processor will not be mapped to the operation
   * input in this method, this will be done later, see evaluate_input_processors for more
   * information. */
  void add_input_processor(StringRef identifier, ProcessorOperation *processor);

  /* Release the results that are mapped to the inputs of the operation. This is called after the
   * evaluation of the operation to declare that the results are no longer needed by this
   * operation. */
  void release_inputs();
};

/* --------------------------------------------------------------------
 * Node Operation.
 */

using namespace nodes::derived_node_tree_types;

/* The operation class that nodes should implement and instantiate in the bNodeType
 * get_compositor_operation, passing the given inputs to the constructor.  */
class NodeOperation : public Operation {
 private:
  /* The node that this operation represents. */
  DNode node_;
  /* A vector storing the results mapped to the inputs that are not linked. */
  Vector<Result> unlinked_inputs_results_;
  /* A mapping between each unlinked input in the node identified by its identifier and its
   * corresponding input socket. */
  Map<StringRef, DInputSocket> unlinked_inputs_sockets_;

 public:
  /* Initialize members by the given arguments, populate the output results based on the node
   * outputs, populate the input types maps based on the node inputs, and add results for unlinked
   * inputs. */
  NodeOperation(Context &context, DNode node);

  /* Returns a reference to the node this operations represents. */
  const bNode &node() const;

 protected:
  /* Returns true if the output identified by the given identifier is needed and should be
   * computed, otherwise returns false. */
  bool is_output_needed(StringRef identifier) const;

  /* Set the values of the results for unlinked inputs. */
  void pre_execute() override;

 private:
  /* For each unlinked input in the node, construct a new result of an appropriate type, add it to
   * the unlinked_inputs_results_ vector, map the input to it, and map the input to its
   * corresponding input socket through the unlinked_inputs_sockets_ map. */
  void populate_results_for_unlinked_inputs();
};

/* --------------------------------------------------------------------
 * Processor Operation.
 */

/* A processor operation is an operation that takes exactly one input and computes exactly one
 * output. */
class ProcessorOperation : public Operation {
 public:
  /* The identifier of the output. This is constant for all operations. */
  static const StringRef output_identifier;
  /* The identifier of the input. This is constant for all operations. */
  static const StringRef input_identifier;

 public:
  using Operation::Operation;

  /* Get a reference to the output result of the processor, this essentially calls the super
   * get_result with the output identifier of the processor. */
  Result &get_result();

  /* Map the input of the processor to the given result, this essentially calls the super
   * map_input_to_result with the input identifier of the processor. */
  void map_input_to_result(Result *result);

 protected:
  /* Processor operations don't need input processors, so override with an empty implementation. */
  void evaluate_input_processors() override;

  /* Get a reference to the input result of the processor, this essentially calls the super
   * get_result with the input identifier of the processor. */
  Result &get_input();

  /* Switch the result mapped to the input with the given result, this essentially calls the super
   * switch_result_mapped_to_input with the input identifier of the processor. */
  void switch_result_mapped_to_input(Result *result);

  /* Populate the result of the processor, this essentially calls the super populate_result method
   * with the output identifier of the processor. */
  void populate_result(Result result);

  /* Declare the descriptor of the input of the processor to be the given descriptor, this
   * essentially calls the super declare_input_descriptor with the input identifier of the
   * processor. */
  void declare_input_descriptor(InputDescriptor descriptor);

  /* Get a reference to the descriptor of the input, this essentially calls the super
   * get_input_descriptor with the input identifier of the processor. */
  InputDescriptor &get_input_descriptor();
};

/* --------------------------------------------------------------------
 *  Reduce To Single Value Processor Operation.
 */

/* A processor that reduces its input result into a single value output result. The input is
 * assumed to be a texture result of size 1x1, that is, a texture composed of a single pixel, the
 * value of which shall serve as the single value of the output result. See
 * add_reduce_to_single_value_input_processor_if_needed. */
class ReduceToSingleValueProcessorOperation : public ProcessorOperation {
 public:
  ReduceToSingleValueProcessorOperation(Context &context, ResultType type);

  void execute() override;
};

/* --------------------------------------------------------------------
 *  Conversion Processor Operation.
 */

/* A conversion processor is a processor that converts a result from a certain type to another. See
   the derived classes for more details. */
class ConversionProcessorOperation : public ProcessorOperation {
 public:
  /* The name of the input sampler in the conversion shader.
   * This is constant for all operations. */
  static const char *shader_input_sampler_name;
  /* The name of the output image in the conversion shader.
   * This is constant for all operations. */
  static const char *shader_output_image_name;

 public:
  using ProcessorOperation::ProcessorOperation;

  void execute() override;

 protected:
  /* Convert the input single value result to the output single value result. */
  virtual void execute_single(const Result &input, Result &output) = 0;

  /* Get the shader the will be used for conversion. It should have an input sampler called
   * shader_input_sampler_name and an output image of an appropriate type called
   * shader_output_image_name. */
  virtual GPUShader *get_conversion_shader() const = 0;
};

/* --------------------------------------------------------------------
 *  Convert Float To Vector Processor Operation.
 */

/* Takes a float result and outputs a vector result. All three components of the output are filled
 * with the input float. */
class ConvertFloatToVectorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertFloatToVectorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Convert Float To Color Processor Operation.
 */

/* Takes a float result and outputs a color result. All three color channels of the output are
 * filled with the input float and the alpha channel is set to 1. */
class ConvertFloatToColorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertFloatToColorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Convert Color To Float Processor Operation.
 */

/* Takes a color result and outputs a float result. The output is the average of the three color
 * channels, the alpha channel is ignored. */
class ConvertColorToFloatProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertColorToFloatProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Convert Vector To Float Processor Operation.
 */

/* Takes a vector result and outputs a float result. The output is the average of the three
 * components. */
class ConvertVectorToFloatProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertVectorToFloatProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Convert Vector To Color Processor Operation.
 */

/* Takes a vector result and outputs a color result. The output is a copy of the three vector
 * components to the three color channels with the alpha channel set to 1. */
class ConvertVectorToColorProcessorOperation : public ConversionProcessorOperation {
 public:
  ConvertVectorToColorProcessorOperation(Context &context);

  void execute_single(const Result &input, Result &output) override;

  GPUShader *get_conversion_shader() const override;
};

/* --------------------------------------------------------------------
 *  Realize On Domain Processor Operation.
 */

/* A realize on domain processor is a processor that projects the input on a certain domain, copies
 * the area of the input that intersects the target domain, and fill the rest with zeros. See the
 * Domain class for more information. */
class RealizeOnDomainProcessorOperation : public ProcessorOperation {
 private:
  /* The target domain to realize the input on. */
  Domain domain_;

 public:
  RealizeOnDomainProcessorOperation(Context &context, Domain domain, ResultType type);

  void execute() override;

 protected:
  /* The operation domain is just the target domain. */
  Domain compute_domain() override;

 private:
  /* Get the realization shader of the appropriate type. */
  GPUShader *get_realization_shader();
};

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

/* A type representing an ordered set of nodes defining a contiguous subset of the node execution
 * schedule. */
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

/* --------------------------------------------------------------------
 * Scheduler.
 */

/* A type representing the ordered set of nodes defining the schedule of node execution. */
using Schedule = VectorSet<DNode>;

/* A class that computes the execution schedule of the nodes. It essentially does a post-order
 * depth first traversal of the node tree from the output node to the leaf input nodes, with
 * informed order of traversal of children based on a heuristic estimation of the number of
 * needed buffers. */
class Scheduler {
 private:
  /* The derived and reference node trees representing the compositor setup. */
  NodeTreeRefMap tree_ref_map_;
  DerivedNodeTree tree_;
  /* A mapping between nodes and heuristic estimations of the number of needed intermediate buffers
   * to compute the nodes and all of their dependencies. */
  Map<DNode, int> needed_buffers_;
  /* An ordered set of nodes defining the schedule of node execution. */
  Schedule schedule_;

 public:
  Scheduler(bNodeTree *node_tree);

  /* Compute the execution schedule of the nodes. */
  void schedule();

  /* Get a reference to the computed schedule. */
  Schedule &get_schedule();

 private:
  /* Computes the output node whose result should be computed and drawn. The output node is the
   * node marked as NODE_DO_OUTPUT. If multiple types of output nodes are marked, then the
   * preference will be CMP_NODE_COMPOSITE > CMP_NODE_VIEWER > CMP_NODE_SPLITVIEWER. */
  DNode compute_output_node() const;

  /* Computes a heuristic estimation of the number of needed intermediate buffers to compute this
   * node and all of its dependencies. The method recursively computes the needed buffers for all
   * node dependencies and stores them in the needed_buffers_ map. So the root/output node can be
   * provided to compute the needed buffers for all nodes. */
  int compute_needed_buffers(DNode node);

  /* Computes the execution schedule of the nodes and stores it in the schedule_. This is
   * essentially a post-order depth first traversal of the node tree from the output node to the
   * leaf input nodes, with informed order of traversal of children based on a heuristic estimation
   * of the number of needed buffers. */
  void compute_schedule(DNode node);
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
  SubSchedule sub_schedule_;

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
  SubSchedule &get_sub_schedule();
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
  /* The scheduler instance used to compute the node execution schedule. */
  Scheduler scheduler_;
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
