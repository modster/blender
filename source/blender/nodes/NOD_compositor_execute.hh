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
 * the node tree. Concrete derived classes are expected to free the textures once the pool is no
 * longer in use. */
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
   * implemented by the compositor engine and should ideally use the DRW texture pool for
   * allocation. */
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
 * is_domain, see InputDescriptor. The realization process simply projects the input domain on the
 * operation domain, copies the area of input that intersects the operation domain, and fill the
 * rest with zeros. This process is illustrated below. It follows that operations should expect all
 * their inputs to have the same domain and consequently size, except possibly for inputs that skip
 * realization.
 *
 *                                   Realized Result
 *             +-------------+       +-------------+
 *             |  Operation  |       |             |
 *             |   Domain    |       |    Zeros    |
 *             |             | ----> |             |
 *       +-----------+       |       |-----+       |
 *       |     |  C  |       |       |  C  |       |
 *       |     +-----|-------+       +-----|-------+
 *       | Input     |
 *       | Domain    |
 *       +-----------+
 *
 * Each operation can define an arbitrary operation domain, but in most cases, the operation domain
 * is inferred from the inputs. By default, the operation domain is computed as follows. Typically,
 * one input of the operation is said to be a domain input and it defines the operation domain. So
 * if the operation have an input whose descriptor sets is_domain and is not a single value input,
 * then the operation domain will be the same domain as the first such input. See the
 * InputDescriptor class. Otherwise, if no domain inputs exists or all are single value inputs,
 * then the first non single value input is used to define the operation domain. If all inputs are
 * single values, then the operation domain is irrelevant and an identity domain is set. See
 * NodeOperation::compute_domain.
 *
 * The aforementioned logic for operation domain computation is only a default that works for most
 * cases, but an operation can override the compute_domain method to implement a different logic.
 * For instance, output nodes have an operation domain the same size as the viewport and with an
 * identity transformation, their operation domain doesn't depend on the inputs at all.
 *
 * For instance, a filter operation have two inputs, a factor and a color, the latter of which
 * is a domain input. If the color input is not a single value, then the domain of this operation
 * is computed to be the same size and transformation as the color input. And if the factor input
 * have a different size and/or transformation from the computed domain of the operation, it will
 * be projected and realized on it to have the same size as described above. It follows that the
 * color input, which is a domain input, will not need to be realized because it already has the
 * same size and transformation as the domain of the operation, because the operation domain is
 * derived from it. On the other hand, if the color input is a single value input, then the
 * operation domain will be the same as the domain of the factor input. Finally, if both inputs are
 * single value inputs, the operation domain will be an identity and is irrelevant. */
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
 * would have been constant over the whole texture. Even if the result is a single value, the
 * texture member should and will be pointing to a valid dummy texture. This dummy texture
 * shouldn't and will not be used and its contents needn't and will not be initialized, it solely
 * exists because operation shaders are built in such a way to operate on both texture and single
 * value input results. Each shader have three uniforms for each input. 1) A boolean that indicates
 * if this input is a single value or a texture. 2) A float that is initialized only if the input
 * is a single value result. 3) A texture that is initialized only if the input is a texture
 * result. Since shaders expect a texture to be bound even if it will not be used in the shader
 * invocation, a dummy texture is bound regardless, hence the need to always store a dummy texture
 * if the result is a single value. */
class Result {
 public:
  /* The base type of the texture or the type of the single value. */
  ResultType type;
  /* If true, the result is a texture, otherwise, the result is a single value. */
  bool is_texture;
  /* If the result is a texture, this member points to a GPU texture storing the result data. If
   * the result is not a texture, this member points to a valid dummy GPU texture. See class
   * description above. */
  GPUTexture *texture = nullptr;
  /* The texture pool used to allocate the texture of the result, this should be initialized during
   * construction. */
  TexturePool &texture_pool;
  /* The number of users currently referencing and using this result. */
  int reference_count = 0;
  /* If the result is a single value, this member stores the value of the result. While this member
   * stores 4 values, only a subset of which could be initialized depending on the type, for
   * instance, a float result will only initialize the first array element and a vector result will
   * only initialize the first three array elements. This member is uninitialized if the result is
   * a texture. */
  float value[4];
  /* The transformation of the result. This only matters if the result was a texture. See the
   * Domain class. */
  Transformation2D transformation = Transformation2D::identity();

 public:
  Result(ResultType type, TexturePool &texture_pool);

  /* Declare the result to be a texture result and allocate a texture of an appropriate type with
   * the given size from the given texture pool. */
  void allocate_texture(int2 size);

  /* Declare the result to be a single value result and allocate a dummy texture of an appropriate
   * type from the given texture pool. See class description for more information. */
  void allocate_single_value();

  /* Bind the texture of the result to the texture image unit with the given name in the currently
   * bound given shader. */
  void bind_as_texture(GPUShader *shader, const char *texture_name) const;

  /* Bind the result as an input that could be single value or a texture to the currently bound
   * given shader. The names of three uniforms corresponding to the input are given as arguments.
   * See class description for more details. */
  void bind_as_generic_input(GPUShader *shader,
                             const char *is_texture_name,
                             const char *value_name,
                             const char *texture_name) const;

  /* Bind the texture of the result to the image unit with the given name in the currently bound
   * given shader. */
  void bind_as_image(GPUShader *shader, const char *image_name) const;

  /* Unbind the texture which was previously bound using bind_as_texture or bind_as_generic_input.
   * This should be called even for single value results due to the use of the dummy texture. */
  void unbind_as_texture() const;

  /* Unbind the texture which was previously bound using bind_as_image. */
  void unbind_as_image() const;

  /* Increment the reference count of the result. This should be called when a user gets a
   * reference to the result to use as an input. */
  void incremenet_reference_count();

  /* Release the result texture back into the texture pool. This should be called when a user that
   * previously referenced and incremented the reference count of the result no longer needs it. */
  void release();

  /* Returns the size of the allocated texture. */
  int2 size() const;

  /* Returns the domain of the result. See the Domain class. */
  Domain domain() const;
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
  /* If true, then the input is considered to be a domain input that is used by default to define
   * the domain of the operation, this is typically the main input of the operation. See the Domain
   * class for more information. */
  bool is_domain = false;
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
   * and added to the map during operation construction. The results should be allocated in the
   * initialization methods. The contents of the results should be computed in the execution
   * methods. */
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

  /* This method should return true if this operation can only operate on buffers, otherwise,
   * return false if the operation can be applied pixel-wise. */
  virtual bool is_buffered() const = 0;

  /* Calls the initialization methods. */
  void initialize();

  /* Calls the execution methods followed by the release methods. */
  void evaluate();

  /* Get a reference to the output result identified by the given identifier. */
  Result &get_result(StringRef identifier);

  /* Map the input identified by the given identifier to the result providing its data. This also
   * increments the reference count of the result. See inputs_to_results_map_ for more details.
   * This should be called by the evaluator to establish links between different operations. */
  void map_input_to_result(StringRef identifier, Result *result);

 protected:
  /* Compute the operation domain of this operation. See the Domain class for more information. */
  virtual Domain compute_domain() = 0;

  /* This method is called before the allocate method and it can be overridden by a derived class
   * to do any necessary internal allocations. */
  virtual void pre_allocate();

  /* This method should allocate all the necessary buffers needed by the operation. This includes
   * the output results as well as any temporary intermediate buffers used by the operation. The
   * texture pool provided by the context should be used to do any texture allocations. */
  virtual void allocate();

  /* This method is called before the execute method and it can be overridden by a derived class
   * to do any necessary internal computations. */
  virtual void pre_execute();

  /* This method should execute the operation, compute its outputs, and write them to the
   * appropriate result. */
  virtual void execute() = 0;

  /* This method is called before the release method and it can be overridden by a derived class
   * to do any necessary internal releases. */
  virtual void pre_release();

  /* This method should release any temporary intermediate buffers that were allocated in the
   * allocation method. */
  virtual void release();

  /* Get a reference to the result connected to the input identified by the given identifier. */
  Result &get_input(StringRef identifier) const;

  /* Switch the result mapped to the input identified by the given identifier with the given
   * result. This will involve releasing original result, but it is assumed that the result will be
   * mapped to something else. */
  void switch_result_mapped_to_input(StringRef identifier, Result *result);

  /* Add the given result to the results_ map identified by the given output identifier. This
   * should be called during operation construction for every output. The provided result shouldn't
   * be allocated or initialized, this will happen later after initialization and execution. */
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
  /* Add all the necessary input processors for each input as needed. Then update the mapped result
   * for each input to be that of the last processor for that input if any input processors exist
   * for it. This is done now in a separate step after all processors were added because the
   * operation might use the original mapped results to determine what processors needs to be
   * added. This method is called in the operation initialization method after pre_allocate but
   * before the allocate method, it needs to happen then and not before at operation construction
   * because the logic for adding input processors can depend on the nature of the input results,
   * but not on their value. */
  void add_input_processors();

  /* Add an implicit conversion input processor for the input identified by the given identifier if
   * needed. */
  void add_implicit_conversion_input_processor_if_needed(StringRef identifier);

  /* Add a realize on domain input processor for the input identified by the given identifier if
   * needed. See the Domain class for more information. */
  void add_realize_on_domain_input_processor_if_needed(StringRef identifier);

  /* Add the given input processor operation to the list of input processors for the input
   * identified by the given identifier. The result of the last input processor will not be mapped
   * to the input in this method, this is done later, see add_input_processors for more
   * information. */
  void add_input_processor(StringRef identifier, ProcessorOperation *processor);

  /* Allocate all input processors in order. This is called before allocating the operation but
   * after the pre_allocate method was called. */
  void allocate_input_processors();

  /* Execute all input processors in order. This is called before executing the operation to
   * prepare its inputs but after the pre_execute method was called. */
  void execute_input_processors();

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

  /* Most node operations are buffered, so return true by default. */
  bool is_buffered() const override;

  /* Returns true if the output identified by the given identifier is needed and should be
   * computed, otherwise returns false. */
  bool is_output_needed(StringRef identifier) const;

  /* Returns a reference to the node this operations represents. */
  const bNode &node() const;

 protected:
  /* Compute the domain of the node operation. This implement the default logic that infers the
   * operation domain from the node, which may be overridden for a different logic. See the Domain
   * class for the inference logic. */
  Domain compute_domain() override;

  /* Allocate the results for unlinked inputs. */
  void pre_allocate() override;

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

  /* Processor operations are always buffered. */
  bool is_buffered() const override;

  /* Get a reference to the output result of the processor, this essentially calls the super
   * get_result with the output identifier of the processor. */
  Result &get_result();

  /* Map the input of the processor to the given result, this essentially calls the super
   * map_input_to_result with the input identifier of the processor. */
  void map_input_to_result(Result *result);

 protected:
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

  void allocate() override;

  void execute() override;

 protected:
  /* The operation domain is just the domain of the input. */
  Domain compute_domain() override;

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

  void allocate() override;

  void execute() override;

 protected:
  /* The operation domain is just the target domain. */
  Domain compute_domain() override;

 private:
  /* Get the realization shader of the appropriate type. */
  GPUShader *get_realization_shader();
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

  /* Compile the node schedule into the stream of operations that will be evaluated in order by the
   * evaluator, and store the result in operations_stream_. */
  void compute_operations_stream(NodeSchedule &node_schedule);

  /* Maps each of the inputs of the node operation to the result of output linked to it. */
  void map_node_inputs_to_results(DNode node);

  /* Initialize the operations in the operations stream in order. */
  void initialize_operations_stream();

  /* Evaluate the operations in the operations stream in order. */
  void evaluate_operations_stream();
};

}  // namespace blender::viewport_compositor
