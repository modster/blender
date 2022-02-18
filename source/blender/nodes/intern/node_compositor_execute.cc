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
#include "BLI_math_vector.h"
#include "BLI_utildefines.h"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "BKE_node.h"

#include "DNA_node_types.h"
#include "DNA_scene_types.h"

#include "GPU_texture.h"

#include "IMB_colormanagement.h"

#include "NOD_compositor_execute.hh"
#include "NOD_derived_node_tree.hh"

namespace blender::viewport_compositor {

/* --------------------------------------------------------------------
 * Texture Pool.
 */

TexturePoolKey::TexturePoolKey(int width, int height, eGPUTextureFormat format)
    : width(width), height(height), format(format)
{
}

TexturePoolKey::TexturePoolKey(const GPUTexture *texture)
{
  width = GPU_texture_width(texture);
  height = GPU_texture_height(texture);
  format = GPU_texture_format(texture);
}

uint64_t TexturePoolKey::hash() const
{
  return get_default_hash_3(width, height, format);
}

bool operator==(const TexturePoolKey &a, const TexturePoolKey &b)
{
  return a.width == b.width && a.height == b.height && a.format == b.format;
}

GPUTexture *TexturePool::acquire(int width, int height, eGPUTextureFormat format)
{
  const TexturePoolKey key = TexturePoolKey(width, height, format);
  Vector<GPUTexture *> &available_textures = textures_.lookup_or_add_default(key);
  if (available_textures.is_empty()) {
    return allocate_texture(width, height, format);
  }
  return available_textures.pop_last();
}

GPUTexture *TexturePool::acquire_color(int width, int height)
{
  return acquire(width, height, GPU_RGBA16F);
}

GPUTexture *TexturePool::acquire_vector(int width, int height)
{
  return acquire(width, height, GPU_RGB16F);
}

GPUTexture *TexturePool::acquire_float(int width, int height)
{
  return acquire(width, height, GPU_R16F);
}

void TexturePool::release(GPUTexture *texture)
{
  /* Since the given texture will always have at least one more user beside the texture pool itself
   * in a well behaved evaluation, this will not actually free the texture, it merely decrement the
   * reference the count. */
  GPU_texture_free(texture);
  /* Don't release if the texture still has more than 1 user. We check if the reference count is
   * more than 1, not zero, because the texture pool itself is considered a user of the texture. */
  if (GPU_texture_get_reference_count(texture) > 1) {
    return;
  }
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
 * Result.
 */

Result::Result(ResultType type, bool is_texture) : type(type), is_texture(is_texture)
{
}

void Result::incremenet_reference_count()
{
  if (is_texture) {
    GPU_texture_ref(data.texture);
  }
}

void Result::release(TexturePool &texture_pool)
{
  if (is_texture) {
    texture_pool.release(data.texture);
  }
}

/* --------------------------------------------------------------------
 * Operation.
 */

Operation::Operation(Context &context) : context_(context)
{
}

void Operation::initialize()
{
}

void Operation::release()
{
}

void Operation::add_result(StringRef identifier, Result result)
{
  results_.add_new(identifier, result);
}

Result &Operation::get_result(StringRef identifier)
{
  return results_.lookup(identifier);
}

void Operation::map_input_to_result(StringRef identifier, Result *result)
{
  inputs_to_results_map_.add_new(identifier, result);
  result->incremenet_reference_count();
}

const Result &Operation::get_input(StringRef identifier) const
{
  return *inputs_to_results_map_.lookup(identifier);
}

void Operation::release_inputs()
{
  for (Result *result : inputs_to_results_map_.values()) {
    result->release(texture_pool());
  }
}

Context &Operation::context()
{
  return context_;
}

TexturePool &Operation::texture_pool()
{
  return context_.texture_pool();
}

/* --------------------------------------------------------------------
 * Node Operation.
 */

using namespace nodes::derived_node_tree_types;

NodeOperation::NodeOperation(Context &context, DNode node) : Operation(context), node_(node)
{
}

/* Node operations are buffered in most cases, but the derived operation can override otherwise. */
bool NodeOperation::is_buffered() const
{
  return true;
}

bool NodeOperation::is_output_needed(StringRef identifier) const
{
  DOutputSocket output = node_.output_by_identifier(identifier);
  if (output->logically_linked_sockets().is_empty()) {
    return false;
  }
  return true;
}

const bNode &NodeOperation::node() const
{
  return *node_->bnode();
}

/* --------------------------------------------------------------------
 * Meta Operation.
 */

MetaOperation::MetaOperation(Context &context) : Operation(context)
{
}

/* --------------------------------------------------------------------
 * Single Value Input Operation.
 */

SingleValueInputOperation::SingleValueInputOperation(Context &context, DInputSocket input)
    : MetaOperation(context), input_(input)
{
}

const StringRef SingleValueInputOperation::output_identifier = StringRef("Output");

bool SingleValueInputOperation::is_buffered() const
{
  return false;
}

/* The result is a single value of the same type as the member socket. */
void SingleValueInputOperation::initialize()
{
  Result result{get_input_result_type(), false};
  add_result(output_identifier, result);
}

/* Copy the default value of the member socket to the output result. */
void SingleValueInputOperation::execute()
{
  Result &result = get_result(output_identifier);
  switch (input_->bsocket()->type) {
    case SOCK_FLOAT:
      result.data.value = input_->default_value<bNodeSocketValueFloat>()->value;
      return;
    case SOCK_VECTOR:
      copy_v3_v3(result.data.vector, input_->default_value<bNodeSocketValueVector>()->value);
      return;
    case SOCK_RGBA:
      copy_v4_v4(result.data.color, input_->default_value<bNodeSocketValueRGBA>()->value);
      return;
    default:
      BLI_assert_unreachable();
  }
}

ResultType SingleValueInputOperation::get_input_result_type() const
{
  switch (input_->bsocket()->type) {
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

  /* Execute the operations stream. */
  execute_operations_stream();
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

/* Emit and initialize all node operations in the same order as the node schedule. */
void Evaluator::compute_operations_stream(NodeSchedule &node_schedule)
{
  for (const DNode &node : node_schedule) {
    emit_node_operation(node);
  }
}

/* First map the inputs to their results, emitting and initializing any meta operations in the
 * process. Then emit and initialize the node operation corresponding to the given node. */
void Evaluator::emit_node_operation(DNode node)
{
  map_node_inputs_to_results(node);

  NodeOperation *node_operation = node_operations_.lookup(node);
  node_operation->initialize();
  operations_stream_.append(node_operation);
}

/* Call map_node_input_to_result for every input in the node. */
void Evaluator::map_node_inputs_to_results(DNode node)
{
  for (const InputSocketRef *input_ref : node->inputs()) {
    const DInputSocket input{node.context(), input_ref};
    map_node_input_to_result(input);
  }
}

/* Either call map_node_linked_input_to_result or map_node_unlinked_input_to_result depending on
 * whether the input is linked to an output or not. */
void Evaluator::map_node_input_to_result(DInputSocket input)
{
  /* The input is unlinked. */
  if (input->logically_linked_sockets().is_empty()) {
    map_node_unlinked_input_to_result(input);
    return;
  }

  input.foreach_origin_socket([&](const DSocket origin) {
    /* The input is linked to an input of a group input node that is actually unlinked, in this
     * case, we pass the input of the group input node. */
    if (origin->is_input()) {
      const DInputSocket group_input{origin.context(), &origin->as_input()};
      map_node_unlinked_input_to_result(group_input);
      return;
    }

    /* The input is linked to an output. */
    const DOutputSocket output{origin.context(), &origin->as_output()};
    map_node_linked_input_to_result(input, output);
  });
}

void Evaluator::map_node_linked_input_to_result(DInputSocket input, DOutputSocket output)
{
  NodeOperation *input_operation = node_operations_.lookup(input.node());
  NodeOperation *output_operation = node_operations_.lookup(output.node());
  /* Map the input to the result we get from the output. */
  Result &result = output_operation->get_result(output->identifier());
  input_operation->map_input_to_result(input->identifier(), &result);
}

void Evaluator::map_node_unlinked_input_to_result(DInputSocket input)
{
  /* Emit a SingleValueInputOperation for that input. */
  SingleValueInputOperation *value_operation = new SingleValueInputOperation(context, input);
  value_operation->initialize();
  operations_stream_.append(value_operation);

  /* Map the input to the result we get from the single value operation. */
  NodeOperation *input_operation = node_operations_.lookup(input.node());
  Result &result = value_operation->get_result(SingleValueInputOperation::output_identifier);
  input_operation->map_input_to_result(input->identifier(), &result);
}

/* Call execute_operation for every operation in the stream in order. */
void Evaluator::execute_operations_stream()
{
  for (Operation *operation : operations_stream_) {
    execute_operation(operation);
  }
}

void Evaluator::execute_operation(Operation *operation)
{
  operation->execute();
  operation->release();
  operation->release_inputs();
}

}  // namespace blender::viewport_compositor
