/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#include "BLI_map.hh"
#include "BLI_stack.hh"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "NOD_derived_node_tree.hh"

#include "VPC_scheduler.hh"
#include "VPC_utilities.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

/* Compute the output node whose result should be computed. The output node is the node marked as
 * NODE_DO_OUTPUT. If multiple types of output nodes are marked, then the preference will be
 * CMP_NODE_COMPOSITE > CMP_NODE_VIEWER > CMP_NODE_SPLITVIEWER. If no output node exists, a null
 * node will be returned. */
static DNode compute_output_node(DerivedNodeTree &tree)
{
  /* Get the top most node tree reference from the derived node tree. */
  const NodeTreeRef &root_tree = tree.root_context().tree();

  /* First search over composite nodes. */
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeComposite")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree.root_context(), node);
    }
  }

  /* Then search over viewer nodes. */
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeViewer")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree.root_context(), node);
    }
  }

  /* Finally search over split viewer nodes. */
  for (const NodeRef *node : root_tree.nodes_by_type("CompositorNodeSplitViewer")) {
    if (node->bnode()->flag & NODE_DO_OUTPUT) {
      return DNode(&tree.root_context(), node);
    }
  }

  /* No output node found, return a null node. */
  return DNode();
}

/* A type representing a mapping that associates each node with a heuristic estimation of the
 * number of intermediate buffers needed to compute it and all of its dependencies. See the
 * compute_number_of_needed_buffers function for more information. */
using NeededBuffers = Map<DNode, int>;

/* Compute a heuristic estimation of the number of intermediate buffers needed to compute each node
 * and all of its dependencies for all nodes that the given node depends on. The output is a map
 * that maps each node with the number of intermediate buffers needed to compute it and all of its
 * dependencies.
 *
 * Consider a node that takes n number of buffers as an input from a number of node dependencies,
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
 * Note that the computed output is not guaranteed to be accurate, and will not be in most cases.
 * The computation is merely a heuristic estimation that works well in most cases. This is due to a
 * number of reasons:
 * - The node tree is actually a graph that allows output sharing, which is not something that was
 *   taken into consideration in this implementation.
 * - Each node may allocate any number of internal buffers, which is not taken into account in this
 *   implementation. */
static NeededBuffers compute_number_of_needed_buffers(DNode output_node)
{
  NeededBuffers needed_buffers;

  /* A stack of nodes used to traverse the node tree in a post order depth first manner starting
   * from the output node. */
  Stack<DNode> node_stack = {output_node};

  /* Traverse the node tree in a post order depth first manner and compute the number of needed
   * buffers for each node. Post order traversal guarantee that all the node dependencies of each
   * node are computed before it. This is done by pushing all the uncomputed node dependencies to
   * the node stack first and only popping and computing the node when all its node dependencies
   * were computed. */
  while (!node_stack.is_empty()) {
    /* Do not pop the node immediately, as it may turn out that we can't compute its number of
     * needed buffers just yet because its dependencies weren't computed, it will be popped later
     * when needed. */
    DNode &node = node_stack.peek();

    /* Go over the node dependencies connected to the inputs of the node and push them to the node
     * stack if they were not computed already. */
    bool any_of_the_node_dependencies_were_pushed = false;
    for (const InputSocketRef *input_ref : node->inputs()) {
      const DInputSocket input{node.context(), input_ref};

      /* Get the origin socket of this input, which will be an output socket if the input is linked
       * to an output. */
      DSocket origin = get_node_input_origin_socket(input);

      /* If the origin socket is an input, that means the input is unlinked and has no dependency
       * node. */
      if (origin->is_input()) {
        continue;
      }

      /* The node dependency was already computed before, so skip it. */
      if (needed_buffers.contains(origin.node())) {
        continue;
      }

      /* The origin node needs to be computed, push the node dependency to the node stack and
       * indicate that it was pushed. */
      node_stack.push(origin.node());
      any_of_the_node_dependencies_were_pushed = true;
    }

    /* If any of the node dependencies were pushed, that means that not all of them were computed
     * and consequently we can't compute the number of needed buffers for this node just yet. */
    if (any_of_the_node_dependencies_were_pushed) {
      continue;
    }

    /* We don't need to store the result of the pop because we already peeked at it before. */
    node_stack.pop();

    /* Compute the number of buffers that the node takes as an input as well as the number of
     * buffers needed to compute the most demanding of the node dependencies. */
    int number_of_input_buffers = 0;
    int buffers_needed_by_dependencies = 0;
    for (const InputSocketRef *input_ref : node->inputs()) {
      const DInputSocket input{node.context(), input_ref};

      /* Get the origin socket of this input, which will be an output socket if the input is linked
       * to an output. */
      DSocket origin = get_node_input_origin_socket(input);

      /* If the origin socket is an input, that means the input is unlinked. Unlinked inputs do not
       * take a buffer, so skip those inputs. */
      if (origin->is_input()) {
        continue;
      }

      /* Since this input is linked, it means that the node takes a buffer through this input and
       * so we increment the number of input buffers. */
      number_of_input_buffers++;

      /* If the number of buffers needed by the node dependency is more than the total number of
       * buffers needed by the dependencies, then update the latter to be the former. This is
       * computing the "d" in the aformentioned equation "max(n + m, d)". */
      const int buffers_needed_by_origin = needed_buffers.lookup(origin.node());
      if (buffers_needed_by_origin > buffers_needed_by_dependencies) {
        buffers_needed_by_dependencies = buffers_needed_by_origin;
      }
    }

    /* Compute the number of buffers that will be computed/output by this node. */
    int number_of_output_buffers = 0;
    for (const OutputSocketRef *output : node->outputs()) {
      if (!output->logically_linked_sockets().is_empty()) {
        number_of_output_buffers++;
      }
    }

    /* Compute the heuristic estimation of the number of needed intermediate buffers to compute
     * this node and all of its dependencies. This is computing the aformentioned equation
     * "max(n + m, d)". */
    const int total_buffers = MAX2(number_of_input_buffers + number_of_output_buffers,
                                   buffers_needed_by_dependencies);
    needed_buffers.add_new(node, total_buffers);
  }

  return needed_buffers;
}

/* There are multiple different possible orders of evaluating a node graph, each of which needs
 * to allocate a number of intermediate buffers to store its intermediate results. It follows
 * that we need to find the evaluation order which uses the least amount of intermediate buffers.
 * For instance, consider a node that takes two input buffers A and B. Each of those buffers is
 * computed through a number of nodes constituting a sub-graph whose root is the node that
 * outputs that buffer. Suppose the number of intermediate buffers needed to compute A and B are
 * N(A) and N(B) respectively and N(A) > N(B). Then evaluating the sub-graph computing A would be
 * a better option than that of B, because had B was computed first, its outputs will need to be
 * stored in extra buffers in addition to the buffers needed by A. The number of buffers needed by
 * each node is estimated as described in the compute_number_of_needed_buffers function.
 *
 * This is a heuristic generalization of the Sethi–Ullman algorithm, a generalization that
 * doesn't always guarantee an optimal evaluation order, as the optimal evaluation order is very
 * difficult to compute, however, this method works well in most cases. Moreover it assumes that
 * all buffers will have roughly the same size, which may not always be the case. */
Schedule compute_schedule(DerivedNodeTree &tree)
{
  Schedule schedule;

  /* Compute the output node whose result should be computed. */
  const DNode output_node = compute_output_node(tree);

  /* No output node, the node tree has no effect, return an empty schedule. */
  if (!output_node) {
    return schedule;
  }

  /* Compute the number of buffers needed by each node connected to the output. */
  const NeededBuffers needed_buffers = compute_number_of_needed_buffers(output_node);

  /* A stack of nodes used to traverse the node tree in a post order depth first manner starting
   * from the output node. */
  Stack<DNode> node_stack = {output_node};

  /* Traverse the node tree in a post order depth first manner, scheduling the nodes in an order
   * informed by the number of buffers needed by each node. Post order traversal guarantee that all
   * the node dependencies of each node are scheduled before it. This is done by pushing all the
   * unscheduled node dependencies to the node stack first and only popping and scheduling the node
   * when all its node dependencies were scheduled. */
  while (!node_stack.is_empty()) {
    /* Do not pop the node immediately, as it may turn out that we can't schedule it just yet
     * because its dependencies weren't scheduled, it will be popped later when needed. */
    DNode &node = node_stack.peek();

    /* Compute the nodes directly connected to the node inputs sorted by their needed buffers such
     * that the node with the lowest number of needed buffers comes first. Note that we actually
     * want the node with the highest number of needed buffers to be schedule first, but since
     * those are pushed to the traversal stack, we need to push them in reverse order. */
    Vector<DNode> sorted_origin_nodes;
    for (const InputSocketRef *input_ref : node->inputs()) {
      const DInputSocket input{node.context(), input_ref};

      /* Get the origin socket of this input, which will be an output socket if the input is linked
       * to an output. */
      DSocket origin = get_node_input_origin_socket(input);

      /* If the origin socket is an input, that means the input is unlinked and has no dependency
       * node, so skip it. */
      if (origin->is_input()) {
        continue;
      }

      /* The origin node was added before, so skip it. The number of origin nodes is very small,
       * typically less than 3, so a linear search is okay. */
      if (sorted_origin_nodes.contains(origin.node())) {
        continue;
      }

      /* The origin node was already schedule, so skip it. */
      if (schedule.contains(origin.node())) {
        continue;
      }

      /* Sort in ascending order on insertion, the number of origin nodes is very small, typically
       * less than 3, so insertion sort is okay. */
      int insertion_position = 0;
      for (int i = 0; i < sorted_origin_nodes.size(); i++) {
        if (needed_buffers.lookup(origin.node()) > needed_buffers.lookup(sorted_origin_nodes[i])) {
          insertion_position++;
        }
        else {
          break;
        }
      }
      sorted_origin_nodes.insert(insertion_position, origin.node());
    }

    /* Push the sorted origin nodes to the node stack in order. */
    for (const DNode &origin_node : sorted_origin_nodes) {
      node_stack.push(origin_node);
    }

    /* If there are no sorted origin nodes, that means they were all already scheduled or that none
     * exists in the first place, so we can pop and schedule the node now. */
    if (sorted_origin_nodes.is_empty()) {
      /* The node might have already been scheduled, so we don't use add_new here and simply don't
       * add it if it was already scheduled. */
      schedule.add(node_stack.pop());
    }
  }

  return schedule;
}

}  // namespace blender::viewport_compositor
