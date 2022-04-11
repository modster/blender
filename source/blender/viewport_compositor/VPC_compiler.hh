/* SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright 2022 Blender Foundation. All rights reserved. */

#pragma once

#include "BLI_vector.hh"

#include "DNA_node_types.h"

#include "NOD_derived_node_tree.hh"

#include "VPC_context.hh"
#include "VPC_gpu_material_operation.hh"
#include "VPC_node_operation.hh"
#include "VPC_operation.hh"

namespace blender::viewport_compositor {

using namespace nodes::derived_node_tree_types;

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
  /* The derived and reference node trees representing the compositor setup. */
  NodeTreeRefMap tree_ref_map_;
  DerivedNodeTree tree_;
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

}  // namespace blender::viewport_compositor
