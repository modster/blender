/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_vector.hh"

#include "DNA_node_types.h"

#include "GPU_material.h"

#include "NOD_derived_node_tree.hh"

namespace blender::realtime_compositor {

using namespace nodes::derived_node_tree_types;

/* ------------------------------------------------------------------------------------------------
 * GPU Material Node
 *
 * A class that represents a node in a GPU material. The GPU node stacks for inputs and outputs are
 * stored and populated during construction. Derived class should implement the compile method to
 * implement the node and link it to the GPU material. The GPU material compiler is expected to
 * initialize the input links of the node before invoking the compile method. */
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
  /* Returns a reference to the derived node that this operation represents. */
  const DNode &node() const;

  /* Returns a reference to the node this operations represents. */
  bNode &bnode() const;

 private:
  /* Populate the inputs of the node. The input link is set to nullptr and is expected to be
   * initialized by the GPU material compiler before calling the compile method. */
  void populate_inputs();
  /* Populate the outputs of the node. The output link is set to nullptr and is expected to be
   * initialized by the compile method. */
  void populate_outputs();
};

}  // namespace blender::realtime_compositor
