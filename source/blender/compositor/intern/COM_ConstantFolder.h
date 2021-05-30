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
 *
 * Copyright 2011, Blender Foundation.
 */

#pragma once

#include "BLI_map.hh"
#include "BLI_set.hh"
#include "BLI_vector.hh"

#include "COM_ExecutionSystem.h"

namespace blender::compositor {
class NodeOperation;
class ConstantOperation;

/**
 */
class ConstantFolder {
 private:
  Vector<NodeOperation *> &all_operations_;
  Map<NodeOperation *, Set<NodeOperation *>> &output_links_;
  ExecutionSystem &exec_system_;

  /**
   * Operations that has been folded (evaluated into constant operations). They are deleted
   * when folding is finished.
   */
  Set<NodeOperation *> folded_operations_;

  /** Created constant operations buffers during folding. */
  Map<ConstantOperation *, MemoryBuffer *> constants_buffers_;

  rcti max_area_;
  rcti first_elem_area_;

 public:
  ConstantFolder(Vector<NodeOperation *> &operations,
                 Map<NodeOperation *, Set<NodeOperation *>> &output_links,
                 ExecutionSystem &exec_system);
  int fold_operations();

 private:
  Vector<ConstantOperation *> try_fold_operations(Span<NodeOperation *> operations);
  ConstantOperation *fold_operation(NodeOperation *op);
  void relink_operation_outputs_to_constant(NodeOperation *from_op, ConstantOperation *to_op);
  void remove_folded_operation(NodeOperation *op);
  MemoryBuffer *create_constant_buffer(DataType data_type);
  Vector<MemoryBuffer *> get_constant_inputs_buffers(NodeOperation *op);
  void delete_folded_operations();
  void delete_constants_buffers();
};

}  // namespace blender::compositor
