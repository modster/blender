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

#include "COM_ConstantFolder.h"
#include "BLI_rect.h"
#include "COM_ConstantOperation.h"
#include "COM_SetColorOperation.h"
#include "COM_SetValueOperation.h"
#include "COM_SetVectorOperation.h"

#include <limits>

namespace blender::compositor {

/**
 * \param operations: Container of operations to fold. Folded operations will be replaced
 * with constant operations.
 * \param output_links_map: Operations output links. Folded operations links will be relinked to
 * their constant operation result.
 * \param exec_system: Execution system.
 */
ConstantFolder::ConstantFolder(Vector<NodeOperation *> &operations,
                               Map<NodeOperation *, Set<NodeOperation *>> &output_links,
                               ExecutionSystem &exec_system)
    : all_operations_(operations), output_links_(output_links), exec_system_(exec_system)
{
  BLI_rcti_init(&max_area_,
                std::numeric_limits<int>::min(),
                std::numeric_limits<int>::max(),
                std::numeric_limits<int>::min(),
                std::numeric_limits<int>::max());
  BLI_rcti_init(&first_elem_area_, 0, 1, 0, 1);
}

static bool is_constant_foldable(NodeOperation *op)
{
  if (!op->get_flags().can_be_constant || op->get_flags().is_constant_operation) {
    return false;
  }
  for (int i = 0; i < op->getNumberOfInputSockets(); i++) {
    if (!op->get_input_operation(i)->get_flags().is_constant_operation) {
      return false;
    }
  }
  return true;
}

static Vector<NodeOperation *> find_constant_foldable_operations(Span<NodeOperation *> operations)
{
  Vector<NodeOperation *> foldable_ops;
  for (NodeOperation *op : operations) {
    if (is_constant_foldable(op)) {
      foldable_ops.append(op);
    }
  }
  return foldable_ops;
}

static ConstantOperation *create_constant_operation(DataType data_type, const float *constant_elem)
{
  switch (data_type) {
    case DataType::Color: {
      SetColorOperation *color_op = new SetColorOperation();
      color_op->setChannels(constant_elem);
      return color_op;
    }
    case DataType::Vector: {
      SetVectorOperation *vector_op = new SetVectorOperation();
      vector_op->setVector(constant_elem);
      return vector_op;
    }
    case DataType::Value: {
      SetValueOperation *value_op = new SetValueOperation();
      value_op->setValue(*constant_elem);
      return value_op;
    }
    default: {
      BLI_assert(!"Non implemented data type");
      return nullptr;
    }
  }
}

ConstantOperation *ConstantFolder::fold_operation(NodeOperation *op)
{
  const DataType data_type = op->getOutputSocket()->getDataType();
  MemoryBuffer *fold_buf = create_constant_buffer(data_type);
  Vector<MemoryBuffer *> inputs_bufs = get_constant_inputs_buffers(op);
  op->render(fold_buf, {first_elem_area_}, inputs_bufs, exec_system_);

  ConstantOperation *constant_op = create_constant_operation(data_type, fold_buf->get_elem(0, 0));
  all_operations_.append(constant_op);
  constants_buffers_.add_new(constant_op, fold_buf);
  relink_operation_outputs_to_constant(op, constant_op);
  remove_folded_operation(op);
  return constant_op;
}

void ConstantFolder::relink_operation_outputs_to_constant(NodeOperation *from_op,
                                                          ConstantOperation *to_op)
{
  if (!output_links_.contains(from_op)) {
    return;
  }
  Set<NodeOperation *> outputs = output_links_.pop(from_op);
  for (NodeOperation *out : outputs) {
    for (int i = 0; i < out->getNumberOfInputSockets(); i++) {
      NodeOperationInput *socket = out->getInputSocket(i);
      NodeOperationOutput *link = socket->getLink();
      if (link && &link->getOperation() == from_op) {
        socket->setLink(to_op->getOutputSocket());
        /* TODO: As resolutions are determined before constant folding we need to manually set
         * constant operations resolutions. Once tiled implementation is removed constant folding
         * should be done first and this code can be removed. */
        uint temp[2];
        uint resolution[2] = {out->getWidth(), out->getHeight()};
        to_op->getOutputSocket()->determineResolution(temp, resolution);
      }
    }
  }
  output_links_.add_new(to_op, std::move(outputs));
}

void ConstantFolder::remove_folded_operation(NodeOperation *op)
{
  output_links_.remove(op);
  folded_operations_.add(op);
  for (int i = 0; i < op->getNumberOfInputSockets(); i++) {
    NodeOperation *input = op->get_input_operation(i);
    BLI_assert(output_links_.contains(input));
    Set<NodeOperation *> &input_outputs = output_links_.lookup(input);
    input_outputs.remove(op);
    if (input_outputs.size() == 0) {
      output_links_.remove(input);
      folded_operations_.add(input);
    }
  }
}

MemoryBuffer *ConstantFolder::create_constant_buffer(DataType data_type)
{
  /* Create a single elem buffer with maximum area possible so readers can read any coordinate
   * returning always same element. */
  return new MemoryBuffer(data_type, max_area_, true);
}

Vector<MemoryBuffer *> ConstantFolder::get_constant_inputs_buffers(NodeOperation *op)
{
  const int num_inputs = op->getNumberOfInputSockets();
  Vector<MemoryBuffer *> inputs_bufs(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    ConstantOperation *constant_op = static_cast<ConstantOperation *>(op->get_input_operation(i));
    MemoryBuffer *constant_buf = constants_buffers_.lookup_or_add_cb(constant_op, [=] {
      MemoryBuffer *buf = create_constant_buffer(constant_op->getOutputSocket()->getDataType());
      constant_op->render(buf, {first_elem_area_}, {}, exec_system_);
      return buf;
    });
    inputs_bufs[i] = constant_buf;
  }
  return inputs_bufs;
}

/** Returns constant operations resulted from folding. */
Vector<ConstantOperation *> ConstantFolder::try_fold_operations(Span<NodeOperation *> operations)
{
  Vector<NodeOperation *> foldable_ops = find_constant_foldable_operations(operations);
  if (foldable_ops.size() == 0) {
    return Vector<ConstantOperation *>();
  }
  Vector<ConstantOperation *> new_folds;
  for (NodeOperation *op : foldable_ops) {
    ConstantOperation *constant_op = fold_operation(op);
    new_folds.append(constant_op);
  }
  return new_folds;
}

/**
 * Evaluate operations that have constant elements values into primitive constant operations.
 */
int ConstantFolder::fold_operations()
{
  Vector<ConstantOperation *> last_folds = try_fold_operations(all_operations_);
  int folds_count = last_folds.size();
  while (last_folds.size() > 0) {
    Vector<NodeOperation *> ops_to_fold;
    for (ConstantOperation *fold : last_folds) {
      Set<NodeOperation *> &outputs = output_links_.lookup(fold);
      ops_to_fold.extend(outputs.begin(), outputs.end());
    }
    last_folds = try_fold_operations(ops_to_fold);
    folds_count += last_folds.size();
  }
  delete_constants_buffers();
  delete_folded_operations();

  return folds_count;
}

void ConstantFolder::delete_constants_buffers()
{
  for (MemoryBuffer *buf : constants_buffers_.values()) {
    delete buf;
  }
  constants_buffers_.clear();
}

void ConstantFolder::delete_folded_operations()
{
  for (NodeOperation *op : folded_operations_) {
    all_operations_.remove_first_occurrence_and_reorder(op);
    delete op;
  }
  folded_operations_.clear();
}

}  // namespace blender::compositor
