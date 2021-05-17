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

#include <cstdio>
#include <memory>
#include <typeinfo>

#include "COM_BufferOperation.h"
#include "COM_ExecutionSystem.h"
#include "COM_ReadBufferOperation.h"
#include "COM_defines.h"

#include "COM_NodeOperation.h" /* own include */

namespace blender::compositor {

/*******************
 **** NodeOperation ****
 *******************/

NodeOperation::NodeOperation()
{
  this->m_resolutionInputSocketIndex = 0;
  this->m_width = 0;
  this->m_height = 0;
  this->m_btree = nullptr;
}

NodeOperationOutput *NodeOperation::getOutputSocket(unsigned int index)
{
  return &m_outputs[index];
}

NodeOperationInput *NodeOperation::getInputSocket(unsigned int index)
{
  return &m_inputs[index];
}

void NodeOperation::addInputSocket(DataType datatype, ResizeMode resize_mode)
{
  m_inputs.append(NodeOperationInput(this, datatype, resize_mode));
}

void NodeOperation::addOutputSocket(DataType datatype)
{
  m_outputs.append(NodeOperationOutput(this, datatype));
}

void NodeOperation::determineResolution(unsigned int resolution[2],
                                        unsigned int preferredResolution[2])
{
  unsigned int used_resolution_index = 0;
  if (m_resolutionInputSocketIndex == RESOLUTION_INPUT_ANY) {
    for (NodeOperationInput &input : m_inputs) {
      unsigned int any_resolution[2] = {0, 0};
      input.determineResolution(any_resolution, preferredResolution);
      if (any_resolution[0] * any_resolution[1] > 0) {
        resolution[0] = any_resolution[0];
        resolution[1] = any_resolution[1];
        break;
      }
      used_resolution_index += 1;
    }
  }
  else if (m_resolutionInputSocketIndex < m_inputs.size()) {
    NodeOperationInput &input = m_inputs[m_resolutionInputSocketIndex];
    input.determineResolution(resolution, preferredResolution);
    used_resolution_index = m_resolutionInputSocketIndex;
  }
  unsigned int temp2[2] = {resolution[0], resolution[1]};

  unsigned int temp[2];
  for (unsigned int index = 0; index < m_inputs.size(); index++) {
    if (index == used_resolution_index) {
      continue;
    }
    NodeOperationInput &input = m_inputs[index];
    if (input.isConnected()) {
      input.determineResolution(temp, temp2);
    }
  }
}

void NodeOperation::setResolutionInputSocketIndex(unsigned int index)
{
  this->m_resolutionInputSocketIndex = index;
}
void NodeOperation::initExecution()
{
  /* pass */
}

void NodeOperation::initMutex()
{
  BLI_mutex_init(&this->m_mutex);
}

void NodeOperation::lockMutex()
{
  BLI_mutex_lock(&this->m_mutex);
}

void NodeOperation::unlockMutex()
{
  BLI_mutex_unlock(&this->m_mutex);
}

void NodeOperation::deinitMutex()
{
  BLI_mutex_end(&this->m_mutex);
}

void NodeOperation::deinitExecution()
{
  /* pass */
}
SocketReader *NodeOperation::getInputSocketReader(unsigned int inputSocketIndex)
{
  return this->getInputSocket(inputSocketIndex)->getReader();
}

NodeOperation *NodeOperation::getInputOperation(unsigned int inputSocketIndex)
{
  NodeOperationInput *input = getInputSocket(inputSocketIndex);
  if (input && input->isConnected()) {
    return &input->getLink()->getOperation();
  }

  return nullptr;
}

bool NodeOperation::determineDependingAreaOfInterest(rcti *input,
                                                     ReadBufferOperation *readOperation,
                                                     rcti *output)
{
  if (m_inputs.size() == 0) {
    BLI_rcti_init(output, input->xmin, input->xmax, input->ymin, input->ymax);
    return false;
  }

  rcti tempOutput;
  bool first = true;
  for (int i = 0; i < getNumberOfInputSockets(); i++) {
    NodeOperation *inputOperation = this->getInputOperation(i);
    if (inputOperation &&
        inputOperation->determineDependingAreaOfInterest(input, readOperation, &tempOutput)) {
      if (first) {
        output->xmin = tempOutput.xmin;
        output->ymin = tempOutput.ymin;
        output->xmax = tempOutput.xmax;
        output->ymax = tempOutput.ymax;
        first = false;
      }
      else {
        output->xmin = MIN2(output->xmin, tempOutput.xmin);
        output->ymin = MIN2(output->ymin, tempOutput.ymin);
        output->xmax = MAX2(output->xmax, tempOutput.xmax);
        output->ymax = MAX2(output->ymax, tempOutput.ymax);
      }
    }
  }
  return !first;
}

/* -------------------------------------------------------------------- */
/** \name Full Frame Methods
 * \{ */

/**
 * Get input area being read by this operation.
 *
 * Implementation don't need to ensure r_input_rect is within operation bounds. The caller must
 * clamp it.
 */
void NodeOperation::get_area_of_interest(const int input_idx,
                                         const rcti &output_rect,
                                         rcti &r_input_rect)
{
  if (get_flags().is_fullframe_operation) {
    r_input_rect = output_rect;
  }
  else {
    /* Non full-frame operations never implement this method. To ensure correctness assume
     * whole area is used. */
    NodeOperation *input_op = getInputOperation(input_idx);
    BLI_rcti_init(&r_input_rect, 0, input_op->getWidth(), 0, input_op->getHeight());
  }
}

/**
 * Renders operation and its inputs. Rendered buffers are saved in the output store.
 */
void NodeOperation::render(ExecutionSystem &exec_system)
{
  SharedOperationBuffers &active_buffers = exec_system.get_active_buffers();
  if (active_buffers.is_operation_rendered(this)) {
    return;
  }

  Vector<MemoryBuffer *> inputs_bufs = get_rendered_inputs_buffers(exec_system);

  const bool has_outputs = getNumberOfOutputSockets() > 0;
  MemoryBuffer *output_buf = has_outputs ? create_output_buffer() : nullptr;

  Span<rcti> render_rects = active_buffers.get_rects_to_render(this);
  if (get_flags().is_fullframe_operation) {
    render_full_frame(output_buf, render_rects, inputs_bufs, exec_system);
  }
  else {
    render_full_frame_fallback(output_buf, render_rects, inputs_bufs, exec_system);
  }
  active_buffers.set_rendered_buffer(this, std::unique_ptr<MemoryBuffer>(output_buf));

  exec_system.operation_finished(this);
}

void NodeOperation::render_full_frame(MemoryBuffer *output_buf,
                                      Span<rcti> render_rects,
                                      Span<MemoryBuffer *> inputs_bufs,
                                      ExecutionSystem &exec_system)
{
  initExecution();
  for (const rcti &render_rect : render_rects) {
    update_memory_buffer(output_buf, render_rect, inputs_bufs, exec_system);
  }
  deinitExecution();
}

Vector<MemoryBuffer *> NodeOperation::get_rendered_inputs_buffers(ExecutionSystem &exec_system)
{
  SharedOperationBuffers &active_buffers = exec_system.get_active_buffers();

  const int num_inputs = getNumberOfInputSockets();
  Vector<MemoryBuffer *> inputs_buffers(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    NodeOperation *input_op = getInputOperation(i);
    if (!active_buffers.is_operation_rendered(input_op)) {
      input_op->render(exec_system);
    }
    inputs_buffers[i] = active_buffers.get_rendered_buffer(input_op);
  }
  return inputs_buffers;
}

MemoryBuffer *NodeOperation::create_output_buffer()
{
  rcti op_rect;
  BLI_rcti_init(&op_rect, 0, getWidth(), 0, getHeight());

  const DataType data_type = getOutputSocket(0)->getDataType();
  /* TODO: We should check if the operation is constant instead of is_set_operation. Finding a way
   * to know if an operation is constant has to be implemented yet. */
  const bool is_a_single_elem = get_flags().is_set_operation;
  return new MemoryBuffer(data_type, op_rect, is_a_single_elem);
}

/**
 * Renders operation using the tiled implementation.
 */
void NodeOperation::render_full_frame_fallback(MemoryBuffer *output_buf,
                                               Span<rcti> render_rects,
                                               Span<MemoryBuffer *> inputs_bufs,
                                               ExecutionSystem &exec_system)
{
  Vector<NodeOperationOutput *> orig_input_links = replace_inputs_with_buffers(inputs_bufs);

  initExecution();
  const bool is_output_operation = getNumberOfOutputSockets() == 0;
  if (!is_output_operation && output_buf->is_a_single_elem()) {
    float *output_elem = output_buf->get_elem(0, 0);
    readSampled(output_elem, 0, 0, PixelSampler::Nearest);
  }
  else {
    for (const rcti &rect : render_rects) {
      exec_system.execute_work(rect, [=](const rcti &split_rect) {
        rcti tile_rect = split_rect;
        if (is_output_operation) {
          executeRegion(&tile_rect, 0);
        }
        else {
          render_tile(output_buf, &tile_rect);
        }
      });
    }
  }
  deinitExecution();

  remove_buffers_and_restore_original_inputs(orig_input_links);
}

void NodeOperation::render_tile(MemoryBuffer *output_buf, rcti *tile_rect)
{
  const bool is_complex = get_flags().complex;
  void *tile_data = is_complex ? initializeTileData(tile_rect) : nullptr;
  const int elem_stride = output_buf->elem_stride;
  for (int y = tile_rect->ymin; y < tile_rect->ymax; y++) {
    float *output_elem = output_buf->get_elem(tile_rect->xmin, y);
    if (is_complex) {
      for (int x = tile_rect->xmin; x < tile_rect->xmax; x++) {
        read(output_elem, x, y, tile_data);
        output_elem += elem_stride;
      }
    }
    else {
      for (int x = tile_rect->xmin; x < tile_rect->xmax; x++) {
        readSampled(output_elem, x, y, PixelSampler::Nearest);
        output_elem += elem_stride;
      }
    }
  }
  if (tile_data) {
    deinitializeTileData(tile_rect, tile_data);
  }
}

/**
 * \return Replaced inputs links.
 */
Vector<NodeOperationOutput *> NodeOperation::replace_inputs_with_buffers(
    Span<MemoryBuffer *> inputs_bufs)
{
  BLI_assert(inputs_bufs.size() == getNumberOfInputSockets());
  Vector<NodeOperationOutput *> orig_links(inputs_bufs.size());
  for (int i = 0; i < inputs_bufs.size(); i++) {
    NodeOperationInput *input_socket = getInputSocket(i);
    BufferOperation *buffer_op = new BufferOperation(inputs_bufs[i], input_socket->getDataType());
    orig_links[i] = input_socket->getLink();
    input_socket->setLink(buffer_op->getOutputSocket());
  }
  return orig_links;
}

void NodeOperation::remove_buffers_and_restore_original_inputs(
    Span<NodeOperationOutput *> original_inputs_links)
{
  BLI_assert(original_inputs_links.size() == getNumberOfInputSockets());
  for (int i = 0; i < original_inputs_links.size(); i++) {
    BLI_assert(typeid(*getInputOperation(i)) == typeid(BufferOperation));

    NodeOperationInput *input_socket = getInputSocket(i);
    delete &input_socket->getLink()->getOperation();
    input_socket->setLink(original_inputs_links[i]);
  }
}

/** \} */

/*****************
 **** OpInput ****
 *****************/

NodeOperationInput::NodeOperationInput(NodeOperation *op, DataType datatype, ResizeMode resizeMode)
    : m_operation(op), m_datatype(datatype), m_resizeMode(resizeMode), m_link(nullptr)
{
}

SocketReader *NodeOperationInput::getReader()
{
  if (isConnected()) {
    return &m_link->getOperation();
  }

  return nullptr;
}

void NodeOperationInput::determineResolution(unsigned int resolution[2],
                                             unsigned int preferredResolution[2])
{
  if (m_link) {
    m_link->determineResolution(resolution, preferredResolution);
  }
}

/******************
 **** OpOutput ****
 ******************/

NodeOperationOutput::NodeOperationOutput(NodeOperation *op, DataType datatype)
    : m_operation(op), m_datatype(datatype)
{
}

void NodeOperationOutput::determineResolution(unsigned int resolution[2],
                                              unsigned int preferredResolution[2])
{
  NodeOperation &operation = getOperation();
  if (operation.get_flags().is_resolution_set) {
    resolution[0] = operation.getWidth();
    resolution[1] = operation.getHeight();
  }
  else {
    operation.determineResolution(resolution, preferredResolution);
    if (resolution[0] > 0 && resolution[1] > 0) {
      operation.setResolution(resolution);
    }
  }
}

std::ostream &operator<<(std::ostream &os, const NodeOperationFlags &node_operation_flags)
{
  if (node_operation_flags.complex) {
    os << "complex,";
  }
  if (node_operation_flags.open_cl) {
    os << "open_cl,";
  }
  if (node_operation_flags.single_threaded) {
    os << "single_threaded,";
  }
  if (node_operation_flags.use_render_border) {
    os << "render_border,";
  }
  if (node_operation_flags.use_viewer_border) {
    os << "view_border,";
  }
  if (node_operation_flags.is_resolution_set) {
    os << "resolution_set,";
  }
  if (node_operation_flags.is_set_operation) {
    os << "set_operation,";
  }
  if (node_operation_flags.is_write_buffer_operation) {
    os << "write_buffer,";
  }
  if (node_operation_flags.is_read_buffer_operation) {
    os << "read_buffer,";
  }
  if (node_operation_flags.is_proxy_operation) {
    os << "proxy,";
  }
  if (node_operation_flags.is_viewer_operation) {
    os << "viewer,";
  }
  if (node_operation_flags.is_preview_operation) {
    os << "preview,";
  }
  if (!node_operation_flags.use_datatype_conversion) {
    os << "no_conversion,";
  }
  if (node_operation_flags.is_fullframe_operation) {
    os << "full_frame,";
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, const NodeOperation &node_operation)
{
  NodeOperationFlags flags = node_operation.get_flags();
  os << "NodeOperation(";
  os << "id=" << node_operation.get_id();
  if (!node_operation.get_name().empty()) {
    os << ",name=" << node_operation.get_name();
  }
  os << ",flags={" << flags << "}";
  if (flags.is_read_buffer_operation) {
    const ReadBufferOperation *read_operation = (const ReadBufferOperation *)&node_operation;
    const MemoryProxy *proxy = read_operation->getMemoryProxy();
    if (proxy) {
      const WriteBufferOperation *write_operation = proxy->getWriteBufferOperation();
      if (write_operation) {
        os << ",write=" << (NodeOperation &)*write_operation;
      }
    }
  }
  os << ")";

  return os;
}

}  // namespace blender::compositor
