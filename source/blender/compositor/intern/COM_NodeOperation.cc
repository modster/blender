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

/**
 * Determines the areas this operation and its inputs need to render. Results are saved in the
 * output manager.
 */
void NodeOperation::determine_rects_to_render(const rcti &render_rect, OutputManager &output_man)
{
  if (!output_man.is_render_registered(this, render_rect)) {
    output_man.register_render(this, render_rect);

    int n_inputs = getNumberOfInputSockets();
    for (int i = 0; i < n_inputs; i++) {
      NodeOperation *op = getInputOperation(i);
      rcti op_rect, input_area;
      BLI_rcti_init(&op_rect, 0, op->getWidth(), 0, op->getHeight());
      get_input_area_of_interest(i, render_rect, input_area);

      /* Ensure input area of interest is within operation bounds. */
      int dummy_offset[2];
      BLI_rcti_clamp(&input_area, &op_rect, dummy_offset);

      op->determine_rects_to_render(input_area, output_man);
    }
  }
}

/**
 * Determines the reads received by this operation and its inputs. Results are saved in the
 * output manager.
 */
void NodeOperation::determine_reads(OutputManager &output_man)
{
  if (!output_man.has_registered_reads(this)) {
    int n_inputs = getNumberOfInputSockets();
    for (int i = 0; i < n_inputs; i++) {
      NodeOperation *input_op = getInputOperation(i);
      input_op->determine_reads(output_man);
      output_man.register_read(input_op);
    }
  }
}

/**
 * Renders this operation and its inputs. Rendered buffers are saved in the output manager.
 */
void NodeOperation::render(ExecutionSystem &exec_system)
{
  OutputManager &output_man = exec_system.get_output_manager();
  if (!output_man.is_output_rendered(this)) {
    /* Ensure inputs are rendered. */
    int n_inputs = getNumberOfInputSockets();
    blender::Vector<NodeOperation *> inputs_ops;
    for (int i = 0; i < n_inputs; i++) {
      NodeOperation *input_op = getInputOperation(i);
      input_op->render(exec_system);
      inputs_ops.append(input_op);
    }

    /* Get input buffers. */
    blender::Vector<MemoryBuffer *> inputs_bufs;
    for (NodeOperation *input_op : inputs_ops) {
      inputs_bufs.append(output_man.get_rendered_output(input_op));
    }

    /* Create output buffer if needed. */
    bool has_output_buffer = getNumberOfOutputSockets() > 0;
    MemoryBuffer *output_buf = nullptr;
    if (has_output_buffer) {
      DataType data_type = getOutputSocket(0)->getDataType();
      rcti rect;
      BLI_rcti_init(&rect, 0, getWidth(), 0, getHeight());
      /* TODO: Check if this operation is a set operation to create a single elem buffer. Need
       * MemoryBuffer constructor for such case yet. */
      output_buf = new MemoryBuffer(data_type, rect);
    }

    /* Render. */
    blender::Span<rcti> render_rects = output_man.get_rects_to_render(this);
    if (get_flags().is_fullframe_operation) {
      initExecution();
      for (const rcti &render_rect : render_rects) {
        update_memory_buffer(output_buf, render_rect, inputs_bufs.as_span(), exec_system);
      }
      deinitExecution();
    }
    else {
      render_non_fullframe(output_buf, render_rects, inputs_bufs.as_span(), exec_system);
    }
    output_man.set_rendered_output(this, std::unique_ptr<MemoryBuffer>(output_buf));

    /* Report inputs reads so that buffers may be freed when all their readers
     * have finished. */
    for (NodeOperation *input_op : inputs_ops) {
      output_man.read_finished(input_op);
    }

    exec_system.operation_finished();
  }
}

/**
 * Renders this operation using the tiled implementation.
 */
void NodeOperation::render_non_fullframe(MemoryBuffer *output_buf,
                                         Span<rcti> render_rects,
                                         blender::Span<MemoryBuffer *> inputs,
                                         ExecutionSystem &exec_system)
{
  /* Set input buffers as input operations. */
  Vector<NodeOperationOutput *> orig_links;
  for (int i = 0; i < inputs.size(); i++) {
    NodeOperationInput *input_socket = getInputSocket(i);
    BufferOperation *buffer_op = new BufferOperation(inputs[i], input_socket->getDataType());
    orig_links.append(input_socket->getLink());
    input_socket->setLink(buffer_op->getOutputSocket());
  }

  /* Execute operation tiled implementation. */
  initExecution();
  bool is_output_operation = getNumberOfOutputSockets() == 0;
  bool is_complex = get_flags().complex;
  NodeOperation *operation = this;
  for (const rcti &rect : render_rects) {
    exec_system.execute_work(rect, [=](const rcti &split_rect) {
      if (is_output_operation) {
        rcti region = split_rect;
        executeRegion(&region, 0);
      }
      else {
        rcti tile_rect = split_rect;
        void *tile_data = initializeTileData(&tile_rect);
        int num_channels = output_buf->get_num_channels();
        /* TODO: Take into account single elem buffers */
        for (int y = split_rect.ymin; y < split_rect.ymax; y++) {
          float *output_elem = output_buf->getBuffer() +
                               y * output_buf->getWidth() * num_channels +
                               split_rect.xmin * num_channels;
          if (is_complex) {
            for (int x = split_rect.xmin; x < split_rect.xmax; x++) {
              operation->read(output_elem, x, y, tile_data);
              output_elem += num_channels;
            }
          }
          else {
            for (int x = split_rect.xmin; x < split_rect.xmax; x++) {
              operation->readSampled(output_elem, x, y, PixelSampler::Nearest);
              output_elem += num_channels;
            }
          }
        }
        if (tile_data) {
          deinitializeTileData(&tile_rect, tile_data);
        }
      }
    });
  }
  deinitExecution();

  /* Delete buffer operations and set original ones. */
  for (int i = 0; i < inputs.size(); i++) {
    NodeOperationInput *input_socket = getInputSocket(i);
    delete &input_socket->getLink()->getOperation();
    input_socket->setLink(orig_links[i]);
  }
}

/**
 * Get input area being read by this operation.
 *
 * Implementation don't need to ensure r_input_rect is within operation bounds. The caller must
 * clamp it.
 */
void NodeOperation::get_input_area_of_interest(int input_idx,
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
    r_input_rect.xmin = 0;
    r_input_rect.ymin = 0;
    r_input_rect.xmax = input_op->getWidth();
    r_input_rect.ymax = input_op->getHeight();
  }
}

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
