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
 * Copyright 2021, Blender Foundation.
 */

#include "COM_SharedOperationBuffers.h"
#include "BLI_rect.h"
#include "COM_NodeOperation.h"

namespace blender::compositor {

SharedOperationBuffers::SharedOperationBuffers() : buffers_()
{
}
SharedOperationBuffers::BufferData::BufferData()
    : buffer(nullptr), render_rects(), registered_reads(0), received_reads(0)
{
}

SharedOperationBuffers::BufferData &SharedOperationBuffers::get_buffer_data(NodeOperation *op)
{
  return buffers_.lookup_or_add_cb(op, []() { return BufferData(); });
}

bool SharedOperationBuffers::is_render_registered(NodeOperation *op, const rcti &rect_to_render)
{
  BufferData &buf_data = get_buffer_data(op);
  for (rcti &reg_rect : buf_data.render_rects) {
    if (BLI_rcti_inside_rcti(&reg_rect, &rect_to_render)) {
      return true;
    }
  }
  return false;
}

void SharedOperationBuffers::register_render(NodeOperation *op, const rcti &rect_to_render)
{
  get_buffer_data(op).render_rects.append(rect_to_render);
}

bool SharedOperationBuffers::has_registered_reads(NodeOperation *op)
{
  return get_buffer_data(op).registered_reads > 0;
}

void SharedOperationBuffers::register_read(NodeOperation *read_op)
{
  get_buffer_data(read_op).registered_reads++;
}

blender::Span<rcti> SharedOperationBuffers::get_rects_to_render(NodeOperation *op)
{
  return get_buffer_data(op).render_rects.as_span();
}

bool SharedOperationBuffers::is_operation_rendered(NodeOperation *op)
{
  return get_buffer_data(op).buffer != nullptr;
}

void SharedOperationBuffers::set_rendered_buffer(NodeOperation *op,
                                                 std::unique_ptr<MemoryBuffer> buffer)
{
  BufferData &buf_data = get_buffer_data(op);
  buf_data.buffer = std::move(buffer);
  BLI_assert(buf_data.received_reads == 0);
}

MemoryBuffer *SharedOperationBuffers::get_rendered_buffer(NodeOperation *op)
{
  BLI_assert(is_operation_rendered(op));
  return get_buffer_data(op).buffer.get();
}

void SharedOperationBuffers::read_finished(NodeOperation *read_op)
{
  BufferData &buf_data = get_buffer_data(read_op);
  buf_data.received_reads++;
  BLI_assert(buf_data.received_reads > 0 && buf_data.received_reads <= buf_data.registered_reads);
  if (buf_data.received_reads == buf_data.registered_reads) {
    /* Dispose buffer. */
    buf_data.buffer = nullptr;
  }
}

}  // namespace blender::compositor
