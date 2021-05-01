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

#include "COM_OutputManager.h"
#include "BLI_rect.h"
#include "COM_NodeOperation.h"

namespace blender::compositor {

OutputManager::OutputManager() : m_outputs()
{
}
OutputManager::OutputData::OutputData()
    : buffer(nullptr), render_rects(), registered_reads(0), received_reads(0)
{
}

OutputManager::OutputData &OutputManager::get_output_data(NodeOperation *op)
{
  return m_outputs.lookup_or_add_cb(op, []() { return OutputData(); });
}

bool OutputManager::is_render_registered(NodeOperation *op, const rcti &render_rect)
{
  OutputData &output = get_output_data(op);
  for (rcti &reg_rect : output.render_rects) {
    if (BLI_rcti_inside_rcti(&reg_rect, &render_rect)) {
      return true;
    }
  }
  return false;
}

void OutputManager::register_render(NodeOperation *op, const rcti &render_rect)
{
  get_output_data(op).render_rects.append(render_rect);
}

bool OutputManager::has_registered_reads(NodeOperation *op)
{
  return get_output_data(op).registered_reads > 0;
}

void OutputManager::register_read(NodeOperation *op)
{
  get_output_data(op).registered_reads++;
}

blender::Span<rcti> OutputManager::get_rects_to_render(NodeOperation *op)
{
  return get_output_data(op).render_rects.as_span();
}

bool OutputManager::is_output_rendered(NodeOperation *op)
{
  return get_output_data(op).buffer != nullptr;
}

void OutputManager::set_rendered_output(NodeOperation *op,
                                        std::unique_ptr<MemoryBuffer> output_buffer)
{
  OutputData &output = get_output_data(op);
  output.buffer = std::move(output_buffer);
  BLI_assert(output.received_reads == 0);
}

MemoryBuffer *OutputManager::get_rendered_output(NodeOperation *op)
{
  BLI_assert(is_output_rendered(op));
  return get_output_data(op).buffer.get();
}

void OutputManager::read_finished(NodeOperation *read_op)
{
  OutputData &output = get_output_data(read_op);
  output.received_reads++;
  BLI_assert(output.received_reads > 0 && output.received_reads <= output.registered_reads);
  if (output.received_reads == output.registered_reads) {
    /* delete buffer */
    output.buffer = nullptr;
  }
}

}  // namespace blender::compositor
