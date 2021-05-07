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

#pragma once

#include "BLI_map.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"
#include "COM_MemoryBuffer.h"
#ifdef WITH_CXX_GUARDEDALLOC
#  include "MEM_guardedalloc.h"
#endif
#include <memory>

namespace blender::compositor {

/**
 * Stores operations output data including rendered buffers. It's responsible for deleting output
 * buffers when all their readers have finished.
 */
class OutputStore {
 private:
  typedef struct OutputData {
   public:
    OutputData();
    std::unique_ptr<MemoryBuffer> buffer;
    blender::Vector<rcti> render_rects;
    int registered_reads;
    int received_reads;
  } OutputData;
  blender::Map<NodeOperation *, OutputData> m_outputs;

 public:
  OutputStore();
  bool is_render_registered(NodeOperation *op, const rcti &render_rect);
  void register_render(NodeOperation *op, const rcti &render_rect);

  bool has_registered_reads(NodeOperation *op);
  void register_read(NodeOperation *op);

  blender::Span<rcti> get_rects_to_render(NodeOperation *op);
  bool is_output_rendered(NodeOperation *op);
  void set_rendered_output(NodeOperation *op, std::unique_ptr<MemoryBuffer> output_buffer);
  MemoryBuffer *get_rendered_output(NodeOperation *op);

  void read_finished(NodeOperation *read_op);

 private:
  OutputData &get_output_data(NodeOperation *op);

#ifdef WITH_CXX_GUARDEDALLOC
  MEM_CXX_CLASS_ALLOC_FUNCS("COM:OutputStore")
#endif
};

}  // namespace blender::compositor
