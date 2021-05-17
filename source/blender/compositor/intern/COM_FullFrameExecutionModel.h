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

#include "COM_ExecutionModel.h"

#ifdef WITH_CXX_GUARDEDALLOC
#  include "MEM_guardedalloc.h"
#endif

namespace blender::compositor {

/* Forward declarations. */
class ExecutionGroup;

/**
 * Fully renders operations in order from inputs to outputs.
 */
class FullFrameExecutionModel : public ExecutionModel {
 private:
  /**
   * Stores operations output data/buffers and dispose them once reader operations are
   * finished.
   */
  OutputStore &output_store_;

  /**
   * Number of operations finished.
   */
  int num_operations_finished_;

  /**
   * Order of priorities for output operations execution.
   */
  Vector<eCompositorPriority> priorities_;

 public:
  FullFrameExecutionModel(CompositorContext &context,
                          OutputStore &output_store,
                          Span<NodeOperation *> operations);

  void execute(ExecutionSystem &exec_system) override;

  void execute_work(const rcti &work_rect,
                    std::function<void(const rcti &split_rect)> work_func) override;

  void operation_finished(NodeOperation *operation) override;

 private:
  void determine_rects_to_render_and_reads();
  void render_operations(ExecutionSystem &exec_system);

  void get_output_render_rect(NodeOperation *output_op, rcti &r_rect);
  void determine_rects_to_render(NodeOperation *operation, const rcti &render_rect);
  void determine_reads(NodeOperation *operation);

  void update_progress_bar();
  bool is_breaked() const;

#ifdef WITH_CXX_GUARDEDALLOC
  MEM_CXX_CLASS_ALLOC_FUNCS("COM:FullFrameExecutionModel")
#endif
};

}  // namespace blender::compositor
