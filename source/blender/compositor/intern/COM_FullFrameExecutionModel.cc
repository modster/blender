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

#include "COM_FullFrameExecutionModel.h"
#include "COM_Debug.h"
#include "COM_ExecutionGroup.h"
#include "COM_ReadBufferOperation.h"
#include "COM_WorkScheduler.h"

#include "BLT_translation.h"

#ifdef WITH_CXX_GUARDEDALLOC
#  include "MEM_guardedalloc.h"
#endif

namespace blender::compositor {

FullFrameExecutionModel::FullFrameExecutionModel(CompositorContext &context,
                                                 OutputStore &output_store,
                                                 Span<NodeOperation *> operations)
    : ExecutionModel(context, operations),
      m_output_store(output_store),
      m_num_operations_finished(0),
      m_priorities()
{
  m_priorities.append(eCompositorPriority::High);
  if (!context.isFastCalculation()) {
    m_priorities.append(eCompositorPriority::Medium);
    m_priorities.append(eCompositorPriority::Low);
  }
}

void FullFrameExecutionModel::execute(ExecutionSystem &exec_system)
{
  const bNodeTree *node_tree = this->m_context.getbNodeTree();
  node_tree->stats_draw(node_tree->sdh, TIP_("Compositing | Initializing execution"));

  DebugInfo::graphviz(&exec_system);

  determine_rects_to_render_and_reads();
  render_operations(exec_system);
}

void FullFrameExecutionModel::determine_rects_to_render_and_reads()
{
  const bool is_rendering = m_context.isRendering();
  const bNodeTree *node_tree = m_context.getbNodeTree();

  rcti render_rect;
  for (eCompositorPriority priority : m_priorities) {
    for (NodeOperation *op : m_operations) {
      op->setbNodeTree(node_tree);
      if (op->isOutputOperation(is_rendering) && op->getRenderPriority() == priority) {
        get_output_render_rect(op, render_rect);
        determine_rects_to_render(op, render_rect);
        determine_reads(op);
      }
    }
  }
}

void FullFrameExecutionModel::render_operations(ExecutionSystem &exec_system)
{
  const bool is_rendering = m_context.isRendering();

  WorkScheduler::start(this->m_context);
  for (eCompositorPriority priority : m_priorities) {
    for (NodeOperation *op : m_operations) {
      if (op->isOutputOperation(is_rendering) && op->getRenderPriority() == priority) {
        op->render(exec_system);
      }
    }
  }
  WorkScheduler::stop();
}

/**
 * Determines the areas given operation and its inputs need to render. Results are saved in output
 * store.
 */
void FullFrameExecutionModel::determine_rects_to_render(NodeOperation *operation,
                                                        const rcti &render_rect)
{
  if (m_output_store.is_render_registered(operation, render_rect)) {
    return;
  }

  m_output_store.register_render(operation, render_rect);

  const int n_inputs = operation->getNumberOfInputSockets();
  for (int i = 0; i < n_inputs; i++) {
    NodeOperation *input_op = &operation->getInputSocket(i)->getLink()->getOperation();
    rcti input_op_rect, input_area;
    BLI_rcti_init(&input_op_rect, 0, input_op->getWidth(), 0, input_op->getHeight());
    operation->get_input_area_of_interest(i, render_rect, input_area);

    /* Ensure input area of interest is within operation bounds. */
    int dummy_offset[2];
    BLI_rcti_clamp(&input_area, &input_op_rect, dummy_offset);

    determine_rects_to_render(input_op, input_area);
  }
}

/**
 * Determines the reads given operation and its inputs will receive. Results are saved in output
 * store.
 */
void FullFrameExecutionModel::determine_reads(NodeOperation *operation)
{
  if (m_output_store.has_registered_reads(operation)) {
    return;
  }

  const int n_inputs = operation->getNumberOfInputSockets();
  for (int i = 0; i < n_inputs; i++) {
    NodeOperation *input_op = &operation->getInputSocket(i)->getLink()->getOperation();
    determine_reads(input_op);
    m_output_store.register_read(input_op);
  }
}

void FullFrameExecutionModel::get_output_render_rect(NodeOperation *output_op, rcti &r_rect)
{
  BLI_assert(output_op->isOutputOperation(m_context.isRendering()));

  /* By default return operation bounds (no border) */
  const int op_width = output_op->getWidth();
  const int op_height = output_op->getHeight();
  BLI_rcti_init(&r_rect, 0, op_width, 0, op_height);

  const bool has_viewer_border = m_border.use_viewer_border &&
                                 (output_op->get_flags().is_viewer_operation ||
                                  output_op->get_flags().is_preview_operation);
  const bool has_render_border = m_border.use_render_border;
  if (has_viewer_border || has_render_border) {
    /* Get border with normalized coordinates */
    const rctf *norm_border = has_viewer_border ? m_border.viewer_border : m_border.render_border;

    /* De-normalize and clamp operation bounds to border */
    rcti border;
    BLI_rcti_init(&border,
                  norm_border->xmin * op_width,
                  norm_border->xmax * op_width,
                  norm_border->ymin * op_height,
                  norm_border->ymax * op_height);
    int dummy_offset[2];
    BLI_rcti_clamp(&r_rect, &border, dummy_offset);
  }
}

/**
 * Multi-threadedly execute given work function passing work_rect splits as argument.
 */
void FullFrameExecutionModel::execute_work(const rcti &work_rect,
                                           std::function<void(const rcti &split_rect)> work_func)
{
  if (is_breaked()) {
    return;
  }

  /* Split work vertically to maximize continuous memory. */
  const int work_height = BLI_rcti_size_y(&work_rect);
  const int n_sub_works = MIN2(WorkScheduler::get_num_cpu_threads(), work_height);
  const int split_height = n_sub_works == 0 ? 0 : work_height / n_sub_works;
  int remaining_height = work_height - split_height * n_sub_works;

  Vector<WorkPackage> sub_works(n_sub_works);
  int sub_work_y = work_rect.ymin;
  for (int i = 0; i < n_sub_works; i++) {
    int sub_work_height = split_height;

    /* Distribute remaining height between sub-works. */
    if (remaining_height > 0) {
      sub_work_height++;
      remaining_height--;
    }

    WorkPackage &sub_work = sub_works[i];
    sub_work.type = eWorkPackageType::CustomFunction;
    sub_work.custom_func = [=, &work_func, &work_rect]() {
      if (is_breaked()) {
        return;
      }
      rcti split_rect;
      BLI_rcti_init(
          &split_rect, work_rect.xmin, work_rect.xmax, sub_work_y, sub_work_y + sub_work_height);
      work_func(split_rect);
    };

    WorkScheduler::schedule(&sub_work);

    sub_work_y += sub_work_height;
  }
  BLI_assert(sub_work_y == work_rect.ymax);

  WorkScheduler::finish();

  /* WorkScheduler::ThreadingModel::Queue needs this code because last work is still running even
   * after calling WorkScheduler::finish(). May be an issue specific to windows. */
  bool work_finished = false;
  while (!work_finished) {
    work_finished = true;
    for (const WorkPackage &sub_work : sub_works) {
      if (!sub_work.finished) {
        work_finished = false;
        WorkScheduler::finish();
        break;
      }
    }
  }
}

void FullFrameExecutionModel::operation_finished(NodeOperation *operation)
{
  /* Report inputs reads so that buffers may be freed/reused. */
  const int n_inputs = operation->getNumberOfInputSockets();
  for (int i = 0; i < n_inputs; i++) {
    m_output_store.read_finished(operation->getInputOperation(i));
  }

  m_num_operations_finished++;
  update_progress_bar();
}

void FullFrameExecutionModel::update_progress_bar()
{
  const bNodeTree *tree = m_context.getbNodeTree();
  if (tree) {
    const float progress = m_num_operations_finished / static_cast<float>(m_operations.size());
    tree->progress(tree->prh, progress);

    char buf[128];
    BLI_snprintf(buf,
                 sizeof(buf),
                 TIP_("Compositing | Operation %i-%li"),
                 m_num_operations_finished + 1,
                 m_operations.size());
    tree->stats_draw(tree->sdh, buf);
  }
}

bool FullFrameExecutionModel::is_breaked() const
{
  const bNodeTree *btree = m_context.getbNodeTree();
  return btree->test_break(btree->tbh);
}

}  // namespace blender::compositor
