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

#include "COM_ExecutionSystem.h"

#include "BLI_utildefines.h"
#include "PIL_time.h"

#include "BKE_node.h"

#include "BLT_translation.h"

#include "COM_BufferOperation.h"
#include "COM_Converter.h"
#include "COM_Debug.h"
#include "COM_ExecutionGroup.h"
#include "COM_NodeOperation.h"
#include "COM_NodeOperationBuilder.h"
#include "COM_OutputStore.h"
#include "COM_ReadBufferOperation.h"
#include "COM_WorkScheduler.h"

#ifdef WITH_CXX_GUARDEDALLOC
#  include "MEM_guardedalloc.h"
#endif

namespace blender::compositor {

ExecutionSystem::ExecutionSystem(RenderData *rd,
                                 Scene *scene,
                                 bNodeTree *editingtree,
                                 bool rendering,
                                 bool fastcalculation,
                                 const ColorManagedViewSettings *viewSettings,
                                 const ColorManagedDisplaySettings *displaySettings,
                                 const char *viewName,
                                 int num_cpu_threads)
{
  this->m_context.setViewName(viewName);
  this->m_context.setScene(scene);
  this->m_context.setbNodeTree(editingtree);
  this->m_context.setPreviewHash(editingtree->previews);
  this->m_context.setFastCalculation(fastcalculation);
  /* initialize the CompositorContext */
  if (rendering) {
    this->m_context.setQuality((eCompositorQuality)editingtree->render_quality);
  }
  else {
    this->m_context.setQuality((eCompositorQuality)editingtree->edit_quality);
  }
  this->m_context.setRendering(rendering);
  this->m_context.setHasActiveOpenCLDevices(WorkScheduler::has_gpu_devices() &&
                                            (editingtree->flag & NTREE_COM_OPENCL));

  this->m_context.setRenderData(rd);
  this->m_context.setViewSettings(viewSettings);
  this->m_context.setDisplaySettings(displaySettings);

  {
    NodeOperationBuilder builder(&m_context, editingtree);
    builder.convertToOperations(this);
  }

  unsigned int resolution[2];

  rctf *viewer_border = &editingtree->viewer_border;
  bool use_viewer_border = (editingtree->flag & NTREE_VIEWER_BORDER) &&
                           viewer_border->xmin < viewer_border->xmax &&
                           viewer_border->ymin < viewer_border->ymax;

  editingtree->stats_draw(editingtree->sdh, TIP_("Compositing | Determining resolution"));

  for (ExecutionGroup *executionGroup : m_groups) {
    resolution[0] = 0;
    resolution[1] = 0;
    executionGroup->determineResolution(resolution);

    if (rendering) {
      /* case when cropping to render border happens is handled in
       * compositor output and render layer nodes
       */
      if ((rd->mode & R_BORDER) && !(rd->mode & R_CROP)) {
        executionGroup->setRenderBorder(
            rd->border.xmin, rd->border.xmax, rd->border.ymin, rd->border.ymax);
      }
    }

    if (use_viewer_border) {
      executionGroup->setViewerBorder(
          viewer_border->xmin, viewer_border->xmax, viewer_border->ymin, viewer_border->ymax);
    }
  }

  m_num_operations_finished = 0;
  m_num_cpu_threads = num_cpu_threads;

  const rctf &render_border = rd->border;
  m_border.use_viewer_border = use_viewer_border;
  m_border.use_render_border = rendering && (rd->mode & R_BORDER) && !(rd->mode & R_CROP);
  BLI_rcti_init(&m_border.viewer_border,
                viewer_border->xmin,
                viewer_border->xmax,
                viewer_border->ymin,
                viewer_border->ymax);
  BLI_rcti_init(&m_border.render_border,
                render_border.xmin,
                render_border.xmax,
                render_border.ymin,
                render_border.ymax);

  if (m_context.get_execution_model() == eExecutionModel::FullFrame) {
    DebugInfo::graphviz(this);
  }
}

ExecutionSystem::~ExecutionSystem()
{
  for (NodeOperation *operation : m_operations) {
    delete operation;
  }
  this->m_operations.clear();

  for (ExecutionGroup *group : m_groups) {
    delete group;
  }
  this->m_groups.clear();
}

void ExecutionSystem::set_operations(const Vector<NodeOperation *> &operations,
                                     const Vector<ExecutionGroup *> &groups)
{
  m_operations = operations;
  m_groups = groups;
}

static void update_read_buffer_offset(Vector<NodeOperation *> &operations)
{
  unsigned int order = 0;
  for (NodeOperation *operation : operations) {
    if (operation->get_flags().is_read_buffer_operation) {
      ReadBufferOperation *readOperation = (ReadBufferOperation *)operation;
      readOperation->setOffset(order);
      order++;
    }
  }
}

static void init_write_operations_for_execution(Vector<NodeOperation *> &operations,
                                                const bNodeTree *bTree)
{
  for (NodeOperation *operation : operations) {
    if (operation->get_flags().is_write_buffer_operation) {
      operation->setbNodeTree(bTree);
      operation->initExecution();
    }
  }
}

static void link_write_buffers(Vector<NodeOperation *> &operations)
{
  for (NodeOperation *operation : operations) {
    if (operation->get_flags().is_read_buffer_operation) {
      ReadBufferOperation *readOperation = static_cast<ReadBufferOperation *>(operation);
      readOperation->updateMemoryBuffer();
    }
  }
}

static void init_non_write_operations_for_execution(Vector<NodeOperation *> &operations,
                                                    const bNodeTree *bTree)
{
  for (NodeOperation *operation : operations) {
    if (!operation->get_flags().is_write_buffer_operation) {
      operation->setbNodeTree(bTree);
      operation->initExecution();
    }
  }
}

static void init_execution_groups_for_execution(Vector<ExecutionGroup *> &groups,
                                                const int chunk_size)
{
  for (ExecutionGroup *execution_group : groups) {
    execution_group->setChunksize(chunk_size);
    execution_group->initExecution();
  }
}

void ExecutionSystem::execute()
{
  const bNodeTree *editingtree = this->m_context.getbNodeTree();
  DebugInfo::execute_started(this);
  if (m_context.get_execution_model() == eExecutionModel::Tiled) {
    editingtree->stats_draw(editingtree->sdh, TIP_("Compositing | Initializing execution"));

    update_read_buffer_offset(m_operations);

    init_write_operations_for_execution(m_operations, m_context.getbNodeTree());
    link_write_buffers(m_operations);
    init_non_write_operations_for_execution(m_operations, m_context.getbNodeTree());
    init_execution_groups_for_execution(m_groups, m_context.getChunksize());

    WorkScheduler::start(this->m_context);
    execute_groups(eCompositorPriority::High);
    if (!this->getContext().isFastCalculation()) {
      execute_groups(eCompositorPriority::Medium);
      execute_groups(eCompositorPriority::Low);
    }
    WorkScheduler::finish();
    WorkScheduler::stop();

    editingtree->stats_draw(editingtree->sdh, TIP_("Compositing | De-initializing execution"));

    for (NodeOperation *operation : m_operations) {
      operation->deinitExecution();
    }

    for (ExecutionGroup *execution_group : m_groups) {
      execution_group->deinitExecution();
    }
  }
  else {
    execute_full_frame();
  }
}

void ExecutionSystem::execute_groups(eCompositorPriority priority)
{
  for (ExecutionGroup *execution_group : m_groups) {
    if (execution_group->get_flags().is_output &&
        execution_group->getRenderPriority() == priority) {
      execution_group->execute(this);
    }
  }
}

/*** FullFrame methods ***/

void ExecutionSystem::execute_full_frame()
{
  /* Set output operations priorities in order. */
  blender::Vector<eCompositorPriority> priorities;
  priorities.append(eCompositorPriority::High);
  if (!this->getContext().isFastCalculation()) {
    priorities.append(eCompositorPriority::Medium);
    priorities.append(eCompositorPriority::Low);
  }

  /* Setup operations. */
  bool is_rendering = m_context.isRendering();
  rcti render_rect;
  const bNodeTree *bNodeTree = m_context.getbNodeTree();
  for (eCompositorPriority priority : priorities) {
    for (NodeOperation *op : m_operations) {
      op->setbNodeTree(bNodeTree);
      if (op->isOutputOperation(is_rendering) && op->getRenderPriority() == priority) {
        get_output_render_rect(op, render_rect);
        determine_rects_to_render(op, render_rect);
        determine_reads(op);
      }
    }
  }

  /* Render operations. */
  WorkScheduler::start(this->m_context);
  for (eCompositorPriority priority : priorities) {
    for (NodeOperation *op : m_operations) {
      if (op->isOutputOperation(is_rendering) && op->getRenderPriority() == priority) {
        render_operation(op);
      }
    }
  }
  WorkScheduler::stop();
}

/**
 * Renders given operation and its inputs. Rendered buffers are saved in the output store.
 */
void ExecutionSystem::render_operation(NodeOperation *operation)
{
  OutputStore &output_store = get_output_store();
  if (!output_store.is_output_rendered(operation)) {
    /* Ensure inputs are rendered. */
    int n_inputs = operation->getNumberOfInputSockets();
    blender::Vector<NodeOperation *> inputs_ops;
    for (int i = 0; i < n_inputs; i++) {
      NodeOperation *input_op = &operation->getInputSocket(i)->getLink()->getOperation();
      render_operation(input_op);
      inputs_ops.append(input_op);
    }

    /* Get input buffers. */
    blender::Vector<MemoryBuffer *> inputs_bufs;
    for (NodeOperation *input_op : inputs_ops) {
      inputs_bufs.append(output_store.get_rendered_output(input_op));
    }

    /* Create output buffer if needed. */
    bool has_output_buffer = operation->getNumberOfOutputSockets() > 0;
    MemoryBuffer *output_buf = nullptr;
    if (has_output_buffer) {
      DataType data_type = operation->getOutputSocket(0)->getDataType();
      rcti rect;
      BLI_rcti_init(&rect, 0, operation->getWidth(), 0, operation->getHeight());
      /* TODO: Check if this operation is a set operation to create a single elem buffer. Need
       * MemoryBuffer constructor for such case yet. */
      output_buf = new MemoryBuffer(data_type, rect);
    }

    /* Render. */
    blender::Span<rcti> render_rects = output_store.get_rects_to_render(operation);
    if (operation->get_flags().is_fullframe_operation) {
      operation->initExecution();
      for (const rcti &render_rect : render_rects) {
        operation->update_memory_buffer(output_buf, render_rect, inputs_bufs.as_span(), *this);
      }
      operation->deinitExecution();
    }
    else {
      render_operation_tiled(operation, output_buf, render_rects, inputs_bufs.as_span());
    }
    output_store.set_rendered_output(operation, std::unique_ptr<MemoryBuffer>(output_buf));

    /* Report inputs reads so that buffers may be freed when all their readers
     * have finished. */
    for (NodeOperation *input_op : inputs_ops) {
      output_store.read_finished(input_op);
    }

    operation_finished();
  }
}

/**
 * Renders given operation using the tiled implementation.
 */
void ExecutionSystem::render_operation_tiled(NodeOperation *operation,
                                             MemoryBuffer *output_buf,
                                             Span<rcti> render_rects,
                                             blender::Span<MemoryBuffer *> inputs)
{
  /* Set input buffers as input operations. */
  Vector<NodeOperationOutput *> orig_links;
  for (int i = 0; i < inputs.size(); i++) {
    NodeOperationInput *input_socket = operation->getInputSocket(i);
    BufferOperation *buffer_op = new BufferOperation(inputs[i], input_socket->getDataType());
    orig_links.append(input_socket->getLink());
    input_socket->setLink(buffer_op->getOutputSocket());
  }

  /* Execute operation tiled implementation. */
  operation->initExecution();
  bool is_output_operation = operation->getNumberOfOutputSockets() == 0;
  bool is_complex = operation->get_flags().complex;
  for (const rcti &rect : render_rects) {
    execute_work(rect, [=](const rcti &split_rect) {
      if (is_output_operation) {
        rcti region = split_rect;
        operation->executeRegion(&region, 0);
      }
      else {
        rcti tile_rect = split_rect;
        void *tile_data = operation->initializeTileData(&tile_rect);
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
          operation->deinitializeTileData(&tile_rect, tile_data);
        }
      }
    });
  }
  operation->deinitExecution();

  /* Delete buffer operations and set original ones. */
  for (int i = 0; i < inputs.size(); i++) {
    NodeOperationInput *input_socket = operation->getInputSocket(i);
    delete &input_socket->getLink()->getOperation();
    input_socket->setLink(orig_links[i]);
  }
}

/**
 * Determines the areas given operation and its inputs need to render. Results are saved in output
 * store.
 */
void ExecutionSystem::determine_rects_to_render(NodeOperation *operation, const rcti &render_rect)
{
  if (!m_output_store.is_render_registered(operation, render_rect)) {
    m_output_store.register_render(operation, render_rect);

    int n_inputs = operation->getNumberOfInputSockets();
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
}

/**
 * Determines the reads given operation and its inputs will receive. Results are saved in output
 * store.
 */
void ExecutionSystem::determine_reads(NodeOperation *operation)
{
  if (!m_output_store.has_registered_reads(operation)) {
    int n_inputs = operation->getNumberOfInputSockets();
    for (int i = 0; i < n_inputs; i++) {
      NodeOperation *input_op = &operation->getInputSocket(i)->getLink()->getOperation();
      determine_reads(input_op);
      m_output_store.register_read(input_op);
    }
  }
}

void ExecutionSystem::get_output_render_rect(NodeOperation *output_op, rcti &r_rect)
{
  BLI_assert(output_op->isOutputOperation(m_context.isRendering()));

  const NodeOperationFlags &op_flags = output_op->get_flags();
  BLI_rcti_init(&r_rect, 0, output_op->getWidth(), 0, output_op->getHeight());

  bool has_viewer_border = m_border.use_viewer_border &&
                           (op_flags.is_viewer_operation || op_flags.is_preview_operation);
  bool has_render_border = m_border.use_render_border;
  if (has_viewer_border || has_render_border) {
    rcti &border = has_viewer_border ? m_border.viewer_border : m_border.render_border;
    r_rect.xmin = border.xmin > r_rect.xmin ? border.xmin : r_rect.xmin;
    r_rect.xmax = border.xmax < r_rect.xmax ? border.xmax : r_rect.xmax;
    r_rect.ymin = border.ymin > r_rect.ymin ? border.ymin : r_rect.ymin;
    r_rect.ymax = border.ymax < r_rect.ymax ? border.ymax : r_rect.ymax;
  }
}

/**
 * Multi-threadedly execute given work function passing work_rect splits as argument.
 */
void ExecutionSystem::execute_work(const rcti &work_rect,
                                   std::function<void(const rcti &split_rect)> work_func)
{
  /* Split work vertically. */
  if (!is_breaked()) {
    int work_height = BLI_rcti_size_y(&work_rect);
    int n_works = m_num_cpu_threads < work_height ? m_num_cpu_threads : work_height;
    int std_split_height = n_works == 0 ? 0 : work_height / n_works;
    int remaining = work_height - std_split_height * n_works;
    Vector<WorkPackage> works(n_works);
    int split_y = work_rect.ymin;
    for (int i = 0; i < n_works; i++) {
      WorkPackage &work = works[i];
      int split_height = std_split_height;
      if (remaining > 0) {
        split_height++;
        remaining--;
      }
      work.custom_func = [=, &work_func, &work_rect]() {
        if (!is_breaked()) {
          rcti split_rect;
          BLI_rcti_init(
              &split_rect, work_rect.xmin, work_rect.xmax, split_y, split_y + split_height);
          work_func(split_rect);
        }
      };
      work.execution_group = nullptr;
      WorkScheduler::schedule(&work);

      split_y += split_height;
    }
    BLI_assert(split_y == work_rect.ymax);

    WorkScheduler::finish();

    /* WorkScheduler::ThreadingModel::Queue needs this code. */
    // bool works_finished = false;
    // while (!works_finished) {
    //  works_finished = true;
    //  for (WorkPackage &work : works) {
    //    if (!work.finished) {
    //      works_finished = false;
    //      WorkScheduler::finish();
    //      break;
    //    }
    //  }
    //}
  }
}

void ExecutionSystem::operation_finished()
{
  m_num_operations_finished++;
  update_progress_bar();
}

void ExecutionSystem::update_progress_bar()
{
  const bNodeTree *tree = m_context.getbNodeTree();
  if (tree) {
    float progress = m_num_operations_finished / static_cast<float>(m_operations.size());
    tree->progress(tree->prh, progress);

    char buf[128];
    BLI_snprintf(buf,
                 sizeof(buf),
                 TIP_("Compositing | Operation %u-%u"),
                 m_num_operations_finished + 1,
                 m_operations.size());
    tree->stats_draw(tree->sdh, buf);
  }
}

bool ExecutionSystem::is_breaked() const
{
  const bNodeTree *btree = m_context.getbNodeTree();
  return btree->test_break(btree->tbh);
}

}  // namespace blender::compositor
