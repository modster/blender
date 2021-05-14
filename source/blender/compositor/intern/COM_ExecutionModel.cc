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

#include "COM_ExecutionModel.h"

namespace blender::compositor {

ExecutionModel::ExecutionModel(CompositorContext &context, Span<NodeOperation *> operations)
    : m_context(context), m_operations(operations)
{
  const bNodeTree *node_tree = m_context.getbNodeTree();

  const rctf *viewer_border = &node_tree->viewer_border;
  m_border.use_viewer_border = (node_tree->flag & NTREE_VIEWER_BORDER) &&
                               viewer_border->xmin < viewer_border->xmax &&
                               viewer_border->ymin < viewer_border->ymax;
  m_border.viewer_border = viewer_border;

  const RenderData *rd = m_context.getRenderData();
  /* Case when cropping to render border happens is handled in
   * compositor output and render layer nodes. */
  m_border.use_render_border = context.isRendering() && (rd->mode & R_BORDER) &&
                               !(rd->mode & R_CROP);
  m_border.render_border = &rd->border;
}

}  // namespace blender::compositor
