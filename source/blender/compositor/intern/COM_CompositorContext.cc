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

#include "COM_CompositorContext.h"

namespace blender::compositor {

CompositorContext::CompositorContext()
{
  scene_ = nullptr;
  rd_ = nullptr;
  quality_ = eCompositorQuality::High;
  hasActiveOpenCLDevices_ = false;
  fastCalculation_ = false;
  viewSettings_ = nullptr;
  displaySettings_ = nullptr;
  bnodetree_ = nullptr;
}

int CompositorContext::getFramenumber() const
{
  BLI_assert(rd_);
  return rd_->cfra;
}

Size2f CompositorContext::get_render_size() const
{
  return {getRenderData()->xsch * getRenderPercentageAsFactor(),
          getRenderData()->ysch * getRenderPercentageAsFactor()};
}

eExecutionModel CompositorContext::get_execution_model() const
{
  if (U.experimental.use_full_frame_compositor) {
    BLI_assert(bnodetree_ != nullptr);
    switch (bnodetree_->execution_mode) {
      case 1:
        return eExecutionModel::FullFrame;
      case 0:
        return eExecutionModel::Tiled;
      default:
        BLI_assert_msg(0, "Invalid execution mode");
    }
  }
  return eExecutionModel::Tiled;
}

}  // namespace blender::compositor
