/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup spoutliner
 */

#include "BLI_listbase.h"

#include "tree_display.hh"

namespace blender::ed::outliner {

TreeDisplayOverrideLibraryHierarchy::TreeDisplayOverrideLibraryHierarchy(
    SpaceOutliner &space_outliner)
    : AbstractTreeDisplay(space_outliner)
{
}

ListBase TreeDisplayOverrideLibraryHierarchy::buildTree(const TreeSourceData & /*source_data*/)
{
  ListBase tree = {nullptr};
  return tree;
}

}  // namespace blender::ed::outliner
