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
 */

#include <cstring>

#include "MEM_guardedalloc.h"

#include "NOD_function.h"

#include "BKE_context.h"
#include "BKE_node.h"
#include "BKE_object.h"

#include "BLT_translation.h"

#include "DNA_modifier_types.h"
#include "DNA_node_types.h"
#include "DNA_space_types.h"

#include "RNA_access.h"

#include "node_common.h"

bNodeTreeType *ntreeType_Function;

static void function_node_tree_update(bNodeTree *ntree)
{
  /* Needed to give correct types to reroutes. */
  ntree_update_reroute_nodes(ntree);
}

static void foreach_nodeclass(Scene *UNUSED(scene), void *calldata, bNodeClassCallback func)
{
  func(calldata, NODE_CLASS_INPUT, N_("Input"));
  func(calldata, NODE_CLASS_OP_COLOR, N_("Color"));
  func(calldata, NODE_CLASS_OP_VECTOR, N_("Vector"));
  func(calldata, NODE_CLASS_CONVERTOR, N_("Convertor"));
  func(calldata, NODE_CLASS_LAYOUT, N_("Layout"));
}

void register_node_tree_type_function(void)
{
  bNodeTreeType *tt = ntreeType_Function = static_cast<bNodeTreeType *>(
      MEM_callocN(sizeof(bNodeTreeType), "function node tree type"));
  tt->type = NTREE_FUNCTION;
  strcpy(tt->idname, "FunctionNodeTree");
  strcpy(tt->ui_name, N_("Function Node Editor"));
  tt->ui_icon = 0; /* defined in drawnode.c */
  strcpy(tt->ui_description, N_("Function nodes"));
  tt->rna_ext.srna = &RNA_FunctionNodeTree;
  tt->update = function_node_tree_update;
  tt->foreach_nodeclass = foreach_nodeclass;

  ntreeTypeAdd(tt);
}
