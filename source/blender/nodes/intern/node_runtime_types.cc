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

/** \file
 * \ingroup nodes
 */

#include "NOD_runtime_types.hh"

#include "BLI_math.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "RNA_define.h"

#include "ED_node.h"

#include "node_util.h"

static bool custom_node_poll_default(bNodeType *UNUSED(ntype),
                                     bNodeTree *UNUSED(ntree),
                                     const char **UNUSED(disabled_hint))
{
  return true;
}

static bool custom_node_poll_instance(bNode *node,
                                      bNodeTree *nodetree,
                                      const char **r_disabled_hint)
{
  return node->typeinfo->poll(node->typeinfo, nodetree, r_disabled_hint);
}

void node_make_runtime_type(bNodeType *ntype,
                            const char *idname,
                            const char *ui_name,
                            const char *ui_description,
                            int ui_icon,
                            short node_class,
                            const StructRNA *rna_base)
{
  const short node_flags = 0;

  /* Basic type setup. */
  node_type_base_custom(ntype, idname, ui_name, node_class, node_flags);
  BLI_strncpy(ntype->ui_description, ui_description, sizeof(ntype->ui_description));
  ntype->ui_icon = ui_icon;

  /* RNA runtime type declaration. */
  ntype->rna_ext.srna = RNA_def_struct_ptr(&BLENDER_RNA, idname, (StructRNA *)rna_base);
  RNA_struct_blender_type_set(ntype->rna_ext.srna, ntype);

  RNA_def_struct_ui_text(ntype->rna_ext.srna, ntype->ui_name, ntype->ui_description);
  RNA_def_struct_ui_icon(ntype->rna_ext.srna, ntype->ui_icon);

  /* Default BKE callbacks. */
  ntype->poll = custom_node_poll_default;
  ntype->poll_instance = custom_node_poll_instance;
  ntype->insert_link = node_insert_link_default;
  ntype->update_internal_links = node_update_internal_links_default;

  /* Default UI callbacks. */
  ED_init_custom_node_type(ntype);
}

void node_free_runtime_type(bNodeType *ntype)
{
  if (!ntype) {
    return;
  }

  RNA_struct_free_extension(ntype->rna_ext.srna, &ntype->rna_ext);
  RNA_struct_free(&BLENDER_RNA, ntype->rna_ext.srna);
}
