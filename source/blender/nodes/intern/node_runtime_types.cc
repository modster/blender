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

void node_make_runtime_type_ex(bNodeType *ntype,
                               const char *idname,
                               const char *ui_name,
                               const char *ui_description,
                               int ui_icon,
                               short node_class,
                               const StructRNA *rna_base,
                               NodePollCb poll_cb,
                               NodeInstancePollCb instance_poll_cb,
                               NodeInitCb init_cb,
                               NodeFreeCb free_cb,
                               NodeCopyCb copy_cb,
                               NodeInsertLinkCb insert_link_cb,
                               NodeUpdateInternalLinksCb update_internal_links_cb,
                               NodeUpdateCb update_cb,
                               NodeGroupUpdateCb group_update_cb,
                               NodeLabelCb label_cb,
                               NodeDrawButtonsCb draw_buttons_cb,
                               NodeDrawButtonsExCb draw_buttons_ex_cb,
                               NodeDrawBackdropCb draw_backdrop_cb,
                               eNodeSizePreset size_preset)
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

  /* BKE callbacks. */
  ntype->poll = poll_cb ? poll_cb : custom_node_poll_default;
  ntype->poll_instance = instance_poll_cb ? instance_poll_cb : custom_node_poll_instance;
  ntype->initfunc = init_cb ? init_cb : nullptr;
  ntype->copyfunc = copy_cb ? copy_cb : nullptr;
  ntype->freefunc = free_cb ? free_cb : nullptr;
  ntype->insert_link = insert_link_cb ? insert_link_cb : node_insert_link_default;
  ntype->update_internal_links = update_internal_links_cb ? update_internal_links_cb :
                                                            node_update_internal_links_default;
  ntype->updatefunc = update_cb ? update_cb : nullptr;
  ntype->group_update_func = group_update_cb ? group_update_cb : nullptr;

  /* UI callbacks. */
  ED_init_custom_node_type(ntype);
  ntype->labelfunc = label_cb ? label_cb : nullptr;
  ntype->draw_buttons = draw_buttons_cb ? draw_buttons_cb : nullptr;
  ntype->draw_buttons_ex = draw_buttons_ex_cb ? draw_buttons_ex_cb : nullptr;
  ntype->draw_backdrop = draw_backdrop_cb ? draw_backdrop_cb : nullptr;
  node_type_size_preset(ntype, size_preset);
}

void node_free_runtime_type(bNodeType *ntype)
{
  if (!ntype) {
    return;
  }

  RNA_struct_free_extension(ntype->rna_ext.srna, &ntype->rna_ext);
  RNA_struct_free(&BLENDER_RNA, ntype->rna_ext.srna);
}

namespace blender::nodes {

}  // namespace blender::nodes
