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
 * \ingroup blenloader
 */
/* allow readfile to use deprecated functionality */
#define DNA_DEPRECATED_ALLOW

#include <string.h>

#include "MEM_guardedalloc.h"

#include "BLI_listbase.h"
#include "BLI_string.h"
#include "BLI_utildefines.h"

#include "DNA_armature_types.h"
#include "DNA_brush_types.h"
#include "DNA_genfile.h"
#include "DNA_modifier_types.h"
#include "DNA_text_types.h"

#include "BKE_idprop.h"
#include "BKE_lib_id.h"
#include "BKE_main.h"

#include "RNA_access.h"
#include "RNA_enum_types.h"

#include "BLO_readfile.h"
#include "readfile.h"

static IDProperty *idproperty_find_ui_container(IDProperty *idprop_group)
{
  LISTBASE_FOREACH (IDProperty *, prop, &idprop_group->data.group) {
    if (prop->type == IDP_GROUP && STREQ(prop->name, "_RNA_UI")) {
      return prop;
    }
  }
  return NULL;
}

static void version_idproperty_move_data_int(IDPropertyUIDataInt *ui_data,
                                             const IDProperty *prop_ui_data)
{
  IDProperty *min = IDP_GetPropertyFromGroup(prop_ui_data, "min");
  if (min != NULL) {
    ui_data->min = ui_data->soft_min = IDP_get_as_int(min);
  }
  IDProperty *max = IDP_GetPropertyFromGroup(prop_ui_data, "max");
  if (max != NULL) {
    ui_data->max = ui_data->soft_max = IDP_get_as_int(max);
  }
  IDProperty *soft_min = IDP_GetPropertyFromGroup(prop_ui_data, "soft_min");
  if (soft_min != NULL) {
    ui_data->soft_min = IDP_get_as_int(soft_min);
    ui_data->soft_min = MIN2(ui_data->soft_min, ui_data->min);
  }
  IDProperty *soft_max = IDP_GetPropertyFromGroup(prop_ui_data, "soft_max");
  if (soft_max != NULL) {
    ui_data->soft_max = IDP_get_as_int(soft_max);
    ui_data->soft_max = MAX2(ui_data->soft_max, ui_data->max);
  }
  IDProperty *step = IDP_GetPropertyFromGroup(prop_ui_data, "step");
  if (step != NULL) {
    ui_data->step = IDP_get_as_int(soft_max);
  }
  IDProperty *default_value = IDP_GetPropertyFromGroup(prop_ui_data, "default");
  if (default_value != NULL) {
    if (default_value->type == IDP_ARRAY) {
      if (default_value->subtype == IDP_INT) {
        ui_data->default_array = MEM_dupallocN(IDP_Array(default_value));
        ui_data->default_array_len = default_value->len;
      }
    }
    else if (default_value->type == IDP_INT) {
      ui_data->default_value = IDP_get_as_int(default_value);
    }
  }
}

static void version_idproperty_move_data_float(IDPropertyUIDataFloat *ui_data,
                                               const IDProperty *prop_ui_data)
{
  IDProperty *min = IDP_GetPropertyFromGroup(prop_ui_data, "min");
  if (min != NULL) {
    ui_data->min = ui_data->soft_min = IDP_get_as_double(min);
  }
  IDProperty *max = IDP_GetPropertyFromGroup(prop_ui_data, "max");
  if (max != NULL) {
    ui_data->max = ui_data->soft_max = IDP_get_as_double(min);
  }
  IDProperty *soft_min = IDP_GetPropertyFromGroup(prop_ui_data, "soft_min");
  if (soft_min != NULL) {
    ui_data->soft_min = IDP_get_as_double(soft_min);
    ui_data->soft_min = MAX2(ui_data->soft_min, ui_data->min);
  }
  IDProperty *soft_max = IDP_GetPropertyFromGroup(prop_ui_data, "soft_max");
  if (soft_max != NULL) {
    ui_data->soft_max = IDP_get_as_double(soft_max);
    ui_data->soft_max = MIN2(ui_data->soft_max, ui_data->max);
  }
  IDProperty *step = IDP_GetPropertyFromGroup(prop_ui_data, "step");
  if (step != NULL) {
    ui_data->step = IDP_get_as_float(step);
  }
  IDProperty *precision = IDP_GetPropertyFromGroup(prop_ui_data, "precision");
  if (precision != NULL) {
    ui_data->precision = IDP_get_as_int(precision);
  }
  IDProperty *default_value = IDP_GetPropertyFromGroup(prop_ui_data, "default");
  if (default_value != NULL) {
    if (default_value->type == IDP_ARRAY) {
      if (ELEM(default_value->subtype, IDP_FLOAT, IDP_DOUBLE)) {
        ui_data->default_array = MEM_dupallocN(IDP_Array(default_value));
        ui_data->default_array_len = default_value->len;
      }
    }
    else if (ELEM(default_value->type, IDP_DOUBLE, IDP_FLOAT)) {
      ui_data->default_value = IDP_get_as_double(default_value);
    }
  }
}

static void version_idproperty_move_data_string(IDPropertyUIDataString *ui_data,
                                                const IDProperty *prop_ui_data)
{
  IDProperty *default_value = IDP_GetPropertyFromGroup(prop_ui_data, "default");
  if (default_value != NULL && default_value->type == IDP_STRING) {
    ui_data->default_value = BLI_strdup(IDP_String(default_value));
  }
}

static void version_idproperty_ui_data(IDProperty *idprop_group)
{
  if (idprop_group == NULL) { /* NULL check here to reduce verbosity of calls to this function. */
    return;
  }

  IDProperty *ui_container = idproperty_find_ui_container(idprop_group);
  if (ui_container == NULL) {
    return;
  }

  LISTBASE_FOREACH (IDProperty *, prop, &idprop_group->data.group) {
    IDProperty *prop_ui_data = IDP_GetPropertyFromGroup(ui_container, prop->name);
    if (prop_ui_data == NULL) {
      continue;
    }

    if (!IDP_supports_ui_data(prop)) {
      continue;
    }

    IDPropertyUIData *ui_data = IDP_ui_data_ensure(prop);

    IDProperty *subtype = IDP_GetPropertyFromGroup(prop_ui_data, "subtype");
    if (subtype != NULL && subtype->type == IDP_STRING) {
      const char *subtype_string = IDP_String(subtype);
      int result = PROP_NONE;
      RNA_enum_value_from_id(rna_enum_property_subtype_items, subtype_string, &result);
      ui_data->rna_subtype = result;
    }

    IDProperty *description = IDP_GetPropertyFromGroup(prop_ui_data, "description");
    if (description != NULL && description->type == IDP_STRING) {
      ui_data->description = BLI_strdup(IDP_String(description));
    }

    /* Type specific data. */
    switch (IDP_ui_data_type(prop)) {
      case IDP_UI_DATA_TYPE_STRING:
        version_idproperty_move_data_string((IDPropertyUIDataString *)ui_data, prop_ui_data);
        break;
      case IDP_UI_DATA_TYPE_ID:
        break;
      case IDP_UI_DATA_TYPE_INT:
        version_idproperty_move_data_int((IDPropertyUIDataInt *)ui_data, prop_ui_data);
        break;
      case IDP_UI_DATA_TYPE_FLOAT:
        version_idproperty_move_data_float((IDPropertyUIDataFloat *)ui_data, prop_ui_data);
        break;
      case IDP_UI_DATA_TYPE_UNSUPPORTED:
        BLI_assert(false);
        break;
    }

    IDP_FreeFromGroup(ui_container, prop_ui_data);
  }

  IDP_FreeFromGroup(idprop_group, ui_container);
}

static void do_versions_idproperty_bones_recursive(Bone *bone)
{
  version_idproperty_ui_data(bone->prop);
  LISTBASE_FOREACH (Bone *, child_bone, &bone->childbase) {
    do_versions_idproperty_bones_recursive(child_bone);
  }
}

/**
 * For every data block that supports them, initialize the new IDProperty UI data struct based on
 * the old more complicated storage. Assumes only the top level of IDProperties below the parent
 * group had UI data in a "_RNA_UI" group.
 *
 * \note The following IDProperty groups in DNA aren't exposed in the UI or are runtime-only, so
 * they don't have UI data: wmOperator, bAddon, bUserMenuItem_Op, wmKeyMapItem, wmKeyConfigPref,
 * uiList, FFMpegCodecData, View3DShading, bToolRef, TimeMarker, ViewLayer, bPoseChannel.
 */
static void do_versions_idproperty_ui_data(Main *bmain)
{
  /* ID data. */
  ID *id;
  FOREACH_MAIN_ID_BEGIN (bmain, id) {
    IDProperty *idprop_group = IDP_GetProperties(id, false);
    if (idprop_group == NULL) {
      continue;
    }
    version_idproperty_ui_data(idprop_group);
  }
  FOREACH_MAIN_ID_END;

  /* Bones. */
  LISTBASE_FOREACH (bArmature *, armature, &bmain->armatures) {
    LISTBASE_FOREACH (Bone *, bone, &armature->bonebase) {
      do_versions_idproperty_bones_recursive(bone);
    }
  }

  /* Nodes and node sockets. */
  LISTBASE_FOREACH (bNodeTree *, ntree, &bmain->nodetrees) {
    LISTBASE_FOREACH (bNode *, node, &ntree->nodes) {
      version_idproperty_ui_data(node->prop);
    }
    LISTBASE_FOREACH (bNodeSocket *, socket, &ntree->inputs) {
      version_idproperty_ui_data(socket->prop);
    }
    LISTBASE_FOREACH (bNodeSocket *, socket, &ntree->outputs) {
      version_idproperty_ui_data(socket->prop);
    }
  }

  /* The UI data from exposed node modifier properties is just copied from the corresponding node
   * group, but the copying only runs when necessary, so we still need to version UI data here. */
  LISTBASE_FOREACH (Object *, ob, &bmain->objects) {
    LISTBASE_FOREACH (ModifierData *, md, &ob->modifiers) {
      if (md->type == eModifierType_Nodes) {
        NodesModifierData *nmd = (NodesModifierData *)md;
        version_idproperty_ui_data(nmd->settings.properties);
      }
    }
  }

  /* Sequences. */
  LISTBASE_FOREACH (Scene *, scene, &bmain->scenes) {
    if (scene->ed != NULL) {
      LISTBASE_FOREACH (Sequence *, seq, &scene->ed->seqbase) {
        version_idproperty_ui_data(seq->prop);
      }
    }
  }
}

void do_versions_after_linking_300(Main *bmain, ReportList *UNUSED(reports))
{
  if (MAIN_VERSION_ATLEAST(bmain, 300, 0) && !MAIN_VERSION_ATLEAST(bmain, 300, 1)) {
    /* Set zero user text objects to have a fake user. */
    LISTBASE_FOREACH (Text *, text, &bmain->texts) {
      if (text->id.us == 0) {
        id_fake_user_set(&text->id);
      }
    }
  }
  /**
   * Versioning code until next subversion bump goes here.
   *
   * \note Be sure to check when bumping the version:
   * - #blo_do_versions_300 in this file.
   * - "versioning_userdef.c", #blo_do_versions_userdef
   * - "versioning_userdef.c", #do_versions_theme
   *
   * \note Keep this message at the bottom of the function.
   */
  {
    /* Keep this block, even when empty. */
    do_versions_idproperty_ui_data(bmain);
  }
}

/* NOLINTNEXTLINE: readability-function-size */
void blo_do_versions_300(FileData *fd, Library *UNUSED(lib), Main *bmain)
{
  if (!MAIN_VERSION_ATLEAST(bmain, 300, 1)) {
    /* Set default value for the new bisect_threshold parameter in the mirror modifier. */
    if (!DNA_struct_elem_find(fd->filesdna, "MirrorModifierData", "float", "bisect_threshold")) {
      LISTBASE_FOREACH (Object *, ob, &bmain->objects) {
        LISTBASE_FOREACH (ModifierData *, md, &ob->modifiers) {
          if (md->type == eModifierType_Mirror) {
            MirrorModifierData *mmd = (MirrorModifierData *)md;
            /* This was the previous hard-coded value. */
            mmd->bisect_threshold = 0.001f;
          }
        }
      }
    }
    /* Grease Pencil: Set default value for dilate pixels. */
    if (!DNA_struct_elem_find(fd->filesdna, "BrushGpencilSettings", "int", "dilate_pixels")) {
      LISTBASE_FOREACH (Brush *, brush, &bmain->brushes) {
        if (brush->gpencil_settings) {
          brush->gpencil_settings->dilate_pixels = 1;
        }
      }
    }
  }
  /**
   * Versioning code until next subversion bump goes here.
   *
   * \note Be sure to check when bumping the version:
   * - "versioning_userdef.c", #blo_do_versions_userdef
   * - "versioning_userdef.c", #do_versions_theme
   *
   * \note Keep this message at the bottom of the function.
   */
  {
    /* Keep this block, even when empty. */
  }
}
