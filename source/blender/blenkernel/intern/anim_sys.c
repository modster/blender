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
 * The Original Code is Copyright (C) 2009 Blender Foundation, Joshua Leung
 * All rights reserved.
 */

/** \file
 * \ingroup bke
 */

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "MEM_guardedalloc.h"

#include "BLI_alloca.h"
#include "BLI_blenlib.h"
#include "BLI_dynstr.h"
#include "BLI_listbase.h"
#include "BLI_math_rotation.h"
#include "BLI_math_vector.h"
#include "BLI_string_utils.h"
#include "BLI_utildefines.h"

#include "BLT_translation.h"

#include "DNA_action_types.h"
#include "DNA_anim_types.h"
#include "DNA_constraint_types.h"
#include "DNA_light_types.h"
#include "DNA_material_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"
#include "DNA_space_types.h"
#include "DNA_texture_types.h"
#include "DNA_world_types.h"

#include "BKE_action.h"
#include "BKE_anim_data.h"
#include "BKE_animsys.h"
#include "BKE_constraint.h"
#include "BKE_context.h"
#include "BKE_fcurve.h"
#include "BKE_global.h"
#include "BKE_lib_id.h"
#include "BKE_main.h"
#include "BKE_material.h"
#include "BKE_nla.h"
#include "BKE_node.h"
#include "BKE_report.h"
#include "BKE_texture.h"

#include "DEG_depsgraph.h"
#include "DEG_depsgraph_query.h"

#include "RNA_access.h"

#include "BLO_read_write.h"

#include "nla_private.h"

#include "atomic_ops.h"

#include "CLG_log.h"

static CLG_LogRef LOG = {"bke.anim_sys"};

/* *********************************** */
/* KeyingSet API */

/* Finding Tools --------------------------- */

/* Find the first path that matches the given criteria */
/* TODO: do we want some method to perform partial matches too? */
KS_Path *BKE_keyingset_find_path(KeyingSet *ks,
                                 ID *id,
                                 const char group_name[],
                                 const char rna_path[],
                                 int array_index,
                                 int UNUSED(group_mode))
{
  KS_Path *ksp;

  /* sanity checks */
  if (ELEM(NULL, ks, rna_path, id)) {
    return NULL;
  }

  /* loop over paths in the current KeyingSet, finding the first one where all settings match
   * (i.e. the first one where none of the checks fail and equal 0)
   */
  for (ksp = ks->paths.first; ksp; ksp = ksp->next) {
    short eq_id = 1, eq_path = 1, eq_index = 1, eq_group = 1;

    /* id */
    if (id != ksp->id) {
      eq_id = 0;
    }

    /* path */
    if ((ksp->rna_path == NULL) || !STREQ(rna_path, ksp->rna_path)) {
      eq_path = 0;
    }

    /* index - need to compare whole-array setting too... */
    if (ksp->array_index != array_index) {
      eq_index = 0;
    }

    /* group */
    if (group_name) {
      /* FIXME: these checks need to be coded... for now, it's not too important though */
    }

    /* if all aspects are ok, return */
    if (eq_id && eq_path && eq_index && eq_group) {
      return ksp;
    }
  }

  /* none found */
  return NULL;
}

/* Defining Tools --------------------------- */

/* Used to create a new 'custom' KeyingSet for the user,
 * that will be automatically added to the stack */
KeyingSet *BKE_keyingset_add(
    ListBase *list, const char idname[], const char name[], short flag, short keyingflag)
{
  KeyingSet *ks;

  /* allocate new KeyingSet */
  ks = MEM_callocN(sizeof(KeyingSet), "KeyingSet");

  BLI_strncpy(
      ks->idname, (idname) ? idname : (name) ? name : DATA_("KeyingSet"), sizeof(ks->idname));
  BLI_strncpy(ks->name, (name) ? name : (idname) ? idname : DATA_("Keying Set"), sizeof(ks->name));

  ks->flag = flag;
  ks->keyingflag = keyingflag;
  /* NOTE: assume that if one is set one way, the other should be too, so that it'll work */
  ks->keyingoverride = keyingflag;

  /* add KeyingSet to list */
  BLI_addtail(list, ks);

  /* Make sure KeyingSet has a unique idname */
  BLI_uniquename(
      list, ks, DATA_("KeyingSet"), '.', offsetof(KeyingSet, idname), sizeof(ks->idname));

  /* Make sure KeyingSet has a unique label (this helps with identification) */
  BLI_uniquename(list, ks, DATA_("Keying Set"), '.', offsetof(KeyingSet, name), sizeof(ks->name));

  /* return new KeyingSet for further editing */
  return ks;
}

/* Add a path to a KeyingSet. Nothing is returned for now...
 * Checks are performed to ensure that destination is appropriate for the KeyingSet in question
 */
KS_Path *BKE_keyingset_add_path(KeyingSet *ks,
                                ID *id,
                                const char group_name[],
                                const char rna_path[],
                                int array_index,
                                short flag,
                                short groupmode)
{
  KS_Path *ksp;

  /* sanity checks */
  if (ELEM(NULL, ks, rna_path)) {
    CLOG_ERROR(&LOG, "no Keying Set and/or RNA Path to add path with");
    return NULL;
  }

  /* ID is required for all types of KeyingSets */
  if (id == NULL) {
    CLOG_ERROR(&LOG, "No ID provided for Keying Set Path");
    return NULL;
  }

  /* don't add if there is already a matching KS_Path in the KeyingSet */
  if (BKE_keyingset_find_path(ks, id, group_name, rna_path, array_index, groupmode)) {
    if (G.debug & G_DEBUG) {
      CLOG_ERROR(&LOG, "destination already exists in Keying Set");
    }
    return NULL;
  }

  /* allocate a new KeyingSet Path */
  ksp = MEM_callocN(sizeof(KS_Path), "KeyingSet Path");

  /* just store absolute info */
  ksp->id = id;
  if (group_name) {
    BLI_strncpy(ksp->group, group_name, sizeof(ksp->group));
  }
  else {
    ksp->group[0] = '\0';
  }

  /* store additional info for relative paths (just in case user makes the set relative) */
  if (id) {
    ksp->idtype = GS(id->name);
  }

  /* just copy path info */
  /* TODO: should array index be checked too? */
  ksp->rna_path = BLI_strdup(rna_path);
  ksp->array_index = array_index;

  /* store flags */
  ksp->flag = flag;
  ksp->groupmode = groupmode;

  /* add KeyingSet path to KeyingSet */
  BLI_addtail(&ks->paths, ksp);

  /* return this path */
  return ksp;
}

/* Free the given Keying Set path */
void BKE_keyingset_free_path(KeyingSet *ks, KS_Path *ksp)
{
  /* sanity check */
  if (ELEM(NULL, ks, ksp)) {
    return;
  }

  /* free RNA-path info */
  if (ksp->rna_path) {
    MEM_freeN(ksp->rna_path);
  }

  /* free path itself */
  BLI_freelinkN(&ks->paths, ksp);
}

/* Copy all KeyingSets in the given list */
void BKE_keyingsets_copy(ListBase *newlist, const ListBase *list)
{
  KeyingSet *ksn;
  KS_Path *kspn;

  BLI_duplicatelist(newlist, list);

  for (ksn = newlist->first; ksn; ksn = ksn->next) {
    BLI_duplicatelist(&ksn->paths, &ksn->paths);

    for (kspn = ksn->paths.first; kspn; kspn = kspn->next) {
      kspn->rna_path = MEM_dupallocN(kspn->rna_path);
    }
  }
}

/* Freeing Tools --------------------------- */

/* Free data for KeyingSet but not set itself */
void BKE_keyingset_free(KeyingSet *ks)
{
  KS_Path *ksp, *kspn;

  /* sanity check */
  if (ks == NULL) {
    return;
  }

  /* free each path as we go to avoid looping twice */
  for (ksp = ks->paths.first; ksp; ksp = kspn) {
    kspn = ksp->next;
    BKE_keyingset_free_path(ks, ksp);
  }
}

/* Free all the KeyingSets in the given list */
void BKE_keyingsets_free(ListBase *list)
{
  KeyingSet *ks, *ksn;

  /* sanity check */
  if (list == NULL) {
    return;
  }

  /* loop over KeyingSets freeing them
   * - BKE_keyingset_free() doesn't free the set itself, but it frees its sub-data
   */
  for (ks = list->first; ks; ks = ksn) {
    ksn = ks->next;
    BKE_keyingset_free(ks);
    BLI_freelinkN(list, ks);
  }
}

void BKE_keyingsets_blend_write(BlendWriter *writer, ListBase *list)
{
  LISTBASE_FOREACH (KeyingSet *, ks, list) {
    /* KeyingSet */
    BLO_write_struct(writer, KeyingSet, ks);

    /* Paths */
    LISTBASE_FOREACH (KS_Path *, ksp, &ks->paths) {
      /* Path */
      BLO_write_struct(writer, KS_Path, ksp);

      if (ksp->rna_path) {
        BLO_write_string(writer, ksp->rna_path);
      }
    }
  }
}

void BKE_keyingsets_blend_read_data(BlendDataReader *reader, ListBase *list)
{
  LISTBASE_FOREACH (KeyingSet *, ks, list) {
    /* paths */
    BLO_read_list(reader, &ks->paths);

    LISTBASE_FOREACH (KS_Path *, ksp, &ks->paths) {
      /* rna path */
      BLO_read_data_address(reader, &ksp->rna_path);
    }
  }
}

void BKE_keyingsets_blend_read_lib(BlendLibReader *reader, ID *id, ListBase *list)
{
  LISTBASE_FOREACH (KeyingSet *, ks, list) {
    LISTBASE_FOREACH (KS_Path *, ksp, &ks->paths) {
      BLO_read_id_address(reader, id->lib, &ksp->id);
    }
  }
}

void BKE_keyingsets_blend_read_expand(BlendExpander *expander, ListBase *list)
{
  LISTBASE_FOREACH (KeyingSet *, ks, list) {
    LISTBASE_FOREACH (KS_Path *, ksp, &ks->paths) {
      BLO_expand(expander, ksp->id);
    }
  }
}

/* ***************************************** */
/* Evaluation Data-Setting Backend */

bool BKE_animsys_store_rna_setting(PointerRNA *ptr,
                                   /* typically 'fcu->rna_path', 'fcu->array_index' */
                                   const char *rna_path,
                                   const int array_index,
                                   PathResolvedRNA *r_result)
{
  bool success = false;
  const char *path = rna_path;

  /* write value to setting */
  if (path) {
    /* get property to write to */
    if (RNA_path_resolve_property(ptr, path, &r_result->ptr, &r_result->prop)) {
      if ((ptr->owner_id == NULL) || RNA_property_animateable(&r_result->ptr, r_result->prop)) {
        int array_len = RNA_property_array_length(&r_result->ptr, r_result->prop);

        if (array_len && array_index >= array_len) {
          if (G.debug & G_DEBUG) {
            CLOG_WARN(&LOG,
                      "Animato: Invalid array index. ID = '%s',  '%s[%d]', array length is %d",
                      (ptr->owner_id) ? (ptr->owner_id->name + 2) : "<No ID>",
                      path,
                      array_index,
                      array_len - 1);
          }
        }
        else {
          r_result->prop_index = array_len ? array_index : -1;
          success = true;
        }
      }
    }
    else {
      /* failed to get path */
      /* XXX don't tag as failed yet though, as there are some legit situations (Action Constraint)
       * where some channels will not exist, but shouldn't lock up Action */
      if (G.debug & G_DEBUG) {
        CLOG_WARN(&LOG,
                  "Animato: Invalid path. ID = '%s',  '%s[%d]'",
                  (ptr->owner_id) ? (ptr->owner_id->name + 2) : "<No ID>",
                  path,
                  array_index);
      }
    }
  }

  return success;
}

/* less than 1.0 evaluates to false, use epsilon to avoid float error */
#define ANIMSYS_FLOAT_AS_BOOL(value) ((value) > ((1.0f - FLT_EPSILON)))

bool BKE_animsys_read_rna_setting(PathResolvedRNA *anim_rna, float *r_value)
{
  PropertyRNA *prop = anim_rna->prop;
  PointerRNA *ptr = &anim_rna->ptr;
  int array_index = anim_rna->prop_index;
  float orig_value;

  /* caller must ensure this is animatable */
  BLI_assert(RNA_property_animateable(ptr, prop) || ptr->owner_id == NULL);

  switch (RNA_property_type(prop)) {
    case PROP_BOOLEAN: {
      if (array_index != -1) {
        const int orig_value_coerce = RNA_property_boolean_get_index(ptr, prop, array_index);
        orig_value = (float)orig_value_coerce;
      }
      else {
        const int orig_value_coerce = RNA_property_boolean_get(ptr, prop);
        orig_value = (float)orig_value_coerce;
      }
      break;
    }
    case PROP_INT: {
      if (array_index != -1) {
        const int orig_value_coerce = RNA_property_int_get_index(ptr, prop, array_index);
        orig_value = (float)orig_value_coerce;
      }
      else {
        const int orig_value_coerce = RNA_property_int_get(ptr, prop);
        orig_value = (float)orig_value_coerce;
      }
      break;
    }
    case PROP_FLOAT: {
      if (array_index != -1) {
        const float orig_value_coerce = RNA_property_float_get_index(ptr, prop, array_index);
        orig_value = (float)orig_value_coerce;
      }
      else {
        const float orig_value_coerce = RNA_property_float_get(ptr, prop);
        orig_value = (float)orig_value_coerce;
      }
      break;
    }
    case PROP_ENUM: {
      const int orig_value_coerce = RNA_property_enum_get(ptr, prop);
      orig_value = (float)orig_value_coerce;
      break;
    }
    default:
      /* nothing can be done here... so it is unsuccessful? */
      return false;
  }

  if (r_value != NULL) {
    *r_value = orig_value;
  }

  /* successful */
  return true;
}

/* Write the given value to a setting using RNA, and return success */
bool BKE_animsys_write_rna_setting(PathResolvedRNA *anim_rna, const float value)
{
  PropertyRNA *prop = anim_rna->prop;
  PointerRNA *ptr = &anim_rna->ptr;
  int array_index = anim_rna->prop_index;

  /* caller must ensure this is animatable */
  BLI_assert(RNA_property_animateable(ptr, prop) || ptr->owner_id == NULL);

  /* Check whether value is new. Otherwise we skip all the updates. */
  float old_value;
  if (!BKE_animsys_read_rna_setting(anim_rna, &old_value)) {
    return false;
  }
  if (old_value == value) {
    return true;
  }

  switch (RNA_property_type(prop)) {
    case PROP_BOOLEAN: {
      const int value_coerce = ANIMSYS_FLOAT_AS_BOOL(value);
      if (array_index != -1) {
        RNA_property_boolean_set_index(ptr, prop, array_index, value_coerce);
      }
      else {
        RNA_property_boolean_set(ptr, prop, value_coerce);
      }
      break;
    }
    case PROP_INT: {
      int value_coerce = (int)value;
      RNA_property_int_clamp(ptr, prop, &value_coerce);
      if (array_index != -1) {
        RNA_property_int_set_index(ptr, prop, array_index, value_coerce);
      }
      else {
        RNA_property_int_set(ptr, prop, value_coerce);
      }
      break;
    }
    case PROP_FLOAT: {
      float value_coerce = value;
      RNA_property_float_clamp(ptr, prop, &value_coerce);
      if (array_index != -1) {
        RNA_property_float_set_index(ptr, prop, array_index, value_coerce);
      }
      else {
        RNA_property_float_set(ptr, prop, value_coerce);
      }
      break;
    }
    case PROP_ENUM: {
      const int value_coerce = (int)value;
      RNA_property_enum_set(ptr, prop, value_coerce);
      break;
    }
    default:
      /* nothing can be done here... so it is unsuccessful? */
      return false;
  }

  /* successful */
  return true;
}

static bool animsys_construct_orig_pointer_rna(const PointerRNA *ptr, PointerRNA *ptr_orig)
{
  *ptr_orig = *ptr;
  /* NOTE: nlastrip_evaluate_controls() creates PointerRNA with ID of NULL. Technically, this is
   * not a valid pointer, but there are exceptions in various places of this file which handles
   * such pointers.
   * We do special trickery here as well, to quickly go from evaluated to original NlaStrip. */
  if (ptr->owner_id == NULL) {
    if (ptr->type != &RNA_NlaStrip) {
      return false;
    }
    NlaStrip *strip = ((NlaStrip *)ptr_orig->data);
    if (strip->orig_strip == NULL) {
      return false;
    }
    ptr_orig->data = strip->orig_strip;
  }
  else {
    ptr_orig->owner_id = ptr_orig->owner_id->orig_id;
    ptr_orig->data = ptr_orig->owner_id;
  }
  return true;
}

static void animsys_write_orig_anim_rna(PointerRNA *ptr,
                                        const char *rna_path,
                                        int array_index,
                                        float value)
{
  PointerRNA ptr_orig;
  if (!animsys_construct_orig_pointer_rna(ptr, &ptr_orig)) {
    return;
  }
  PathResolvedRNA orig_anim_rna;
  /* TODO(sergey): Should be possible to cache resolved path in dependency graph somehow. */
  if (BKE_animsys_store_rna_setting(&ptr_orig, rna_path, array_index, &orig_anim_rna)) {
    BKE_animsys_write_rna_setting(&orig_anim_rna, value);
  }
}

/**
 * Evaluate all the F-Curves in the given list
 * This performs a set of standard checks. If extra checks are required,
 * separate code should be used.
 */
static void animsys_evaluate_fcurves(PointerRNA *ptr,
                                     ListBase *list,
                                     const AnimationEvalContext *anim_eval_context,
                                     bool flush_to_original)
{
  /* Calculate then execute each curve. */
  LISTBASE_FOREACH (FCurve *, fcu, list) {
    /* Check if this F-Curve doesn't belong to a muted group. */
    if ((fcu->grp != NULL) && (fcu->grp->flag & AGRP_MUTED)) {
      continue;
    }
    /* Check if this curve should be skipped. */
    if ((fcu->flag & (FCURVE_MUTED | FCURVE_DISABLED))) {
      continue;
    }
    /* Skip empty curves, as if muted. */
    if (BKE_fcurve_is_empty(fcu)) {
      continue;
    }
    PathResolvedRNA anim_rna;
    if (BKE_animsys_store_rna_setting(ptr, fcu->rna_path, fcu->array_index, &anim_rna)) {
      const float curval = calculate_fcurve(&anim_rna, fcu, anim_eval_context);
      BKE_animsys_write_rna_setting(&anim_rna, curval);
      if (flush_to_original) {
        animsys_write_orig_anim_rna(ptr, fcu->rna_path, fcu->array_index, curval);
      }
    }
  }
}

/* ***************************************** */
/* Driver Evaluation */

AnimationEvalContext BKE_animsys_eval_context_construct(struct Depsgraph *depsgraph,
                                                        float eval_time)
{
  AnimationEvalContext ctx = {
      .depsgraph = depsgraph,
      .eval_time = eval_time,
  };
  return ctx;
}

AnimationEvalContext BKE_animsys_eval_context_construct_at(
    const AnimationEvalContext *anim_eval_context, float eval_time)
{
  return BKE_animsys_eval_context_construct(anim_eval_context->depsgraph, eval_time);
}

/* Evaluate Drivers */
static void animsys_evaluate_drivers(PointerRNA *ptr,
                                     AnimData *adt,
                                     const AnimationEvalContext *anim_eval_context)
{
  FCurve *fcu;

  /* drivers are stored as F-Curves, but we cannot use the standard code, as we need to check if
   * the depsgraph requested that this driver be evaluated...
   */
  for (fcu = adt->drivers.first; fcu; fcu = fcu->next) {
    ChannelDriver *driver = fcu->driver;
    bool ok = false;

    /* check if this driver's curve should be skipped */
    if ((fcu->flag & (FCURVE_MUTED | FCURVE_DISABLED)) == 0) {
      /* check if driver itself is tagged for recalculation */
      /* XXX driver recalc flag is not set yet by depsgraph! */
      if ((driver) && !(driver->flag & DRIVER_FLAG_INVALID)) {
        /* evaluate this using values set already in other places
         * NOTE: for 'layering' option later on, we should check if we should remove old value
         * before adding new to only be done when drivers only changed. */
        PathResolvedRNA anim_rna;
        if (BKE_animsys_store_rna_setting(ptr, fcu->rna_path, fcu->array_index, &anim_rna)) {
          const float curval = calculate_fcurve(&anim_rna, fcu, anim_eval_context);
          ok = BKE_animsys_write_rna_setting(&anim_rna, curval);
        }

        /* set error-flag if evaluation failed */
        if (ok == 0) {
          driver->flag |= DRIVER_FLAG_INVALID;
        }
      }
    }
  }
}

/* ***************************************** */
/* Actions Evaluation */

/* strictly not necessary for actual "evaluation", but it is a useful safety check
 * to reduce the amount of times that users end up having to "revive" wrongly-assigned
 * actions
 */
static void action_idcode_patch_check(ID *id, bAction *act)
{
  int idcode = 0;

  /* just in case */
  if (ELEM(NULL, id, act)) {
    return;
  }

  idcode = GS(id->name);

  /* the actual checks... hopefully not too much of a performance hit in the long run... */
  if (act->idroot == 0) {
    /* use the current root if not set already
     * (i.e. newly created actions and actions from 2.50-2.57 builds).
     * - this has problems if there are 2 users, and the first one encountered is the invalid one
     *   in which case, the user will need to manually fix this (?)
     */
    act->idroot = idcode;
  }
  else if (act->idroot != idcode) {
    /* only report this error if debug mode is enabled (to save performance everywhere else) */
    if (G.debug & G_DEBUG) {
      printf(
          "AnimSys Safety Check Failed: Action '%s' is not meant to be used from ID-Blocks of "
          "type %d such as '%s'\n",
          act->id.name + 2,
          idcode,
          id->name);
    }
  }
}

/* ----------------------------------------- */

/* Evaluate Action Group */
void animsys_evaluate_action_group(PointerRNA *ptr,
                                   bAction *act,
                                   bActionGroup *agrp,
                                   const AnimationEvalContext *anim_eval_context)
{
  FCurve *fcu;

  /* check if mapper is appropriate for use here (we set to NULL if it's inappropriate) */
  if (ELEM(NULL, act, agrp)) {
    return;
  }

  action_idcode_patch_check(ptr->owner_id, act);

  /* if group is muted, don't evaluated any of the F-Curve */
  if (agrp->flag & AGRP_MUTED) {
    return;
  }

  /* calculate then execute each curve */
  for (fcu = agrp->channels.first; (fcu) && (fcu->grp == agrp); fcu = fcu->next) {
    /* check if this curve should be skipped */
    if ((fcu->flag & (FCURVE_MUTED | FCURVE_DISABLED)) == 0 && !BKE_fcurve_is_empty(fcu)) {
      PathResolvedRNA anim_rna;
      if (BKE_animsys_store_rna_setting(ptr, fcu->rna_path, fcu->array_index, &anim_rna)) {
        const float curval = calculate_fcurve(&anim_rna, fcu, anim_eval_context);
        BKE_animsys_write_rna_setting(&anim_rna, curval);
      }
    }
  }
}

/* Evaluate Action (F-Curve Bag) */
static void animsys_evaluate_action_ex(PointerRNA *ptr,
                                       bAction *act,
                                       const AnimationEvalContext *anim_eval_context,
                                       const bool flush_to_original)
{
  /* check if mapper is appropriate for use here (we set to NULL if it's inappropriate) */
  if (act == NULL) {
    return;
  }

  action_idcode_patch_check(ptr->owner_id, act);

  /* calculate then execute each curve */
  animsys_evaluate_fcurves(ptr, &act->curves, anim_eval_context, flush_to_original);
}

void animsys_evaluate_action(PointerRNA *ptr,
                             bAction *act,
                             const AnimationEvalContext *anim_eval_context,
                             const bool flush_to_original)
{
  animsys_evaluate_action_ex(ptr, act, anim_eval_context, flush_to_original);
}

/* ***************************************** */
/* NLA System - Evaluation */

/* calculate influence of strip based for given frame based on blendin/out values */
static float nlastrip_get_influence(NlaStrip *strip, float cframe)
{
  /* sanity checks - normalize the blendin/out values? */
  strip->blendin = fabsf(strip->blendin);
  strip->blendout = fabsf(strip->blendout);

  /* result depends on where frame is in respect to blendin/out values */
  if (IS_EQF(strip->blendin, 0.0f) == false && (cframe <= (strip->start + strip->blendin))) {
    /* there is some blend-in */
    return fabsf(cframe - strip->start) / (strip->blendin);
  }
  if (IS_EQF(strip->blendout, 0.0f) == false && (cframe >= (strip->end - strip->blendout))) {
    /* there is some blend-out */
    return fabsf(strip->end - cframe) / (strip->blendout);
  }

  /* in the middle of the strip, we should be full strength */
  return 1.0f;
}

/* evaluate the evaluation time and influence for the strip, storing the results in the strip */
static void nlastrip_evaluate_controls(NlaStrip *strip,
                                       const AnimationEvalContext *anim_eval_context,
                                       const bool flush_to_original)
{
  /* now strip's evaluate F-Curves for these settings (if applicable) */
  if (strip->fcurves.first) {
    PointerRNA strip_ptr;

    /* create RNA-pointer needed to set values */
    RNA_pointer_create(NULL, &RNA_NlaStrip, strip, &strip_ptr);

    /* execute these settings as per normal */
    animsys_evaluate_fcurves(&strip_ptr, &strip->fcurves, anim_eval_context, flush_to_original);
  }

  /* analytically generate values for influence and time (if applicable)
   * - we do this after the F-Curves have been evaluated to override the effects of those
   *   in case the override has been turned off.
   */
  if ((strip->flag & NLASTRIP_FLAG_USR_INFLUENCE) == 0) {
    strip->influence = nlastrip_get_influence(strip, anim_eval_context->eval_time);
  }

  /* Bypass evaluation time computation if time mapping is disabled. */
  if ((strip->flag & NLASTRIP_FLAG_NO_TIME_MAP) != 0) {
    strip->strip_time = anim_eval_context->eval_time;
    return;
  }

  if ((strip->flag & NLASTRIP_FLAG_USR_TIME) == 0) {
    strip->strip_time = nlastrip_get_frame(
        strip, anim_eval_context->eval_time, NLATIME_CONVERT_EVAL);
  }

  /* if user can control the evaluation time (using F-Curves), consider the option which allows
   * this time to be clamped to lie within extents of the action-clip, so that a steady changing
   * rate of progress through several cycles of the clip can be achieved easily.
   */
  /* NOTE: if we add any more of these special cases, we better group them up nicely... */
  if ((strip->flag & NLASTRIP_FLAG_USR_TIME) && (strip->flag & NLASTRIP_FLAG_USR_TIME_CYCLIC)) {
    strip->strip_time = fmod(strip->strip_time - strip->actstart, strip->actend - strip->actstart);
  }
}

/* gets the strip active at the current time for a list of strips for evaluation purposes */
NlaEvalStrip *nlastrips_ctime_get_strip(ListBase *list,
                                        ListBase *strips,
                                        short index,
                                        const AnimationEvalContext *anim_eval_context,
                                        const bool flush_to_original)
{
  NlaStrip *strip, *estrip = NULL;
  NlaEvalStrip *nes;
  short side = 0;
  float ctime = anim_eval_context->eval_time;

  /* loop over strips, checking if they fall within the range */
  for (strip = strips->first; strip; strip = strip->next) {
    /* check if current time occurs within this strip  */
    if (IN_RANGE_INCL(ctime, strip->start, strip->end) ||
        (strip->flag & NLASTRIP_FLAG_NO_TIME_MAP)) {
      /* this strip is active, so try to use it */
      estrip = strip;
      side = NES_TIME_WITHIN;
      break;
    }

    /* if time occurred before current strip... */
    if (ctime < strip->start) {
      if (strip == strips->first) {
        /* before first strip - only try to use it if it extends backwards in time too */
        if (strip->extendmode == NLASTRIP_EXTEND_HOLD) {
          estrip = strip;
        }

        /* side is 'before' regardless of whether there's a useful strip */
        side = NES_TIME_BEFORE;
      }
      else {
        /* before next strip - previous strip has ended, but next hasn't begun,
         * so blending mode depends on whether strip is being held or not...
         * - only occurs when no transition strip added, otherwise the transition would have
         *   been picked up above...
         */
        strip = strip->prev;

        if (strip->extendmode != NLASTRIP_EXTEND_NOTHING) {
          estrip = strip;
        }
        side = NES_TIME_AFTER;
      }
      break;
    }

    /* if time occurred after current strip... */
    if (ctime > strip->end) {
      /* only if this is the last strip should we do anything, and only if that is being held */
      if (strip == strips->last) {
        if (strip->extendmode != NLASTRIP_EXTEND_NOTHING) {
          estrip = strip;
        }

        side = NES_TIME_AFTER;
        break;
      }

      /* otherwise, skip... as the 'before' case will catch it more elegantly! */
    }
  }

  /* check if a valid strip was found
   * - must not be muted (i.e. will have contribution
   */
  if ((estrip == NULL) || (estrip->flag & NLASTRIP_FLAG_MUTED)) {
    return NULL;
  }

  /* if ctime was not within the boundaries of the strip, clamp! */
  switch (side) {
    case NES_TIME_BEFORE: /* extend first frame only */
      ctime = estrip->start;
      break;
    case NES_TIME_AFTER: /* extend last frame only */
      ctime = estrip->end;
      break;
  }

  /* evaluate strip's evaluation controls
   * - skip if no influence (i.e. same effect as muting the strip)
   * - negative influence is not supported yet... how would that be defined?
   */
  /* TODO: this sounds a bit hacky having a few isolated F-Curves
   * stuck on some data it operates on... */
  AnimationEvalContext clamped_eval_context = BKE_animsys_eval_context_construct_at(
      anim_eval_context, ctime);
  nlastrip_evaluate_controls(estrip, &clamped_eval_context, flush_to_original);
  if (estrip->influence <= 0.0f) {
    return NULL;
  }

  /* check if strip has valid data to evaluate,
   * and/or perform any additional type-specific actions
   */
  switch (estrip->type) {
    case NLASTRIP_TYPE_CLIP:
      /* clip must have some action to evaluate */
      if (estrip->act == NULL) {
        return NULL;
      }
      break;
    case NLASTRIP_TYPE_TRANSITION:
      /* there must be strips to transition from and to (i.e. prev and next required) */
      if (ELEM(NULL, estrip->prev, estrip->next)) {
        return NULL;
      }

      /* evaluate controls for the relevant extents of the bordering strips... */
      AnimationEvalContext start_eval_context = BKE_animsys_eval_context_construct_at(
          anim_eval_context, estrip->start);
      AnimationEvalContext end_eval_context = BKE_animsys_eval_context_construct_at(
          anim_eval_context, estrip->end);
      nlastrip_evaluate_controls(estrip->prev, &start_eval_context, flush_to_original);
      nlastrip_evaluate_controls(estrip->next, &end_eval_context, flush_to_original);
      break;
  }

  /* add to list of strips we need to evaluate */
  nes = MEM_callocN(sizeof(NlaEvalStrip), "NlaEvalStrip");

  nes->strip = estrip;
  nes->strip_mode = side;
  nes->track_index = index;
  nes->strip_time = estrip->strip_time;

  if (list) {
    BLI_addtail(list, nes);
  }

  return nes;
}

static NlaEvalStrip *nlastrips_ctime_get_strip_single(
    ListBase *estrips,
    NlaStrip *single_strip,
    const AnimationEvalContext *anim_eval_context,
    const bool flush_to_original)
{
  ListBase dummy_trackslist;
  dummy_trackslist.first = dummy_trackslist.last = single_strip;

  return nlastrips_ctime_get_strip(
      estrips, &dummy_trackslist, -1, anim_eval_context, flush_to_original);
}

/* ---------------------- */

/* Initialize a valid mask, allocating memory if necessary. */
static void nlavalidmask_init(NlaValidMask *mask, int bits)
{
  if (BLI_BITMAP_SIZE(bits) > sizeof(mask->buffer)) {
    mask->ptr = BLI_BITMAP_NEW(bits, "NlaValidMask");
  }
  else {
    mask->ptr = mask->buffer;
  }
}

/* Free allocated memory for the mask. */
static void nlavalidmask_free(NlaValidMask *mask)
{
  if (mask->ptr != mask->buffer) {
    MEM_freeN(mask->ptr);
  }
}

/* ---------------------- */

/* Hashing functions for NlaEvalChannelKey. */
static uint nlaevalchan_keyhash(const void *ptr)
{
  const NlaEvalChannelKey *key = ptr;
  uint hash = BLI_ghashutil_ptrhash(key->ptr.data);
  return hash ^ BLI_ghashutil_ptrhash(key->prop);
}

static bool nlaevalchan_keycmp(const void *a, const void *b)
{
  const NlaEvalChannelKey *A = a;
  const NlaEvalChannelKey *B = b;

  return ((A->ptr.data != B->ptr.data) || (A->prop != B->prop));
}

/* ---------------------- */

/* Allocate a new blending value snapshot for the channel. */
static NlaEvalChannelSnapshot *nlaevalchan_snapshot_new(NlaEvalChannel *nec)
{
  int length = nec->base_snapshot.length;

  size_t byte_size = sizeof(NlaEvalChannelSnapshot) + sizeof(float) * length;
  NlaEvalChannelSnapshot *nec_snapshot = MEM_callocN(byte_size, "NlaEvalChannelSnapshot");

  nec_snapshot->channel = nec;
  nec_snapshot->length = length;
  nlavalidmask_init(&nec_snapshot->invertible, length);
  nlavalidmask_init(&nec_snapshot->raw_value_sampled, length);

  return nec_snapshot;
}

/* Free a channel's blending value snapshot. */
static void nlaevalchan_snapshot_free(NlaEvalChannelSnapshot *nec_snapshot)
{
  BLI_assert(!nec_snapshot->is_base);

  nlavalidmask_free(&nec_snapshot->invertible);
  nlavalidmask_free(&nec_snapshot->raw_value_sampled);
  MEM_freeN(nec_snapshot);
}

/* Copy all data in the snapshot. */
static void nlaevalchan_snapshot_copy(NlaEvalChannelSnapshot *dst,
                                      const NlaEvalChannelSnapshot *src)
{
  BLI_assert(dst->channel == src->channel);

  memcpy(dst->values, src->values, sizeof(float) * dst->length);
}

/* ---------------------- */

/* Initialize a blending state snapshot structure. */
static void nlaeval_snapshot_init(NlaEvalSnapshot *snapshot,
                                  NlaEvalData *nlaeval,
                                  NlaEvalSnapshot *base)
{
  snapshot->base = base;
  snapshot->size = MAX2(16, nlaeval->num_channels);
  snapshot->channels = MEM_callocN(sizeof(*snapshot->channels) * snapshot->size,
                                   "NlaEvalSnapshot::channels");
}

/* Retrieve the individual channel snapshot. */
static NlaEvalChannelSnapshot *nlaeval_snapshot_get(NlaEvalSnapshot *snapshot, int index)
{
  return (index < snapshot->size) ? snapshot->channels[index] : NULL;
}

/* Ensure at least this number of slots exists. */
static void nlaeval_snapshot_ensure_size(NlaEvalSnapshot *snapshot, int size)
{
  if (size > snapshot->size) {
    snapshot->size *= 2;
    CLAMP_MIN(snapshot->size, size);
    CLAMP_MIN(snapshot->size, 16);

    size_t byte_size = sizeof(*snapshot->channels) * snapshot->size;
    snapshot->channels = MEM_recallocN_id(
        snapshot->channels, byte_size, "NlaEvalSnapshot::channels");
  }
}

/* Retrieve the address of a slot in the blending state snapshot for this channel (may realloc). */
static NlaEvalChannelSnapshot **nlaeval_snapshot_ensure_slot(NlaEvalSnapshot *snapshot,
                                                             NlaEvalChannel *nec)
{
  nlaeval_snapshot_ensure_size(snapshot, nec->owner->num_channels);
  return &snapshot->channels[nec->index];
}

/* Retrieve the blending snapshot for the specified channel, with fallback to base. */
static NlaEvalChannelSnapshot *nlaeval_snapshot_find_channel(NlaEvalSnapshot *snapshot,
                                                             NlaEvalChannel *nec)
{
  while (snapshot != NULL) {
    NlaEvalChannelSnapshot *nec_snapshot = nlaeval_snapshot_get(snapshot, nec->index);
    if (nec_snapshot != NULL) {
      return nec_snapshot;
    }
    snapshot = snapshot->base;
  }

  return &nec->base_snapshot;
}

/* Retrieve or create the channel value snapshot, copying from the other snapshot
 * (or default values) */
static NlaEvalChannelSnapshot *nlaeval_snapshot_ensure_channel(NlaEvalSnapshot *snapshot,
                                                               NlaEvalChannel *nec)
{
  NlaEvalChannelSnapshot **slot = nlaeval_snapshot_ensure_slot(snapshot, nec);

  if (*slot == NULL) {
    NlaEvalChannelSnapshot *base_snapshot, *nec_snapshot;

    nec_snapshot = nlaevalchan_snapshot_new(nec);
    base_snapshot = nlaeval_snapshot_find_channel(snapshot->base, nec);

    nlaevalchan_snapshot_copy(nec_snapshot, base_snapshot);

    *slot = nec_snapshot;
  }

  return *slot;
}

/* Free all memory owned by this blending snapshot structure. */
static void nlaeval_snapshot_free_data(NlaEvalSnapshot *snapshot)
{
  if (snapshot->channels != NULL) {
    for (int i = 0; i < snapshot->size; i++) {
      NlaEvalChannelSnapshot *nec_snapshot = snapshot->channels[i];
      if (nec_snapshot != NULL) {
        nlaevalchan_snapshot_free(nec_snapshot);
      }
    }

    MEM_freeN(snapshot->channels);
  }

  snapshot->base = NULL;
  snapshot->size = 0;
  snapshot->channels = NULL;
}

/* ---------------------- */

/* Free memory owned by this evaluation channel. */
static void nlaevalchan_free_data(NlaEvalChannel *nec)
{
  nlavalidmask_free(&nec->domain);

  if (nec->blend_snapshot != NULL) {
    nlaevalchan_snapshot_free(nec->blend_snapshot);
  }
}

/* Initialize a full NLA evaluation state structure. */
static void nlaeval_init(NlaEvalData *nlaeval)
{
  memset(nlaeval, 0, sizeof(*nlaeval));

  nlaeval->path_hash = BLI_ghash_str_new("NlaEvalData::path_hash");
  nlaeval->key_hash = BLI_ghash_new(
      nlaevalchan_keyhash, nlaevalchan_keycmp, "NlaEvalData::key_hash");
}

static void nlaeval_free(NlaEvalData *nlaeval)
{
  /* Delete base snapshot - its channels are part of NlaEvalChannel and shouldn't be freed. */
  MEM_SAFE_FREE(nlaeval->base_snapshot.channels);

  /* Delete result snapshot. */
  nlaeval_snapshot_free_data(&nlaeval->eval_snapshot);

  /* Delete channels. */
  LISTBASE_FOREACH (NlaEvalChannel *, nec, &nlaeval->channels) {
    nlaevalchan_free_data(nec);
  }

  BLI_freelistN(&nlaeval->channels);
  BLI_ghash_free(nlaeval->path_hash, NULL, NULL);
  BLI_ghash_free(nlaeval->key_hash, NULL, NULL);
}

/* ---------------------- */

static int nlaevalchan_validate_index(const NlaEvalChannel *nec, const int index)
{
  if (nec->is_array) {
    if (index >= 0 && index < nec->base_snapshot.length) {
      return index;
    }

    return -1;
  }
  return 0;
}

static bool nlaevalchan_validate_index_ex(const NlaEvalChannel *nec, const int array_index)
{
  /** Although array_index comes from fcurve, that doesn't necessarily mean the property has that
   * many elements. */
  const int index = nlaevalchan_validate_index(nec, array_index);

  if (index < 0) {
    if (G.debug & G_DEBUG) {
      ID *id = nec->key.ptr.owner_id;
      CLOG_WARN(&LOG,
                "Animation: Invalid array index. ID = '%s',  '%s[%d]', array length is %d",
                id ? (id->name + 2) : "<No ID>",
                nec->rna_path,
                array_index,
                nec->base_snapshot.length);
    }

    return false;
  }
  return true;
}

/* Initialize default values for NlaEvalChannel from the property data. */
static void nlaevalchan_get_default_values(NlaEvalChannel *nec, float *r_values)
{
  PointerRNA *ptr = &nec->key.ptr;
  PropertyRNA *prop = nec->key.prop;
  int length = nec->base_snapshot.length;

  /* Use unit quaternion for quaternion properties. */
  if (nec->mix_mode == NEC_MIX_QUATERNION) {
    unit_qt(r_values);
    return;
  }
  /* Use all zero for Axis-Angle properties. */
  if (nec->mix_mode == NEC_MIX_AXIS_ANGLE) {
    zero_v4(r_values);
    return;
  }

  /* NOTE: while this doesn't work for all RNA properties as default values aren't in fact
   * set properly for most of them, at least the common ones (which also happen to get used
   * in NLA strips a lot, e.g. scale) are set correctly.
   */
  if (RNA_property_array_check(prop)) {
    BLI_assert(length == RNA_property_array_length(ptr, prop));
    bool *tmp_bool;
    int *tmp_int;

    switch (RNA_property_type(prop)) {
      case PROP_BOOLEAN:
        tmp_bool = MEM_malloc_arrayN(sizeof(*tmp_bool), length, __func__);
        RNA_property_boolean_get_default_array(ptr, prop, tmp_bool);
        for (int i = 0; i < length; i++) {
          r_values[i] = (float)tmp_bool[i];
        }
        MEM_freeN(tmp_bool);
        break;
      case PROP_INT:
        tmp_int = MEM_malloc_arrayN(sizeof(*tmp_int), length, __func__);
        RNA_property_int_get_default_array(ptr, prop, tmp_int);
        for (int i = 0; i < length; i++) {
          r_values[i] = (float)tmp_int[i];
        }
        MEM_freeN(tmp_int);
        break;
      case PROP_FLOAT:
        RNA_property_float_get_default_array(ptr, prop, r_values);
        break;
      default:
        memset(r_values, 0, sizeof(float) * length);
    }
  }
  else {
    BLI_assert(length == 1);

    switch (RNA_property_type(prop)) {
      case PROP_BOOLEAN:
        *r_values = (float)RNA_property_boolean_get_default(ptr, prop);
        break;
      case PROP_INT:
        *r_values = (float)RNA_property_int_get_default(ptr, prop);
        break;
      case PROP_FLOAT:
        *r_values = RNA_property_float_get_default(ptr, prop);
        break;
      case PROP_ENUM:
        *r_values = (float)RNA_property_enum_get_default(ptr, prop);
        break;
      default:
        *r_values = 0.0f;
    }
  }

  /* Ensure multiplicative properties aren't reset to 0. */
  if (nec->mix_mode == NEC_MIX_MULTIPLY) {
    for (int i = 0; i < length; i++) {
      if (r_values[i] == 0.0f) {
        r_values[i] = 1.0f;
      }
    }
  }
}

static char nlaevalchan_detect_mix_mode(NlaEvalChannelKey *key, int length)
{
  PropertySubType subtype = RNA_property_subtype(key->prop);

  if (subtype == PROP_QUATERNION && length == 4) {
    return NEC_MIX_QUATERNION;
  }
  if (subtype == PROP_AXISANGLE && length == 4) {
    return NEC_MIX_AXIS_ANGLE;
  }
  if (RNA_property_flag(key->prop) & PROP_PROPORTIONAL) {
    return NEC_MIX_MULTIPLY;
  }
  return NEC_MIX_ADD;
}

/* Verify that an appropriate NlaEvalChannel for this property exists. */
static NlaEvalChannel *nlaevalchan_verify_key(NlaEvalData *nlaeval,
                                              const char *path,
                                              NlaEvalChannelKey *key)
{
  /* Look it up in the key hash. */
  NlaEvalChannel **p_key_nec;
  NlaEvalChannelKey **p_key;
  bool found_key = BLI_ghash_ensure_p_ex(
      nlaeval->key_hash, key, (void ***)&p_key, (void ***)&p_key_nec);

  if (found_key) {
    return *p_key_nec;
  }

  /* Create the channel. */
  bool is_array = RNA_property_array_check(key->prop);
  int length = is_array ? RNA_property_array_length(&key->ptr, key->prop) : 1;

  NlaEvalChannel *nec = MEM_callocN(sizeof(NlaEvalChannel) + sizeof(float) * length,
                                    "NlaEvalChannel");

  /* Initialize the channel. */
  nec->rna_path = path;
  nec->key = *key;

  nec->owner = nlaeval;
  nec->index = nlaeval->num_channels++;
  nec->is_array = is_array;

  nec->mix_mode = nlaevalchan_detect_mix_mode(key, length);

  nlavalidmask_init(&nec->domain, length);

  nec->base_snapshot.channel = nec;
  nec->base_snapshot.length = length;
  nec->base_snapshot.is_base = true;

  nlaevalchan_get_default_values(nec, nec->base_snapshot.values);

  /* Store channel in data structures. */
  BLI_addtail(&nlaeval->channels, nec);

  *nlaeval_snapshot_ensure_slot(&nlaeval->base_snapshot, nec) = &nec->base_snapshot;

  *p_key_nec = nec;
  *p_key = &nec->key;

  return nec;
}

/** Unlike nlaevalchan_verify(), this will not create a channel if it does not exist. */
static bool nlaevalchan_try_get(NlaEvalData *nlaeval, const char *path, NlaEvalChannel **r_nec)
{
  if (path == NULL) {
    return false;
  }

  /* Lookup the path in the path based hash. */
  NlaEvalChannel *p_path_nec = (NlaEvalChannel *)BLI_ghash_lookup(nlaeval->path_hash,
                                                                  (void *)path);

  if (p_path_nec) {
    *r_nec = p_path_nec;
  }
  return p_path_nec != NULL;
}

/* Verify that an appropriate NlaEvalChannel for this path exists. */
static NlaEvalChannel *nlaevalchan_verify(PointerRNA *ptr, NlaEvalData *nlaeval, const char *path)
{
  if (path == NULL) {
    return NULL;
  }

  /* Lookup the path in the path based hash. */
  NlaEvalChannel **p_path_nec;
  bool found_path = BLI_ghash_ensure_p(nlaeval->path_hash, (void *)path, (void ***)&p_path_nec);

  if (found_path) {
    return *p_path_nec;
  }

  /* Cache NULL result for now. */
  *p_path_nec = NULL;

  /* Resolve the property and look it up in the key hash. */
  NlaEvalChannelKey key;

  if (!RNA_path_resolve_property(ptr, path, &key.ptr, &key.prop)) {
    /* Report failure to resolve the path. */
    if (G.debug & G_DEBUG) {
      CLOG_WARN(&LOG,
                "Animato: Invalid path. ID = '%s',  '%s'",
                (ptr->owner_id) ? (ptr->owner_id->name + 2) : "<No ID>",
                path);
    }

    return NULL;
  }

  /* Check that the property can be animated. */
  if (ptr->owner_id != NULL && !RNA_property_animateable(&key.ptr, key.prop)) {
    return NULL;
  }

  NlaEvalChannel *nec = nlaevalchan_verify_key(nlaeval, path, &key);

  if (nec->rna_path == NULL) {
    nec->rna_path = path;
  }

  return *p_path_nec = nec;
}

/* ---------------------- */

/* Accumulate the lower strip and fcurve values of a channel according to mode and influence. */
static float nla_blend_value(const int blendmode,
                             const float lower_value,
                             const float fcurve_value,
                             const float inf)
{
  /* Optimization: no need to try applying if there is no influence. */
  if (IS_EQF(inf, 0.0f)) {
    return lower_value;
  }

  /* Perform blending. */
  switch (blendmode) {
    case NLASTRIP_MODE_ADD:
      /* Simply add the scaled value on to the stack. */
      return lower_value + (fcurve_value * inf);

    case NLASTRIP_MODE_SUBTRACT:
      /* Simply subtract the scaled value from the stack. */
      return lower_value - (fcurve_value * inf);

    case NLASTRIP_MODE_MULTIPLY:
      /* Multiply the scaled value with the stack. */
      return inf * (lower_value * fcurve_value) + (1 - inf) * lower_value;

    case NLASTRIP_MODE_COMBINE:
      BLI_assert(!"combine mode");
      ATTR_FALLTHROUGH;

    default:
      /* TODO: Do we really want to blend by default? it seems more uses might prefer add... */
      /* Do linear interpolation. The influence of the accumulated data (elsewhere, that is called
       * dstweight) is 1 - influence, since the strip's influence is srcweight.
       */
      return lower_value * (1.0f - inf) + (fcurve_value * inf);
  }
}

/* Accumulate the lower and fcurve values of a channel according to mode and influence. */
static float nla_combine_value(const int mix_mode,
                               float base_value,
                               const float lower_value,
                               const float fcurve_value,
                               const float inf)
{
  /* Optimization: No need to try applying if there is no influence. */
  if (IS_EQF(inf, 0.0f)) {
    return lower_value;
  }

  /* Perform blending */
  switch (mix_mode) {
    case NEC_MIX_ADD:
    case NEC_MIX_AXIS_ANGLE:
      return lower_value + (fcurve_value - base_value) * inf;

    case NEC_MIX_MULTIPLY:
      if (IS_EQF(base_value, 0.0f)) {
        base_value = 1.0f;
      }
      return lower_value * powf(fcurve_value / base_value, inf);

    default:
      BLI_assert(!"invalid mix mode");
      return lower_value;
  }
}

/** \returns true if solution exists and output is written to. */
static bool nla_blend_value_invert_get_fcurve_value(const int blend_mode,
                                                    const float lower_value,
                                                    const float blended_value,
                                                    const float influence,
                                                    float *r_fcurve_value)
{

  /** No solution if fcurve value had 0 influence. */
  if (IS_EQF(0, influence)) {
    return false;
  }

  switch (blend_mode) {
    case NLASTRIP_MODE_ADD:
      *r_fcurve_value = (blended_value - lower_value) / influence;
      return true;

    case NLASTRIP_MODE_SUBTRACT:
      *r_fcurve_value = (lower_value - blended_value) / influence;
      return true;

    case NLASTRIP_MODE_MULTIPLY:

      /** Division by zero. */
      if (IS_EQF(0.0f, lower_value)) {
        /* Resolve 0/0 to 1. */
        if (IS_EQF(0.0f, blended_value)) {
          *r_fcurve_value = 1.0f;
          return true;
        }
        /* Division by zero. */
        return false;
      }

      /** Math:
       *
       *  blended_value = inf * (lower_value * fcurve_value) + (1 - inf) * lower_value
       *  blended_value - (1 - inf) * lower_value = inf * (lower_value * fcurve_value)
       *  (blended_value - (1 - inf) * lower_value) / (inf * lower_value) =  fcurve_value
       *  (blended_value - lower_value + inf * lower_value) / (inf * lower_value) =  fcurve_value
       *  ((blended_value - lower_value) / (inf * lower_value)) + 1 =  fcurve_value
       *
       *  fcurve_value = ((blended_value - lower_value) / (inf * lower_value)) + 1
       */
      *r_fcurve_value = ((blended_value - lower_value) / (influence * lower_value)) + 1.0f;
      return true;

    case NLASTRIP_MODE_COMBINE:
      BLI_assert(!"combine mode");
      ATTR_FALLTHROUGH;

    default:

      /** Math:
       *
       *  blended_value = lower_value * (1.0f - inf) + (fcurve_value * inf)
       *  blended_value - lower_value * (1.0f - inf) = (fcurve_value * inf)
       *  (blended_value - lower_value * (1.0f - inf)) / inf = fcurve_value
       *
       *  fcurve_value = (blended_value - lower_value * (1.0f - inf)) / inf
       */
      *r_fcurve_value = (blended_value - lower_value * (1.0f - influence)) / influence;
      return true;
  }
}

/** Compute the value that would blend to the desired target value using nla_combine_value.
 * \returns true if solution exists and output is written to.  */
static bool nla_combine_value_invert_get_fcurve_value(const int mix_mode,
                                                      float base_value,
                                                      const float lower_value,
                                                      const float blended_value,
                                                      const float influence,
                                                      float *r_fcurve_value)
{
  /* No solution if fcurve had no influence. */
  if (IS_EQF(0, influence)) {
    return false;
  }

  switch (mix_mode) {
    case NEC_MIX_ADD:
    case NEC_MIX_AXIS_ANGLE:
      *r_fcurve_value = base_value + (blended_value - lower_value) / influence;
      return true;

    case NEC_MIX_MULTIPLY:
      if (IS_EQF(base_value, 0.0f)) {
        base_value = 1.0f;
      }
      /* Divison by zero. */
      if (IS_EQF(lower_value, 0.0f)) {
        /* Resolve 0/0 to 1. */
        if (IS_EQF(blended_value, 0.0f)) {
          *r_fcurve_value = base_value;
          return true;
        }
        /* Division by zero. */
        return false;
      }

      *r_fcurve_value = base_value * powf(blended_value / lower_value, 1.0f / influence);
      return true;

    default:
      BLI_assert(!"invalid mix mode");
      return false;
  }
}

/** Accumulate quaternion channels for Combine mode according to influence.
 * \returns blended_value = lower_values @ fcurve_values^infl
 */
static void nla_combine_quaternion(const float lower_values[4],
                                   const float fcurve_values[4],
                                   const float influence,
                                   float r_blended_value[4])
{
  float tmp_lower[4], tmp_fcurve_value[4];

  normalize_qt_qt(tmp_lower, lower_values);
  normalize_qt_qt(tmp_fcurve_value, fcurve_values);

  pow_qt_fl_normalized(tmp_fcurve_value, influence);
  mul_qt_qtqt(r_blended_value, tmp_lower, tmp_fcurve_value);
}

/** \returns true if solution exists and output written to. */
static bool nla_combine_quaternion_invert_get_fcurve_values(const float lower_values[4],
                                                            const float blended_values[4],
                                                            const float influence,
                                                            float r_fcurve_value[4])
{
  /* blended_value = lower_values @ fcurve_values^infl
   * inv(lower_values) @ blended_value = fcurve_value^infl
   * (inv(lower_values) @ blended_value) ^ (1/inf) = fcurve_value
   *
   * Returns: fcurve_value = (inv(lower_values) @ blended_value) ^ (1/inf) */
  if (IS_EQF(0, influence)) {
    return false;
  }
  float tmp_lower[4], tmp_blended[4];

  normalize_qt_qt(tmp_lower, lower_values);
  normalize_qt_qt(tmp_blended, blended_values);
  invert_qt_normalized(tmp_lower);

  mul_qt_qtqt(r_fcurve_value, tmp_lower, tmp_blended);
  pow_qt_fl_normalized(r_fcurve_value, 1.0f / influence);

  return true;
}

/** \returns true if solution exists and output written to. */
static bool nla_blend_value_invert_get_lower_value(const int blendmode,
                                                   const float fcurve_value,
                                                   const float blended_value,
                                                   const float influence,
                                                   float *r_lower_value)
{
  switch (blendmode) {
    case NLASTRIP_MODE_ADD:
      /* Simply subtract the scaled value on to the stack. */
      *r_lower_value = blended_value - (fcurve_value * influence);
      return true;

    case NLASTRIP_MODE_SUBTRACT:
      /* Simply add the scaled value from the stack. */
      *r_lower_value = blended_value + (fcurve_value * influence);
      return true;

    case NLASTRIP_MODE_MULTIPLY:

      /** Division by zero. */
      if (IS_EQF(-fcurve_value * influence, 1.0f - influence)) {
        /** Resolve 0/0 to 1. */
        if (IS_EQF(0.0f, blended_value)) {
          *r_lower_value = 1;
          return true;
        }
        /** Division by zero. */
        return false;
      }
      /* Math:
       *     blended_value = inf * (lower_value * fcurve_value) + (1 - inf) * lower_value
       *                   = lower_value * (inf * fcurve_value + (1-inf))
       *         lower_value = blended_value / (inf * fcurve_value + (1-inf))
       */
      *r_lower_value = blended_value / (influence * fcurve_value + (1.0f - influence));
      return true;

    case NLASTRIP_MODE_COMBINE:
      BLI_assert(!"combine mode");
      return false;

    default:

      /** No solution if lower strip has 0 influence. */
      if (IS_EQF(1.0f, influence)) {
        return false;
      }

      /** Math:
       *
       *  blended_value = lower_value * (1.0f - inf) + (fcurve_value * inf)
       *  blended_value - (fcurve_value * inf) = lower_value * (1.0f - inf)
       *  blended_value - (fcurve_value * inf) / (1.0f - inf) = lower_value
       *
       *  lower_value = blended_value - (fcurve_value * inf) / (1.0f - inf)
       */
      *r_lower_value = (blended_value - (fcurve_value * influence)) / (1.0f - influence);
      return true;
  }
}

/** \returns true if solution exists and output written to. */
static bool nla_combine_value_invert_get_lower_value(const int mix_mode,
                                                     float base_value,
                                                     const float fcurve_value,
                                                     const float blended_value,
                                                     const float inf,
                                                     float *r_lower_value)
{
  /* Perform blending. */
  switch (mix_mode) {
    case NEC_MIX_ADD:
    case NEC_MIX_AXIS_ANGLE:
      *r_lower_value = blended_value - (fcurve_value - base_value) * inf;
      return true;
    case NEC_MIX_MULTIPLY:
      /** Division by zero. */
      if (IS_EQF(0.0f, fcurve_value)) {
        /** Resolve 0/0 to 1. */
        if (IS_EQF(0.0f, blended_value)) {
          *r_lower_value = 1.0f;
          return true;
        }
        return false;
      }

      if (IS_EQF(0.0f, base_value)) {
        base_value = 1.0f;
      }

      *r_lower_value = blended_value / powf(fcurve_value / base_value, inf);
      return true;

    default:
      BLI_assert(!"invalid mix mode");
      return false;
  }
}

static void nla_combine_quaternion_invert_get_lower_values(const float fcurve_values[4],
                                                           const float blended_values[4],
                                                           const float influence,
                                                           float r_lower_value[4])
{
  /* blended_value = lower_values @ fcurve_values^infl
   * blended_value @ inv(fcurve_values^inf) = lower_values
   *
   * Returns: lower_values = blended_value @ inv(fcurve_values^inf) */

  float tmp_fcurve[4], tmp_blended[4];

  normalize_qt_qt(tmp_fcurve, fcurve_values);
  normalize_qt_qt(tmp_blended, blended_values);

  pow_qt_fl_normalized(tmp_fcurve, influence);
  invert_qt_normalized(tmp_fcurve);

  mul_qt_qtqt(r_lower_value, tmp_blended, tmp_fcurve);
}

/* Data about the current blend mode. */
typedef struct NlaBlendData {
  NlaEvalSnapshot *snapshot;
  int mode;
  float influence;

  NlaEvalChannel *blend_queue;
} NlaBlendData;

/* Queue the channel for deferred blending. */
static NlaEvalChannelSnapshot *nlaevalchan_queue_blend(NlaBlendData *blend, NlaEvalChannel *nec)
{
  if (!nec->in_blend) {
    if (nec->blend_snapshot == NULL) {
      nec->blend_snapshot = nlaevalchan_snapshot_new(nec);
    }

    nec->in_blend = true;
    nlaevalchan_snapshot_copy(nec->blend_snapshot, &nec->base_snapshot);

    nec->next_blend = blend->blend_queue;
    blend->blend_queue = nec;
  }

  return nec->blend_snapshot;
}

/* Accumulate (i.e. blend) the given value on to the channel it affects. */
static bool nlaeval_blend_value(NlaBlendData *blend,
                                NlaEvalChannel *nec,
                                int array_index,
                                float value)
{
  if (nec == NULL) {
    return false;
  }

  if (!nlaevalchan_validate_index_ex(nec, array_index)) {
    return false;
  }

  NlaEvalChannelSnapshot *nec_snapshot = nlaeval_snapshot_ensure_channel(blend->snapshot, nec);
  float *p_value = &nec_snapshot->values[array_index];

  if (blend->mode == NLASTRIP_MODE_COMBINE) {
    /* Quaternion blending is deferred until all sub-channel values are known. */
    if (nec->mix_mode == NEC_MIX_QUATERNION) {
      NlaEvalChannelSnapshot *blend_snapshot = nlaevalchan_queue_blend(blend, nec);

      blend_snapshot->values[array_index] = value;
    }
    else {
      float base_value = nec->base_snapshot.values[array_index];

      *p_value = nla_combine_value(nec->mix_mode, base_value, *p_value, value, blend->influence);
    }
  }
  else {
    *p_value = nla_blend_value(blend->mode, *p_value, value, blend->influence);
  }

  return true;
}

/* Storing lower values within snapshot's necs->values if invertible. Marks non-invertible channels
 * and defers quaternion combine inversion. */
static void nlaeval_blend_value_invert_get_lower_value(NlaBlendData *blend,
                                                       NlaEvalChannelSnapshot *necs,
                                                       const int array_index,
                                                       const float fcurve_value)
{
  NlaEvalChannel *nec = necs->channel;
  if (!nlaevalchan_validate_index_ex(nec, array_index)) {
    /** Note: no need to disable bits. If index invalid, then the fcurve wouldn't contribute
     * anyways. */
    return;
  }

  float *const p_value = &necs->values[array_index];

  if (blend->mode == NLASTRIP_MODE_COMBINE) {
    /* Quaternion blending is deferred until all sub-channel values are known. */
    if (nec->mix_mode == NEC_MIX_QUATERNION) {
      NlaEvalChannelSnapshot *blend_snapshot = nlaevalchan_queue_blend(blend, nec);

      blend_snapshot->values[array_index] = fcurve_value;
    }
    else {
      float base_value = nec->base_snapshot.values[array_index];

      if (!nla_combine_value_invert_get_lower_value(
              nec->mix_mode, base_value, fcurve_value, *p_value, blend->influence, p_value)) {
        BLI_BITMAP_DISABLE(necs->invertible.ptr, array_index);
      }
    }
  }
  else {
    if (!nla_blend_value_invert_get_lower_value(
            blend->mode, fcurve_value, *p_value, blend->influence, p_value)) {
      BLI_BITMAP_DISABLE(necs->invertible.ptr, array_index);
    }
  }
}

/* Finish deferred quaternion blending. */
static void nlaeval_blend_flush(NlaBlendData *blend)
{
  NlaEvalChannel *nec;

  while ((nec = blend->blend_queue)) {
    blend->blend_queue = nec->next_blend;
    nec->in_blend = false;

    NlaEvalChannelSnapshot *nec_snapshot = nlaeval_snapshot_ensure_channel(blend->snapshot, nec);
    NlaEvalChannelSnapshot *blend_snapshot = nec->blend_snapshot;

    BLI_assert(nec->mix_mode == NEC_MIX_QUATERNION);
    nla_combine_quaternion(
        nec_snapshot->values, blend_snapshot->values, blend->influence, nec_snapshot->values);
  }
}

/* Finish deferred quaternion combine inversion. */
static void nlaeval_blend_flush_invert_get_lower_value(NlaBlendData *blend)
{
  NlaEvalChannel *nec;
  while ((nec = blend->blend_queue)) {
    blend->blend_queue = nec->next_blend;
    nec->in_blend = false;

    NlaEvalChannelSnapshot *nec_snapshot = nlaeval_snapshot_ensure_channel(blend->snapshot, nec);
    NlaEvalChannelSnapshot *blend_snapshot = nec->blend_snapshot;

    BLI_assert(nec->mix_mode == NEC_MIX_QUATERNION);
    nla_combine_quaternion_invert_get_lower_values(
        blend_snapshot->values, nec_snapshot->values, blend->influence, nec_snapshot->values);
  }
}

/* Blend the specified snapshots into the target, and free the input snapshots. */
static void nlaeval_snapshot_mix_and_free(NlaEvalData *nlaeval,
                                          NlaEvalSnapshot *out,
                                          NlaEvalSnapshot *in1,
                                          NlaEvalSnapshot *in2,
                                          float alpha)
{
  BLI_assert(in1->base == out && in2->base == out);

  nlaeval_snapshot_ensure_size(out, nlaeval->num_channels);

  for (int i = 0; i < nlaeval->num_channels; i++) {
    NlaEvalChannelSnapshot *c_in1 = nlaeval_snapshot_get(in1, i);
    NlaEvalChannelSnapshot *c_in2 = nlaeval_snapshot_get(in2, i);

    if (c_in1 || c_in2) {
      NlaEvalChannelSnapshot *c_out = out->channels[i];

      /* Steal the entry from one of the input snapshots. */
      if (c_out == NULL) {
        if (c_in1 != NULL) {
          c_out = c_in1;
          in1->channels[i] = NULL;
        }
        else {
          c_out = c_in2;
          in2->channels[i] = NULL;
        }
      }

      if (c_in1 == NULL) {
        c_in1 = nlaeval_snapshot_find_channel(in1->base, c_out->channel);
      }
      if (c_in2 == NULL) {
        c_in2 = nlaeval_snapshot_find_channel(in2->base, c_out->channel);
      }

      out->channels[i] = c_out;

      for (int j = 0; j < c_out->length; j++) {
        c_out->values[j] = c_in1->values[j] * (1.0f - alpha) + c_in2->values[j] * alpha;
      }
    }
  }

  nlaeval_snapshot_free_data(in1);
  nlaeval_snapshot_free_data(in2);
}

/* ---------------------- */
/* F-Modifier stack joining/separation utilities -
 * should we generalize these for BLI_listbase.h interface? */

/* Temporarily join two lists of modifiers together, storing the result in a third list */
static void nlaeval_fmodifiers_join_stacks(ListBase *result, ListBase *list1, ListBase *list2)
{
  FModifier *fcm1, *fcm2;

  /* if list1 is invalid...  */
  if (ELEM(NULL, list1, list1->first)) {
    if (list2 && list2->first) {
      result->first = list2->first;
      result->last = list2->last;
    }
  }
  /* if list 2 is invalid... */
  else if (ELEM(NULL, list2, list2->first)) {
    result->first = list1->first;
    result->last = list1->last;
  }
  else {
    /* list1 should be added first, and list2 second,
     * with the endpoints of these being the endpoints for result
     * - the original lists must be left unchanged though, as we need that fact for restoring.
     */
    result->first = list1->first;
    result->last = list2->last;

    fcm1 = list1->last;
    fcm2 = list2->first;

    fcm1->next = fcm2;
    fcm2->prev = fcm1;
  }
}

/* Split two temporary lists of modifiers */
static void nlaeval_fmodifiers_split_stacks(ListBase *list1, ListBase *list2)
{
  FModifier *fcm1, *fcm2;

  /* if list1/2 is invalid... just skip */
  if (ELEM(NULL, list1, list2)) {
    return;
  }
  if (ELEM(NULL, list1->first, list2->first)) {
    return;
  }

  /* get endpoints */
  fcm1 = list1->last;
  fcm2 = list2->first;

  /* clear their links */
  fcm1->next = NULL;
  fcm2->prev = NULL;
}

/* ---------------------- */

static bool is_fcurve_evaluatable(FCurve *fcu)
{
  if (fcu->flag & (FCURVE_MUTED | FCURVE_DISABLED)) {
    return false;
  }
  if ((fcu->grp) && (fcu->grp->flag & AGRP_MUTED)) {
    return false;
  }
  if (BKE_fcurve_is_empty(fcu)) {
    return false;
  }
  return true;
}

/** Evaluate action-clip strip and accumulate within snapshot.
 * \param allow_alloc_channels: If true, new NlaEvalChannels allocated and evaluated if needed.
 * Otherwise only channels existing within the NlaEvalData are evaluated.
 */
static void nlastrip_evaluate_actionclip(PointerRNA *ptr,
                                         NlaEvalData *channels,
                                         ListBase *modifiers,
                                         NlaEvalStrip *nes,
                                         NlaEvalSnapshot *snapshot,
                                         bool allow_alloc_channels)
{
  ListBase tmp_modifiers = {NULL, NULL};
  NlaStrip *strip = nes->strip;
  FCurve *fcu;
  float evaltime;

  if (strip == NULL) {
    return;
  }

  if (strip->act == NULL) {
    CLOG_ERROR(&LOG, "NLA-Strip Eval Error: Strip '%s' has no Action", strip->name);
    return;
  }

  action_idcode_patch_check(ptr->owner_id, strip->act);

  /* Join this strip's modifiers to the parent's modifiers (own modifiers first). */
  nlaeval_fmodifiers_join_stacks(&tmp_modifiers, &strip->modifiers, modifiers);

  /* Evaluate strip's modifiers which modify time to evaluate the base curves at. */
  FModifiersStackStorage storage;
  storage.modifier_count = BLI_listbase_count(&tmp_modifiers);
  storage.size_per_modifier = evaluate_fmodifiers_storage_size_per_modifier(&tmp_modifiers);
  storage.buffer = alloca(storage.modifier_count * storage.size_per_modifier);

  evaltime = evaluate_time_fmodifiers(&storage, &tmp_modifiers, NULL, 0.0f, strip->strip_time);

  NlaBlendData blend = {
      .snapshot = snapshot,
      .mode = strip->blendmode,
      .influence = strip->influence,
  };

  /* Evaluate all the F-Curves in the action,
   * saving the relevant pointers to data that will need to be used. */
  for (fcu = strip->act->curves.first; fcu; fcu = fcu->next) {

    if (!is_fcurve_evaluatable(fcu)) {
      continue;
    }

    /* Get an NLA evaluation channel to work with, and accumulate the evaluated value with the
     * value(s) stored in this channel if it has been used already. */
    NlaEvalChannel *nec = NULL;
    if (allow_alloc_channels) {
      /** Guarantees NlaEvalChannel. */
      nec = nlaevalchan_verify(ptr, channels, fcu->rna_path);
    }
    else {
      /** Only get NlaEvalChannel if it exists. */
      nlaevalchan_try_get(channels, fcu->rna_path, &nec);
    }

    if (!nec) {
      /** Skip since caller only wants to fill values for existing channels in snapshot. */
      continue;
    }

    /* Evaluate the F-Curve's value for the time given in the strip
     * NOTE: we use the modified time here, since strip's F-Curve Modifiers
     * are applied on top of this.
     */
    float value = evaluate_fcurve(fcu, evaltime);

    /* Apply strip's F-Curve Modifiers on this value
     * NOTE: we apply the strip's original evaluation time not the modified one
     * (as per standard F-Curve eval).
     */
    evaluate_value_fmodifiers(&storage, &tmp_modifiers, fcu, &value, strip->strip_time);

    nlaeval_blend_value(&blend, nec, fcu->array_index, value);
  }

  nlaeval_blend_flush(&blend);

  /* Unlink this strip's modifiers from the parent's modifiers again. */
  nlaeval_fmodifiers_split_stacks(&strip->modifiers, modifiers);
}

/** Remove the strip's effect on the snapshot. Results overwrite snapshot values and marks
 * non-invertible channels. If not invertible, then we cannot obtain the lower value needed for
 * keyframe remapping. */
static void nlastrip_evaluate_actionclip_invert_get_lower_values(PointerRNA *ptr,
                                                                 NlaEvalData *channels,
                                                                 ListBase *modifiers,
                                                                 NlaEvalStrip *nes,
                                                                 NlaEvalSnapshot *snapshot)
{
  ListBase tmp_modifiers = {NULL, NULL};
  NlaStrip *strip = nes->strip;
  FCurve *fcu;
  float evaltime;

  /* Sanity checks for action. */
  if (strip == NULL) {
    return;
  }

  if (strip->act == NULL) {
    CLOG_ERROR(&LOG, "NLA-Strip Eval Error: Strip '%s' has no Action", strip->name);
    return;
  }

  action_idcode_patch_check(ptr->owner_id, strip->act);

  /* Join this strip's modifiers to the parent's modifiers (own modifiers first). */
  nlaeval_fmodifiers_join_stacks(&tmp_modifiers, &strip->modifiers, modifiers);

  /* Evaluate strip's modifiers which modify time to evaluate the base curves at. */
  FModifiersStackStorage storage;
  storage.modifier_count = BLI_listbase_count(&tmp_modifiers);
  storage.size_per_modifier = evaluate_fmodifiers_storage_size_per_modifier(&tmp_modifiers);
  storage.buffer = alloca(storage.modifier_count * storage.size_per_modifier);

  evaltime = evaluate_time_fmodifiers(&storage, &tmp_modifiers, NULL, 0.0f, strip->strip_time);

  NlaBlendData blend = {
      .snapshot = snapshot,
      .mode = strip->blendmode,
      .influence = strip->influence,
  };

  NlaEvalChannelSnapshot *necs;
  float value = 0.0f;
  NlaEvalChannel *nec;

  /* Go through each snapshot value and remove strip's effect from it. */
  for (fcu = strip->act->curves.first; fcu; fcu = fcu->next) {

    if (!is_fcurve_evaluatable(fcu)) {
      continue;
    }

    /** Only fill values for existing channels in snapshot, those that caller wants inverted. */
    if (nlaevalchan_try_get(channels, fcu->rna_path, &nec)) {
      necs = nlaeval_snapshot_ensure_channel(snapshot, nec);

      /** If an upper strip failed to invert this value, then we can't do anything with it. */
      if (!BLI_BITMAP_TEST_BOOL(necs->invertible.ptr, fcu->array_index)) {
        continue;
      }

      /* Evaluate the F-Curve's value for the time given in the strip
       * NOTE: we use the modified time here, since strip's F-Curve Modifiers
       * are applied on top of this. */
      value = evaluate_fcurve(fcu, evaltime);

      /* Apply strip's F-Curve Modifiers on this value
       * NOTE: we apply the strip's original evaluation time not the modified one
       * (as per standard F-Curve eval). */
      evaluate_value_fmodifiers(&storage, &tmp_modifiers, fcu, &value, strip->strip_time);
      nlaeval_blend_value_invert_get_lower_value(&blend, necs, fcu->array_index, value);
    }
  }
  nlaeval_blend_flush_invert_get_lower_value(&blend);

  /* Unlink this strip's modifiers from the parent's modifiers again. */
  nlaeval_fmodifiers_split_stacks(&strip->modifiers, modifiers);
}

/** Fills snapshot with the action clip's evaluated fcurve values with modifiers applied. No
 * Blending. */
static void nlastrip_evaluate_actionclip_raw_value(PointerRNA *ptr,
                                                   NlaEvalData *upper_eval_data,
                                                   ListBase *modifiers,
                                                   NlaEvalStrip *nes,
                                                   NlaEvalSnapshot *snapshot,
                                                   short *r_blendmode,
                                                   float *r_influence)
{
  ListBase tmp_modifiers = {NULL, NULL};
  NlaStrip *strip = nes->strip;
  FCurve *fcu;
  float evaltime;

  /* Sanity checks for action. */
  if (strip == NULL) {
    return;
  }

  if (strip->act == NULL) {
    CLOG_ERROR(&LOG, "NLA-Strip Eval Error: Strip '%s' has no Action", strip->name);
    return;
  }

  *r_blendmode = strip->blendmode;
  *r_influence = strip->influence;

  action_idcode_patch_check(ptr->owner_id, strip->act);

  /* Join this strip's modifiers to the parent's modifiers (own modifiers first). */
  nlaeval_fmodifiers_join_stacks(&tmp_modifiers, &strip->modifiers, modifiers);

  /* Evaluate strip's modifiers which modify time to evaluate the base curves at. */
  FModifiersStackStorage storage;
  storage.modifier_count = BLI_listbase_count(&tmp_modifiers);
  storage.size_per_modifier = evaluate_fmodifiers_storage_size_per_modifier(&tmp_modifiers);
  storage.buffer = alloca(storage.modifier_count * storage.size_per_modifier);

  evaltime = evaluate_time_fmodifiers(&storage, &tmp_modifiers, NULL, 0.0f, strip->strip_time);

  NlaEvalChannelSnapshot *necs;
  /* Go through each snapshot value and remove strip's effect from it. */
  float value = 0.0f;
  NlaEvalChannel *nec = NULL;

  bool allow_alloc_channels = true;
  for (fcu = strip->act->curves.first; fcu; fcu = fcu->next) {

    if (!is_fcurve_evaluatable(fcu))
      continue;

    /** Only fill values for existing channels in snapshot, those that caller wants inverted. */
    // if (nlaevalchan_try_get(upper_eval_data, fcu->rna_path, &nec)) {
    if (true) {

      /* Get an NLA evaluation channel to work with, and accumulate the evaluated value with the
       * value(s) stored in this channel if it has been used already. */
      NlaEvalChannel *nec = NULL;
      if (allow_alloc_channels) {
        /** Guarantees NlaEvalChannel. */
        nec = nlaevalchan_verify(ptr, upper_eval_data, fcu->rna_path);
      }
      else {
        /** Only get NlaEvalChannel if it exists. */
        nlaevalchan_try_get(upper_eval_data, fcu->rna_path, &nec);
      }

      if (!nec) {
        /** Skip since caller only wants to fill values for existing channels in snapshot. */
        continue;
      }

      necs = nlaeval_snapshot_ensure_channel(snapshot, nec);
      if (!nlaevalchan_validate_index_ex(nec, fcu->array_index))
        continue;

      /* Evaluate the F-Curve's value for the time given in the strip
       * NOTE: we use the modified time here, since strip's F-Curve Modifiers
       * are applied on top of this.
       */
      value = evaluate_fcurve(fcu, evaltime);

      /* Apply strip's F-Curve Modifiers on this value
       * NOTE: we apply the strip's original evaluation time not the modified one
       * (as per standard F-Curve eval).
       */
      evaluate_value_fmodifiers(&storage, &tmp_modifiers, fcu, &value, strip->strip_time);

      necs->values[fcu->array_index] = value;
      /** If a channel isn't sampled, then its value defaults to the lower snapshot value. Note,
       * the snapshot sent to this function is a temporary one, not the original one sent to
       * nlastrip_evaluate_transition_invert_get_lower_values(). */
      if (necs->channel->mix_mode == NEC_MIX_QUATERNION) {
        /* For quaternion properties, always output all sub-channels. */
        BLI_bitmap_set_all(necs->raw_value_sampled.ptr, true, 4);
      }
      else {
        BLI_BITMAP_ENABLE(necs->raw_value_sampled.ptr, fcu->array_index);
      }
    }
  }

  /* Unlink this strip's modifiers from the parent's modifiers again. */
  nlaeval_fmodifiers_split_stacks(&strip->modifiers, modifiers);
}

/* evaluate transition strip */
static void nlastrip_evaluate_transition(PointerRNA *ptr,
                                         NlaEvalData *channels,
                                         ListBase *modifiers,
                                         NlaEvalStrip *nes,
                                         NlaEvalSnapshot *snapshot,
                                         const AnimationEvalContext *anim_eval_context,
                                         const bool flush_to_original,
                                         bool allow_alloc_channels)
{
  ListBase tmp_modifiers = {NULL, NULL};
  NlaEvalSnapshot snapshot1, snapshot2;
  NlaEvalStrip tmp_nes;
  NlaStrip *s1, *s2;

  /* join this strip's modifiers to the parent's modifiers (own modifiers first) */
  nlaeval_fmodifiers_join_stacks(&tmp_modifiers, &nes->strip->modifiers, modifiers);

  /* get the two strips to operate on
   * - we use the endpoints of the strips directly flanking our strip
   *   using these as the endpoints of the transition (destination and source)
   * - these should have already been determined to be valid...
   * - if this strip is being played in reverse, we need to swap these endpoints
   *   otherwise they will be interpolated wrong
   */
  if (nes->strip->flag & NLASTRIP_FLAG_REVERSE) {
    s1 = nes->strip->next;
    s2 = nes->strip->prev;
  }
  else {
    s1 = nes->strip->prev;
    s2 = nes->strip->next;
  }

  /* prepare template for 'evaluation strip'
   * - based on the transition strip's evaluation strip data
   * - strip_mode is NES_TIME_TRANSITION_* based on which endpoint
   * - strip_time is the 'normalized' (i.e. in-strip) time for evaluation,
   *   which doubles up as an additional weighting factor for the strip influences
   *   which allows us to appear to be 'interpolating' between the two extremes
   */
  tmp_nes = *nes;

  /* evaluate these strips into a temp-buffer (tmp_channels) */
  /* FIXME: modifier evaluation here needs some work... */
  /* first strip */
  tmp_nes.strip_mode = NES_TIME_TRANSITION_START;
  tmp_nes.strip = s1;
  tmp_nes.strip_time = s1->strip_time;
  nlaeval_snapshot_init(&snapshot1, channels, snapshot);
  nlastrip_evaluate(ptr,
                    channels,
                    &tmp_modifiers,
                    &tmp_nes,
                    &snapshot1,
                    anim_eval_context,
                    flush_to_original,
                    allow_alloc_channels);

  /* second strip */
  tmp_nes.strip_mode = NES_TIME_TRANSITION_END;
  tmp_nes.strip = s2;
  tmp_nes.strip_time = s2->strip_time;
  nlaeval_snapshot_init(&snapshot2, channels, snapshot);
  nlastrip_evaluate(ptr,
                    channels,
                    &tmp_modifiers,
                    &tmp_nes,
                    &snapshot2,
                    anim_eval_context,
                    flush_to_original,
                    allow_alloc_channels);

  /* accumulate temp-buffer and full-buffer, using the 'real' strip */
  nlaeval_snapshot_mix_and_free(channels, snapshot, &snapshot1, &snapshot2, nes->strip_time);

  /* unlink this strip's modifiers from the parent's modifiers again */
  nlaeval_fmodifiers_split_stacks(&nes->strip->modifiers, modifiers);
}

/** Stores lower values within necs->values if invertible. */
static void transition_qt_combine_to_combine_get_lower_values(NlaEvalChannelSnapshot *necs,
                                                              const NlaEvalChannelSnapshot *necs1,
                                                              const NlaEvalChannelSnapshot *necs2,
                                                              const float transition_influence,
                                                              const float inf1,
                                                              const float inf2)
{
  /** Invert for:
   *  blended_result = lerp( combine(lower,left, inf1), combine(lower, right, inf2), tinf)
   */

  /** left_quat = (1 - tinf) * necs1^[inf1]  */
  float left_quat[4];
  copy_v4_v4(left_quat, necs1->values);
  normalize_qt(left_quat);
  pow_qt_fl_normalized(left_quat, inf1);
  mul_v4_fl(left_quat, (1.0 - transition_influence));

  /** right_quat = tinf * necs2^[inf2]  */
  float right_quat[4];
  copy_v4_v4(right_quat, necs2->values);
  normalize_qt(right_quat);
  pow_qt_fl_normalized(right_quat, inf2);
  mul_v4_fl(right_quat, transition_influence);

  /** blended_quat = necs  */
  float blended_quat[4];
  copy_v4_v4(blended_quat, necs->values);
  normalize_qt(blended_quat);

  /** result_quat =  blended_quat @ invert(left_quat + right_quat)  */
  float result_quat[4];
  add_v4_v4v4(result_quat, left_quat, right_quat);
  normalize_qt(result_quat);
  invert_qt_normalized(result_quat);
  mul_qt_qtqt(result_quat, blended_quat, result_quat);

  copy_qt_qt(necs->values, result_quat);
}

/** Stores lower values within necs->values if invertible. */
static void transition_qt_combine_to_empty_get_lower_values(NlaEvalChannelSnapshot *necs,
                                                            const NlaEvalChannelSnapshot *necs1,
                                                            const float transition_influence,
                                                            const float inf1)
{
  /** Invert for:
   *  blended_result = lerp( combine( lower, left, inf1),  lower, tinf)
   */

  /** blended_quat = (1/tinf) * necs */
  float blended_quat[4];
  copy_v4_v4(blended_quat, necs->values);
  normalize_qt(blended_quat);
  mul_v4_fl(blended_quat, (1.0 / transition_influence));

  /** left_quat = (1/tinf) * (1 - tinf) * necs1^[inf1]  */
  float left_quat[4];
  copy_v4_v4(left_quat, necs1->values);
  normalize_qt(left_quat);
  pow_qt_fl_normalized(left_quat, inf1);
  mul_v4_fl(left_quat, (1.0 / transition_influence) * (1.0 - transition_influence));

  float identity_quat[4];
  unit_qt(identity_quat);

  /** result_quat = blended_quat @ inv(left_quat + identity) */
  float result_quat[4];
  add_v4_v4v4(result_quat, left_quat, identity_quat);
  normalize_qt(result_quat);
  invert_qt_normalized(result_quat);
  mul_qt_qtqt(result_quat, blended_quat, result_quat);
  normalize_qt(result_quat);

  copy_qt_qt(necs->values, result_quat);
}

/** Stores lower values within necs->values if invertible.
 * If not invertible, disables necs->invertible bits. Not invertible for transitions between
 * non-empty Combine and any non-empty strips of (Replace, Add, Subtract, Multiply). Only
 * invertible for transitions between Combine and (Combine or Empty)
 */
static void transition_qt_get_lower_values(NlaEvalChannelSnapshot *necs,
                                           const NlaEvalChannelSnapshot *necs1,
                                           const NlaEvalChannelSnapshot *necs2,
                                           const short blendmode1,
                                           const short blendmode2,
                                           const float transition_influence,
                                           const float inf1,
                                           const float inf2)
{

  /** Transition didn't include nec's property. */
  if (!necs1 && !necs2) {
    return;
  }

  /** If upper strip failed to invert this value, then we can't do anything with it.
   * We only check one index since quaternions are enabled and disabled as a whole. */
  if (!BLI_BITMAP_TEST_BOOL(necs->invertible.ptr, 0)) {
    return;
  }

  /** (Wayde Moss) These two variables kept separate in case anyone ever figures out the math
   * for inverting transitions between Combine and non-Combine strips. I've done the math countless
   * times but haven't figured it out yet.
   */
  bool is_combine1 = blendmode1 == NLASTRIP_MODE_COMBINE;
  bool is_combine2 = blendmode2 == NLASTRIP_MODE_COMBINE;

  /** Currently, no support for inverting through quaternion transitions that blend non-empty
   * (Combine) with non-empty strips of (Replace, Add, Subtract, Multiply). */
  if ((necs1 && !is_combine1) || (necs2 && !is_combine2)) {
    BLI_bitmap_set_all(necs->invertible.ptr, false, 4);
    return;
  }

  if (necs1) {
    if (necs2) {
      transition_qt_combine_to_combine_get_lower_values(
          necs, necs1, necs2, transition_influence, inf1, inf2);
    }
    else {
      transition_qt_combine_to_empty_get_lower_values(necs, necs1, transition_influence, inf1);
    }
  }
  else { /**  !necs1 && necs2 && is_combine2 */
    transition_qt_combine_to_empty_get_lower_values(necs, necs2, 1.0 - transition_influence, inf2);
  }
}

/** For strip blend modes of [Add, Subtract, Multiply (legacy), Multiply (pow), Replace],
 * they can all be represented in a common linear form of:
 * blended_result = lower_strip_blend_result * r_factor + r_offset */
static void transition_linear_get_coefficients(const int strip_blendmode,
                                               const int channel_mixmode,
                                               const float strip_value,
                                               float strip_base_value,
                                               const float strip_influence,
                                               float *r_factor,
                                               float *r_offset)
{
  switch (strip_blendmode) {
    case NLASTRIP_MODE_ADD:
      *r_factor = 1;
      *r_offset = strip_value * strip_influence;
      break;

    case NLASTRIP_MODE_SUBTRACT:
      *r_factor = 1;
      *r_offset = -strip_value * strip_influence;
      break;

    case NLASTRIP_MODE_MULTIPLY:
      *r_factor = (1.0f - strip_influence) + strip_value * strip_influence;
      *r_offset = 0;
      break;

    case NLASTRIP_MODE_COMBINE:
      switch (channel_mixmode) {
        case NEC_MIX_ADD:
        case NEC_MIX_AXIS_ANGLE:
          *r_factor = 1;
          *r_offset = (strip_value - strip_base_value) * strip_influence;
          break;

        case NEC_MIX_MULTIPLY:
          if (IS_EQF(strip_base_value, 0.0f)) {
            strip_base_value = 1.0f;
          }
          *r_factor = powf(strip_value / strip_base_value, strip_influence);
          *r_offset = 0;
          break;

        default:
          BLI_assert(!"combine mode");
          break;
      }
      break;

    default:
      *r_factor = (1.0f - strip_influence);
      *r_offset = strip_value * strip_influence;
      break;
  }
}

/** Stores lower values within necs->values if invertible. If not invertible, disables
 * necs->invertible bits. */
static void transition_linear_get_lower_values(NlaEvalChannelSnapshot *necs,
                                               const NlaEvalChannelSnapshot *necs1,
                                               const NlaEvalChannelSnapshot *necs2,
                                               const short blendmode1,
                                               const short blendmode2,
                                               const float transition_influence,
                                               const float inf1,
                                               const float inf2)
{
  float factor1, offset1;
  float factor2, offset2;

  for (int array_index = 0; array_index < necs->length; array_index++) {

    /** If upper strip failed to invert this value, then we can't do anything with it. */
    if (!BLI_BITMAP_TEST_BOOL(necs->invertible.ptr, array_index)) {
      continue;
    }

    /** Default factor and offset is solution for when necs1/2 NULL or not written to. The
     * strip didn't have the channel so the transition blended with the lower snapshot
     * directly. */
    factor1 = 1;
    offset1 = 0;
    if (necs1 && BLI_BITMAP_TEST_BOOL(necs1->raw_value_sampled.ptr, array_index))
      transition_linear_get_coefficients(blendmode1,
                                         necs1->channel->mix_mode,
                                         necs1->values[array_index],
                                         necs1->channel->base_snapshot.values[array_index],
                                         inf1,
                                         &factor1,
                                         &offset1);

    factor2 = 1;
    offset2 = 0;
    if (necs2 && BLI_BITMAP_TEST_BOOL(necs2->raw_value_sampled.ptr, array_index))
      transition_linear_get_coefficients(blendmode2,
                                         necs2->channel->mix_mode,
                                         necs2->values[array_index],
                                         necs2->channel->base_snapshot.values[array_index],
                                         inf2,
                                         &factor2,
                                         &offset2);

    /**
     *  blended_result = (lower * factor1 + offset1) * (1-t) + (lower * factor2 + offset2) * t
     *  blended_result = lower * factor1*(1-t) + offset1*(1-t) + lower * factor2*t + offset2*t
     *  blended_result = lower * (factor1*(1-t) + factor2*t) + [offset1*(1-t) + offset2*t]
     *  blended_result = lower * lerped_factor + lerped_offset
     *
     *  lower = (blended_result - lerped_offset) / lerped_factor
     */
    const float lerped_factor = (factor1 * (1.0f - transition_influence)) +
                                (factor2 * transition_influence);
    const float lerped_offset = (offset1 * (1.0f - transition_influence)) +
                                (offset2 * transition_influence);

    if (IS_EQF(0.0f, lerped_factor)) {
      /**
       * case: blended_result = lower * lerped_factor + lerped_offset
       *       blended_result = lower * 0 + lerped_offset
       *       blended_result = lerped_offset
       *
       * No solution. necs's value has been fully replaced
       *  or irrelevant when strip is multiply zero.
       *
       * Don't break to allow inverting as many channels as possible.
       */
      BLI_BITMAP_DISABLE(necs->invertible.ptr, array_index);
    }
    else {
      /** Fill snapshot with the lower blended values. */
      necs->values[array_index] = (necs->values[array_index] - lerped_offset) / lerped_factor;
    }
  }
}

/** Stores lower values within snapshot's NlaEvalChannel->values if invertible. */
static void nlastrip_evaluate_transition_invert_get_lower_values(
    PointerRNA *ptr,
    NlaEvalData *channels,
    ListBase *modifiers,
    NlaEvalStrip *nes,
    NlaEvalSnapshot *snapshot,
    const AnimationEvalContext *anim_eval_context)
{
  ListBase tmp_modifiers = {NULL, NULL};
  NlaEvalSnapshot snapshot1, snapshot2;
  NlaEvalStrip tmp_nes;
  NlaStrip *s1, *s2;

  /* Join this strip's modifiers to the parent's modifiers (own modifiers first). */
  nlaeval_fmodifiers_join_stacks(&tmp_modifiers, &nes->strip->modifiers, modifiers);

  /* get the two strips to operate on
   * - we use the endpoints of the strips directly flanking our strip
   *   using these as the endpoints of the transition (destination and source)
   * - these should have already been determined to be valid...
   * - if this strip is being played in reverse, we need to swap these endpoints
   *   otherwise they will be interpolated wrong
   */
  if (nes->strip->flag & NLASTRIP_FLAG_REVERSE) {
    s1 = nes->strip->next;
    s2 = nes->strip->prev;
  }
  else {
    s1 = nes->strip->prev;
    s2 = nes->strip->next;
  }

  /* prepare template for 'evaluation strip'
   * - based on the transition strip's evaluation strip data
   * - strip_mode is NES_TIME_TRANSITION_* based on which endpoint
   * - strip_time is the 'normalized' (i.e. in-strip) time for evaluation,
   *   which doubles up as an additional weighting factor for the strip influences
   *   which allows us to appear to be 'interpolating' between the two extremes
   */
  tmp_nes = *nes;

  float transition_influence = nes->strip_time;
  /* evaluate these strips into a temp-buffer (tmp_channels) */
  /* FIXME: modifier evaluation here needs some work... */
  /* first strip */
  tmp_nes.strip_mode = NES_TIME_TRANSITION_START;
  tmp_nes.strip = s1;
  tmp_nes.strip_time = s1->strip_time;
  nlaeval_snapshot_init(&snapshot1, channels, NULL);
  /** Get raw strip value without blending. */
  short blendmode1 = NLASTRIP_MODE_REPLACE;
  float inf1 = 0;
  nlastrip_evaluate_raw_value(
      ptr, channels, &tmp_modifiers, &tmp_nes, &snapshot1, anim_eval_context, &blendmode1, &inf1);

  /* second strip */
  tmp_nes.strip_mode = NES_TIME_TRANSITION_END;
  tmp_nes.strip = s2;
  tmp_nes.strip_time = s2->strip_time;
  nlaeval_snapshot_init(&snapshot2, channels, NULL);
  /** Get raw strip value without blending. */
  short blendmode2 = NLASTRIP_MODE_REPLACE;
  float inf2 = 0;
  nlastrip_evaluate_raw_value(
      ptr, channels, &tmp_modifiers, &tmp_nes, &snapshot2, anim_eval_context, &blendmode2, &inf2);

  nlaeval_snapshot_ensure_size(snapshot, channels->num_channels);

  NlaEvalChannelSnapshot *necs, *necs1, *necs2;
  for (int i = 0; i < snapshot->size; i++) {
    necs = snapshot->channels[i];
    /** We only care about inverting the channels that exist within snapshot.
     * Assumes channels to invert are filled and non-null contiguous. */
    if (!necs)
      break;

    /** If necs1 is NULL then snapshot1 does not contribute to snapshot. So snapshot is then a
     * result of the lower strip output blended with snapshot2. Vice versa for necs2. */
    necs1 = nlaeval_snapshot_get(&snapshot1, i);
    necs2 = nlaeval_snapshot_get(&snapshot2, i);

    /** Neither contributed to snapshot. */
    if (!necs1 && !necs2)
      continue;

    if ((necs->channel->mix_mode == NEC_MIX_QUATERNION) &&
        ((necs1 && ELEM(blendmode1, NLASTRIP_MODE_COMBINE)) ||
         (necs2 && ELEM(blendmode2, NLASTRIP_MODE_COMBINE)))) {
      /** Note: For Combine transitions, there is only proper quaternion keyframing support through
       * transitions between Combine and (Combine or Empty) strips. This function will properly
       * disable bits for malformed quaternion transitions.
       *
       * The other branch properly handles the case of both strips being any of (replace, add,
       * subtract, multiply), even if mix mode is quaternion.
       */
      transition_qt_get_lower_values(
          necs, necs1, necs2, blendmode1, blendmode2, transition_influence, inf1, inf2);
    }
    else {
      transition_linear_get_lower_values(
          necs, necs1, necs2, blendmode1, blendmode2, transition_influence, inf1, inf2);
    }
  }

  nlaeval_snapshot_free_data(&snapshot1);
  nlaeval_snapshot_free_data(&snapshot2);

  /* Unlink this strip's modifiers from the parent's modifiers again. */
  nlaeval_fmodifiers_split_stacks(&nes->strip->modifiers, modifiers);
}

/* evaluate meta-strip */
static void nlastrip_evaluate_meta(PointerRNA *ptr,
                                   NlaEvalData *channels,
                                   ListBase *modifiers,
                                   NlaEvalStrip *nes,
                                   NlaEvalSnapshot *snapshot,
                                   const AnimationEvalContext *anim_eval_context,
                                   const bool flush_to_original,
                                   bool allow_alloc_channels)
{
  ListBase tmp_modifiers = {NULL, NULL};
  NlaStrip *strip = nes->strip;
  NlaEvalStrip *tmp_nes;
  float evaltime;

  /* meta-strip was calculated normally to have some time to be evaluated at
   * and here we 'look inside' the meta strip, treating it as a decorated window to
   * its child strips, which get evaluated as if they were some tracks on a strip
   * (but with some extra modifiers to apply).
   *
   * NOTE: keep this in sync with animsys_evaluate_nla()
   */

  /* join this strip's modifiers to the parent's modifiers (own modifiers first) */
  nlaeval_fmodifiers_join_stacks(&tmp_modifiers, &strip->modifiers, modifiers);

  /* find the child-strip to evaluate */
  evaltime = (nes->strip_time * (strip->end - strip->start)) + strip->start;
  AnimationEvalContext child_context = BKE_animsys_eval_context_construct_at(anim_eval_context,
                                                                             evaltime);
  tmp_nes = nlastrips_ctime_get_strip(NULL, &strip->strips, -1, &child_context, flush_to_original);

  /* directly evaluate child strip into accumulation buffer...
   * - there's no need to use a temporary buffer (as it causes issues [T40082])
   */
  if (tmp_nes) {
    nlastrip_evaluate(ptr,
                      channels,
                      &tmp_modifiers,
                      tmp_nes,
                      snapshot,
                      &child_context,
                      flush_to_original,
                      allow_alloc_channels);

    /* free temp eval-strip */
    MEM_freeN(tmp_nes);
  }

  /* unlink this strip's modifiers from the parent's modifiers again */
  nlaeval_fmodifiers_split_stacks(&strip->modifiers, modifiers);
}

/** Stores lower values within snapshot's NlaEvalChannel->values if invertible. */
static void nlastrip_evaluate_meta_invert_get_lower_values(
    PointerRNA *ptr,
    NlaEvalData *upper_eval_data,
    ListBase *modifiers,
    NlaEvalStrip *nes,
    NlaEvalSnapshot *snapshot,
    const AnimationEvalContext *anim_eval_context)
{
  ListBase tmp_modifiers = {NULL, NULL};
  NlaStrip *strip = nes->strip;
  NlaEvalStrip *tmp_nes;
  float evaltime;

  /* meta-strip was calculated normally to have some time to be evaluated at
   * and here we 'look inside' the meta strip, treating it as a decorated window to
   * it's child strips, which get evaluated as if they were some tracks on a strip
   * (but with some extra modifiers to apply).
   *
   * NOTE: keep this in sync with animsys_evaluate_nla()
   */

  /* join this strip's modifiers to the parent's modifiers (own modifiers first) */
  nlaeval_fmodifiers_join_stacks(&tmp_modifiers, &strip->modifiers, modifiers);

  /* find the child-strip to evaluate */
  evaltime = (nes->strip_time * (strip->end - strip->start)) + strip->start;

  AnimationEvalContext child_context = BKE_animsys_eval_context_construct_at(anim_eval_context,
                                                                             evaltime);
  tmp_nes = nlastrips_ctime_get_strip(NULL, &strip->strips, -1, &child_context, false);
  /* directly evaluate child strip into accumulation buffer...
   * - there's no need to use a temporary buffer (as it causes issues [T40082])
   */
  if (tmp_nes) {
    nlastrip_evaluate_invert_get_lower_values(
        ptr, upper_eval_data, &tmp_modifiers, tmp_nes, snapshot, &child_context);

    /* free temp eval-strip */
    MEM_freeN(tmp_nes);
  }

  /* unlink this strip's modifiers from the parent's modifiers again */
  nlaeval_fmodifiers_split_stacks(&strip->modifiers, modifiers);
}

/** Stores non-blended raw values within snapshot's NlaEvalChannel->values and marks
 * NlaEvalChannel->raw_value_sampled if sampled from strip. */
static void nlastrip_evaluate_meta_raw_value(PointerRNA *ptr,
                                             NlaEvalData *upper_eval_data,
                                             ListBase *modifiers,
                                             NlaEvalStrip *nes,
                                             NlaEvalSnapshot *snapshot,
                                             const AnimationEvalContext *anim_eval_context,
                                             short *r_blendmode,
                                             float *r_influence)
{
  ListBase tmp_modifiers = {NULL, NULL};
  NlaStrip *strip = nes->strip;
  NlaEvalStrip *tmp_nes;
  float evaltime;

  /* meta-strip was calculated normally to have some time to be evaluated at
   * and here we 'look inside' the meta strip, treating it as a decorated window to
   * it's child strips, which get evaluated as if they were some tracks on a strip
   * (but with some extra modifiers to apply).
   *
   * NOTE: keep this in sync with animsys_evaluate_nla()
   */

  /* join this strip's modifiers to the parent's modifiers (own modifiers first) */
  nlaeval_fmodifiers_join_stacks(&tmp_modifiers, &strip->modifiers, modifiers);

  /* find the child-strip to evaluate */
  evaltime = (nes->strip_time * (strip->end - strip->start)) + strip->start;
  AnimationEvalContext child_context = BKE_animsys_eval_context_construct_at(anim_eval_context,
                                                                             evaltime);
  tmp_nes = nlastrips_ctime_get_strip(NULL, &strip->strips, -1, &child_context, false);

  /* directly evaluate child strip into accumulation buffer...
   * - there's no need to use a temporary buffer (as it causes issues [T40082])
   */
  if (tmp_nes) {
    nlastrip_evaluate_raw_value(ptr,
                                upper_eval_data,
                                &tmp_modifiers,
                                tmp_nes,
                                snapshot,
                                &child_context,
                                r_blendmode,
                                r_influence);

    /* free temp eval-strip */
    MEM_freeN(tmp_nes);
  }

  /* unlink this strip's modifiers from the parent's modifiers again */
  nlaeval_fmodifiers_split_stacks(&strip->modifiers, modifiers);
}
/*
 * lower is also output
 */
static void nlaeval_snapshot_blend(NlaEvalData *nlaeval,
                                   NlaEvalSnapshot *raw_upper,
                                   short upper_blendmode,
                                   float upper_influence,
                                   NlaEvalSnapshot *lower)
{
  nlaeval_snapshot_ensure_size(lower, nlaeval->num_channels);

  for (int i = 0; i < nlaeval->num_channels; i++) {
    NlaEvalChannelSnapshot *c_upper = nlaeval_snapshot_get(raw_upper, i);
    if (c_upper == NULL) {
      continue;
    }

    NlaEvalChannel *nec = c_upper->channel;
    NlaEvalChannelSnapshot *c_lower = nlaeval_snapshot_ensure_channel(lower, nec);

    int mix_mode = c_lower->channel->mix_mode;
    if (upper_blendmode == NLASTRIP_MODE_COMBINE) {
      if (mix_mode == NEC_MIX_QUATERNION) {
        nla_combine_quaternion(c_lower->values, c_upper->values, upper_influence, c_lower->values);
      }
      else {
        float *base_values = nec->base_snapshot.values;
        for (int j = 0; j < c_lower->length; j++) {
          c_lower->values[j] = nla_combine_value(
              mix_mode, base_values[j], c_lower->values[j], c_upper->values[j], upper_influence);
        }
      }
    }
    else {

      {
        // blend, user quat slerp through shortest angle if non full replace.
        // combine and the other blend modes shouldnt need this behavior since replace
        // defines an absolute orientation while the others are offset and proportional types.
        if (upper_blendmode == NLASTRIP_MODE_REPLACE && upper_influence < 1.0f) {
          PropertySubType subtype = RNA_property_subtype(nec->key.prop);
          int length = nec->base_snapshot.length;

          if (subtype == PROP_QUATERNION && length == 4) {
            // Todo: quaternion slerp through shorter angle
            float lower_qt[4], upper_qt[4];
            copy_qt_qt(upper_qt, c_upper->values);
            copy_qt_qt(lower_qt, c_lower->values);

            float result_qt[4];
            interp_qt_qtqt(result_qt, lower_qt, upper_qt, upper_influence);

            copy_qt_qt(c_lower->values, result_qt);
            continue;
          }
          else if (subtype == PROP_AXISANGLE && length == 4) {
            // Todo: quaternion slerp through shorter angle
            float lower_qt[4], upper_qt[4];
            axis_angle_to_quat(upper_qt, c_upper->values, c_upper->values[3]);
            axis_angle_to_quat(lower_qt, c_lower->values, c_lower->values[3]);

            float result_qt[4];
            interp_qt_qtqt(result_qt, lower_qt, upper_qt, upper_influence);

            quat_to_axis_angle(c_lower->values, &c_lower->values[3], result_qt);
            continue;
          }
          else if (subtype == PROP_EULER && length == 3) {
            // Todo: quaternion slerp through shorter angle
            float lower_qt[4], upper_qt[4];
            eul_to_quat(upper_qt, c_upper->values);
            eul_to_quat(lower_qt, c_lower->values);

            float result_qt[4];
            interp_qt_qtqt(result_qt, lower_qt, upper_qt, upper_influence);

            quat_to_eul(c_lower->values, result_qt);
            continue;
          }
        }
        for (int j = 0; j < c_lower->length; j++) {
          c_lower->values[j] = nla_blend_value(
              upper_blendmode, c_lower->values[j], c_upper->values[j], upper_influence);
        }
      }
    }
  }
}
/* evaluates the given evaluation strip */
void nlastrip_evaluate(PointerRNA *ptr,
                       NlaEvalData *channels,
                       ListBase *modifiers,
                       NlaEvalStrip *nes,
                       NlaEvalSnapshot *snapshot,
                       const AnimationEvalContext *anim_eval_context,
                       const bool flush_to_original,
                       bool allow_alloc_channels)
{
  /**
   * Todo: for blending rotation channels, its sometimes important to choose whether to ipo
   * through shorter or longer angle- maybe make it a toggle on blend data? Otherwise can't
   * properly align character flips due to it rotating in wrong direction... would need to be
   * applied using quaternion which means doing similar rotation grabbing and manipulation from/to
   * nla channels again which sucks..but doable..
   *    -....and maybe it would need to be a per bone's rotation component set toggle?
   *    -because some bones may flip while others may not...?
   *
   * What if we just stored the rotation channels as NLA quaternion channels (conversion on read
   * from fcurves) Then right before flushing to property, we convert to proper rotation type?
   * Unsure how itll affect keyframe remapping exactly
   *
   */
  NlaEvalSnapshot snapshot_raw;
  nlaeval_snapshot_init(&snapshot_raw, channels, NULL);

  short blendmode = 0;
  float influence = 0;
  nlastrip_evaluate_raw_value(
      ptr, channels, NULL, nes, &snapshot_raw, anim_eval_context, &blendmode, &influence);

  /* Apply blend transforms to each bone's raw snapshot values.  */
  Object *object = (Object *)ptr->data;
  bPose *pose = object->pose;
  /**
   * Assumes blend xformed bones are root bones with no parents. ( I think that would affect
   * conversion to bone local space?). If it has an animated parent, then blend xforms generally
   * won't make sense anyways (not a usecase situation).
   *
   * Q: maybe the blend xform should be stored per bone and already in local space?
   *
   * todo: make blend xform UI in python... alot easier.
   * todo: if strip has cycled, then apply blend xform based on how far each bone moves per
   * cycle. Probably need a toggle per blend xform for whether cyclic offset if applied (no need
   * to be per bone).
   */
  LISTBASE_FOREACH (NlaBlendTransform *, blend, &nes->strip->blend_transforms) {
    float world[4][4];
    loc_eul_size_to_mat4(world, blend->location, blend->euler, blend->scale);

    LISTBASE_FOREACH (NlaBlendTransform_BoneTarget *, bone, &blend->bones) {

      char name_esc[sizeof(bone->name) * 2];
      BLI_strescape(name_esc, bone->name, sizeof(name_esc));

      bPoseChannel *pose_channel = BKE_pose_channel_find_name(pose, name_esc);

      float rest_matrix[4][4];
      unit_m4(rest_matrix);
      /* Converting to world space works fine since the bone is in rest pose.
       * and we want to apply a World-space delta transform but applied in local space,
       * which is what the following matrix inversion and multiplication does.
       * */
      BKE_constraint_mat_convertspace(object,
                                      pose_channel,
                                      rest_matrix,
                                      CONSTRAINT_SPACE_LOCAL,
                                      CONSTRAINT_SPACE_WORLD,
                                      false);
      float rest_matrix_inv[4][4];
      invert_m4_m4(rest_matrix_inv, rest_matrix);

      /* Get blend transform in bone's non-animated local space. */
      float bone_blend_matrix[4][4];
      mul_m4_m4m4(bone_blend_matrix, rest_matrix_inv, world);
      mul_m4_m4m4(bone_blend_matrix, bone_blend_matrix, rest_matrix);
      // copy_m4_m4(bone_blend_matrix, world);
      // BKE_constraint_mat_convertspace(object,
      //                                pose_channel,
      //                                bone_blend_matrix,
      //                                CONSTRAINT_SPACE_WORLD,
      //                                CONSTRAINT_SPACE_LOCAL,
      //                                false);

      char *location_path = BLI_sprintfN("pose.bones[\"%s\"].location", name_esc);
      NlaEvalChannel *location_channel = nlaevalchan_verify(ptr, channels, location_path);
      float *location_values =
          nlaeval_snapshot_ensure_channel(&snapshot_raw, location_channel)->values;

      char *rotation_path;
      switch (pose_channel->rotmode) {
        case ROT_MODE_QUAT:
          rotation_path = BLI_sprintfN("pose.bones[\"%s\"].rotation_quaternion", name_esc);
          break;
        case ROT_MODE_AXISANGLE:
          rotation_path = BLI_sprintfN("pose.bones[\"%s\"].rotation_axis_angle", name_esc);
          break;
        default:
          rotation_path = BLI_sprintfN("pose.bones[\"%s\"].rotation_euler", name_esc);
          break;
      }
      NlaEvalChannel *rotation_channel = nlaevalchan_verify(ptr, channels, rotation_path);
      float *rotation_values =
          nlaeval_snapshot_ensure_channel(&snapshot_raw, rotation_channel)->values;

      char *scale_path = BLI_sprintfN("pose.bones[\"%s\"].scale", name_esc);
      NlaEvalChannel *scale_channel = nlaevalchan_verify(ptr, channels, scale_path);
      float *scale_values = nlaeval_snapshot_ensure_channel(&snapshot_raw, scale_channel)->values;

      /* Apply blend transform as a parent transform to bone's action channels.
       * Results written directly back to raw snapshot. */
      float raw_snapshot_matrix[4][4];
      float decomposed_quat[4];
      switch (pose_channel->rotmode) {
        case ROT_MODE_QUAT:
          loc_quat_size_to_mat4(
              raw_snapshot_matrix, location_values, rotation_values, scale_values);
          mul_m4_m4m4(raw_snapshot_matrix, bone_blend_matrix, raw_snapshot_matrix);
          mat4_decompose(location_values, rotation_values, scale_values, raw_snapshot_matrix);

          break;
        case ROT_MODE_AXISANGLE:
          loc_axisangle_size_to_mat4(raw_snapshot_matrix,
                                     location_values,
                                     rotation_values,
                                     rotation_values[3],
                                     scale_values);
          mul_m4_m4m4(raw_snapshot_matrix, bone_blend_matrix, raw_snapshot_matrix);
          mat4_decompose(location_values, decomposed_quat, scale_values, raw_snapshot_matrix);
          quat_to_axis_angle(rotation_values, &rotation_values[3], decomposed_quat);
          break;
        default:
          loc_eul_size_to_mat4(
              raw_snapshot_matrix, location_values, rotation_values, scale_values);
          mul_m4_m4m4(raw_snapshot_matrix, bone_blend_matrix, raw_snapshot_matrix);
          quat_to_eul(rotation_values, decomposed_quat);

          break;
      }

      MEM_freeN(location_path);  // todo: verify correct call
      MEM_freeN(rotation_path);  // todo: verify correct call
      MEM_freeN(scale_path);     // todo: verify correct call
    }
  }

  /** Blend raw snapshot with lower snapshot. */
  nlaeval_snapshot_blend(channels, &snapshot_raw, blendmode, influence, snapshot);
  nlaeval_snapshot_free_data(&snapshot_raw);
}

/* Removes the effect of the evaluation strip from the snapshot value,
 * storing lower values within snapshot's NlaEvalChannel->values if invertible. */
void nlastrip_evaluate_invert_get_lower_values(PointerRNA *ptr,
                                               NlaEvalData *upper_eval_data,
                                               ListBase *modifiers,
                                               NlaEvalStrip *nes,
                                               NlaEvalSnapshot *snapshot,
                                               const AnimationEvalContext *anim_eval_context)
{
  NlaStrip *strip = nes->strip;

  /* To prevent potential infinite recursion problems
   * (i.e. transition strip, beside meta strip containing a transition
   * several levels deep inside it),
   * we tag the current strip as being evaluated, and clear this when we leave.
   */
  /* TODO: be careful with this flag, since some edit tools may be running and have
   * set this while animation playback was running. */
  if (strip->flag & NLASTRIP_FLAG_EDIT_TOUCHED) {
    return;
  }
  strip->flag |= NLASTRIP_FLAG_EDIT_TOUCHED;

  /* actions to take depend on the type of strip */
  switch (strip->type) {
    case NLASTRIP_TYPE_CLIP: /* action-clip */
      nlastrip_evaluate_actionclip_invert_get_lower_values(
          ptr, upper_eval_data, modifiers, nes, snapshot);
      break;
    case NLASTRIP_TYPE_TRANSITION: /* transition */
      nlastrip_evaluate_transition_invert_get_lower_values(
          ptr, upper_eval_data, modifiers, nes, snapshot, anim_eval_context);
      break;
    case NLASTRIP_TYPE_META: /* meta */
      nlastrip_evaluate_meta_invert_get_lower_values(
          ptr, upper_eval_data, modifiers, nes, snapshot, anim_eval_context);
      break;

    default: /* do nothing */
      break;
  }

  /* clear temp recursion safe-check */
  strip->flag &= ~NLASTRIP_FLAG_EDIT_TOUCHED;
}

/* Evaluates strip without blending, stored within snapshot. NLAEvalChannelSnapshot
 * raw_value_sampled bits are set for channels sampled. This is only called by transition's
 * invert function. */
void nlastrip_evaluate_raw_value(PointerRNA *ptr,
                                 NlaEvalData *upper_eval_data,
                                 ListBase *modifiers,
                                 NlaEvalStrip *nes,
                                 NlaEvalSnapshot *snapshot,
                                 const AnimationEvalContext *anim_eval_context,
                                 short *r_blendmode,
                                 float *r_influence)
{
  NlaStrip *strip = nes->strip;

  /* To prevent potential infinite recursion problems
   * (i.e. transition strip, beside meta strip containing a transition
   * several levels deep inside it),
   * we tag the current strip as being evaluated, and clear this when we leave.
   */
  /* TODO: be careful with this flag, since some edit tools may be running and have
   * set this while animation playback was running. */
  if (strip->flag & NLASTRIP_FLAG_EDIT_TOUCHED) {
    return;
  }
  strip->flag |= NLASTRIP_FLAG_EDIT_TOUCHED;

  /* actions to take depend on the type of strip */
  switch (strip->type) {
    case NLASTRIP_TYPE_CLIP: /* action-clip */
      nlastrip_evaluate_actionclip_raw_value(
          ptr, upper_eval_data, modifiers, nes, snapshot, r_blendmode, r_influence);
      break;
    case NLASTRIP_TYPE_TRANSITION:
      /*
       * If nes is eventually a transition, then nothing is written and no value bitmask is
       * set. This matches the existing behavior that adjacent transitions evaluate to default
       * (lower snapshot or property type default).
       *
       * In normal cases, this case should never happen. This is only called by
       * nlastrip_evaluate_transition_invert_get_lower_values() and transitions never reference
       * other transitions for prev/next strips. Metas containing a transition won't call this
       * either in the case of [Action strip][Evaluated Transition][Meta(Transition, ...)].
       * */
      break;
    case NLASTRIP_TYPE_META: /* meta */
      nlastrip_evaluate_meta_raw_value(ptr,
                                       upper_eval_data,
                                       modifiers,
                                       nes,
                                       snapshot,
                                       anim_eval_context,
                                       r_blendmode,
                                       r_influence);
      break;

    default: /* do nothing */
      break;
  }

  /* clear temp recursion safe-check */
  strip->flag &= ~NLASTRIP_FLAG_EDIT_TOUCHED;
}

/* write the accumulated settings to */
void nladata_flush_channels(PointerRNA *ptr,
                            NlaEvalData *channels,
                            NlaEvalSnapshot *snapshot,
                            const bool flush_to_original)
{
  /* sanity checks */
  if (channels == NULL) {
    return;
  }

  /* for each channel with accumulated values, write its value on the property it affects */
  LISTBASE_FOREACH (NlaEvalChannel *, nec, &channels->channels) {
    /**
     * The bitmask is set for all channels touched by NLA due to the domain() function.
     * Channels touched by current set of evaluated strips will have a snapshot channel
     * directly from the evaluation snapshot.
     *
     * This function falls back to the default value if the snapshot channel doesn't exist.
     * Thus channels, touched by NLA but not by the current set of evaluated strips, will be
     * reset to default. If channel not touched by NLA then it's value is unchanged.
     */
    NlaEvalChannelSnapshot *nec_snapshot = nlaeval_snapshot_find_channel(snapshot, nec);

    PathResolvedRNA rna = {nec->key.ptr, nec->key.prop, -1};

    for (int i = 0; i < nec_snapshot->length; i++) {
      if (BLI_BITMAP_TEST(nec->domain.ptr, i)) {
        float value = nec_snapshot->values[i];
        if (nec->is_array) {
          rna.prop_index = i;
        }
        BKE_animsys_write_rna_setting(&rna, value);
        if (flush_to_original) {
          animsys_write_orig_anim_rna(ptr, nec->rna_path, rna.prop_index, value);
        }
      }
    }
  }
}

/* ---------------------- */

static void nla_eval_domain_action(PointerRNA *ptr,
                                   NlaEvalData *channels,
                                   bAction *act,
                                   GSet *touched_actions)
{
  if (!BLI_gset_add(touched_actions, act)) {
    return;
  }

  LISTBASE_FOREACH (FCurve *, fcu, &act->curves) {
    /* check if this curve should be skipped */
    if (fcu->flag & (FCURVE_MUTED | FCURVE_DISABLED)) {
      continue;
    }
    if ((fcu->grp) && (fcu->grp->flag & AGRP_MUTED)) {
      continue;
    }
    if (BKE_fcurve_is_empty(fcu)) {
      continue;
    }

    NlaEvalChannel *nec = nlaevalchan_verify(ptr, channels, fcu->rna_path);

    if (nec != NULL) {
      /* For quaternion properties, enable all sub-channels. */
      if (nec->mix_mode == NEC_MIX_QUATERNION) {
        BLI_bitmap_set_all(nec->domain.ptr, true, 4);
        continue;
      }

      int idx = nlaevalchan_validate_index(nec, fcu->array_index);

      if (idx >= 0) {
        BLI_BITMAP_ENABLE(nec->domain.ptr, idx);
      }
    }
  }
}

static void nla_eval_domain_strips(PointerRNA *ptr,
                                   NlaEvalData *channels,
                                   ListBase *strips,
                                   GSet *touched_actions)
{
  LISTBASE_FOREACH (NlaStrip *, strip, strips) {
    /* check strip's action */
    if (strip->act) {
      nla_eval_domain_action(ptr, channels, strip->act, touched_actions);
    }

    /* check sub-strips (if metas) */
    nla_eval_domain_strips(ptr, channels, &strip->strips, touched_actions);
  }
}

/**
 * Ensure that all channels touched by any of the actions in enabled tracks exist.
 * This is necessary to ensure that evaluation result depends only on current frame.
 */
static void animsys_evaluate_nla_domain(PointerRNA *ptr, NlaEvalData *channels, AnimData *adt)
{
  GSet *touched_actions = BLI_gset_ptr_new(__func__);

  if (adt->action) {
    nla_eval_domain_action(ptr, channels, adt->action, touched_actions);
  }
  /** If upper tracks are evaluated then need to include tmpact. */
  if (adt->tmpact && (adt->flag & ADT_NLA_EVAL_UPPER_TRACKS)) {
    nla_eval_domain_action(ptr, channels, adt->tmpact, touched_actions);
  }

  /* NLA Data - Animation Data for Strips */
  LISTBASE_FOREACH (NlaTrack *, nlt, &adt->nla_tracks) {
    /* solo and muting are mutually exclusive... */
    if (adt->flag & ADT_NLA_SOLO_TRACK) {
      /* skip if there is a solo track, but this isn't it */
      if ((nlt->flag & NLATRACK_SOLO) == 0) {
        continue;
      }
      /* else - mute doesn't matter */
    }
    else {
      /* no solo tracks - skip track if muted */
      if (nlt->flag & NLATRACK_MUTED) {
        continue;
      }
    }

    nla_eval_domain_strips(ptr, channels, &nlt->strips, touched_actions);
  }

  BLI_gset_free(touched_actions, NULL);
}

/* ---------------------- */

/** Tweaked strip is evaluated differently from other strips. Adjacent strips are ignored
 * and includes a workaround for when user is not editing in place. */
static NlaEvalStrip *animsys_append_tweaked_strip(ListBase *list,
                                                  const AnimData *adt,
                                                  const AnimationEvalContext *anim_eval_context,
                                                  NlaStrip *dummy_strip,
                                                  const bool flush_to_original,
                                                  const bool for_keyframing)

{
  /** Single strip list to prevent adjacent strips from being evaluated. */
  ListBase dummy_trackslist;
  dummy_trackslist.first = dummy_trackslist.last = dummy_strip;

  /* Copy active strip so we can modify how it evaluates without affecting user data. */
  memcpy(dummy_strip, adt->actstrip, sizeof(NlaStrip));
  dummy_strip->next = dummy_strip->prev = NULL;

  /* If tweaked strip is syncing action length, then evaluate using action length. */
  if (dummy_strip->flag & NLASTRIP_FLAG_SYNC_LENGTH) {
    BKE_nlastrip_recalculate_bounds_sync_action(dummy_strip);
  }

  /* Strips with a user-defined time curve don't get properly remapped for editing
   * at the moment, so mapping them just for display may be confusing. */
  bool is_inplace_tweak = !(adt->flag & ADT_NLA_EDIT_NOMAP) &&
                          !(adt->actstrip->flag & NLASTRIP_FLAG_USR_TIME);

  if (!is_inplace_tweak) {
    /* Use Hold due to no proper remapping yet (the note above). */
    dummy_strip->extendmode = NLASTRIP_EXTEND_HOLD;

    /* Disable range. */
    dummy_strip->flag |= NLASTRIP_FLAG_NO_TIME_MAP;
  }

  /** Controls whether able to keyframe outside range of tweaked strip. */
  if (for_keyframing) {
    dummy_strip->extendmode = (is_inplace_tweak &&
                               !(dummy_strip->flag & NLASTRIP_FLAG_SYNC_LENGTH)) ?
                                  NLASTRIP_EXTEND_NOTHING :
                                  NLASTRIP_EXTEND_HOLD;
  }

  /* Add this to our list of evaluation strips. */
  return nlastrips_ctime_get_strip(
      list, &dummy_trackslist, -1, anim_eval_context, flush_to_original);
}

static void nonstrip_action_fill_strip_data(const AnimData *adt,
                                            NlaStrip *action_strip,
                                            const bool keyframing_to_strip)
{
  memset(action_strip, 0, sizeof(*action_strip));

  bAction *action = adt->action;

  if ((adt->flag & ADT_NLA_EDIT_ON)) {
    action = adt->tmpact;
  }

  /* Set settings of dummy NLA strip from AnimData settings. */
  action_strip->act = action;

  /* Action range is calculated taking F-Modifiers into account
   * (which making new strips doesn't do due to the troublesome nature of that). */
  calc_action_range(action_strip->act, &action_strip->actstart, &action_strip->actend, 1);
  action_strip->start = action_strip->actstart;
  action_strip->end = (IS_EQF(action_strip->actstart, action_strip->actend)) ?
                          (action_strip->actstart + 1.0f) :
                          (action_strip->actend);

  action_strip->blendmode = adt->act_blendmode;
  action_strip->extendmode = adt->act_extendmode;
  action_strip->influence = adt->act_influence;

  /* NOTE: must set this, or else the default setting overrides,
   * and this setting doesn't work. */
  action_strip->flag |= NLASTRIP_FLAG_USR_INFLUENCE;

  /* Unless extendmode is Nothing (might be useful for flattening NLA evaluation), disable
   * range. Extendmode Nothing and Hold will behave as normal. Hold Forward will behave just
   * like Hold.
   */
  if (action_strip->extendmode != NLASTRIP_EXTEND_NOTHING) {
    action_strip->flag |= NLASTRIP_FLAG_NO_TIME_MAP;
  }

  const bool tweakoff = (adt->flag & ADT_NLA_EDIT_ON) == 0;
  const bool tweakon_eval_upper = ((adt->flag & ADT_NLA_EDIT_ON) != 0) &&
                                  ((adt->flag & ADT_NLA_EVAL_UPPER_TRACKS) != 0);
  const bool soloing = (adt->flag & ADT_NLA_SOLO_TRACK) != 0;
  const bool actionstrip_evaluated = action_strip->act && !soloing &&
                                     (tweakoff || tweakon_eval_upper);
  if (!actionstrip_evaluated) {
    action_strip->flag |= NLASTRIP_FLAG_MUTED;
  }

  /** If we're keyframing, then we must allow keyframing outside fcurve bounds. */
  if (keyframing_to_strip) {
    action_strip->extendmode = NLASTRIP_EXTEND_HOLD;
  }
}

bool is_nlatrack_evaluatable(const AnimData *adt, const NlaTrack *nlt)
{
  /* Skip disabled tracks unless it contains the tweaked strip. */
  if ((adt->flag & ADT_NLA_EDIT_ON) && (nlt->flag & NLATRACK_DISABLED)) {
    if (nlt->index != adt->act_track->index) {
      return false;
    }
  }

  /* Solo and muting are mutually exclusive. */
  if (adt->flag & ADT_NLA_SOLO_TRACK) {
    /* Skip if there is a solo track, but this isn't it. */
    if ((nlt->flag & NLATRACK_SOLO) == 0) {
      return false;
    }
  }
  else {
    /* Skip track if muted. */
    if (nlt->flag & NLATRACK_MUTED) {
      return false;
    }
  }

  return true;
}

/** Check for special case of non-pushed action being evaluated with no NLA influence (off and
 * no strips evaluated) nor NLA interference (ensure NLA not soloing). */
static bool is_nonstrip_action_evaluated_without_nla(const AnimData *adt,
                                                     const bool any_strip_evaluated)
{
  if (adt->action) {
    if ((adt->flag & (ADT_NLA_SOLO_TRACK | ADT_NLA_EDIT_ON)) == 0 && !any_strip_evaluated) {
      /* Evaluate as if there isn't any NLA data. */
      return true;
    }
  }
  return false;
}

/** XXX Wayde Moss: BKE_nlatrack_find_tweaked() exists within nla.c, but it doesn't appear to
 * work as expected. From animsys_evaluate_nla_for_flush(), it returns NULL in tweak mode. I'm
 * not sure why. Preferably, it would be as simple as checking for (adt->act_Track == nlt) but
 * that doesn't work either, neither does comparing indices.
 *
 *  This function is a temporary work around. The first disabled track is always the tweaked
 * track.
 */
static NlaTrack *nlatrack_find_tweaked(const AnimData *adt)
{
  NlaTrack *nlt;

  if (adt == NULL) {
    return NULL;
  }

  /* Since the track itself gets disabled, we want the first disabled... */
  for (nlt = adt->nla_tracks.first; nlt; nlt = nlt->next) {
    if (nlt->flag & NLATRACK_DISABLED) {
      return nlt;
    }
  }

  /* Not found! */
  return NULL;
}

/**
 * NLA Evaluation function - values are calculated and stored in temporary "NlaEvalChannels"
 * \param[out] echannels: Evaluation channels with calculated values
 */
static bool animsys_evaluate_nla_for_flush(NlaEvalData *echannels,
                                           PointerRNA *ptr,
                                           const AnimData *adt,
                                           const AnimationEvalContext *anim_eval_context,
                                           const bool flush_to_original)
{
  NlaTrack *nlt;
  short track_index = 0;
  bool has_strips = false;
  ListBase estrips = {NULL, NULL};
  NlaEvalStrip *nes;

  NlaStrip tweak_strip;

  memset(&tweak_strip, 0, sizeof(tweak_strip));

  NlaTrack *tweaked_track = nlatrack_find_tweaked(adt);

  /* Get the stack of strips to evaluate at current time (influence calculated here). */
  for (nlt = adt->nla_tracks.first; nlt; nlt = nlt->next, track_index++) {

    if (!is_nlatrack_evaluatable(adt, nlt)) {
      continue;
    }

    if (nlt->strips.first) {
      has_strips = true;
    }

    if (nlt == tweaked_track) {
      /** Tweaked strip is evaluated differently. */
      nes = animsys_append_tweaked_strip(
          &estrips, adt, anim_eval_context, &tweak_strip, flush_to_original, false);
    }
    else {
      /** Append strip to evaluate for this track. */
      nes = nlastrips_ctime_get_strip(
          &estrips, &nlt->strips, track_index, anim_eval_context, flush_to_original);
    }
    if (nes) {
      nes->track = nlt;
    }
  }

  if (is_nonstrip_action_evaluated_without_nla(adt, has_strips)) {
    BLI_freelistN(&estrips);
    return false;
  }

  NlaStrip action_strip = {0};
  nonstrip_action_fill_strip_data(adt, &action_strip, false);
  nlastrips_ctime_get_strip_single(&estrips, &action_strip, anim_eval_context, flush_to_original);

  /*
   * create and group loc/rot/scale nla channels for blend bones
   *  -need to also calc blend xform in bone's local space (blend transform in world
  (pose?)
   *      space, need it in bone local space) use convert_space?
   *
   *
   * Q: how to get pchan_index from bone name?
      bPoseChannel *BKE_pose_channel_find_name(const bPose *pose, const char *name)
  bPoseChannel *pchan = pose_pchan_get_indexed(object, pchan_index);

  BKE_constraint_mat_convertspace(ob, pchan, (float(*)[4])mat_ret, from, to, false);

   * will also have to compute each strip into raw-value snapshots.
   *
   * before blending snapshot wtih lower, we apply blend transforms -use blend
   * grouped data
   * ______
   *
   * Prep:
   *    1) Get all strips that will evaluate on current frame.
   *    2) Create and group specified bones' nla channels. This is used to conveniently get
  channels to apply blend to. Grouped using  hash: bonename to nlachannels (should blend
  xform affect domain? should it xform as if bone has identity if no channels exist? for now
  assuming trivial all TRS exist)

   * For each strip:
   *   1) store fcurve raw values into a snapshot
   *   2)
   * 1) Convert World space blend xforms to each specified bone's local space
    2.5) hash? for bone name to bone's blend xform matrix?
   * 4) Apply blend xforms to raw snapshot as a parent xform
   * 5) apply nla blend
   */
  /**
   * todo: name escape bone names
   * /

  // TODO: blend xform should be part of nalstrip_evaluate()? (so remap() and resample()
  don't
  // have to change anything)
  //  should eval and create raw snapshot per call, then blend as expected before finishing
  GHash *nla_blend_from_bonename = BLI_ghash_str_new("blend_xform_from_bonename");

  // PointerRNA id_ptr;
  // RNA_id_pointer_create(prop_ptr->owner_id, &id_ptr);

  // NlaEvalChannel *nec = nlaevalchan_verify(&id_ptr, &upper_eval_data, rna_path);

  for (nes = estrips.first; nes; nes = nes->next) {
  }

  /* Per strip, evaluate and accumulate on top of existing channels. */
  for (nes = estrips.first; nes; nes = nes->next) {
    nlastrip_evaluate(ptr,
                      echannels,
                      NULL,
                      nes,
                      &echannels->eval_snapshot,
                      anim_eval_context,
                      flush_to_original,
                      true);
  }

  /* Free temporary evaluation data that's not used elsewhere. */
  BLI_freelistN(&estrips);
  return true;
}

/** Lower blended values are calculated and accumulated into r_context->lower_nla_channels.
 * Upper NlaEvalStrips are stored in r_context->upper_estrips. */
static void animsys_evaluate_nla_for_keyframing(NlaKeyframingContext *r_context,
                                                PointerRNA *ptr,
                                                const AnimData *adt,
                                                const AnimationEvalContext *anim_eval_context)
{
  if (!r_context) {
    return;
  }

  /* Early out. If NLA track is soloing and tweaked action isn't it, then don't allow keyframe
   * insertion. */
  if (adt->flag & ADT_NLA_SOLO_TRACK) {
    if (!(adt->act_track && (adt->act_track->flag & NLATRACK_SOLO))) {
      r_context->eval_strip = NULL;
      return;
    }
  }

  NlaTrack *nlt;
  short track_index = 0;
  bool has_strips = false;

  ListBase *upper_estrips = &r_context->upper_estrips;
  ListBase lower_estrips = {NULL, NULL};
  NlaEvalStrip *nes;

  NlaTrack *tweaked_track = nlatrack_find_tweaked(adt);

  /* Get the lower stack of strips to evaluate at current time (influence calculated here). */
  for (nlt = adt->nla_tracks.first; nlt; nlt = nlt->next, track_index++) {

    if (!is_nlatrack_evaluatable(adt, nlt)) {
      continue;
    }

    /* Tweaked strip effect should not be stored in any snapshot. */
    if (nlt == tweaked_track) {
      break;
    }

    if (nlt->strips.first) {
      has_strips = true;
    }

    /* Get strip to evaluate for this channel. */
    nes = nlastrips_ctime_get_strip(
        &lower_estrips, &nlt->strips, track_index, anim_eval_context, false);
    if (nes) {
      nes->track = nlt;
    }
  }

  /* Get the upper stack of strips to evaluate at current time (influence calculated here). */
  /* Var nlt exists only if tweak strip exists. */
  if (nlt) {

    /* Skip tweaked strip. */
    nlt = nlt->next;
    track_index++;

    for (; nlt; nlt = nlt->next, track_index++) {

      if (!is_nlatrack_evaluatable(adt, nlt)) {
        continue;
      }

      if (nlt->strips.first) {
        has_strips = true;
      }

      /* Get strip to evaluate for this channel. */
      nes = nlastrips_ctime_get_strip(
          upper_estrips, &nlt->strips, track_index, anim_eval_context, false);
      if (nes) {
        nes->track = nlt;
      }
    }
  }

  /** Note: Although we early out, we can still keyframe to the non-pushed action since the
   * keyframe remap function detects (adt->strip.act == NULL) and will keyframe without
   * remapping.
   */
  if (is_nonstrip_action_evaluated_without_nla(adt, has_strips)) {
    BLI_freelistN(&lower_estrips);
    return;
  }

  /* Write r_context->eval_strip. */
  if (adt->flag & ADT_NLA_EDIT_ON) {

    /* Append nonstrip action to upper snapshot. */
    NlaStrip *action_strip = &r_context->nonuser_act_strip;
    nonstrip_action_fill_strip_data(adt, action_strip, (adt->flag & ADT_NLA_EDIT_ON) == 0);
    nlastrips_ctime_get_strip_single(upper_estrips, action_strip, anim_eval_context, false);

    NlaStrip *dummy_strip = &r_context->strip;
    memset(dummy_strip, 0, sizeof(*dummy_strip));
    r_context->eval_strip = animsys_append_tweaked_strip(
        NULL, adt, anim_eval_context, dummy_strip, false, true);
  }
  else {

    NlaStrip *action_strip = &r_context->strip;
    nonstrip_action_fill_strip_data(adt, action_strip, true);
    /* Note: If NLA track is soloing, then NULL returned and no keyframe will be inserted. */
    r_context->eval_strip = nlastrips_ctime_get_strip_single(
        NULL, action_strip, anim_eval_context, false);
  }

  /* If NULL, then keyframing will fail. No need to do any more processing. */
  if (!r_context->eval_strip) {
    BLI_freelistN(upper_estrips);
    BLI_freelistN(&lower_estrips);
    return;
  }

  /* If tweak strip is full REPLACE, then lower strips not needed. */
  if (r_context->strip.blendmode == NLASTRIP_MODE_REPLACE &&
      IS_EQF(r_context->strip.influence, 1)) {
    BLI_freelistN(&lower_estrips);
    return;
  }

  /* For each strip, evaluate then accumulate on top of existing channels. */
  for (nes = lower_estrips.first; nes; nes = nes->next) {
    nlastrip_evaluate(ptr,
                      &r_context->lower_nla_channels,
                      NULL,
                      nes,
                      &r_context->lower_nla_channels.eval_snapshot,
                      anim_eval_context,
                      false,
                      true);
  }

  /* Free temporary evaluation data that's not used elsewhere. */
  BLI_freelistN(&lower_estrips);
}

/* NLA Evaluation function (mostly for use through do_animdata)
 * - All channels that will be affected are not cleared anymore. Instead, we just evaluate into
 *   some temp channels, where values can be accumulated in one go.
 */
static void animsys_calculate_nla(PointerRNA *ptr,
                                  AnimData *adt,
                                  const AnimationEvalContext *anim_eval_context,
                                  const bool flush_to_original)
{
  NlaEvalData echannels;

  nlaeval_init(&echannels);

  /* evaluate the NLA stack, obtaining a set of values to flush */
  if (animsys_evaluate_nla_for_flush(&echannels, ptr, adt, anim_eval_context, flush_to_original)) {
    /* reset any channels touched by currently inactive actions to default value */
    animsys_evaluate_nla_domain(ptr, &echannels, adt);

    /* flush effects of accumulating channels in NLA to the actual data they affect */
    nladata_flush_channels(ptr, &echannels, &echannels.eval_snapshot, flush_to_original);
  }
  else {
    /* special case - evaluate as if there isn't any NLA data */
    /* TODO: this is really just a stop-gap measure... */
    if (G.debug & G_DEBUG) {
      CLOG_WARN(&LOG, "NLA Eval: Stopgap for active action on NLA Stack - no strips case");
    }

    animsys_evaluate_action(ptr, adt->action, anim_eval_context, flush_to_original);
  }

  /* free temp data */
  nlaeval_free(&echannels);
}

/* ---------------------- */

/**
 * Prepare data necessary to compute correct keyframe values for NLA strips
 * with non-Replace mode or influence different from 1.
 *
 * \param cache: List used to cache contexts for reuse when keying
 * multiple channels in one operation.
 * \param ptr: RNA pointer to the Object with the animation.
 * \return Keyframing context, or NULL if not necessary.
 */
NlaKeyframingContext *BKE_animsys_get_nla_keyframing_context(
    struct ListBase *cache,
    struct PointerRNA *ptr,
    struct AnimData *adt,
    const AnimationEvalContext *anim_eval_context,
    const bool flush_to_original)
{
  /* No remapping needed if NLA is off or no action. */
  if ((adt == NULL) || (adt->action == NULL) || (adt->nla_tracks.first == NULL) ||
      (adt->flag & ADT_NLA_EVAL_OFF)) {
    return NULL;
  }

  /* No remapping if editing an ordinary Replace action with full influence. */
  if (!(adt->flag & ADT_NLA_EDIT_ON) &&
      (adt->act_blendmode == NLASTRIP_MODE_REPLACE && adt->act_influence == 1.0f)) {
    return NULL;
  }

  /* Try to find a cached context. */
  NlaKeyframingContext *ctx = BLI_findptr(cache, adt, offsetof(NlaKeyframingContext, adt));

  if (ctx == NULL) {
    /* Allocate and evaluate a new context. */
    ctx = MEM_callocN(sizeof(*ctx), "NlaKeyframingContext");
    ctx->adt = adt;

    nlaeval_init(&ctx->lower_nla_channels);
    animsys_evaluate_nla_for_keyframing(ctx, ptr, adt, anim_eval_context);

    BLI_assert(ELEM(ctx->strip.act, NULL, adt->action));
    BLI_addtail(cache, ctx);
  }

  return ctx;
}

/**
 * Apply correction from the NLA context to the values about to be keyframed.
 *
 * \param context: Context to use (may be NULL).
 * \param prop_ptr: Property about to be keyframed.
 * \param[in,out] values: Array of property values to adjust.
 * \param count: Number of values in the array.
 * \param index: Index of the element about to be updated, or -1.
 * \param[out] r_force_all: Set to true if all channels must be inserted. May be NULL.
 * \return False if correction fails due to a division by zero,
 * or null r_force_all when all channels are required.
 */
bool BKE_animsys_nla_remap_keyframe_values(struct NlaKeyframingContext *context,
                                           const AnimationEvalContext *anim_eval_context,
                                           struct PointerRNA *prop_ptr,
                                           struct PropertyRNA *prop,
                                           char rna_path[],
                                           float *values,
                                           int count,
                                           int index,
                                           bool *r_force_all)
{
  if (r_force_all != NULL) {
    *r_force_all = false;
  }

  /* No context means no correction. */
  if (context == NULL || context->strip.act == NULL) {
    return true;
  }

  /* If the strip is not evaluated, it is the same as zero influence. */
  if (context->eval_strip == NULL) {
    return false;
  }

  /* Full influence Replace strips also require no correction. */
  const int blend_mode = context->strip.blendmode;
  const float influence = context->strip.influence;

  /* Zero influence is division by zero. */
  if (influence <= 0.0f) {
    return false;
  }

  /* Find the evaluation channel for the NLA stack below current strip. */
  NlaEvalChannelKey key = {
      .ptr = *prop_ptr,
      .prop = prop,
  };

  /**
   * Remove upper NLA stack effects.
   *
   * Each iteration solves for the blended result from the next lower strip. When all upper
   * estrips inverted, the snapshot will contain the blended result of the tweak strip.
   *
   * This block's only output is to overwrite values or early out if remapping fails.
   */
  {
    /* Create the channels for the upper strips to output to. */
    NlaEvalData upper_eval_data;
    nlaeval_init(&upper_eval_data);

    NlaEvalSnapshot *blended_values_snapshot = &upper_eval_data.eval_snapshot;

    PointerRNA id_ptr;
    RNA_id_pointer_create(prop_ptr->owner_id, &id_ptr);

    NlaEvalChannel *nec = nlaevalchan_verify(&id_ptr, &upper_eval_data, rna_path);
    NlaEvalChannelSnapshot *nec_snapshot = nlaeval_snapshot_ensure_channel(blended_values_snapshot,
                                                                           nec);

    /* Fill user values. */
    for (int i = 0; i < count; i++) {
      nec_snapshot->values[i] = values[i];
    }

    /** Bitmap used to know which indices were successfully inverted. */
    BLI_bitmap_set_all(nec_snapshot->invertible.ptr, true, count);

    /** Per iteration, remove effect of current strip which gives output of strip below it. */
    ListBase *upper_estrips = &context->upper_estrips;
    LISTBASE_FOREACH_BACKWARD (NlaEvalStrip *, nes, upper_estrips) {
      /** This will disable nec_snapshot->invertible bits if an upper strip is not invertible
       * (full replace, multiply zero, or non-invertible transition). Then there is no
       * inversion solution. */
      nlastrip_evaluate_invert_get_lower_values(
          &id_ptr, &upper_eval_data, NULL, nes, blended_values_snapshot, anim_eval_context);
    }

    /* At this point, the snapshot contains the output values after the tweak strip is applied.
     */
    bool all_invertible = true;
    if (index == -1 || nec->mix_mode == NEC_MIX_QUATERNION) {
      for (int i = 0; i < count; i++) {
        values[i] = nec_snapshot->values[i];
        all_invertible &= BLI_BITMAP_TEST_BOOL(nec_snapshot->invertible.ptr, i);
      }
    }
    else {
      values[index] = nec_snapshot->values[index];
      all_invertible &= BLI_BITMAP_TEST_BOOL(nec_snapshot->invertible.ptr, index);
    }

    nlaeval_free(&upper_eval_data);

    /** To match existing implementation, only succeeds if all desired indices are invertible.
     */
    if (!all_invertible) {
      return false;
    }
  }

  /**
   * Remove lower NLA stack effects.
   *
   * Using the tweak strip's blended result and the lower snapshot value, we can solve for the
   * fcurve values of the tweak strip.
   */
  NlaEvalData *const lower_nlaeval = &context->lower_nla_channels;
  NlaEvalChannel *const lower_nec = nlaevalchan_verify_key(lower_nlaeval, NULL, &key);

  if ((lower_nec->base_snapshot.length != count)) {
    BLI_assert(!"invalid value count");
    return false;
  }

  /* Invert the blending operation to compute the desired key values. */
  NlaEvalChannelSnapshot *const lower_nec_snapshot = nlaeval_snapshot_find_channel(
      &lower_nlaeval->eval_snapshot, lower_nec);

  float *old_lower_values = lower_nec_snapshot->values;

  if (blend_mode == NLASTRIP_MODE_COMBINE) {
    /* Quaternion combine handles all sub-channels as a unit. */
    if (lower_nec->mix_mode == NEC_MIX_QUATERNION) {
      if (r_force_all == NULL) {
        return false;
      }

      *r_force_all = true;

      if (!nla_combine_quaternion_invert_get_fcurve_values(
              old_lower_values, values, influence, values)) {
        return false;
      }
    }
    else {
      float *base_values = lower_nec->base_snapshot.values;

      for (int i = 0; i < count; i++) {
        if (ELEM(index, i, -1)) {
          if (!nla_combine_value_invert_get_fcurve_value(lower_nec->mix_mode,
                                                         base_values[i],
                                                         old_lower_values[i],
                                                         values[i],
                                                         influence,
                                                         &values[i])) {
            return false;
          }
        }
      }
    }
  }
  else {
    for (int i = 0; i < count; i++) {
      if (ELEM(index, i, -1)) {
        if (!nla_blend_value_invert_get_fcurve_value(
                blend_mode, old_lower_values[i], values[i], influence, &values[i])) {
          return false;
        }
      }
    }
  }

  return true;
}

/**
 * Free all cached contexts from the list.
 */
void BKE_animsys_free_nla_keyframing_context_cache(struct ListBase *cache)
{
  LISTBASE_FOREACH (NlaKeyframingContext *, ctx, cache) {
    MEM_SAFE_FREE(ctx->eval_strip);
    BLI_freelistN(&ctx->upper_estrips);
    nlaeval_free(&ctx->lower_nla_channels);
  }

  BLI_freelistN(cache);
}

/** Allocates per added action clip strip.
 * TODO: Wayde Moss: place this in proper location, so other functions may use it in future.
 */
void nlastrip_append_actionclip_recursive(ListBase *dst, NlaStrip *strip)
{
  BLI_assert(!ELEM(NULL, dst, strip));

  switch (strip->type) {
    case NLASTRIP_TYPE_CLIP: {
      LinkData *ld = MEM_callocN(sizeof(LinkData), __func__);
      ld->data = strip;
      BLI_addtail(dst, ld);

      break;
    }

    case NLASTRIP_TYPE_TRANSITION:
      /** NOTE: Transitions never reference other transitions so this only results in infinite
       * recursion when there's a bug elsewhere. */
      if (strip->prev && strip->next) {
        nlastrip_append_actionclip_recursive(dst, strip->prev);
        nlastrip_append_actionclip_recursive(dst, strip->next);
      }
      break;

    case NLASTRIP_TYPE_META:
      LISTBASE_FOREACH (NlaStrip *, inner_strip, &strip->strips) {
        nlastrip_append_actionclip_recursive(dst, inner_strip);
      }
      break;

    default:
      break;
  }
}

/** Mute selected NLA strips and resample into a new track. The final Nla stack result will be
 * preserved when possible. New resampled strip will be selected. Previously selected strips
 * will be muted and deselected afterward.
 *
 * \param resample_blendmode: Resulting resampled strip's blend mode.
 * \param resample_influence: Resulting resampled strip's influence. above.
 * \param resample_insertion_nlt_index: NlaTrack to insert the resample track above or below.
 * \param insert_track_lower: Side of resample_insertion_nlt_index to place resample track.
 * \returns: The new resample track. Returns NULL and does nothing if in tweak mode, resample
 * influence zero, or no fcurves are involved in the resample.
 */
NlaTrack *BKE_animsys_resample_selected_strips(Main *main,
                                               Depsgraph *depsgraph,
                                               AnimData *adt,
                                               PointerRNA *id_ptr,
                                               char resample_name[],
                                               short resample_blendmode,
                                               float resample_influence,
                                               int resample_insertion_nlt_index,
                                               bool insert_track_lower)
{

  /**
   * ***************************** Intended Uses ******************************************
   *
   * Merge Strips: User selects a block of NlaStrips and Resamples.
   *
   * Convert Strips: User selects a single NlaStrip and Resamples with a different blendmode
   * and/or influence.
   *
   * ********************* Potential improvements/changes *********************************
   *
   * For frames where user had a keyframe, make them non-selected. Select non-user keys. This
   * allows a follow-up op to do an Fcurve simplify or decimate of only the baked keyframes.
   * Effectively it allows a follow-up Smart Bake that preserves user keys. Perhaps this can be
   * done by the caller.
   *
   * Allow user to somehow select channels to be resampled. Currently all channels found in all
   * selected strips are resampled. Though a simple work around is to delete the undesired
   * channels after the resample.
   *
   * ********************* Limitations and potential problems *****************************
   *
   * Design: When resample strip value not valid, what should we do? Currently we write a
   * default value. Nothing we write will preserve the animation. This leaves the problem as a
   * "Known Issue".
   *
   * This function will not properly resample outside of the resample bounds. Generally, it's
   * not possible since multiple strips with non-None extend modes can not be represented by a
   * single strip of any extend mode.. Maybe it's possible by properly setting the pre and post
   * extrapolation for individual fcurves?
   */

  /**
   * General Resample Implementation:
   *
   * 1) Get evaluatable tracks and selected strips, excluding non-evaluated strips.
   *
   * 2) resampled_bounds: Calculate bounds of strips. For initial implementation, we take the
   * simple approach. This will be a single range that includes intervals where none of the
   * selected strips evaluate.
   *
   * 3) Create a new track and strip for the resample data using resample_bounds. We use a new
   * track to ensure there is sufficient horizontal space for all the keyframes. Resample track
   * used as marker for lower and upper evaluation stacks.
   *
   * 4) Calculate resample keyframes. Per frame in resample_bounds:
   *
   * whole_snapshot: Evaluate the whole NLA stack. Selected strips are included since we're
   * preserving this snapshot result.
   *
   * lower_snapshot: Evaluate the NLA stack from the base track up to the resample track,
   * excluding selected tracks and the resampled track.
   *
   * upper_strips: The strips from the resampled strip, exclusive, to the topmost track,
   * excluding selected strips.
   *
   * With these three sets of data, we can solve for the value that resample strip must
   * evaluate to that satisfies whole_snapshot when the selected strips are muted. It's the
   * same way keyframe remapping works where the tweak strip is substituted with the resample
   * strip.
   */

  BLI_assert(!ELEM(NULL, main, depsgraph, adt, id_ptr));

  if (IS_EQF(0, resample_influence) || (adt->flag & ADT_NLA_EDIT_ON) != 0) {
    return NULL;
  }

  if (strlen(resample_name) == 0) {
    resample_name = "Resample";
  }

  /*************** 1) Get evaluatable selected strips that will be resampled.************ */

  ListBase evaluatable_tracks = {NULL, NULL};
  LinkData *insertion_track_target_ld = NULL;
  int nlt_index;
  LISTBASE_FOREACH_INDEX (NlaTrack *, nlt, &adt->nla_tracks, nlt_index) {

    /** Insertion index may not refer to evaluatable track. We add it to the list anyways
     * so we can properly place the resample track in the list. If it's not evaluateable, it
     * will be removed from the list later. */
    if (nlt_index == resample_insertion_nlt_index) {

      LinkData *ld = MEM_callocN(sizeof(LinkData), __func__);
      ld->data = nlt;
      BLI_addtail(&evaluatable_tracks, ld);

      insertion_track_target_ld = ld;
      continue;
    }

    if (!is_nlatrack_evaluatable(adt, nlt) || !nlt->strips.first) {
      continue;
    }

    LinkData *ld = MEM_callocN(sizeof(LinkData), __func__);
    ld->data = nlt;
    BLI_addtail(&evaluatable_tracks, ld);
  }
  BLI_assert(insertion_track_target_ld);

  ListBase selected_strips = {NULL, NULL};
  LISTBASE_FOREACH (LinkData *, ld, &evaluatable_tracks) {
    NlaTrack *nlt = ld->data;

    for (NlaStrip *strip = nlt->strips.first; strip; strip = strip->next) {
      if ((strip->flag & NLASTRIP_FLAG_SELECT) != 0 && (strip->flag & NLASTRIP_FLAG_MUTED) == 0) {

        LinkData *ld = MEM_callocN(sizeof(LinkData), __func__);
        ld->data = strip;
        BLI_addtail(&selected_strips, ld);

        // printf("selected strip: %s\n", strip->name);
      }
    }
  }

  /** Early Out. No selected strips to resample. */
  if (BLI_listbase_count(&selected_strips) == 0) {
    // printf("\n Early out. No strips selected.\n");
    BLI_freelistN(&evaluatable_tracks);
    return NULL;
  }

  /********* (Earlier than used) Create NlaEvalData, NlaEvalSnapshots, bActionGroup hashes.
   * *****/
  /** NOTE: We obtain the following data this early to prevent allocating the resample track
   * and strip prematurely. It's mostly out of preference. Otherwise, this block can be placed
   * right before calculating the resample keyframe values. */

  NlaEvalData eval_data_buffer;
  NlaEvalData *eval_data = &eval_data_buffer;
  nlaeval_init(eval_data);
  NlaEvalSnapshot *whole_snapshot = &eval_data->eval_snapshot;

  NlaEvalSnapshot lower_snapshot_buffer = {0};
  NlaEvalSnapshot *lower_snapshot = &lower_snapshot_buffer;
  nlaeval_snapshot_init(lower_snapshot, eval_data, NULL);

  /** Key: NlaEvalChannel's rna_path
   *  Value: array of original fcurves*/
  GHash *nec_to_fcurve_array = BLI_ghash_str_new("nec_to_fcurve_array");
  /** Key: group name
   *  Value: bActionGroup */
  GHash *groupname_to_group = BLI_ghash_str_new("groupname_to_group");

  /** Visit all the involved fcurves. An fcurve pair (rna_path, array_index) may be visited
   * multiple times.
   *
   * Allocate all the necessary NlaEvalChannels and NlaEvalChannelSnapshots. This is the only
   *time we allow new NlaEvalChannels and snapshots to be allocated.
   *
   * NlaEvalChannel->domain bitmap will be used to store whether the fcurve channel exists
   * among the selected strips. Effectively it marks whether the fcurve is involved in the
   * resample. For Quaternions, all 4 fcurves are marked as involved if any exists. Later, we
   *only allocate enough memory to store resample values for involved fcurves.
   *
   * We use hashes to preserve fcurve groups. For NlaEvalChannels, we assume that not all
   *elements may be in the same bActionGroup. It's an unlikely case but still possible. For now
   *we just store the original bActionGroup and later we'll duplicate it.
   **/
  LISTBASE_FOREACH (LinkData *, link_data, &selected_strips) {
    NlaStrip *outter_strip = link_data->data;

    ListBase action_clips = {NULL, NULL};
    nlastrip_append_actionclip_recursive(&action_clips, outter_strip);

    LISTBASE_FOREACH (LinkData *, ld_clip, &action_clips) {
      NlaStrip *strip = ld_clip->data;

      if (!strip->act) {
        continue;
      }

      LISTBASE_FOREACH (FCurve *, fcurve, &strip->act->curves) {

        if (!is_fcurve_evaluatable(fcurve)) {
          continue;
        }

        NlaEvalChannel *nec = nlaevalchan_verify(id_ptr, eval_data, fcurve->rna_path);
        nlaeval_snapshot_ensure_channel(whole_snapshot, nec);
        nlaeval_snapshot_ensure_channel(lower_snapshot, nec);

        if (nec->mix_mode == NEC_MIX_QUATERNION) {
          BLI_bitmap_set_all(nec->domain.ptr, true, 4);
        }
        else {
          BLI_BITMAP_ENABLE(nec->domain.ptr, fcurve->array_index);
        }

        if (fcurve->grp) {

          FCurve ***p_fcurve_array;
          if (!BLI_ghash_ensure_p(
                  nec_to_fcurve_array, fcurve->rna_path, (void ***)&p_fcurve_array)) {
            *p_fcurve_array = MEM_callocN(sizeof(FCurve *) * nec->base_snapshot.length, __func__);
          }

          if (nlaevalchan_validate_index_ex(nec, fcurve->array_index)) {
            FCurve **fcurve_array = *p_fcurve_array;
            fcurve_array[fcurve->array_index] = fcurve;
          }

          bActionGroup **p_group;
          if (!BLI_ghash_ensure_p(groupname_to_group, fcurve->grp->name, (void ***)&p_group)) {
            *p_group = fcurve->grp;
          }
        }
      }
    }

    BLI_freelistN(&action_clips);
  }

  /** Count the total number of fcurves involve in the resample.
   *
   * We have to do this separately from the above loop because the above will visit the same
   * fcurve (rna_path, array_index) pair multiple times. */
  int total_fcurves = 0;
  LISTBASE_FOREACH (NlaEvalChannel *, nec, &eval_data->channels) {
    for (int array_index = 0; array_index < nec->base_snapshot.length; array_index++) {
      if (BLI_BITMAP_TEST_BOOL(nec->domain.ptr, array_index)) {
        total_fcurves++;
      }
    }
  }

  /** Early Out. No evaluatable FCurves to resample. */
  if (total_fcurves == 0) {
    // printf("Early out. No fcurves involved in resample.\n");
    BLI_freelistN(&evaluatable_tracks);
    BLI_freelistN(&selected_strips);
    nlaeval_free(eval_data);
    nlaeval_snapshot_free_data(lower_snapshot);
    BLI_ghash_free(nec_to_fcurve_array, NULL, MEM_freeN);
    BLI_ghash_free(groupname_to_group, NULL, NULL);
    return NULL;
  }

  // printf("total fcurves resampled: %i\n", total_fcurves);

  /*************** 2) Calculate resample bounds of selected strips.***************** */

  /** REFACTOR: as BKE_nlastrips_calculate_bounds() */
  int resample_start = INT_MAX;
  int resample_end = INT_MIN;
  LISTBASE_FOREACH (LinkData *, link_data, &selected_strips) {
    NlaStrip *strip = link_data->data;

    resample_start = min(resample_start, strip->start);
    resample_end = max(resample_end, strip->end);
  }

  /************** 3) Allocate resample strip. *************************************** */

  NlaTrack *resample_track = BKE_nlatrack_add(adt, NULL);
  bAction *resample_action = BKE_action_add(main, resample_name);
  NlaStrip *resample_strip = BKE_nlastrip_new(resample_action);
  BKE_nlatrack_add_strip(resample_track, resample_strip);

  /** REFACTOR: as BKE_nla_track_rename() */
  strcpy(resample_track->name, resample_name);
  BLI_uniquename(&adt->nla_tracks,
                 resample_track,
                 DATA_("NlaTrack"),
                 '.',
                 offsetof(NlaTrack, name),
                 sizeof(resample_track->name));

  /** REFACTOR: as BKE_nlastrip_rename() */
  strcpy(resample_strip->name, resample_name);
  BKE_nlastrip_validate_name(adt, resample_strip);

  resample_strip->blendmode = resample_blendmode;
  resample_strip->start = resample_strip->actstart = resample_start;
  resample_strip->end = resample_strip->actend = resample_end;
  resample_strip->influence = resample_influence;
  resample_strip->extendmode = NLASTRIP_EXTEND_NOTHING;

  /** REFACTOR: as BKE_nlastrip_set_influence */
  if (!IS_EQF(1, resample_influence)) {
    resample_strip->flag |= NLASTRIP_FLAG_USR_INFLUENCE;
  }

  /** Insert resample track to correct location to relevant lists. Resample track added to
   * evaluatable_tracks to mark and separate the lower and upper evaluated strips. */
  LinkData *resample_track_link = MEM_callocN(sizeof(LinkData), __func__);
  resample_track_link->data = resample_track;

  BLI_poptail(&adt->nla_tracks);
  if (insert_track_lower) {
    BLI_insertlinkbefore(&adt->nla_tracks, insertion_track_target_ld->data, resample_track);
    BLI_insertlinkbefore(&evaluatable_tracks, insertion_track_target_ld, resample_track_link);
  }
  else {
    BLI_insertlinkafter(&adt->nla_tracks, insertion_track_target_ld->data, resample_track);
    BLI_insertlinkafter(&evaluatable_tracks, insertion_track_target_ld, resample_track_link);
  }

  if (!is_nlatrack_evaluatable(adt, insertion_track_target_ld->data)) {
    BLI_remlink(&evaluatable_tracks, insertion_track_target_ld);
    MEM_freeN(insertion_track_target_ld);
  }

  /**************  4) Calculate resample keyframes. ***************************** */

  /** Instead of calculating the resample strip value and creating a keyframe (and fcurve) for
   * it per iteration, we create a buffer of all the resample values and create the keyframes
   * after the resampling completes. This is partly for optimization and partly to simplify the
   * code involved in the loop.
   *
   * Memory layout:
   *
   *   foreach(NlaEvalChannel nec){
   *      foreach(involved fcurve){
   *         foreach(frame from start..end){
   *             memory: value0, value1, ..., valueN
   *         }
   *      }
   *    }
   *
   * So we store all the keyframe co values for an fcurve as a single contiguous block. That
   * way we can create the fcurve and sequentially write all of its co values in one go. For
   * our current implementation, each write is offsetted by total_frames amount of floats. A
   * potential efficiency improvement would be to calculate the eval data in the same order and
   * optimizing the eval calculation for single channels (4 channels for quaternions).
   */
  const int total_frames = resample_end - resample_start + 1;
  const int total_allocated_values = total_fcurves * total_frames;
  float *const resample_values_buffer = MEM_mallocN(total_allocated_values * sizeof(float),
                                                    "nla_resample_values");
  /** If there are no involved fcurves, then we would've early outted already. The min total
   * frames is one. At this point, we're always asking for nonzero memory to be allocated. */
  BLI_assert(resample_values_buffer);
  // printf("total frames: %i\n", total_frames);
  // printf("total allocated resample values: %i\n", total_allocated_values);

  const int fcurve_stride = total_frames;
  const bool full_replace = ELEM(resample_blendmode, NLASTRIP_MODE_REPLACE) &&
                            IS_EQF(resample_influence, 1);
  ListBase upper_estrips = {NULL, NULL};
  ListBase lower_estrips = {NULL, NULL};

  NlaStrip action_strip = {0};
  nonstrip_action_fill_strip_data(adt, &action_strip, false);

  /** XXX: Proceeding code assumes no NlaEvalChannel/Snapshot ever created or deleted. Doing so
   * is unexpected and will lead to a crash since resample_values_buffer is not allocated for
   * non resampled channels.
   *
   * The only allocated data per frame are NlaEvalStrips into upper_estrips and lower_estrips,
   * freed per iteration. A more optimal solution is noted inside at the end of the loop. */
  const int total_eval_channels = eval_data->num_channels;
  for (int cfra = resample_start; cfra <= resample_end; cfra++) {

    AnimationEvalContext cfra_context = BKE_animsys_eval_context_construct(depsgraph, cfra);
    /************** Gather upper and lower evaluated NlaEvalStrips. *************/
    {
      LinkData *ld;
      /* Get the lower stack of strips to evaluate at current time (influence calculated here).
       */
      for (ld = evaluatable_tracks.first; ld; ld = ld->next) {
        NlaTrack *nlt = ld->data;

        /* Resample strip should not be evaluated by any snapshot. */
        if (nlt == resample_track) {
          break;
        }

        nlastrips_ctime_get_strip(&lower_estrips, &nlt->strips, -1, &cfra_context, false);
      }

      /* Skip resampled strip. */
      ld = ld->next;

      /* Get the upper stack of strips to evaluate at current time (influence calculated here).
       */
      for (; ld; ld = ld->next) {
        NlaTrack *nlt = ld->data;
        nlastrips_ctime_get_strip(&upper_estrips, &nlt->strips, -1, &cfra_context, false);
      }

      nlastrips_ctime_get_strip_single(&upper_estrips, &action_strip, &cfra_context, false);
    }

    /************** Calculate snapshots. ***************************** */
    {
      /** Reset snapshot channels.
       *
       * REFACTOR: as animsys_reset_snapshot_channels()
       */
      LISTBASE_FOREACH (NlaEvalChannel *, nec, &eval_data->channels) {
        int nec_index = nec->index;

        BLI_assert(nec_index >= 0 && nec_index < whole_snapshot->size);
        BLI_assert(nec_index >= 0 && nec_index < lower_snapshot->size);

        nlaevalchan_snapshot_copy(whole_snapshot->channels[nec_index], &nec->base_snapshot);
        nlaevalchan_snapshot_copy(lower_snapshot->channels[nec_index], &nec->base_snapshot);
      }

      /** Calculate the whole_snapshot.
       *
       * Whole snapshot generally cannot use lower snapshot as starting point since whole
       * snapshot must include the selected strips, which are excluded from the lower
       * snapshot. */
      for (NlaEvalStrip *nes = lower_estrips.first; nes; nes = nes->next) {
        nlastrip_evaluate(
            id_ptr, eval_data, NULL, nes, whole_snapshot, &cfra_context, false, false);
      }
      for (NlaEvalStrip *nes = upper_estrips.first; nes; nes = nes->next) {
        nlastrip_evaluate(
            id_ptr, eval_data, NULL, nes, whole_snapshot, &cfra_context, false, false);
      }

      /** Calculate lower_eval_data. Exclude selected strips.
       *
       * Optimization: If resample strip is full REPLACE, then lower strips not needed to solve
       * for resample result. This works because resample strip will replace every channel in
       * the lower strips. */
      if (!full_replace) {
        for (NlaEvalStrip *nes = lower_estrips.first; nes; nes = nes->next) {
          if ((nes->strip->flag & NLASTRIP_FLAG_SELECT) == 0) {
            nlastrip_evaluate(
                id_ptr, eval_data, NULL, nes, lower_snapshot, &cfra_context, false, false);
          }
        }
      }
    }

    /************** Solve for resample values. ***************************** */
    {
      /** Mark bitmaps, used to know which indices were successfully inverted. */
      LISTBASE_FOREACH (NlaEvalChannel *, nec, &eval_data->channels) {
        int nec_index = nec->index;
        BLI_assert(nec_index >= 0 && nec_index < whole_snapshot->size);

        NlaEvalChannelSnapshot *necs = whole_snapshot->channels[nec_index];
        BLI_bitmap_set_all(necs->invertible.ptr, true, necs->length);
      }

      /** Per iteration, remove effect of current strip which gives output of strip below it.
       *
       * REFACTOR: as animsys_snapshot_invert_upper_strips()
       */
      LISTBASE_FOREACH_BACKWARD (NlaEvalStrip *, nes, &upper_estrips) {
        if ((nes->strip->flag & NLASTRIP_FLAG_SELECT) == 0) {
          /** This will disable nec_snapshot->invertible bits if an upper strip is not
           * invertible (full replace, multiply zero, or non-invertible transition). Then there
           * is no inversion solution. */
          nlastrip_evaluate_invert_get_lower_values(
              id_ptr, eval_data, NULL, nes, whole_snapshot, &cfra_context);
        }
      }

      /** XXX: Proceeding code assumes no eval channel ever created or deleted. Doing so is
       * unexpected and will lead to a crash since resample_values_buffer is not allocated for
       * non-resampled channels. */
      BLI_assert(eval_data->num_channels == total_eval_channels);

      /** Since each NlaEvalChannel can be associated with a different varying number of
       * fcurve channels we have to track the write offset. (NlaEvalChannelSnapshot->length
       * varies and so does the total fcurves actually involved for resampling, tagged by
       * NlaEvalChannel->domain.) */
      int running_keyframe_write_offset = cfra - resample_start;
      LISTBASE_FOREACH (NlaEvalChannel *, nec, &eval_data->channels) {
        int nec_index = nec->index;

        BLI_assert(nec_index >= 0 && nec_index < whole_snapshot->size);
        BLI_assert(nec_index >= 0 && nec_index < lower_snapshot->size);

        NlaEvalChannelSnapshot *upper_nec_snapshot = whole_snapshot->channels[nec_index];
        NlaEvalChannelSnapshot *lower_nec_snapshot = lower_snapshot->channels[nec_index];

        const short blendmode = resample_blendmode;
        const float influence = resample_influence;
        const short mix_mode = nec->mix_mode;
        const int count = lower_nec_snapshot->length;
        float *const base_values = nec->base_snapshot.values;
        float *const values = upper_nec_snapshot->values;
        float *const old_lower_values = lower_nec_snapshot->values;
        float *const r_values = values;

        BLI_assert(ELEM(
            lower_nec_snapshot->length, upper_nec_snapshot->length, nec->base_snapshot.length));

        /** REFACTOR: void nlastrip_invert_remove_lower_stack(...) */
        if (blendmode == NLASTRIP_MODE_COMBINE) {
          /* Quaternion combine handles all sub-channels as a unit. */
          if (mix_mode == NEC_MIX_QUATERNION) {
            if (!nla_combine_quaternion_invert_get_fcurve_values(
                    old_lower_values, values, influence, values)) {
              BLI_bitmap_set_all(upper_nec_snapshot->invertible.ptr, false, 4);
            }
          }
          else {
            for (int i = 0; i < count; i++) {
              if (!nla_combine_value_invert_get_fcurve_value(mix_mode,
                                                             base_values[i],
                                                             old_lower_values[i],
                                                             values[i],
                                                             influence,
                                                             &values[i])) {
                BLI_BITMAP_DISABLE(upper_nec_snapshot->invertible.ptr, i);
              }
            }
          }
        }
        else {
          for (int i = 0; i < count; i++) {
            if (!nla_blend_value_invert_get_fcurve_value(
                    blendmode, old_lower_values[i], values[i], influence, &values[i])) {
              BLI_BITMAP_DISABLE(upper_nec_snapshot->invertible.ptr, i);
            }
          }
        }

        /** Store the resample value. We'll add them as keyframes later. */
        for (int fcurve_array_index = 0; fcurve_array_index < count; fcurve_array_index++) {
          /** We do not write the value if the associated fcurve array index has no resample
           * memory allocated for it. */
          if (!BLI_BITMAP_TEST_BOOL(nec->domain.ptr, fcurve_array_index)) {
            continue;
          }

          BLI_assert(running_keyframe_write_offset >= 0 &&
                     running_keyframe_write_offset < total_allocated_values);

          if (BLI_BITMAP_TEST_BOOL(upper_nec_snapshot->invertible.ptr, fcurve_array_index)) {
            resample_values_buffer[running_keyframe_write_offset] = r_values[fcurve_array_index];
          }
          else {
            resample_values_buffer[running_keyframe_write_offset] =
                base_values[fcurve_array_index];
          }

          running_keyframe_write_offset += fcurve_stride;
        }
      }
    }

    /************** Free data. ***************************** */

    /** Potential optimization: Instead of freeing the list, we can maintain a dynamic
     * list. Strips would be added and removed as the current frame passes their bounds.
     * Currently every NlaEvalStrip is searched for and found per frame. Reminder that
     * we would still have to calculate estrip->strip_time and influence every frame if it's
     * animated (can it be driven?). */
    BLI_freelistN(&upper_estrips);
    BLI_freelistN(&lower_estrips);
  }

  /** Create bActionGroups by duplicating and replacing the original in the hash. */
  GHashIterator gh_iter;
  GHASH_ITER (gh_iter, groupname_to_group) {
    bActionGroup *original_group = BLI_ghashIterator_getValue(&gh_iter);
    bActionGroup *resample_group = action_groups_add_new(resample_action, original_group->name);
    BLI_ghash_reinsert(groupname_to_group, original_group->name, resample_group, NULL, NULL);

    resample_group->flag = 0;
    resample_group->customCol = original_group->customCol;
    action_group_colors_sync(resample_group, original_group);
  }

  /** Create fcurves and add resampled keyframes. */
  int running_keyframe_read_offset = 0;
  LISTBASE_FOREACH (NlaEvalChannel *, nec, &eval_data->channels) {
    for (int array_index = 0; array_index < nec->base_snapshot.length; array_index++) {
      if (BLI_BITMAP_TEST_BOOL(nec->domain.ptr, array_index)) {

        FCurve *fcurve = BKE_fcurve_create();
        BLI_assert(fcurve);
        fcurve->rna_path = BLI_strdup(nec->rna_path);
        fcurve->array_index = array_index;

        /** Assign fcurve group. Validity check since we only allocated for valid indices. */
        FCurve *original_fcurve = NULL;
        bActionGroup *action_group = NULL;
        FCurve **fcurve_array = BLI_ghash_lookup(nec_to_fcurve_array, fcurve->rna_path);
        if (nlaevalchan_validate_index_ex(nec, fcurve->array_index)) {
          original_fcurve = fcurve_array[fcurve->array_index];

          if (original_fcurve->grp) {
            action_group = BLI_ghash_lookup(groupname_to_group, original_fcurve->grp->name);
          }
        }

        /** Copy some data from original fcurve. */
        if (original_fcurve) {
          /** Only these two flags are important to copy from the original. */
          fcurve->flag = (original_fcurve->flag &
                          (FCURVE_INT_VALUES | FCURVE_DISCRETE_VALUES | FCURVE_VISIBLE));
          fcurve->extend = original_fcurve->extend;
          fcurve->auto_smoothing = original_fcurve->auto_smoothing;
          fcurve->color_mode = original_fcurve->color_mode;
          for (int i = 0; i < sizeof(fcurve->color) / sizeof(float); i++) {
            fcurve->color[i] = original_fcurve->color[i];
          }
        }

        if (action_group) {
          action_groups_add_channel(resample_action, action_group, fcurve);
        }
        else {
          BLI_addtail(&resample_action->curves, fcurve);
        }

        /** Add Keyframes. */
        fcurve->bezt = MEM_callocN(sizeof(BezTriple) * total_frames, "beztriple");
        BLI_assert(fcurve->bezt);
        fcurve->totvert = total_frames;
        for (int cfra = resample_start; cfra <= resample_end; cfra++) {
          const int index_cfra = cfra - resample_start;

          BLI_assert(running_keyframe_read_offset >= 0 &&
                     running_keyframe_read_offset < total_allocated_values);
          BLI_assert(index_cfra >= 0 && index_cfra < fcurve->totvert);

          BezTriple *beztr = (fcurve->bezt + index_cfra);
          beztr->h1 = beztr->h2 = HD_AUTO_ANIM;
          beztr->ipo = BEZT_IPO_BEZ;
          /** TODO: What's convention for creating macros? Should I take it out of
           * ED_anim_api.h and put it in here? What about things that rely on the same file for
           * that define? Should I recreate the define in this file? Seems worst to copy what
           * Macro does, but I can ask that question in the patch.
           */
          // BEZKEYTYPE(&beztr) = keyframe_type;
          beztr->hide = BEZT_KEYTYPE_KEYFRAME;

          float y = resample_values_buffer[running_keyframe_read_offset];
          beztr->vec[0][0] = cfra - 1.0f;
          beztr->vec[0][1] = y;
          beztr->vec[1][0] = cfra;
          beztr->vec[1][1] = y;
          beztr->vec[2][0] = cfra + 1.0f;
          beztr->vec[2][1] = y;
          running_keyframe_read_offset++;
        }
      }
    }
  }

  for (FCurve *fcurve = resample_action->curves.first; fcurve; fcurve = fcurve->next) {
    calchandles_fcurve(fcurve);
  }

  /** Deselect all tracks and strips. */
  LISTBASE_FOREACH (NlaTrack *, nlt, &adt->nla_tracks) {
    nlt->flag &= ~(NLATRACK_SELECTED | NLATRACK_ACTIVE);
    LISTBASE_FOREACH (NlaStrip *, strip, &nlt->strips) {
      strip->flag &= ~(NLASTRIP_FLAG_SELECT | NLASTRIP_FLAG_ACTIVE);
    }
  }

  /**  Mute the prior selected strips that has been resampled. */
  LISTBASE_FOREACH (LinkData *, link_data, &selected_strips) {
    NlaStrip *strip = link_data->data;
    strip->flag |= NLASTRIP_FLAG_MUTED;
  }

  /** Select resampled tracka and strip. */
  resample_track->flag |= NLATRACK_ACTIVE | NLATRACK_SELECTED;
  resample_strip->flag |= NLASTRIP_FLAG_ACTIVE | NLASTRIP_FLAG_SELECT;

  BLI_freelistN(&selected_strips);
  BLI_freelistN(&evaluatable_tracks);

  /** This occurs per frame above. Comment left as reminder. */
  // BLI_freelistN(&upper_estrips);
  // BLI_freelistN(&lower_estrips);

  MEM_freeN(resample_values_buffer);

  nlaeval_free(eval_data);
  nlaeval_snapshot_free_data(lower_snapshot);
  /** Whole snapshot already freed when eval_data freed. Comment left as reminder. */
  // nlaeval_snapshot_free_data(whole_snapshot);

  BLI_ghash_free(nec_to_fcurve_array, NULL, MEM_freeN);
  BLI_ghash_free(groupname_to_group, NULL, NULL);

  return resample_track;
}

/* ***************************************** */
/* Overrides System - Public API */

/* Evaluate Overrides */
static void animsys_evaluate_overrides(PointerRNA *ptr, AnimData *adt)
{
  AnimOverride *aor;

  /* for each override, simply execute... */
  for (aor = adt->overrides.first; aor; aor = aor->next) {
    PathResolvedRNA anim_rna;
    if (BKE_animsys_store_rna_setting(ptr, aor->rna_path, aor->array_index, &anim_rna)) {
      BKE_animsys_write_rna_setting(&anim_rna, aor->value);
    }
  }
}

/* ***************************************** */
/* Evaluation System - Public API */

/* Overview of how this system works:
 * 1) Depsgraph sorts data as necessary, so that data is in an order that means
 *     that all dependencies are resolved before dependents.
 * 2) All normal animation is evaluated, so that drivers have some basis values to
 *    work with
 *    a.  NLA stacks are done first, as the Active Actions act as 'tweaking' tracks
 *        which modify the effects of the NLA-stacks
 *    b.  Active Action is evaluated as per normal, on top of the results of the NLA tracks
 *
 * --------------< often in a separate phase... >------------------
 *
 * 3) Drivers/expressions are evaluated on top of this, in an order where dependencies are
 *    resolved nicely.
 *    Note: it may be necessary to have some tools to handle the cases where some higher-level
 *          drivers are added and cause some problematic dependencies that
 *          didn't exist in the local levels...
 *
 * --------------< always executed >------------------
 *
 * Maintenance of editability of settings (XXX):
 * - In order to ensure that settings that are animated can still be manipulated in the UI
 * without requiring that keyframes are added to prevent these values from being overwritten,
 *   we use 'overrides'.
 *
 * Unresolved things:
 * - Handling of multi-user settings (i.e. time-offset, group-instancing) -> big cache grids
 *   or nodal system? but stored where?
 * - Multiple-block dependencies
 *   (i.e. drivers for settings are in both local and higher levels) -> split into separate
 * lists?
 *
 * Current Status:
 * - Currently (as of September 2009), overrides we haven't needed to (fully) implement
 * overrides. However, the code for this is relatively harmless, so is left in the code for
 * now.
 */

/* Evaluation loop for evaluation animation data
 *
 * This assumes that the animation-data provided belongs to the ID block in question,
 * and that the flags for which parts of the anim-data settings need to be recalculated
 * have been set already by the depsgraph. Now, we use the recalc
 */
void BKE_animsys_evaluate_animdata(ID *id,
                                   AnimData *adt,
                                   const AnimationEvalContext *anim_eval_context,
                                   eAnimData_Recalc recalc,
                                   const bool flush_to_original)
{
  PointerRNA id_ptr;

  /* sanity checks */
  if (ELEM(NULL, id, adt)) {
    return;
  }

  /* get pointer to ID-block for RNA to use */
  RNA_id_pointer_create(id, &id_ptr);

  /* recalculate keyframe data:
   * - NLA before Active Action, as Active Action behaves as 'tweaking track'
   *   that overrides 'rough' work in NLA
   */
  /* TODO: need to double check that this all works correctly */
  if (recalc & ADT_RECALC_ANIM) {
    /* evaluate NLA data */
    if ((adt->nla_tracks.first) && !(adt->flag & ADT_NLA_EVAL_OFF)) {
      /* evaluate NLA-stack
       * - active action is evaluated as part of the NLA stack as the last item
       */
      animsys_calculate_nla(&id_ptr, adt, anim_eval_context, flush_to_original);
    }
    /* evaluate Active Action only */
    else if (adt->action) {
      animsys_evaluate_action_ex(&id_ptr, adt->action, anim_eval_context, flush_to_original);
    }
  }

  /* recalculate drivers
   * - Drivers need to be evaluated afterwards, as they can either override
   *   or be layered on top of existing animation data.
   * - Drivers should be in the appropriate order to be evaluated without problems...
   */
  if (recalc & ADT_RECALC_DRIVERS) {
    animsys_evaluate_drivers(&id_ptr, adt, anim_eval_context);
  }

  /* always execute 'overrides'
   * - Overrides allow editing, by overwriting the value(s) set from animation-data, with the
   *   value last set by the user (and not keyframed yet).
   * - Overrides are cleared upon frame change and/or keyframing
   * - It is best that we execute this every time, so that no errors are likely to occur.
   */
  animsys_evaluate_overrides(&id_ptr, adt);
}

/* Evaluation of all ID-blocks with Animation Data blocks - Animation Data Only
 *
 * This will evaluate only the animation info available in the animation data-blocks
 * encountered. In order to enforce the system by which some settings controlled by a
 * 'local' (i.e. belonging in the nearest ID-block that setting is related to, not a
 * standard 'root') block are overridden by a larger 'user'
 */
void BKE_animsys_evaluate_all_animation(Main *main, Depsgraph *depsgraph, float ctime)
{
  ID *id;

  if (G.debug & G_DEBUG) {
    printf("Evaluate all animation - %f\n", ctime);
  }

  const bool flush_to_original = DEG_is_active(depsgraph);
  const AnimationEvalContext anim_eval_context = BKE_animsys_eval_context_construct(depsgraph,
                                                                                    ctime);

  /* macros for less typing
   * - only evaluate animation data for id if it has users (and not just fake ones)
   * - whether animdata exists is checked for by the evaluation function, though taking
   *   this outside of the function may make things slightly faster?
   */
#define EVAL_ANIM_IDS(first, aflag) \
  for (id = first; id; id = id->next) { \
    if (ID_REAL_USERS(id) > 0) { \
      AnimData *adt = BKE_animdata_from_id(id); \
      BKE_animsys_evaluate_animdata(id, adt, &anim_eval_context, aflag, flush_to_original); \
    } \
  } \
  (void)0

  /* another macro for the "embedded" nodetree cases
   * - this is like EVAL_ANIM_IDS, but this handles the case "embedded nodetrees"
   *   (i.e. scene/material/texture->nodetree) which we need a special exception
   *   for, otherwise they'd get skipped
   * - 'ntp' stands for "node tree parent" = data-block where node tree stuff resides
   */
#define EVAL_ANIM_NODETREE_IDS(first, NtId_Type, aflag) \
  for (id = first; id; id = id->next) { \
    if (ID_REAL_USERS(id) > 0) { \
      AnimData *adt = BKE_animdata_from_id(id); \
      NtId_Type *ntp = (NtId_Type *)id; \
      if (ntp->nodetree) { \
        AnimData *adt2 = BKE_animdata_from_id((ID *)ntp->nodetree); \
        BKE_animsys_evaluate_animdata( \
            &ntp->nodetree->id, adt2, &anim_eval_context, ADT_RECALC_ANIM, flush_to_original); \
      } \
      BKE_animsys_evaluate_animdata(id, adt, &anim_eval_context, aflag, flush_to_original); \
    } \
  } \
  (void)0

  /* optimization:
   * when there are no actions, don't go over database and loop over heaps of data-blocks,
   * which should ultimately be empty, since it is not possible for now to have any animation
   * without some actions, and drivers wouldn't get affected by any state changes
   *
   * however, if there are some curves, we will need to make sure that their 'ctime' property
   * gets set correctly, so this optimization must be skipped in that case...
   */
  if (BLI_listbase_is_empty(&main->actions) && BLI_listbase_is_empty(&main->curves)) {
    if (G.debug & G_DEBUG) {
      printf("\tNo Actions, so no animation needs to be evaluated...\n");
    }

    return;
  }

  /* nodes */
  EVAL_ANIM_IDS(main->nodetrees.first, ADT_RECALC_ANIM);

  /* textures */
  EVAL_ANIM_NODETREE_IDS(main->textures.first, Tex, ADT_RECALC_ANIM);

  /* lights */
  EVAL_ANIM_NODETREE_IDS(main->lights.first, Light, ADT_RECALC_ANIM);

  /* materials */
  EVAL_ANIM_NODETREE_IDS(main->materials.first, Material, ADT_RECALC_ANIM);

  /* cameras */
  EVAL_ANIM_IDS(main->cameras.first, ADT_RECALC_ANIM);

  /* shapekeys */
  EVAL_ANIM_IDS(main->shapekeys.first, ADT_RECALC_ANIM);

  /* metaballs */
  EVAL_ANIM_IDS(main->metaballs.first, ADT_RECALC_ANIM);

  /* curves */
  EVAL_ANIM_IDS(main->curves.first, ADT_RECALC_ANIM);

  /* armatures */
  EVAL_ANIM_IDS(main->armatures.first, ADT_RECALC_ANIM);

  /* lattices */
  EVAL_ANIM_IDS(main->lattices.first, ADT_RECALC_ANIM);

  /* meshes */
  EVAL_ANIM_IDS(main->meshes.first, ADT_RECALC_ANIM);

  /* particles */
  EVAL_ANIM_IDS(main->particles.first, ADT_RECALC_ANIM);

  /* speakers */
  EVAL_ANIM_IDS(main->speakers.first, ADT_RECALC_ANIM);

  /* movie clips */
  EVAL_ANIM_IDS(main->movieclips.first, ADT_RECALC_ANIM);

  /* linestyles */
  EVAL_ANIM_IDS(main->linestyles.first, ADT_RECALC_ANIM);

  /* grease pencil */
  EVAL_ANIM_IDS(main->gpencils.first, ADT_RECALC_ANIM);

  /* palettes */
  EVAL_ANIM_IDS(main->palettes.first, ADT_RECALC_ANIM);

  /* cache files */
  EVAL_ANIM_IDS(main->cachefiles.first, ADT_RECALC_ANIM);

  /* hairs */
  EVAL_ANIM_IDS(main->hairs.first, ADT_RECALC_ANIM);

  /* pointclouds */
  EVAL_ANIM_IDS(main->pointclouds.first, ADT_RECALC_ANIM);

  /* volumes */
  EVAL_ANIM_IDS(main->volumes.first, ADT_RECALC_ANIM);

  /* simulations */
  EVAL_ANIM_IDS(main->simulations.first, ADT_RECALC_ANIM);

  /* objects */
  /* ADT_RECALC_ANIM doesn't need to be supplied here, since object AnimData gets
   * this tagged by Depsgraph on framechange. This optimization means that objects
   * linked from other (not-visible) scenes will not need their data calculated.
   */
  EVAL_ANIM_IDS(main->objects.first, 0);

  /* masks */
  EVAL_ANIM_IDS(main->masks.first, ADT_RECALC_ANIM);

  /* worlds */
  EVAL_ANIM_NODETREE_IDS(main->worlds.first, World, ADT_RECALC_ANIM);

  /* scenes */
  EVAL_ANIM_NODETREE_IDS(main->scenes.first, Scene, ADT_RECALC_ANIM);
}

/* ***************************************** */

/* ************** */
/* Evaluation API */

void BKE_animsys_eval_animdata(Depsgraph *depsgraph, ID *id)
{
  float ctime = DEG_get_ctime(depsgraph);
  AnimData *adt = BKE_animdata_from_id(id);
  /* XXX: this is only needed for flushing RNA updates,
   * which should get handled as part of the dependency graph instead. */
  DEG_debug_print_eval_time(depsgraph, __func__, id->name, id, ctime);
  const bool flush_to_original = DEG_is_active(depsgraph);

  const AnimationEvalContext anim_eval_context = BKE_animsys_eval_context_construct(depsgraph,
                                                                                    ctime);
  BKE_animsys_evaluate_animdata(id, adt, &anim_eval_context, ADT_RECALC_ANIM, flush_to_original);
}

void BKE_animsys_update_driver_array(ID *id)
{
  AnimData *adt = BKE_animdata_from_id(id);

  /* Runtime driver map to avoid O(n^2) lookups in BKE_animsys_eval_driver.
   * Ideally the depsgraph could pass a pointer to the COW driver directly,
   * but this is difficult in the current design. */
  if (adt && adt->drivers.first) {
    BLI_assert(!adt->driver_array);

    int num_drivers = BLI_listbase_count(&adt->drivers);
    adt->driver_array = MEM_mallocN(sizeof(FCurve *) * num_drivers, "adt->driver_array");

    int driver_index = 0;
    LISTBASE_FOREACH (FCurve *, fcu, &adt->drivers) {
      adt->driver_array[driver_index++] = fcu;
    }
  }
}

void BKE_animsys_eval_driver(Depsgraph *depsgraph, ID *id, int driver_index, FCurve *fcu_orig)
{
  BLI_assert(fcu_orig != NULL);

  /* TODO(sergey): De-duplicate with BKE animsys. */
  PointerRNA id_ptr;
  bool ok = false;

  /* Lookup driver, accelerated with driver array map. */
  const AnimData *adt = BKE_animdata_from_id(id);
  FCurve *fcu;

  if (adt->driver_array) {
    fcu = adt->driver_array[driver_index];
  }
  else {
    fcu = BLI_findlink(&adt->drivers, driver_index);
  }

  DEG_debug_print_eval_subdata_index(
      depsgraph, __func__, id->name, id, "fcu", fcu->rna_path, fcu, fcu->array_index);

  RNA_id_pointer_create(id, &id_ptr);

  /* check if this driver's curve should be skipped */
  if ((fcu->flag & (FCURVE_MUTED | FCURVE_DISABLED)) == 0) {
    /* check if driver itself is tagged for recalculation */
    /* XXX driver recalc flag is not set yet by depsgraph! */
    ChannelDriver *driver_orig = fcu_orig->driver;
    if ((driver_orig) && !(driver_orig->flag & DRIVER_FLAG_INVALID)) {
      /* evaluate this using values set already in other places
       * NOTE: for 'layering' option later on, we should check if we should remove old value
       * before adding new to only be done when drivers only changed */
      // printf("\told val = %f\n", fcu->curval);

      PathResolvedRNA anim_rna;
      if (BKE_animsys_store_rna_setting(&id_ptr, fcu->rna_path, fcu->array_index, &anim_rna)) {
        /* Evaluate driver, and write results to COW-domain destination */
        const float ctime = DEG_get_ctime(depsgraph);
        const AnimationEvalContext anim_eval_context = BKE_animsys_eval_context_construct(
            depsgraph, ctime);
        const float curval = calculate_fcurve(&anim_rna, fcu, &anim_eval_context);
        ok = BKE_animsys_write_rna_setting(&anim_rna, curval);

        /* Flush results & status codes to original data for UI (T59984) */
        if (ok && DEG_is_active(depsgraph)) {
          animsys_write_orig_anim_rna(&id_ptr, fcu->rna_path, fcu->array_index, curval);

          /* curval is displayed in the UI, and flag contains error-status codes */
          fcu_orig->curval = fcu->curval;
          driver_orig->curval = fcu->driver->curval;
          driver_orig->flag = fcu->driver->flag;

          DriverVar *dvar_orig = driver_orig->variables.first;
          DriverVar *dvar = fcu->driver->variables.first;
          for (; dvar_orig && dvar; dvar_orig = dvar_orig->next, dvar = dvar->next) {
            DriverTarget *dtar_orig = &dvar_orig->targets[0];
            DriverTarget *dtar = &dvar->targets[0];
            for (int i = 0; i < MAX_DRIVER_TARGETS; i++, dtar_orig++, dtar++) {
              dtar_orig->flag = dtar->flag;
            }

            dvar_orig->curval = dvar->curval;
            dvar_orig->flag = dvar->flag;
          }
        }
      }

      /* set error-flag if evaluation failed */
      if (ok == 0) {
        CLOG_WARN(&LOG, "invalid driver - %s[%d]", fcu->rna_path, fcu->array_index);
        driver_orig->flag |= DRIVER_FLAG_INVALID;
      }
    }
  }
}
