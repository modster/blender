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
 * \ingroup wm
 *
 * \name Window-Manager XR Actions
 *
 * Uses the Ghost-XR API to manage OpenXR actions.
 * All functions are designed to be usable by RNA / the Python API.
 */

#include "BLI_ghash.h"
#include "BLI_math.h"

#include "GHOST_C-api.h"

#include "MEM_guardedalloc.h"

#include "WM_api.h"
#include "WM_types.h"

#include "wm_xr_intern.h"

/* -------------------------------------------------------------------- */
/** \name XR-Action API
 *
 * API functions for managing OpenXR actions.
 *
 * \{ */

static wmXrActionSet *action_set_find(wmXrData *xr, const char *action_set_name)
{
  GHash *action_sets = xr->runtime->session_state.action_sets;
  return action_sets ? BLI_ghash_lookup(action_sets, action_set_name) : NULL;
}

static wmXrActionSet *action_set_create(const char *action_set_name)
{
  wmXrActionSet *action_set = MEM_callocN(sizeof(wmXrActionSet), __func__);
  action_set->name = MEM_mallocN(strlen(action_set_name) + 1, __func__);
  strcpy(action_set->name, action_set_name);

  return action_set;
}

static void action_set_destroy(void *val)
{
  wmXrActionSet *action_set = val;

  if (action_set->name) {
    MEM_freeN(action_set->name);
  }

  MEM_freeN(action_set);
}

static wmXrAction *action_find(wmXrActionSet *action_set, const char *action_name)
{
  GHash *actions = action_set->actions;
  return actions ? BLI_ghash_lookup(actions, action_name) : NULL;
}

static wmXrAction *action_create(const char *action_name,
                                 eXrActionType type,
                                 unsigned int count_subaction_paths,
                                 const char **subaction_paths,
                                 float threshold,
                                 wmOperatorType *ot,
                                 IDProperty *op_properties,
                                 eXrOpFlag op_flag)
{
  wmXrAction *action = MEM_callocN(sizeof(wmXrAction), __func__);
  action->name = MEM_mallocN(strlen(action_name) + 1, __func__);
  strcpy(action->name, action_name);
  action->type = type;

  const unsigned int count = count_subaction_paths;
  action->count_subaction_paths = count;

  action->subaction_paths = MEM_mallocN(sizeof(char *) * count, __func__);
  for (unsigned int i = 0; i < count; ++i) {
    action->subaction_paths[i] = MEM_mallocN(strlen(subaction_paths[i]) + 1, __func__);
    strcpy(action->subaction_paths[i], subaction_paths[i]);
  }

  size_t size;
  switch (type) {
    case XR_BOOLEAN_INPUT:
      size = sizeof(bool);
      break;
    case XR_FLOAT_INPUT:
      size = sizeof(float);
      break;
    case XR_VECTOR2F_INPUT:
      size = sizeof(float[2]);
      break;
    case XR_POSE_INPUT:
      size = sizeof(GHOST_XrPose);
      break;
    case XR_VIBRATION_OUTPUT:
      return action;
  }
  action->states = MEM_calloc_arrayN(count, size, __func__);
  action->states_prev = MEM_calloc_arrayN(count, size, __func__);

  action->threshold = threshold;
  CLAMP(action->threshold, 0.0f, 1.0f);

  action->ot = ot;
  action->op_properties = op_properties;
  action->op_flag = op_flag;

  return action;
}

static void action_destroy(void *val)
{
  wmXrAction *action = val;

  if (action->name) {
    MEM_freeN(action->name);
  }

  const unsigned int count = action->count_subaction_paths;
  char **subaction_paths = action->subaction_paths;
  if (subaction_paths) {
    for (unsigned int i = 0; i < count; ++i) {
      if (subaction_paths[i]) {
        MEM_freeN(subaction_paths[i]);
      }
    }
    MEM_freeN(subaction_paths);
  }

  if (action->states) {
    MEM_freeN(action->states);
  }
  if (action->states_prev) {
    MEM_freeN(action->states_prev);
  }

  MEM_freeN(action);
}

bool WM_xr_action_set_create(wmXrData *xr, const char *action_set_name)
{
  if (action_set_find(xr, action_set_name)) {
    return false;
  }

  if (!GHOST_XrCreateActionSet(xr->runtime->context, action_set_name)) {
    return false;
  }

  GHash *action_sets = xr->runtime->session_state.action_sets;
  if (!action_sets) {
    action_sets = xr->runtime->session_state.action_sets = BLI_ghash_str_new(__func__);
  }

  wmXrActionSet *action_set = action_set_create(action_set_name);
  BLI_ghash_insert(
      action_sets,
      action_set->name,
      action_set); /* Important to use action_set->name, since only a pointer is stored. */

  return true;
}

void WM_xr_action_set_destroy(wmXrData *xr, const char *action_set_name, bool remove_reference)
{
  GHOST_XrContextHandle context = xr->runtime->context;
  if (context && GHOST_XrSessionIsRunning(context)) {
    GHOST_XrDestroyActionSet(context, action_set_name);
  }

  wmXrSessionState *session_state = &xr->runtime->session_state;
  GHash *action_sets = session_state->action_sets;
  wmXrActionSet *action_set = BLI_ghash_lookup(action_sets, action_set_name);
  if (!action_set) {
    return;
  }

  if (action_set == session_state->active_action_set) {
    if (action_set->controller_pose_action) {
      wm_xr_session_controller_data_clear(&xr->runtime->session_state);
      action_set->controller_pose_action = NULL;
    }
    if (action_set->active_modal_action) {
      action_set->active_modal_action = NULL;
    }
    session_state->active_action_set = NULL;
  }

  if (action_set->actions) {
    BLI_ghash_free(action_set->actions, NULL, action_destroy);
  }

  if (remove_reference) {
    BLI_ghash_remove(action_sets, action_set_name, NULL, action_set_destroy);
  }
  else {
    action_set_destroy(action_set);
  }
}

bool WM_xr_action_create(wmXrData *xr,
                         const char *action_set_name,
                         const char *action_name,
                         eXrActionType type,
                         unsigned int count_subaction_paths,
                         const char **subaction_paths,
                         float threshold,
                         wmOperatorType *ot,
                         IDProperty *op_properties,
                         eXrOpFlag op_flag)
{
  wmXrActionSet *action_set = action_set_find(xr, action_set_name);
  if (!action_set) {
    return false;
  }

  GHOST_XrActionInfo info = {
      .name = action_name,
      .count_subaction_paths = count_subaction_paths,
      .subaction_paths = subaction_paths,
  };

  switch (type) {
    case XR_BOOLEAN_INPUT:
      info.type = GHOST_kXrActionTypeBooleanInput;
      break;
    case XR_FLOAT_INPUT:
      info.type = GHOST_kXrActionTypeFloatInput;
      break;
    case XR_VECTOR2F_INPUT:
      info.type = GHOST_kXrActionTypeVector2fInput;
      break;
    case XR_POSE_INPUT:
      info.type = GHOST_kXrActionTypePoseInput;
      break;
    case XR_VIBRATION_OUTPUT:
      info.type = GHOST_kXrActionTypeVibrationOutput;
      break;
  }

  if (!GHOST_XrCreateActions(xr->runtime->context, action_set_name, 1, &info)) {
    return false;
  }

  GHash *actions = action_set->actions;
  if (!actions) {
    actions = action_set->actions = BLI_ghash_str_new(__func__);
  }

  if (!action_find(action_set, action_name)) {
    wmXrAction *action = action_create(action_name,
                                       type,
                                       count_subaction_paths,
                                       subaction_paths,
                                       threshold,
                                       ot,
                                       op_properties,
                                       op_flag);
    if (action) {
      BLI_ghash_insert(
          actions,
          action->name,
          action); /* Important to use action->name, since only a pointer is stored. */
    }
  }

  return true;
}

void WM_xr_action_destroy(wmXrData *xr, const char *action_set_name, const char *action_name)
{
  wmXrActionSet *action_set = action_set_find(xr, action_set_name);
  if (!action_set) {
    return;
  }

  GHOST_XrDestroyActions(xr->runtime->context, action_set_name, 1, &action_name);

  GHash *actions = action_set->actions;
  char controller_pose_name[64];
  char active_modal_name[64];

  /* Save names of controller pose and active modal actions in case they are removed from the
   * GHash. */
  if (action_set->controller_pose_action &&
      STREQ(action_set->controller_pose_action->name, action_name)) {
    strcpy(controller_pose_name, action_set->controller_pose_action->name);
  }
  else {
    controller_pose_name[0] = '\0';
  }
  if (action_set->active_modal_action &&
      STREQ(action_set->active_modal_action->name, action_name)) {
    strcpy(active_modal_name, action_set->active_modal_action->name);
  }
  else {
    active_modal_name[0] = '\0';
  }

  BLI_ghash_remove(actions, action_name, NULL, action_destroy);

  if (controller_pose_name[0] != '\0') {
    if (action_set == xr->runtime->session_state.active_action_set) {
      wm_xr_session_controller_data_clear(&xr->runtime->session_state);
    }
    action_set->controller_pose_action = NULL;
  }
  if (active_modal_name[0] != '\0') {
    action_set->active_modal_action = NULL;
  }
}

bool WM_xr_action_space_create(wmXrData *xr,
                               const char *action_set_name,
                               const char *action_name,
                               unsigned int count_subaction_paths,
                               const char **subaction_paths,
                               const float (*poses)[7])
{
  GHOST_XrActionSpaceInfo info = {
      .action_name = action_name,
      .count_subaction_paths = count_subaction_paths,
      .subaction_paths = subaction_paths,
      .poses = (const GHOST_XrPose *)poses,
  };
  BLI_STATIC_ASSERT(sizeof(*info.poses) == sizeof(*poses), "GHOST_XrPose size mismatch.");

  return GHOST_XrCreateActionSpaces(xr->runtime->context, action_set_name, 1, &info) ? true :
                                                                                       false;
}

void WM_xr_action_space_destroy(wmXrData *xr,
                                const char *action_set_name,
                                const char *action_name,
                                unsigned int count_subaction_paths,
                                const char **subaction_paths)
{
  GHOST_XrActionSpaceInfo info = {
      .action_name = action_name,
      .count_subaction_paths = count_subaction_paths,
      .subaction_paths = subaction_paths,
  };

  GHOST_XrDestroyActionSpaces(xr->runtime->context, action_set_name, 1, &info);
}

bool WM_xr_action_binding_create(wmXrData *xr,
                                 const char *action_set_name,
                                 const char *interaction_profile_path,
                                 const char *action_name,
                                 unsigned int count_interaction_paths,
                                 const char **interaction_paths)
{
  GHOST_XrActionBinding binding = {
      .action_name = action_name,
      .count_interaction_paths = count_interaction_paths,
      .interaction_paths = interaction_paths,
  };

  GHOST_XrActionBindingsInfo info = {
      .interaction_profile_path = interaction_profile_path,
      .count_bindings = 1,
      .bindings = &binding,
  };

  return GHOST_XrCreateActionBindings(xr->runtime->context, action_set_name, 1, &info);
}

void WM_xr_action_binding_destroy(wmXrData *xr,
                                  const char *action_set_name,
                                  const char *interaction_profile_path,
                                  const char *action_name,
                                  unsigned int count_interaction_paths,
                                  const char **interaction_paths)
{
  GHOST_XrActionBinding binding = {
      .action_name = action_name,
      .count_interaction_paths = count_interaction_paths,
      .interaction_paths = interaction_paths,
  };

  GHOST_XrActionBindingsInfo info = {
      .interaction_profile_path = interaction_profile_path,
      .count_bindings = 1,
      .bindings = &binding,
  };

  GHOST_XrDestroyActionBindings(xr->runtime->context, action_set_name, 1, &info);
}

bool WM_xr_active_action_set_set(wmXrData *xr, const char *action_set_name)
{
  wmXrActionSet *action_set = action_set_find(xr, action_set_name);
  if (!action_set) {
    return false;
  }

  {
    /* Unset active modal action (if any). */
    wmXrActionSet *active_action_set = xr->runtime->session_state.active_action_set;
    if (active_action_set) {
      wmXrAction *active_modal_action = active_action_set->active_modal_action;
      if (active_modal_action) {
        if (active_modal_action->active_modal_path) {
          active_modal_action->active_modal_path = NULL;
        }
        active_action_set->active_modal_action = NULL;
      }
    }
  }

  xr->runtime->session_state.active_action_set = action_set;

  if (action_set->controller_pose_action) {
    wm_xr_session_controller_data_populate(action_set->controller_pose_action, xr);
  }

  return true;
}

bool WM_xr_controller_pose_action_set(wmXrData *xr,
                                      const char *action_set_name,
                                      const char *action_name)
{
  wmXrActionSet *action_set = action_set_find(xr, action_set_name);
  if (!action_set) {
    return false;
  }

  wmXrAction *action = action_find(action_set, action_name);
  if (!action) {
    return false;
  }

  action_set->controller_pose_action = action;

  if (action_set == xr->runtime->session_state.active_action_set) {
    wm_xr_session_controller_data_populate(action, xr);
  }

  return true;
}

bool WM_xr_action_state_get(const wmXrData *xr,
                            const char *action_set_name,
                            const char *action_name,
                            eXrActionType type,
                            const char *subaction_path,
                            void *r_state)
{
  const wmXrActionSet *action_set = action_set_find((wmXrData *)xr, action_set_name);
  if (!action_set) {
    return false;
  }

  const wmXrAction *action = action_find((wmXrActionSet *)action_set, action_name);
  if (!action) {
    return false;
  }

  BLI_assert(action->type == type);

  /* Find the action state corresponding to the subaction path. */
  for (unsigned int i = 0; i < action->count_subaction_paths; ++i) {
    if (STREQ(subaction_path, action->subaction_paths[i])) {
      switch (type) {
        case XR_BOOLEAN_INPUT:
          *(bool *)r_state = ((bool *)action->states)[i];
          break;
        case XR_FLOAT_INPUT:
          *(float *)r_state = ((float *)action->states)[i];
          break;
        case XR_VECTOR2F_INPUT:
          copy_v2_v2(((float(*)[2])r_state)[0], ((float(*)[2])action->states)[i]);
          break;
        case XR_POSE_INPUT: {
          /* Safety check, since r_state may be float[7] instead of GHOST_XrPose *. */
          BLI_STATIC_ASSERT(sizeof(GHOST_XrPose) == sizeof(float[7]),
                            "GHOST_XrPose size mismatch.");
          memcpy(
              (GHOST_XrPose *)r_state, &((GHOST_XrPose *)action->states)[i], sizeof(GHOST_XrPose));
          break;
        }
        case XR_VIBRATION_OUTPUT:
          break;
      }
      return true;
    }
  }

  return false;
}

bool WM_xr_haptic_action_apply(wmXrData *xr,
                               const char *action_set_name,
                               const char *action_name,
                               unsigned int count,
                               const char *const *subaction_paths,
                               const long long *duration,
                               const float *frequency,
                               const float *amplitude)
{
  return GHOST_XrApplyHapticAction(xr->runtime->context,
                                   action_set_name,
                                   action_name,
                                   count,
                                   subaction_paths,
                                   duration,
                                   frequency,
                                   amplitude) ?
             true :
             false;
}

void WM_xr_haptic_action_stop(wmXrData *xr,
                              const char *action_set_name,
                              const char *action_name,
                              unsigned int count,
                              const char *const *subaction_paths)
{
  GHOST_XrStopHapticAction(
      xr->runtime->context, action_set_name, action_name, count, subaction_paths);
}

/** \} */ /* XR-Action API */
