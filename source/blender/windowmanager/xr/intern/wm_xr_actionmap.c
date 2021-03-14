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
 * \name Window-Manager XR Action Maps
 *
 * XR actionmap API, similar to WM keymap API.
 */

#include <math.h>
#include <string.h>

#include "BKE_context.h"
#include "BKE_global.h"
#include "BKE_idprop.h"

#include "BLI_listbase.h"
#include "BLI_string.h"

#include "MEM_guardedalloc.h"

#include "WM_api.h"
#include "WM_types.h"

#define XR_ACTIONCONF_STR_DEFAULT "blender"
#define XR_ACTIONMAP_STR_DEFAULT "actionmap"
#define XR_AMI_STR_DEFAULT "action"

/* Replacement for U.keyconfigstr for actionconfigs. */
static char g_xr_actionconfigstr[64];

/* Actionconfig update flag. */
enum {
  WM_XR_ACTIONCONF_UPDATE_ACTIVE = (1 << 0), /* Update active actionconfig. */
  WM_XR_ACTIONCONF_UPDATE_ALL = (1 << 1),    /* Update all actionconfigs. */
  WM_XR_ACTIONCONF_UPDATE_ENSURE =
      (1 << 2), /* Skip actionmap/item flag checks when updating actionconfigs. */
};

static char g_xr_actionconfig_update_flag = 0;

/* -------------------------------------------------------------------- */

void WM_xr_actionconfig_update_tag(XrActionMap *actionmap, XrActionMapItem *ami)
{
  g_xr_actionconfig_update_flag |= WM_XR_ACTIONCONF_UPDATE_ACTIVE;

  if (actionmap) {
    actionmap->flag |= XR_ACTIONMAP_UPDATE;
  }
  if (ami) {
    ami->flag |= XR_AMI_UPDATE;
  }
}

#if 0 /* Currently unused. */
static void wm_xr_actionconfig_update_active(XrActionMap *actionmap, XrActionMapItem *ami, bool ensure)
{
  WM_xr_actionconfig_update_tag(actionmap, ami);
  if (ensure) {
    g_xr_actionconfig_update_flag |= WM_XR_ACTIONCONF_UPDATE_ENSURE;
  }
}

static void wm_xr_actionconfig_update_all(bool ensure)
{
  g_xr_actionconfig_update_flag |= WM_XR_ACTIONCONF_UPDATE_ALL;
  if (ensure) {
    g_xr_actionconfig_update_flag |= WM_XR_ACTIONCONF_UPDATE_ENSURE;
  }
}
#endif

/* -------------------------------------------------------------------- */
/** \name Actionmap Item
 *
 * Item in a XR actionmap, that maps an XR event to an operator, pose, or haptic output.
 * \{ */

static void wm_xr_actionmap_item_properties_set(XrActionMapItem *ami)
{
  WM_operator_properties_alloc(&(ami->op_properties_ptr), &(ami->op_properties), ami->op);
  WM_operator_properties_sanitize(ami->op_properties_ptr, 1);
}

/**
 * Similar to #wm_xr_actionmap_item_properties_set()
 * but checks for the #eXrActionType and #wmOperatorType having changed.
 */
void WM_xr_actionmap_item_properties_update_ot(XrActionMapItem *ami)
{
  switch (ami->type) {
    case XR_BOOLEAN_INPUT:
    case XR_FLOAT_INPUT:
    case XR_VECTOR2F_INPUT:
      break;
    case XR_POSE_INPUT:
    case XR_VIBRATION_OUTPUT:
      WM_xr_actionmap_item_properties_free(ami);
      memset(ami->op, 0, sizeof(ami->op));
      return;
  }

  if (ami->op[0] == 0) {
    WM_xr_actionmap_item_properties_free(ami);
    return;
  }

  if (ami->op_properties_ptr == NULL) {
    wm_xr_actionmap_item_properties_set(ami);
  }
  else {
    wmOperatorType *ot = WM_operatortype_find(ami->op, 0);
    if (ot) {
      if (ot->srna != ami->op_properties_ptr->type) {
        /* Matches wm_xr_actionmap_item_properties_set() but doesn't alloc new ptr. */
        WM_operator_properties_create_ptr(ami->op_properties_ptr, ot);
        if (ami->op_properties) {
          ami->op_properties_ptr->data = ami->op_properties;
        }
        WM_operator_properties_sanitize(ami->op_properties_ptr, 1);
      }
    }
    else {
      WM_xr_actionmap_item_properties_free(ami);
    }
  }
}

static void wm_xr_actionmap_item_properties_update_ot_from_list(ListBase *am_lb, bool ensure)
{
  if (ensure) {
    LISTBASE_FOREACH (XrActionMap *, am, am_lb) {
      LISTBASE_FOREACH (XrActionMapItem *, ami, &am->items) {
        WM_xr_actionmap_item_properties_update_ot(ami);
        ami->flag &= ~XR_AMI_UPDATE;
      }
      am->flag &= ~XR_ACTIONMAP_UPDATE;
    }
  }
  else {
    LISTBASE_FOREACH (XrActionMap *, am, am_lb) {
      if ((am->flag & XR_ACTIONMAP_UPDATE) != 0) {
        LISTBASE_FOREACH (XrActionMapItem *, ami, &am->items) {
          if ((ami->flag & XR_AMI_UPDATE) != 0) {
            WM_xr_actionmap_item_properties_update_ot(ami);
            ami->flag &= ~XR_AMI_UPDATE;
          }
        }
        am->flag &= ~XR_ACTIONMAP_UPDATE;
      }
    }
  }
}

XrActionMapItem *WM_xr_actionmap_item_new(XrActionMap *actionmap,
                                          const char *idname,
                                          bool replace_existing)
{
  XrActionMapItem *ami_prev = WM_xr_actionmap_item_list_find(&actionmap->items, idname);
  if (ami_prev && replace_existing) {
    WM_xr_actionmap_item_properties_free(ami_prev);
    return ami_prev;
  }

  /* Create new item. */
  XrActionMapItem *ami = MEM_callocN(sizeof(XrActionMapItem), __func__);
  BLI_strncpy(ami->idname, idname, XR_AMI_MAX_NAME);
  if (ami_prev) {
    WM_xr_actionmap_item_ensure_unique(actionmap, ami);
  }

  BLI_addtail(&actionmap->items, ami);

  /* Set type to float (button) input by default. */
  ami->type = XR_FLOAT_INPUT;
  ami->threshold = 0.3f;

  WM_xr_actionconfig_update_tag(actionmap, ami);

  return ami;
}

static XrActionMapItem *wm_xr_actionmap_item_list_find_except(ListBase *lb,
                                                              const char *idname,
                                                              XrActionMapItem *amiexcept)
{
  LISTBASE_FOREACH (XrActionMapItem *, ami, lb) {
    if (STREQLEN(idname, ami->idname, XR_AMI_MAX_NAME) && (ami != amiexcept)) {
      return ami;
    }
  }

  return NULL;
}

void WM_xr_actionmap_item_ensure_unique(XrActionMap *actionmap, XrActionMapItem *ami)
{
  /* Ensure unique name. */
  char name[XR_AMI_MAX_NAME];
  char *suffix;
  size_t baselen;
  size_t idx = 0;

  BLI_strncpy(name, ami->idname, XR_AMI_MAX_NAME);
  baselen = BLI_strnlen(name, XR_AMI_MAX_NAME);
  suffix = &name[baselen];

  while (wm_xr_actionmap_item_list_find_except(&actionmap->items, name, ami)) {
    if ((baselen + 1) + (log10(++idx) + 1) > XR_AMI_MAX_NAME) {
      /* Use default base name. */
      BLI_strncpy(name, XR_AMI_STR_DEFAULT, XR_AMI_MAX_NAME);
      baselen = BLI_strnlen(name, XR_AMI_MAX_NAME);
      suffix = &name[baselen];
      idx = 0;
    }
    else {
      BLI_snprintf(suffix, XR_AMI_MAX_NAME, "%i", idx);
    }
  }

  BLI_strncpy(ami->idname, name, XR_AMI_MAX_NAME);
}

static XrActionMapItem *wm_xr_actionmap_item_copy(XrActionMapItem *ami)
{
  XrActionMapItem *amin = MEM_dupallocN(ami);

  amin->prev = amin->next = NULL;
  amin->flag &= ~XR_AMI_UPDATE;

  if (amin->op_properties) {
    amin->op_properties_ptr = MEM_callocN(sizeof(PointerRNA), "wmOpItemPtr");
    WM_operator_properties_create(amin->op_properties_ptr, amin->op);

    amin->op_properties = IDP_CopyProperty(amin->op_properties);
    amin->op_properties_ptr->data = amin->op_properties;
  }
  else {
    amin->op_properties = NULL;
    amin->op_properties_ptr = NULL;
  }

  return amin;
}

XrActionMapItem *WM_xr_actionmap_item_add_copy(XrActionMap *actionmap, XrActionMapItem *ami_src)
{
  XrActionMapItem *ami_dst = wm_xr_actionmap_item_copy(ami_src);

  WM_xr_actionmap_item_ensure_unique(actionmap, ami_dst);

  BLI_addtail(&actionmap->items, ami_dst);

  WM_xr_actionconfig_update_tag(actionmap, ami_dst);

  return ami_dst;
}

bool WM_xr_actionmap_item_remove(XrActionMap *actionmap, XrActionMapItem *ami)
{
  int idx = BLI_findindex(&actionmap->items, ami);

  if (idx != -1) {
    if (ami->op_properties_ptr) {
      WM_operator_properties_free(ami->op_properties_ptr);
      MEM_freeN(ami->op_properties_ptr);
    }
    BLI_freelinkN(&actionmap->items, ami);

    if (idx <= actionmap->selitem) {
      if (--actionmap->selitem < 0) {
        actionmap->selitem = 0;
      }
    }

    WM_xr_actionconfig_update_tag(actionmap, NULL);

    return true;
  }
  return false;
}

XrActionMapItem *WM_xr_actionmap_item_list_find(ListBase *lb, const char *idname)
{
  LISTBASE_FOREACH (XrActionMapItem *, ami, lb) {
    if (STREQLEN(idname, ami->idname, XR_AMI_MAX_NAME)) {
      return ami;
    }
  }

  return NULL;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Actionmap
 *
 * List of XR actionmap items.
 * \{ */

XrActionMap *WM_xr_actionmap_new(XrActionConfig *actionconf,
                                 const char *idname,
                                 bool replace_existing)
{
  XrActionMap *am_prev = WM_xr_actionmap_list_find(&actionconf->actionmaps, idname);
  if (am_prev && replace_existing) {
    WM_xr_actionmap_clear(am_prev);
    return am_prev;
  }

  /* Create new actionmap. */
  XrActionMap *am = MEM_callocN(sizeof(struct XrActionMap), __func__);
  BLI_strncpy(am->idname, idname, XR_ACTIONMAP_MAX_NAME);
  if (am_prev) {
    WM_xr_actionmap_ensure_unique(actionconf, am);
  }

  BLI_addtail(&actionconf->actionmaps, am);

  WM_xr_actionconfig_update_tag(am, NULL);

  return am;
}

static XrActionMap *wm_xr_actionmap_list_find_except(ListBase *lb,
                                                     const char *idname,
                                                     XrActionMap *am_except)
{
  LISTBASE_FOREACH (XrActionMap *, am, lb) {
    if (STREQLEN(idname, am->idname, XR_ACTIONMAP_MAX_NAME) && (am != am_except)) {
      return am;
    }
  }

  return NULL;
}

void WM_xr_actionmap_ensure_unique(XrActionConfig *actionconf, XrActionMap *actionmap)
{
  /* Ensure unique name. */
  char name[XR_ACTIONMAP_MAX_NAME];
  char *suffix;
  size_t baselen;
  size_t idx = 0;

  BLI_strncpy(name, actionmap->idname, XR_ACTIONMAP_MAX_NAME);
  baselen = BLI_strnlen(name, XR_ACTIONMAP_MAX_NAME);
  suffix = &name[baselen];

  while (wm_xr_actionmap_list_find_except(&actionconf->actionmaps, name, actionmap)) {
    if ((baselen + 1) + (log10(++idx) + 1) > XR_ACTIONMAP_MAX_NAME) {
      /* Use default base name. */
      BLI_strncpy(name, XR_ACTIONMAP_STR_DEFAULT, XR_ACTIONMAP_MAX_NAME);
      baselen = BLI_strnlen(name, XR_ACTIONMAP_MAX_NAME);
      suffix = &name[baselen];
      idx = 0;
    }
    else {
      BLI_snprintf(suffix, XR_ACTIONMAP_MAX_NAME, "%i", idx);
    }
  }

  BLI_strncpy(actionmap->idname, name, XR_ACTIONMAP_MAX_NAME);
}

static XrActionMap *wm_xr_actionmap_copy(XrActionMap *am_src)
{
  XrActionMap *am_dst = MEM_dupallocN(am_src);

  BLI_listbase_clear(&am_dst->items);
  am_dst->flag &= ~(XR_ACTIONMAP_UPDATE);

  LISTBASE_FOREACH (XrActionMapItem *, ami, &am_src->items) {
    XrActionMapItem *ami_new = wm_xr_actionmap_item_copy(ami);
    BLI_addtail(&am_dst->items, ami_new);
  }

  return am_dst;
}

XrActionMap *WM_xr_actionmap_add_copy(XrActionConfig *actionconf, XrActionMap *am_src)
{
  XrActionMap *am_dst = wm_xr_actionmap_copy(am_src);

  WM_xr_actionmap_ensure_unique(actionconf, am_dst);

  BLI_addtail(&actionconf->actionmaps, am_dst);

  WM_xr_actionconfig_update_tag(am_dst, NULL);

  return am_dst;
}

bool WM_xr_actionmap_remove(XrActionConfig *actionconf, XrActionMap *actionmap)
{
  int idx = BLI_findindex(&actionconf->actionmaps, actionmap);

  if (idx != -1) {
    WM_xr_actionmap_clear(actionmap);
    BLI_remlink(&actionconf->actionmaps, actionmap);
    MEM_freeN(actionmap);

    if (idx <= actionconf->actactionmap) {
      if (--actionconf->actactionmap < 0) {
        actionconf->actactionmap = 0;
      }
    }
    if (idx <= actionconf->selactionmap) {
      if (--actionconf->selactionmap < 0) {
        actionconf->selactionmap = 0;
      }
    }

    return true;
  }
  return false;
}

XrActionMap *WM_xr_actionmap_list_find(ListBase *lb, const char *idname)
{
  LISTBASE_FOREACH (XrActionMap *, am, lb) {
    if (STREQLEN(idname, am->idname, XR_ACTIONMAP_MAX_NAME)) {
      return am;
    }
  }

  return NULL;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Action Configuration
 *
 * List of XR actionmaps.
 * \{ */

XrActionConfig *WM_xr_actionconfig_new(XrSessionSettings *settings,
                                       const char *idname,
                                       bool user_defined)
{
  XrActionConfig *actionconf = BLI_findstring(
      &settings->actionconfigs, idname, offsetof(XrActionConfig, idname));
  if (actionconf) {
    WM_xr_actionconfig_clear(actionconf);
    return actionconf;
  }

  /* Create new configuration. */
  actionconf = MEM_callocN(sizeof(XrActionConfig), __func__);
  BLI_strncpy(actionconf->idname, idname, sizeof(actionconf->idname));
  BLI_addtail(&settings->actionconfigs, actionconf);

  if (user_defined) {
    actionconf->flag |= XR_ACTIONCONF_USER;
  }

  return actionconf;
}

bool WM_xr_actionconfig_remove(XrSessionSettings *settings, XrActionConfig *actionconf)
{
  if (BLI_findindex(&settings->actionconfigs, actionconf) != -1) {
    if (STREQLEN(g_xr_actionconfigstr, actionconf->idname, sizeof(g_xr_actionconfigstr))) {
      BLI_strncpy(
          g_xr_actionconfigstr, settings->defaultconf->idname, sizeof(g_xr_actionconfigstr));
      WM_xr_actionconfig_update_tag(NULL, NULL);
    }

    BLI_remlink(&settings->actionconfigs, actionconf);
    WM_xr_actionconfig_free(actionconf);

    return true;
  }
  return false;
}

XrActionConfig *WM_xr_actionconfig_active_get(XrSessionSettings *settings)
{
  XrActionConfig *actionconf;

  /* First try from preset. */
  actionconf = BLI_findstring(
      &settings->actionconfigs, g_xr_actionconfigstr, offsetof(XrActionConfig, idname));
  if (actionconf) {
    return actionconf;
  }

  /* Otherwise use default. */
  return settings->defaultconf;
}

void WM_xr_actionconfig_active_set(XrSessionSettings *settings, const char *idname)
{
  /* Setting a different action configuration as active: we ensure all is
   * updated properly before and after making the change. */

  WM_xr_actionconfig_update(settings);

  BLI_strncpy(g_xr_actionconfigstr, idname, sizeof(g_xr_actionconfigstr));

  WM_xr_actionconfig_update_tag(NULL, NULL);
  WM_xr_actionconfig_update(settings);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name XR Action Configuration API
 *
 * API functions for managing XR action configurations.
 *
 * \{ */

void WM_xr_actionconfig_init(bContext *C)
{
  wmWindowManager *wm = CTX_wm_manager(C);
  XrSessionSettings *settings = &wm->xr.session_settings;

  /* Create standard action configs. */
  if (settings->defaultconf == NULL) {
    settings->defaultconf = WM_xr_actionconfig_new(settings, XR_ACTIONCONF_STR_DEFAULT, false);
  }
  if (settings->addonconf == NULL) {
    settings->addonconf = WM_xr_actionconfig_new(
        settings, XR_ACTIONCONF_STR_DEFAULT " addon", false);
  }
  if (settings->userconf == NULL) {
    /* Treat user config as user-defined so its actionmaps can be saved to files. */
    settings->userconf = WM_xr_actionconfig_new(settings, XR_ACTIONCONF_STR_DEFAULT " user", true);
  }
}

void WM_xr_actionconfig_update(XrSessionSettings *settings)
{
  if (G.background) {
    return;
  }

  if (g_xr_actionconfig_update_flag == 0) {
    return;
  }

  const bool ensure = ((g_xr_actionconfig_update_flag & WM_XR_ACTIONCONF_UPDATE_ENSURE) != 0);

  if ((g_xr_actionconfig_update_flag & WM_XR_ACTIONCONF_UPDATE_ALL) != 0) {
    /* Update properties for all actionconfigs. */
    LISTBASE_FOREACH (XrActionConfig *, ac, &settings->actionconfigs) {
      wm_xr_actionmap_item_properties_update_ot_from_list(&ac->actionmaps, ensure);
    }

    g_xr_actionconfig_update_flag &= ~(WM_XR_ACTIONCONF_UPDATE_ALL |
                                       WM_XR_ACTIONCONF_UPDATE_ACTIVE);
  }

  if ((g_xr_actionconfig_update_flag & WM_XR_ACTIONCONF_UPDATE_ACTIVE) != 0) {
    /* Update properties for active actionconfig. */
    XrActionConfig *ac = WM_xr_actionconfig_active_get(settings);
    if (ac) {
      wm_xr_actionmap_item_properties_update_ot_from_list(&ac->actionmaps, ensure);
    }

    g_xr_actionconfig_update_flag &= ~WM_XR_ACTIONCONF_UPDATE_ACTIVE;
  }

  if (ensure) {
    g_xr_actionconfig_update_flag &= ~WM_XR_ACTIONCONF_UPDATE_ENSURE;
  }

  BLI_assert(g_xr_actionconfig_update_flag == 0);
}

/** \} */ /* XR-Action Configuration API */
