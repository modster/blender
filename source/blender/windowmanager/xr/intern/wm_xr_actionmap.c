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

#include <string.h>

#include "BKE_context.h"
#include "BKE_global.h"
#include "BKE_idprop.h"

#include "BLI_listbase.h"
#include "BLI_string.h"

#include "MEM_guardedalloc.h"

#include "WM_api.h"
#include "WM_types.h"

 /* Replacement for U.keyconfigstr for actionconfigs. */
static char g_xr_actionconfigstr[64];

/* Actionconfig update flag. */
enum {
  WM_XR_ACTIONCONF_UPDATE_ACTIVE = (1 << 0), /* Update active actionconfig. */
  WM_XR_ACTIONCONF_UPDATE_ALL = (1 << 1), /* Update all actionconfigs. */
  WM_XR_ACTIONCONF_UPDATE_ENSURE = (1 << 2), /* Skip actionmap/item flag checks when updating actionconfigs. */
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
 * Item in a XR actionmap, that maps from an event to an operator.
 * \{ */

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

static void wm_xr_actionmap_item_properties_free(XrActionMapItem *ami)
{
  if (ami->op_properties_ptr) {
    WM_operator_properties_free(ami->op_properties_ptr);
    MEM_freeN(ami->op_properties_ptr);
    ami->op_properties_ptr = NULL;
    ami->op_properties = NULL;
  }
  else {
    BLI_assert(ami->op_properties == NULL);
  }
}

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
      wm_xr_actionmap_item_properties_free(ami);
      memset(ami->op, 0, sizeof(ami->op));
      return;
  }

  if (ami->op[0] == 0) {
    wm_xr_actionmap_item_properties_free(ami);
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
      /* Zombie actionmap item. */
      wm_xr_actionmap_item_properties_free(ami);
    }
  }
}

static void wm_xr_actionmap_item_properties_update_ot_from_list(ListBase *am_lb, bool ensure)
{
  if (ensure) {
    LISTBASE_FOREACH(XrActionMap *, am, am_lb) {
      LISTBASE_FOREACH(XrActionMapItem *, ami, &am->items) {
        WM_xr_actionmap_item_properties_update_ot(ami);
        ami->flag &= ~XR_AMI_UPDATE;
      }
      am->flag &= ~XR_ACTIONMAP_UPDATE;
    }
  }
  else {
    LISTBASE_FOREACH(XrActionMap *, am, am_lb) {
      if ((am->flag & XR_ACTIONMAP_UPDATE) != 0) {
        LISTBASE_FOREACH(XrActionMapItem *, ami, &am->items) {
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

static XrActionMapItem *wm_xr_actionmap_item_new(const char *idname)
{
  XrActionMapItem *ami = MEM_callocN(sizeof(XrActionMapItem), __func__);

  BLI_strncpy(ami->idname, idname, XR_AMI_MAX_NAME);

  return ami;
}

XrActionMapItem *WM_xr_actionmap_item_ensure(XrActionMap *actionmap, const char *idname)
{
  XrActionMapItem *ami = WM_xr_actionmap_item_list_find(&actionmap->items, idname);

  if (ami == NULL) {
    ami = wm_xr_actionmap_item_new(idname);
    BLI_addtail(&actionmap->items, ami);

    /* Set type to float (button) input by default. */
    ami->type = XR_FLOAT_INPUT;

    WM_xr_actionconfig_update_tag(actionmap, ami);
  }

  return ami;
}

XrActionMapItem *WM_xr_actionmap_item_add_copy(XrActionMap *actionmap, const char *idname, XrActionMapItem *ami_src)
{
  XrActionMapItem *ami_dst = wm_xr_actionmap_item_copy(ami_src);
  BLI_strncpy(ami_dst->idname, idname, XR_AMI_MAX_NAME);

  BLI_addtail(&actionmap->items, ami_dst);

  WM_xr_actionconfig_update_tag(actionmap, ami_dst);

  return ami_dst;
}

bool WM_xr_actionmap_item_remove(XrActionMap *actionmap, XrActionMapItem *ami)
{
  if (BLI_findindex(&actionmap->items, ami) != -1) {
    if (ami->op_properties_ptr) {
      WM_operator_properties_free(ami->op_properties_ptr);
      MEM_freeN(ami->op_properties_ptr);
    }
    BLI_freelinkN(&actionmap->items, ami);

    WM_xr_actionconfig_update_tag(actionmap, NULL);
    return true;
  }
  return false;
}

XrActionMapItem *WM_xr_actionmap_item_list_find(ListBase *lb, const char *idname)
{
  LISTBASE_FOREACH(XrActionMapItem *, ami, lb) {
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

static XrActionMap *wm_xr_actionmap_new(const char *idname)
{
  XrActionMap *am = MEM_callocN(sizeof(struct XrActionMap), __func__);

  BLI_strncpy(am->idname, idname, XR_ACTIONMAP_MAX_NAME);

  return am;
}

XrActionMap *WM_xr_actionmap_ensure(XrActionConfig *actionconf, const char *idname)
{
  XrActionMap *am = WM_xr_actionmap_list_find(&actionconf->actionmaps, idname);

  if (am == NULL) {
    am = wm_xr_actionmap_new(idname);
    BLI_addtail(&actionconf->actionmaps, am);

    WM_xr_actionconfig_update_tag(am, NULL);
  }

  return am;
}

static XrActionMap *wm_xr_actionmap_copy(XrActionMap *am_src)
{
  XrActionMap *am_dst = MEM_dupallocN(am_src);

  BLI_listbase_clear(&am_dst->items);
  am_dst->flag &= ~(XR_ACTIONMAP_UPDATE);

  LISTBASE_FOREACH(XrActionMapItem *, ami, &am_src->items) {
    XrActionMapItem *ami_new = wm_xr_actionmap_item_copy(ami);
    BLI_addtail(&am_dst->items, ami_new);
  }

  return am_dst;
}

XrActionMap *WM_xr_actionmap_add_copy(XrActionConfig *actionconf, const char *idname, XrActionMap *am_src)
{
  XrActionMap *am_dst = wm_xr_actionmap_copy(am_src);
  BLI_strncpy(am_dst->idname, idname, XR_ACTIONMAP_MAX_NAME);

  BLI_addtail(&actionconf->actionmaps, am_dst);

  WM_xr_actionconfig_update_tag(am_dst, NULL);

  return am_dst;
}

void WM_xr_actionmap_clear(XrActionMap *actionmap)
{
  LISTBASE_FOREACH(XrActionMapItem *, ami, &actionmap->items) {
    wm_xr_actionmap_item_properties_free(ami);
  }

  BLI_freelistN(&actionmap->items);
}

bool WM_xr_actionmap_remove(XrActionConfig *actionconf, XrActionMap *actionmap)
{
  if (BLI_findindex(&actionconf->actionmaps, actionmap) != -1) {

    WM_xr_actionmap_clear(actionmap);
    BLI_remlink(&actionconf->actionmaps, actionmap);
    MEM_freeN(actionmap);

    return true;
  }
  return false;
}

XrActionMap *WM_xr_actionmap_list_find(ListBase *lb, const char *idname)
{
  LISTBASE_FOREACH(XrActionMap *, am, lb) {
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
 * There is a builtin default action configuration,
 * a user action configuration, and possibly other preset configurations.
 * \{ */

XrActionConfig *WM_xr_actionconfig_new(XrSessionSettings *settings, const char *idname, bool user_defined)
{
  XrActionConfig *actionconf = BLI_findstring(&settings->actionconfigs, idname, offsetof(XrActionConfig, idname));
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

XrActionConfig *WM_xr_actionconfig_new_user(XrSessionSettings *settings, const char *idname)
{
  return WM_xr_actionconfig_new(settings, idname, true);
}

void WM_xr_actionconfig_clear(XrActionConfig *actionconf)
{
  LISTBASE_FOREACH(XrActionMap *, am, &actionconf->actionmaps) {
    WM_xr_actionmap_clear(am);
  }

  BLI_freelistN(&actionconf->actionmaps);
}

void WM_xr_actionconfig_free(XrActionConfig *actionconf)
{
  WM_xr_actionconfig_clear(actionconf);
  MEM_freeN(actionconf);
}

bool WM_xr_actionconfig_remove(XrSessionSettings *settings, XrActionConfig *actionconf)
{
  if (BLI_findindex(&settings->actionconfigs, actionconf) != -1) {
    if (STREQLEN(g_xr_actionconfigstr, actionconf->idname, sizeof(g_xr_actionconfigstr))) {
      BLI_strncpy(g_xr_actionconfigstr, settings->defaultconf->idname, sizeof(g_xr_actionconfigstr));
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
  actionconf = BLI_findstring(&settings->actionconfigs, g_xr_actionconfigstr, offsetof(XrActionConfig, idname));
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
/** \name XR-Action Map API
 *
 * API functions for managing XR action maps.
 *
 * \{ */

void WM_xr_actionconfig_init(bContext *C)
{
  wmWindowManager *wm = CTX_wm_manager(C);
  XrSessionSettings *settings = &wm->xr.session_settings;

  /* Create standard action configs. */
  if (settings->defaultconf == NULL) {
    /* Keep lowercase to match the preset filename. */
    settings->defaultconf = WM_xr_actionconfig_new(settings, XR_ACTIONCONF_STR_DEFAULT, false);
  }
  if (settings->addonconf == NULL) {
    settings->addonconf = WM_xr_actionconfig_new(settings, XR_ACTIONCONF_STR_DEFAULT " addon", false);
  }
  if (settings->userconf == NULL) {
    settings->userconf = WM_xr_actionconfig_new(settings, XR_ACTIONCONF_STR_DEFAULT " user", false);
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
    LISTBASE_FOREACH(XrActionConfig *, ac, &settings->actionconfigs) {
      wm_xr_actionmap_item_properties_update_ot_from_list(&ac->actionmaps, ensure);
    }

    g_xr_actionconfig_update_flag &= ~(WM_XR_ACTIONCONF_UPDATE_ALL | WM_XR_ACTIONCONF_UPDATE_ACTIVE);
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

/** \} */ /* XR-Action Map API */
