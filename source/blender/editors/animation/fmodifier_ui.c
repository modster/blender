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
 * The Original Code is Copyright (C) 2009 Blender Foundation.
 * All rights reserved.
 */

/** \file
 * \ingroup edanimation
 */

/* User-Interface Stuff for F-Modifiers:
 * This file defines the (C-Coded) templates + editing callbacks needed
 * by the interface stuff or F-Modifiers, as used by F-Curves in the Graph Editor,
 * and NLA-Strips in the NLA Editor.
 *
 * Copy/Paste Buffer for F-Modifiers:
 * For now, this is also defined in this file so that it can be shared between the
 */

#include <string.h>

#include "DNA_anim_types.h"
#include "DNA_scene_types.h"

#include "MEM_guardedalloc.h"

#include "BLT_translation.h"

#include "BLI_blenlib.h"
#include "BLI_utildefines.h"

#include "BKE_context.h"
#include "BKE_fcurve.h"
#include "BKE_screen.h"

#include "WM_api.h"
#include "WM_types.h"

#include "RNA_access.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "ED_anim_api.h"
#include "ED_undo.h"

#include "DEG_depsgraph.h"

#include "anim_intern.h"

typedef void (*PanelDrawFn)(const bContext *, struct Panel *);
static void deg_update(bContext *C, void *owner_id, void *UNUSED(var2));
static void fmodifier_panel_header(const bContext *C, Panel *panel);

/* -------------------------------------------------------------------- */
/** \name Panel Registering and Panel Callbacks
 * \{ */

static bAnimListElem *get_active_fcurve_channel(bAnimContext *ac)
{
  ListBase anim_data = {NULL, NULL};
  int filter = (ANIMFILTER_DATA_VISIBLE | ANIMFILTER_FOREDIT | ANIMFILTER_ACTIVE);
  size_t items = ANIM_animdata_filter(ac, &anim_data, filter, ac->data, ac->datatype);

  /* We take the first F-Curve only, since some other ones may have had 'active' flag set
   * if they were from linked data.
   */
  if (items) {
    bAnimListElem *ale = (bAnimListElem *)anim_data.first;

    /* remove first item from list, then free the rest of the list and return the stored one */
    BLI_remlink(&anim_data, ale);
    ANIM_animdata_freelist(&anim_data);

    return ale;
  }

  /* no active F-Curve */
  return NULL;
}

static int graph_panel_context(const bContext *C, bAnimListElem **ale, FCurve **fcu)
{
  bAnimContext ac;
  bAnimListElem *elem = NULL;

  /* For now, only draw if we could init the anim-context info
   * (necessary for all animation-related tools)
   * to work correctly is able to be correctly retrieved.
   * There's no point showing empty panels?
   */
  if (ANIM_animdata_get_context(C, &ac) == 0) {
    return 0;
  }

  /* try to find 'active' F-Curve */
  elem = get_active_fcurve_channel(&ac);
  if (elem == NULL) {
    return 0;
  }

  if (fcu) {
    *fcu = (FCurve *)elem->data;
  }
  if (ale) {
    *ale = elem;
  }
  else {
    MEM_freeN(elem);
  }

  return 1;
}

static void fmodifier_get_pointers(const bContext *C,
                                   Panel *panel,
                                   FModifier **r_fcm,
                                   ID **r_owner_id)
{
  bAnimListElem *ale;
  FCurve *fcu;
  if (!graph_panel_context(C, &ale, &fcu)) {
    return;
  }
  ListBase *modifiers = &fcu->modifiers;

  *r_fcm = BLI_findlink(modifiers, panel->runtime.list_index);
  *r_owner_id = ale->fcurve_owner_id;

  uiLayoutSetActive(panel->layout, !(fcu->flag & FCURVE_MOD_OFF));
}

static bool graph_panel_poll(const bContext *C, PanelType *UNUSED(pt))
{
  return graph_panel_context(C, NULL, NULL);
}

/**
 * Move an FModifier to the index it's moved to after a drag and drop.
 */
static void fmodifier_reorder(bContext *C, Panel *panel, int new_index)
{
  bAnimListElem *ale;
  FCurve *fcu;
  if (!graph_panel_context(C, &ale, &fcu)) {
    return;
  }

  int current_index = panel->runtime.list_index;
  if (current_index == new_index) {
    return;
  }

  ListBase *modifiers = &fcu->modifiers;
  FModifier *fcm = BLI_findlink(modifiers, current_index);
  if (fcm == NULL) {
    return;
  }

  /* Cycles modifier has to be the first, so make sure it's kept that way. */
  if (fcm->type == FMODIFIER_TYPE_CYCLES) {
    return;
  }
  FModifier *fcm_first = modifiers->first;
  if (fcm_first->type == FMODIFIER_TYPE_CYCLES && new_index == 0) {
    return;
  }

  BLI_assert(current_index >= 0);
  BLI_assert(new_index >= 0);

  /* Move the FModifier in the list. */
  BLI_listbase_link_move(modifiers, fcm, new_index - current_index);

  ED_undo_push(C, "Move F-Curve Modifier");

  ID *fcurve_owner_id = ale->fcurve_owner_id;
  WM_event_add_notifier(C, NC_ANIMATION | ND_KEYFRAME | NA_EDITED, NULL);
  DEG_id_tag_update(fcurve_owner_id, ID_RECALC_ANIMATION);
}

#define FMODIFIER_TYPE_PANEL_PREFIX "ANIM_PT_"
void ANIM_fmodifier_type_panel_id(int type, char *r_idname)
{
  const FModifierTypeInfo *fmi = get_fmodifier_typeinfo(type);

  strcpy(r_idname, FMODIFIER_TYPE_PANEL_PREFIX);
  strcat(r_idname, fmi->name);
}

static short get_fmodifier_expand_flag(const bContext *C, Panel *panel)
{
  FCurve *fcu;
  if (!graph_panel_context(C, NULL, &fcu)) {
    return 1;
  }
  FModifier *fcm = BLI_findlink(&fcu->modifiers, panel->runtime.list_index);
  return fcm->ui_expand_flag;
}

static void set_fmodifier_expand_flag(const bContext *C, Panel *panel, short expand_flag)
{
  FCurve *fcu;
  if (!graph_panel_context(C, NULL, &fcu)) {
    return;
  }
  FModifier *fcm = BLI_findlink(&fcu->modifiers, panel->runtime.list_index);
  fcm->ui_expand_flag = expand_flag;
}

static PanelType *fmodifier_panel_register(ARegionType *region_type,
                                           eFModifier_Types type,
                                           PanelDrawFn draw)
{
  /* Get the name for the modifier's panel. */
  char panel_idname[BKE_ST_MAXNAME];
  ANIM_fmodifier_type_panel_id(type, panel_idname);

  PanelType *panel_type = MEM_callocN(sizeof(PanelType), panel_idname);

  strcpy(panel_type->idname, panel_idname);
  strcpy(panel_type->label, "");
  strcpy(panel_type->category, "Modifiers");
  strcpy(panel_type->translation_context, BLT_I18NCONTEXT_DEFAULT_BPYRNA);

  panel_type->draw_header = fmodifier_panel_header;
  panel_type->draw = draw;
  panel_type->poll = graph_panel_poll;

  /* Give the panel the special flag that says it was built here and corresponds to a
   * modifer rather than a PanelType. */
  panel_type->flag = PNL_LAYOUT_HEADER_EXPAND | PNL_DRAW_BOX | PNL_INSTANCED;
  panel_type->reorder = fmodifier_reorder;
  panel_type->get_list_data_expand_flag = get_fmodifier_expand_flag;
  panel_type->set_list_data_expand_flag = set_fmodifier_expand_flag;

  BLI_addtail(&region_type->paneltypes, panel_type);

  return panel_type;
}

/**
 * Add a child panel to the parent.
 *
 * \note To create the panel type's idname, it appends the \a name argument to the \a parent's
 * idname.
 */
static PanelType *fmodifier_subpanel_register(ARegionType *region_type,
                                              const char *name,
                                              const char *label,
                                              PanelDrawFn draw_header,
                                              PanelDrawFn draw,
                                              PanelType *parent)
{
  /* Create the subpanel's ID name. */
  char panel_idname[BKE_ST_MAXNAME];
  strcpy(panel_idname, parent->idname);
  strcat(panel_idname, "_");
  strcat(panel_idname, name);

  PanelType *panel_type = MEM_callocN(sizeof(PanelType), panel_idname);

  strcpy(panel_type->idname, panel_idname);
  strcpy(panel_type->label, label);
  strcpy(panel_type->category, "Modifiers");
  strcpy(panel_type->translation_context, BLT_I18NCONTEXT_DEFAULT_BPYRNA);

  panel_type->draw_header = draw_header;
  panel_type->draw = draw;
  panel_type->poll = graph_panel_poll;
  panel_type->flag = (PNL_DEFAULT_CLOSED | PNL_DRAW_BOX);

  BLI_assert(parent != NULL);
  strcpy(panel_type->parent_id, parent->idname);
  panel_type->parent = parent;
  BLI_addtail(&parent->children, BLI_genericNodeN(panel_type));
  BLI_addtail(&region_type->paneltypes, panel_type);

  return panel_type;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name General UI Callbacks and Drawing
 * \{ */

// XXX! --------------------------------
/* Temporary definition for limits of float number buttons
 * (FLT_MAX tends to infinity with old system). */
#define UI_FLT_MAX 10000.0f

#define B_REDR 1
#define B_FMODIFIER_REDRAW 20

/* callback to update depsgraph on value changes */
static void deg_update(bContext *C, void *owner_id, void *UNUSED(var2))
{
  /* send notifiers */
  /* XXX for now, this is the only way to get updates in all the right places...
   * but would be nice to have a special one in this case. */
  WM_event_add_notifier(C, NC_ANIMATION | ND_KEYFRAME | NA_EDITED, NULL);
  DEG_id_tag_update(owner_id, ID_RECALC_ANIMATION);
}

/* callback to verify modifier data */
static void validate_fmodifier_cb(bContext *C, void *fcm_v, void *owner_id)
{
  FModifier *fcm = (FModifier *)fcm_v;
  const FModifierTypeInfo *fmi = fmodifier_get_typeinfo(fcm);

  /* call the verify callback on the modifier if applicable */
  if (fmi && fmi->verify_data) {
    fmi->verify_data(fcm);
  }
  if (owner_id) {
    deg_update(C, owner_id, NULL);
  }
}

/* callback to remove the given modifier  */
typedef struct FModifierDeleteContext {
  ID *fcurve_owner_id;
  ListBase *modifiers;
} FModifierDeleteContext;
static void delete_fmodifier_cb(bContext *C, void *ctx_v, void *fcm_v)
{
  FModifierDeleteContext *ctx = (FModifierDeleteContext *)ctx_v;
  ListBase *modifiers = ctx->modifiers;
  FModifier *fcm = (FModifier *)fcm_v;

  /* remove the given F-Modifier from the active modifier-stack */
  remove_fmodifier(modifiers, fcm);

  ED_undo_push(C, "Delete F-Curve Modifier");

  deg_update(C, ctx->fcurve_owner_id, NULL);
}

static void fmodifier_influence_draw(uiLayout *layout, ID *fcurve_owner_id, FModifier *fcm)
{
  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifier, fcm, &ptr);

  uiItemS(layout);

  uiLayout *row = uiLayoutRowWithHeading(layout, true, IFACE_("Influence"));
  uiItemR(row, &ptr, "use_influence", 0, "", ICON_NONE);
  uiLayout *sub = uiLayoutRow(row, true);
  uiLayoutSetActive(sub, fcm->flag & FMODIFIER_FLAG_USEINFLUENCE);
  uiItemR(sub, &ptr, "influence", 0, "", ICON_NONE);
}

static void fmodifier_frame_range_header_draw(const bContext *C, Panel *panel)
{
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifier, fcm, &ptr);
  uiItemR(layout, &ptr, "use_restricted_range", 0, "", ICON_NONE);
}

static void fmodifier_frame_range_draw(const bContext *C, Panel *panel)
{
  uiLayout *col;
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifier, fcm, &ptr);

  uiLayoutSetPropSep(layout, true);

  uiLayoutSetActive(layout, fcm->flag & FMODIFIER_FLAG_RANGERESTRICT);

  col = uiLayoutColumn(layout, true);
  uiItemR(col, &ptr, "frame_start", 0, IFACE_("Frame Start"), ICON_NONE);
  uiItemR(col, &ptr, "frame_end", 0, IFACE_("End"), ICON_NONE);

  col = uiLayoutColumn(layout, true);
  uiItemR(col, &ptr, "blend_in", 0, IFACE_("Blend In"), ICON_NONE);
  uiItemR(col, &ptr, "blend_out", 0, IFACE_("Out"), ICON_NONE);
}

static void fmodifier_panel_header(const bContext *C, Panel *panel)
{
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &owner_id);

  PointerRNA ptr;
  RNA_pointer_create(owner_id, &RNA_FModifier, fcm, &ptr);
  const FModifierTypeInfo *fmi = fmodifier_get_typeinfo(fcm);

  /* get layout-row + UI-block for this */

  uiBlock *block = uiLayoutGetBlock(layout);  // err...

  /* left-align -------------------------------------------- */
  uiLayout *sub = uiLayoutRow(layout, true);
  uiLayoutSetAlignment(sub, UI_LAYOUT_ALIGN_LEFT);

  UI_block_emboss_set(block, UI_EMBOSS_NONE);

  /* checkbox for 'active' status (for now) */
  uiItemR(sub, &ptr, "active", UI_ITEM_R_ICON_ONLY, "", ICON_NONE);

  /* name */
  if (fmi) {
    uiItemL(sub, IFACE_(fmi->name), ICON_NONE);
  }
  else {
    uiItemL(sub, IFACE_("<Unknown Modifier>"), ICON_NONE);
  }

  /* right-align ------------------------------------------- */
  sub = uiLayoutRow(layout, true);
  uiLayoutSetAlignment(sub, UI_LAYOUT_ALIGN_RIGHT);

  /* 'mute' button */
  uiItemR(sub, &ptr, "mute", UI_ITEM_R_ICON_ONLY, "", ICON_NONE);

  UI_block_emboss_set(block, UI_EMBOSS_NONE);

  /* delete button */
  uiBut *but = uiDefIconBut(block,
                            UI_BTYPE_BUT,
                            B_REDR,
                            ICON_X,
                            0,
                            0,
                            UI_UNIT_X,
                            UI_UNIT_Y,
                            NULL,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            TIP_("Delete F-Curve Modifier"));
  FModifierDeleteContext *ctx = MEM_mallocN(sizeof(FModifierDeleteContext), "fmodifier ctx");
  ctx->fcurve_owner_id = owner_id;
  FCurve *fcu;
  if (!graph_panel_context(C, NULL, &fcu)) {
    return;
  }
  ctx->modifiers = &fcu->modifiers;
  UI_but_funcN_set(but, delete_fmodifier_cb, ctx, fcm);

  UI_block_emboss_set(block, UI_EMBOSS);

  uiItemS(layout);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Generator Modifier
 * \{ */

static void generator_panel_draw(const bContext *C, Panel *panel)
{
  uiLayout *row;
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  uiBut *but;
  short bwidth = 314 - 1.5 * UI_UNIT_X; /* max button width */

  FMod_Generator *data = (FMod_Generator *)fcm->data;

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifierFunctionGenerator, fcm, &ptr);

  /* basic settings (backdrop + mode selector + some padding) */
  /* col = uiLayoutColumn(layout, true); */ /* UNUSED */
  uiBlock *block = uiLayoutGetBlock(layout);
  but = uiDefButR(block,
                  UI_BTYPE_MENU,
                  B_FMODIFIER_REDRAW,
                  NULL,
                  0,
                  0,
                  bwidth,
                  UI_UNIT_Y,
                  &ptr,
                  "mode",
                  -1,
                  0,
                  0,
                  -1,
                  -1,
                  NULL);
  UI_but_func_set(but, validate_fmodifier_cb, fcm, NULL);

  uiItemR(layout, &ptr, "use_additive", 0, NULL, ICON_NONE);

  /* Settings for individual modes. */
  switch (data->mode) {
    case FCM_GENERATOR_POLYNOMIAL: /* polynomial expression */
    {
      const uiFontStyle *fstyle = UI_FSTYLE_WIDGET;
      float *cp = NULL;
      char xval[32];
      uint i;
      int maxXWidth;

      /* draw polynomial order selector */
      row = uiLayoutRow(layout, false);
      block = uiLayoutGetBlock(row);

      but = uiDefButI(
          block,
          UI_BTYPE_NUM,
          B_FMODIFIER_REDRAW,
          IFACE_("Poly Order:"),
          0.5f * UI_UNIT_X,
          0,
          bwidth,
          UI_UNIT_Y,
          &data->poly_order,
          1,
          100,
          1,
          0,
          TIP_("'Order' of the Polynomial (for a polynomial with n terms, 'order' is n-1)"));
      UI_but_func_set(but, validate_fmodifier_cb, fcm, fcurve_owner_id);

      /* calculate maximum width of label for "x^n" labels */
      if (data->arraysize > 2) {
        BLI_snprintf(xval, sizeof(xval), "x^%u", data->arraysize);
        /* XXX: UI_fontstyle_string_width is not accurate */
        maxXWidth = UI_fontstyle_string_width(fstyle, xval) + 0.5 * UI_UNIT_X;
      }
      else {
        /* basic size (just "x") */
        maxXWidth = UI_fontstyle_string_width(fstyle, "x") + 0.5 * UI_UNIT_X;
      }

      /* draw controls for each coefficient and a + sign at end of row */
      row = uiLayoutRow(layout, true);
      block = uiLayoutGetBlock(row);

      /* Update depsgraph when values change */
      UI_block_func_set(block, deg_update, fcurve_owner_id, NULL);

      cp = data->coefficients;
      for (i = 0; (i < data->arraysize) && (cp); i++, cp++) {
        /* To align with first line... */
        if (i) {
          uiDefBut(block,
                   UI_BTYPE_LABEL,
                   1,
                   "   ",
                   0,
                   0,
                   2 * UI_UNIT_X,
                   UI_UNIT_Y,
                   NULL,
                   0.0,
                   0.0,
                   0,
                   0,
                   "");
        }
        else {
          uiDefBut(block,
                   UI_BTYPE_LABEL,
                   1,
                   "y =",
                   0,
                   0,
                   2 * UI_UNIT_X,
                   UI_UNIT_Y,
                   NULL,
                   0.0,
                   0.0,
                   0,
                   0,
                   "");
        }

        /* coefficient */
        uiDefButF(block,
                  UI_BTYPE_NUM,
                  B_FMODIFIER_REDRAW,
                  "",
                  0,
                  0,
                  bwidth / 2,
                  UI_UNIT_Y,
                  cp,
                  -UI_FLT_MAX,
                  UI_FLT_MAX,
                  10,
                  3,
                  TIP_("Coefficient for polynomial"));

        /* 'x' param (and '+' if necessary) */
        if (i == 0) {
          BLI_strncpy(xval, " ", sizeof(xval));
        }
        else if (i == 1) {
          BLI_strncpy(xval, "x", sizeof(xval));
        }
        else {
          BLI_snprintf(xval, sizeof(xval), "x^%u", i);
        }
        uiDefBut(block,
                 UI_BTYPE_LABEL,
                 1,
                 xval,
                 0,
                 0,
                 maxXWidth,
                 UI_UNIT_Y,
                 NULL,
                 0.0,
                 0.0,
                 0,
                 0,
                 TIP_("Power of x"));

        if ((i != (data->arraysize - 1)) || ((i == 0) && data->arraysize == 2)) {
          uiDefBut(
              block, UI_BTYPE_LABEL, 1, "+", 0, 0, UI_UNIT_X, UI_UNIT_Y, NULL, 0.0, 0.0, 0, 0, "");

          /* next coefficient on a new row */
          row = uiLayoutRow(layout, true);
          block = uiLayoutGetBlock(row);
        }
        else {
          /* For alignment in UI! */
          uiDefBut(
              block, UI_BTYPE_LABEL, 1, " ", 0, 0, UI_UNIT_X, UI_UNIT_Y, NULL, 0.0, 0.0, 0, 0, "");
        }
      }
      break;
    }

    case FCM_GENERATOR_POLYNOMIAL_FACTORISED: /* Factorized polynomial expression */
    {
      float *cp = NULL;
      uint i;

      /* draw polynomial order selector */
      row = uiLayoutRow(layout, false);
      block = uiLayoutGetBlock(row);

      but = uiDefButI(
          block,
          UI_BTYPE_NUM,
          B_FMODIFIER_REDRAW,
          IFACE_("Poly Order:"),
          0,
          0,
          314 - 1.5 * UI_UNIT_X,
          UI_UNIT_Y,
          &data->poly_order,
          1,
          100,
          1,
          0,
          TIP_("'Order' of the Polynomial (for a polynomial with n terms, 'order' is n-1)"));
      UI_but_func_set(but, validate_fmodifier_cb, fcm, fcurve_owner_id);

      /* draw controls for each pair of coefficients */
      row = uiLayoutRow(layout, true);
      block = uiLayoutGetBlock(row);

      /* Update depsgraph when values change */
      UI_block_func_set(block, deg_update, fcurve_owner_id, NULL);

      cp = data->coefficients;
      for (i = 0; (i < data->poly_order) && (cp); i++, cp += 2) {
        /* To align with first line */
        if (i) {
          uiDefBut(block,
                   UI_BTYPE_LABEL,
                   1,
                   "   ",
                   0,
                   0,
                   2.5 * UI_UNIT_X,
                   UI_UNIT_Y,
                   NULL,
                   0.0,
                   0.0,
                   0,
                   0,
                   "");
        }
        else {
          uiDefBut(block,
                   UI_BTYPE_LABEL,
                   1,
                   "y =",
                   0,
                   0,
                   2.5 * UI_UNIT_X,
                   UI_UNIT_Y,
                   NULL,
                   0.0,
                   0.0,
                   0,
                   0,
                   "");
        }
        /* opening bracket */
        uiDefBut(
            block, UI_BTYPE_LABEL, 1, "(", 0, 0, UI_UNIT_X, UI_UNIT_Y, NULL, 0.0, 0.0, 0, 0, "");

        /* coefficients */
        uiDefButF(block,
                  UI_BTYPE_NUM,
                  B_FMODIFIER_REDRAW,
                  "",
                  0,
                  0,
                  5 * UI_UNIT_X,
                  UI_UNIT_Y,
                  cp,
                  -UI_FLT_MAX,
                  UI_FLT_MAX,
                  10,
                  3,
                  TIP_("Coefficient of x"));

        uiDefBut(block,
                 UI_BTYPE_LABEL,
                 1,
                 "x +",
                 0,
                 0,
                 2 * UI_UNIT_X,
                 UI_UNIT_Y,
                 NULL,
                 0.0,
                 0.0,
                 0,
                 0,
                 "");

        uiDefButF(block,
                  UI_BTYPE_NUM,
                  B_FMODIFIER_REDRAW,
                  "",
                  0,
                  0,
                  5 * UI_UNIT_X,
                  UI_UNIT_Y,
                  cp + 1,
                  -UI_FLT_MAX,
                  UI_FLT_MAX,
                  10,
                  3,
                  TIP_("Second coefficient"));

        /* closing bracket and multiplication sign */
        if ((i != (data->poly_order - 1)) || ((i == 0) && data->poly_order == 2)) {
          uiDefBut(block,
                   UI_BTYPE_LABEL,
                   1,
                   ") \xc3\x97",
                   0,
                   0,
                   2 * UI_UNIT_X,
                   UI_UNIT_Y,
                   NULL,
                   0.0,
                   0.0,
                   0,
                   0,
                   "");

          /* set up new row for the next pair of coefficients */
          row = uiLayoutRow(layout, true);
          block = uiLayoutGetBlock(row);
        }
        else {
          uiDefBut(block,
                   UI_BTYPE_LABEL,
                   1,
                   ")  ",
                   0,
                   0,
                   2 * UI_UNIT_X,
                   UI_UNIT_Y,
                   NULL,
                   0.0,
                   0.0,
                   0,
                   0,
                   "");
        }
      }
      break;
    }
  }

  fmodifier_influence_draw(layout, fcurve_owner_id, fcm);
}

void ANIM_fcm_generator_panel_register(ARegionType *region_type)
{
  PanelType *panel_type = fmodifier_panel_register(
      region_type, FMODIFIER_TYPE_GENERATOR, generator_panel_draw);
  fmodifier_subpanel_register(region_type,
                              "frame_range",
                              "Restrict Frame Range",
                              fmodifier_frame_range_header_draw,
                              fmodifier_frame_range_draw,
                              panel_type);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Function Generator Modifier
 * \{ */

static void fn_generator_panel_draw(const bContext *C, Panel *panel)
{
  uiLayout *col;
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifierFunctionGenerator, fcm, &ptr);

  uiItemR(layout, &ptr, "function_type", 0, "", ICON_NONE);

  uiLayoutSetPropSep(layout, true);

  col = uiLayoutColumn(layout, false);
  uiItemR(col, &ptr, "use_additive", 0, NULL, ICON_NONE);

  col = uiLayoutColumn(layout, false);
  uiItemR(col, &ptr, "amplitude", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "phase_multiplier", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "phase_offset", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "value_offset", 0, NULL, ICON_NONE);

  fmodifier_influence_draw(layout, fcurve_owner_id, fcm);
}

void ANIM_fcm_fn_generator_panel_register(ARegionType *region_type)
{
  PanelType *panel_type = fmodifier_panel_register(
      region_type, FMODIFIER_TYPE_FN_GENERATOR, fn_generator_panel_draw);
  fmodifier_subpanel_register(region_type,
                              "frame_range",
                              "Restrict Frame Range",
                              fmodifier_frame_range_header_draw,
                              fmodifier_frame_range_draw,
                              panel_type);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Cycles Modifier
 * \{ */

static void cycles_panel_draw(const bContext *C, Panel *panel)
{
  uiLayout *col;
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifierCycles, fcm, &ptr);

  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);

  /* Before. */
  col = uiLayoutColumn(layout, false);
  uiItemR(col, &ptr, "mode_before", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "cycles_before", 0, IFACE_("Count"), ICON_NONE);

  /* After. */
  col = uiLayoutColumn(layout, false);
  uiItemR(col, &ptr, "mode_after", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "cycles_after", 0, IFACE_("Count"), ICON_NONE);

  fmodifier_influence_draw(layout, fcurve_owner_id, fcm);
}

void ANIM_fcm_cycles_panel_register(ARegionType *region_type)
{
  fmodifier_panel_register(region_type, FMODIFIER_TYPE_CYCLES, cycles_panel_draw);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Noise Modifier
 * \{ */

static void noise_panel_draw(const bContext *C, Panel *panel)
{
  uiLayout *col;
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifierNoise, fcm, &ptr);

  uiLayoutSetPropSep(layout, true);
  uiLayoutSetPropDecorate(layout, false);

  uiItemR(layout, &ptr, "blend_type", 0, NULL, ICON_NONE);

  col = uiLayoutColumn(layout, false);
  uiItemR(col, &ptr, "scale", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "strength", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "offset", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "phase", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "depth", 0, NULL, ICON_NONE);

  fmodifier_influence_draw(layout, fcurve_owner_id, fcm);
}

void ANIM_fcm_noise_panel_register(ARegionType *region_type)
{
  PanelType *panel_type = fmodifier_panel_register(
      region_type, FMODIFIER_TYPE_NOISE, noise_panel_draw);
  fmodifier_subpanel_register(region_type,
                              "frame_range",
                              "Restrict Frame Range",
                              fmodifier_frame_range_header_draw,
                              fmodifier_frame_range_draw,
                              panel_type);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Enevelope Modifier
 * \{ */

static void fmod_envelope_addpoint_cb(bContext *C, void *fcm_dv, void *UNUSED(arg))
{
  Scene *scene = CTX_data_scene(C);
  FMod_Envelope *env = (FMod_Envelope *)fcm_dv;
  FCM_EnvelopeData *fedn;
  FCM_EnvelopeData fed;

  /* init template data */
  fed.min = -1.0f;
  fed.max = 1.0f;
  fed.time = (float)scene->r.cfra;  // XXX make this int for ease of use?
  fed.f1 = fed.f2 = 0;

  /* check that no data exists for the current frame... */
  if (env->data) {
    bool exists;
    int i = BKE_fcm_envelope_find_index(env->data, (float)(scene->r.cfra), env->totvert, &exists);

    /* binarysearch_...() will set exists by default to 0,
     * so if it is non-zero, that means that the point exists already */
    if (exists) {
      return;
    }

    /* add new */
    fedn = MEM_callocN((env->totvert + 1) * sizeof(FCM_EnvelopeData), "FCM_EnvelopeData");

    /* add the points that should occur before the point to be pasted */
    if (i > 0) {
      memcpy(fedn, env->data, i * sizeof(FCM_EnvelopeData));
    }

    /* add point to paste at index i */
    *(fedn + i) = fed;

    /* add the points that occur after the point to be pasted */
    if (i < env->totvert) {
      memcpy(fedn + i + 1, env->data + i, (env->totvert - i) * sizeof(FCM_EnvelopeData));
    }

    /* replace (+ free) old with new */
    MEM_freeN(env->data);
    env->data = fedn;

    env->totvert++;
  }
  else {
    env->data = MEM_callocN(sizeof(FCM_EnvelopeData), "FCM_EnvelopeData");
    *(env->data) = fed;

    env->totvert = 1;
  }
}

/* callback to remove envelope data point */
// TODO: should we have a separate file for things like this?
static void fmod_envelope_deletepoint_cb(bContext *UNUSED(C), void *fcm_dv, void *ind_v)
{
  FMod_Envelope *env = (FMod_Envelope *)fcm_dv;
  FCM_EnvelopeData *fedn;
  int index = POINTER_AS_INT(ind_v);

  /* check that no data exists for the current frame... */
  if (env->totvert > 1) {
    /* allocate a new smaller array */
    fedn = MEM_callocN(sizeof(FCM_EnvelopeData) * (env->totvert - 1), "FCM_EnvelopeData");

    memcpy(fedn, env->data, sizeof(FCM_EnvelopeData) * (index));
    memcpy(fedn + index,
           env->data + (index + 1),
           sizeof(FCM_EnvelopeData) * ((env->totvert - index) - 1));

    /* free old array, and set the new */
    MEM_freeN(env->data);
    env->data = fedn;
    env->totvert--;
  }
  else {
    /* just free array, since the only vert was deleted */
    if (env->data) {
      MEM_freeN(env->data);
      env->data = NULL;
    }
    env->totvert = 0;
  }
}

/* draw settings for envelope modifier */
static void envelope_panel_draw(const bContext *C, Panel *panel)
{
  uiLayout *row, *col;
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  FMod_Envelope *env = (FMod_Envelope *)fcm->data;

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifierEnvelope, fcm, &ptr);

  uiLayoutSetPropSep(layout, true);

  /* General settings. */
  col = uiLayoutColumn(layout, true);
  uiItemR(col, &ptr, "reference_value", 0, IFACE_("Reference"), ICON_NONE);
  uiItemR(col, &ptr, "default_min", 0, IFACE_("Min"), ICON_NONE);
  uiItemR(col, &ptr, "default_max", 0, IFACE_("Max"), ICON_NONE);

  /* Control points list. */

  row = uiLayoutRow(layout, false);
  uiBlock *block = uiLayoutGetBlock(row);

  uiBut *but = uiDefBut(block,
                        UI_BTYPE_BUT,
                        B_FMODIFIER_REDRAW,
                        IFACE_("Add Control Point"),
                        0,
                        0,
                        7.5 * UI_UNIT_X,
                        UI_UNIT_Y,
                        NULL,
                        0,
                        0,
                        0,
                        0,
                        TIP_("Add a new control-point to the envelope on the current frame"));
  UI_but_func_set(but, fmod_envelope_addpoint_cb, env, NULL);

  col = uiLayoutColumn(layout, false);
  uiLayoutSetPropSep(col, false);

  FCM_EnvelopeData *fed = env->data;
  for (int i = 0; i < env->totvert; i++, fed++) {
    PointerRNA ctrl_ptr;
    RNA_pointer_create(fcurve_owner_id, &RNA_FModifierEnvelopeControlPoint, fed, &ctrl_ptr);

    /* get a new row to operate on */
    row = uiLayoutRow(col, true);
    block = uiLayoutGetBlock(row);

    uiItemR(row, &ctrl_ptr, "frame", 0, NULL, ICON_NONE);
    uiItemR(row, &ctrl_ptr, "min", 0, IFACE_("Min"), ICON_NONE);
    uiItemR(row, &ctrl_ptr, "max", 0, IFACE_("Max"), ICON_NONE);

    but = uiDefIconBut(block,
                       UI_BTYPE_BUT,
                       B_FMODIFIER_REDRAW,
                       ICON_X,
                       0,
                       0,
                       0.9 * UI_UNIT_X,
                       UI_UNIT_Y,
                       NULL,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       TIP_("Delete envelope control point"));
    UI_but_func_set(but, fmod_envelope_deletepoint_cb, env, POINTER_FROM_INT(i));
    UI_block_align_begin(block);
  }

  fmodifier_influence_draw(layout, fcurve_owner_id, fcm);
}

void ANIM_fcm_envelope_panel_register(ARegionType *region_type)
{
  PanelType *panel_type = fmodifier_panel_register(
      region_type, FMODIFIER_TYPE_ENVELOPE, envelope_panel_draw);
  fmodifier_subpanel_register(region_type,
                              "frame_range",
                              "Restrict Frame Range",
                              fmodifier_frame_range_header_draw,
                              fmodifier_frame_range_draw,
                              panel_type);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Limits Modifier
 * \{ */

static void limits_panel_draw(const bContext *C, Panel *panel)
{
  uiLayout *col, *row, *sub;
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifierLimits, fcm, &ptr);

  uiLayoutSetPropSep(layout, true);

  /* Minimums. */
  col = uiLayoutColumn(layout, false);
  row = uiLayoutRowWithHeading(col, true, IFACE_("Minumum X"));
  uiItemR(row, &ptr, "use_min_x", 0, "", ICON_NONE);
  sub = uiLayoutColumn(row, true);
  uiLayoutSetActive(sub, RNA_boolean_get(&ptr, "use_min_x"));
  uiItemR(sub, &ptr, "min_x", 0, "", ICON_NONE);

  row = uiLayoutRowWithHeading(col, true, IFACE_("Y"));
  uiItemR(row, &ptr, "use_min_y", 0, "", ICON_NONE);
  sub = uiLayoutColumn(row, true);
  uiLayoutSetActive(sub, RNA_boolean_get(&ptr, "use_min_y"));
  uiItemR(sub, &ptr, "min_y", 0, "", ICON_NONE);

  /* Maximums. */
  col = uiLayoutColumn(layout, false);
  row = uiLayoutRowWithHeading(col, true, IFACE_("Maximum X"));
  uiItemR(row, &ptr, "use_max_x", 0, "", ICON_NONE);
  sub = uiLayoutColumn(row, true);
  uiLayoutSetActive(sub, RNA_boolean_get(&ptr, "use_max_x"));
  uiItemR(sub, &ptr, "max_x", 0, "", ICON_NONE);

  row = uiLayoutRowWithHeading(col, true, IFACE_("Y"));
  uiItemR(row, &ptr, "use_max_y", 0, "", ICON_NONE);
  sub = uiLayoutColumn(row, true);
  uiLayoutSetActive(sub, RNA_boolean_get(&ptr, "use_max_y"));
  uiItemR(sub, &ptr, "max_y", 0, "", ICON_NONE);

  fmodifier_influence_draw(layout, fcurve_owner_id, fcm);
}

void ANIM_fcm_limits_panel_register(ARegionType *region_type)
{
  PanelType *panel_type = fmodifier_panel_register(
      region_type, FMODIFIER_TYPE_LIMITS, limits_panel_draw);
  fmodifier_subpanel_register(region_type,
                              "frame_range",
                              "Restrict Frame Range",
                              fmodifier_frame_range_header_draw,
                              fmodifier_frame_range_draw,
                              panel_type);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Stepped Interpolation Modifier
 * \{ */

static void stepped_panel_draw(const bContext *C, Panel *panel)
{
  uiLayout *col, *sub, *row;
  uiLayout *layout = panel->layout;

  FModifier *fcm;
  ID *fcurve_owner_id;
  fmodifier_get_pointers(C, panel, &fcm, &fcurve_owner_id);

  PointerRNA ptr;
  RNA_pointer_create(fcurve_owner_id, &RNA_FModifierStepped, fcm, &ptr);

  uiLayoutSetPropSep(layout, true);

  /* Stepping Settings. */
  col = uiLayoutColumn(layout, false);
  uiItemR(col, &ptr, "frame_step", 0, NULL, ICON_NONE);
  uiItemR(col, &ptr, "frame_offset", 0, NULL, ICON_NONE);

  /* Start range settings. */
  row = uiLayoutRowWithHeading(layout, true, IFACE_("Start Frame"));
  uiItemR(row, &ptr, "use_frame_start", 0, "", ICON_NONE);
  sub = uiLayoutColumn(row, true);
  uiLayoutSetActive(sub, RNA_boolean_get(&ptr, "use_frame_start"));
  uiItemR(sub, &ptr, "frame_start", 0, "", ICON_NONE);

  /* End range settings. */
  row = uiLayoutRowWithHeading(layout, true, IFACE_("End Frame"));
  uiItemR(row, &ptr, "use_frame_end", 0, "", ICON_NONE);
  sub = uiLayoutColumn(row, true);
  uiLayoutSetActive(sub, RNA_boolean_get(&ptr, "use_frame_end"));
  uiItemR(sub, &ptr, "frame_end", 0, "", ICON_NONE);

  fmodifier_influence_draw(layout, fcurve_owner_id, fcm);
}

void ANIM_fcm_stepped_panel_register(ARegionType *region_type)
{
  PanelType *panel_type = fmodifier_panel_register(
      region_type, FMODIFIER_TYPE_STEPPED, stepped_panel_draw);
  fmodifier_subpanel_register(region_type,
                              "frame_range",
                              "Restrict Frame Range",
                              fmodifier_frame_range_header_draw,
                              fmodifier_frame_range_draw,
                              panel_type);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Panel Creator "Template"
 *
 * Checks if the panels match the FCurve's modifiers, rebubilds them if they don't.
 * \{ */

static void fmodifier_panel_id(void *fcm_link, char *r_name)
{
  FModifier *fcm = (FModifier *)fcm_link;
  ANIM_fmodifier_type_panel_id(fcm->type, r_name);
}

void ANIM_fmodifier_panels(const bContext *C, ListBase *fmodifiers)
{
  ScrArea *sa = CTX_wm_area(C);
  ARegion *region = CTX_wm_region(C);

  bool panels_match = UI_panel_list_matches_data(region, fmodifiers, fmodifier_panel_id);

  if (!panels_match) {
    UI_panels_free_instanced(C, region);
    FModifier *fcm = fmodifiers->first;
    for (int i = 0; fcm; i++, fcm = fcm->next) {
      const FModifierTypeInfo *fmi = get_fmodifier_typeinfo(fcm->type);
      // if (fmi->panelRegister) {
      char panel_idname[MAX_NAME];
      fmodifier_panel_id(fcm, panel_idname);

      Panel *new_panel = UI_panel_add_instanced(sa, region, &region->panels, panel_idname, i);
      if (new_panel != NULL) {
        UI_panel_set_expand_from_list_data(C, new_panel);
      }
      // }
    }
  }
  else {
    /* The expansion might have been changed elsewhere, so we still need to set it. */
    LISTBASE_FOREACH (Panel *, panel, &region->panels) {
      if ((panel->type != NULL) && (panel->type->flag & PNL_INSTANCED))
        UI_panel_set_expand_from_list_data(C, panel);
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Copy / Paste Buffer Code
 * \{ */

/* Copy/Paste Buffer itself (list of FModifier 's) */
static ListBase fmodifier_copypaste_buf = {NULL, NULL};

/* ---------- */

/* free the copy/paste buffer */
void ANIM_fmodifiers_copybuf_free(void)
{
  /* just free the whole buffer */
  free_fmodifiers(&fmodifier_copypaste_buf);
}

/* copy the given F-Modifiers to the buffer, returning whether anything was copied or not
 * assuming that the buffer has been cleared already with ANIM_fmodifiers_copybuf_free()
 * - active: only copy the active modifier
 */
bool ANIM_fmodifiers_copy_to_buf(ListBase *modifiers, bool active)
{
  bool ok = true;

  /* sanity checks */
  if (ELEM(NULL, modifiers, modifiers->first)) {
    return 0;
  }

  /* copy the whole list, or just the active one? */
  if (active) {
    FModifier *fcm = find_active_fmodifier(modifiers);

    if (fcm) {
      FModifier *fcmN = copy_fmodifier(fcm);
      BLI_addtail(&fmodifier_copypaste_buf, fcmN);
    }
    else {
      ok = 0;
    }
  }
  else {
    copy_fmodifiers(&fmodifier_copypaste_buf, modifiers);
  }

  /* did we succeed? */
  return ok;
}

/* 'Paste' the F-Modifier(s) from the buffer to the specified list
 * - replace: free all the existing modifiers to leave only the pasted ones
 */
bool ANIM_fmodifiers_paste_from_buf(ListBase *modifiers, bool replace, FCurve *curve)
{
  FModifier *fcm;
  bool ok = false;

  /* sanity checks */
  if (modifiers == NULL) {
    return 0;
  }

  bool was_cyclic = curve && BKE_fcurve_is_cyclic(curve);

  /* if replacing the list, free the existing modifiers */
  if (replace) {
    free_fmodifiers(modifiers);
  }

  /* now copy over all the modifiers in the buffer to the end of the list */
  for (fcm = fmodifier_copypaste_buf.first; fcm; fcm = fcm->next) {
    /* make a copy of it */
    FModifier *fcmN = copy_fmodifier(fcm);

    fcmN->curve = curve;

    /* make sure the new one isn't active, otherwise the list may get several actives */
    fcmN->flag &= ~FMODIFIER_FLAG_ACTIVE;

    /* now add it to the end of the list */
    BLI_addtail(modifiers, fcmN);
    ok = 1;
  }

  /* adding or removing the Cycles modifier requires an update to handles */
  if (curve && BKE_fcurve_is_cyclic(curve) != was_cyclic) {
    calchandles_fcurve(curve);
  }

  /* did we succeed? */
  return ok;
}

/** \} */
