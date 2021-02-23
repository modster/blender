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

#include "BLI_index_range.hh"
#include "BLI_listbase.h"
#include "BLI_rect.h"
#include "BLI_utildefines.h"

#include "BKE_geometry_set.hh"
#include "BKE_screen.h"

#include "ED_screen.h"
#include "ED_space_api.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"
#include "DNA_space_types.h"

#include "MEM_guardedalloc.h"

#include "UI_interface.h"
#include "UI_resources.h"
#include "UI_view2d.h"

#include "DEG_depsgraph_query.h"

#include "RNA_access.h"

#include "WM_types.h"

#include "spreadsheet_intern.hh"

using blender::IndexRange;
using blender::MutableSpan;
using blender::Set;
using blender::Span;
using blender::StringRef;
using blender::StringRefNull;
using blender::Vector;
using blender::bke::ReadAttributePtr;
using blender::fn::CPPType;
using blender::fn::GMutableSpan;
using blender::fn::GSpan;

static SpaceLink *spreadsheet_create(const ScrArea *UNUSED(area), const Scene *UNUSED(scene))
{
  SpaceSpreadsheet *spreadsheet_space = (SpaceSpreadsheet *)MEM_callocN(sizeof(SpaceSpreadsheet),
                                                                        "spreadsheet space");
  spreadsheet_space->spacetype = SPACE_SPREADSHEET;

  {
    /* header */
    ARegion *region = (ARegion *)MEM_callocN(sizeof(ARegion), "spreadsheet header");
    BLI_addtail(&spreadsheet_space->regionbase, region);
    region->regiontype = RGN_TYPE_HEADER;
    region->alignment = (U.uiflag & USER_HEADER_BOTTOM) ? RGN_ALIGN_BOTTOM : RGN_ALIGN_TOP;
  }

  {
    /* main window */
    ARegion *region = (ARegion *)MEM_callocN(sizeof(ARegion), "spreadsheet main region");
    BLI_addtail(&spreadsheet_space->regionbase, region);
    region->regiontype = RGN_TYPE_WINDOW;
  }

  return (SpaceLink *)spreadsheet_space;
}

static void spreadsheet_free(SpaceLink *UNUSED(sl))
{
}

static void spreadsheet_init(wmWindowManager *UNUSED(wm), ScrArea *UNUSED(area))
{
}

static SpaceLink *spreadsheet_duplicate(SpaceLink *sl)
{
  return (SpaceLink *)MEM_dupallocN(sl);
}

static void spreadsheet_keymap(wmKeyConfig *UNUSED(keyconf))
{
}

static void spreadsheet_main_region_init(wmWindowManager *UNUSED(wm), ARegion *region)
{
  region->v2d.scroll = V2D_SCROLL_RIGHT | V2D_SCROLL_BOTTOM | V2D_SCROLL_HORIZONTAL_HIDE |
                       V2D_SCROLL_VERTICAL_HIDE;
  region->v2d.align = V2D_ALIGN_NO_NEG_X | V2D_ALIGN_NO_POS_Y;
  region->v2d.keepzoom = V2D_LOCKZOOM_X | V2D_LOCKZOOM_Y | V2D_LIMITZOOM | V2D_KEEPASPECT;
  region->v2d.keeptot = V2D_KEEPTOT_STRICT;
  region->v2d.minzoom = region->v2d.maxzoom = 1.0f;

  UI_view2d_region_reinit(&region->v2d, V2D_COMMONVIEW_LIST, region->winx, region->winy);
}

static void spreadsheet_draw_readonly_table(uiBlock *block,
                                            const GeometryComponent &component,
                                            const AttributeDomain domain)
{
  Set<std::string> attribute_names = component.attribute_names();
  struct AttributeWithName {
    ReadAttributePtr attribute;
    std::string name;
  };
  Vector<AttributeWithName> attribute_columns;
  for (StringRef attribute_name : attribute_names) {
    ReadAttributePtr attribute = component.attribute_try_get_for_read(attribute_name);
    if (!attribute) {
      continue;
    }
    if (attribute->domain() == domain) {
      attribute_columns.append({std::move(attribute), attribute_name});
    }
  }
  std::sort(
      attribute_columns.begin(),
      attribute_columns.end(),
      [](const AttributeWithName &a, const AttributeWithName &b) { return a.name < b.name; });

  int current_x = UI_UNIT_X * 2;
  int current_y = -UI_UNIT_Y;
  for (const AttributeWithName &data : attribute_columns) {
    const int width = 5 * UI_UNIT_X;
    const int height = UI_UNIT_Y;
    uiDefIconTextBut(block,
                     UI_BTYPE_LABEL,
                     0,
                     ICON_NONE,
                     data.name.c_str(),
                     current_x,
                     current_y,
                     width,
                     height,
                     nullptr,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     nullptr);
    current_x += width;
  }

  const int domain_size = component.attribute_domain_size(domain);
  for (const int i : IndexRange(domain_size)) {
    const int x = 0;
    const int y = -UI_UNIT_Y * (i + 2);
    const int width = UI_UNIT_X;
    const int height = UI_UNIT_Y;
    uiDefIconTextBut(block,
                     UI_BTYPE_LABEL,
                     0,
                     ICON_NONE,
                     std::to_string(i).c_str(),
                     x,
                     y,
                     width,
                     height,
                     nullptr,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     nullptr);
  }
}

static void spreadsheet_main_region_draw(const bContext *C, ARegion *region)
{
  UI_ThemeClearColor(TH_BACK);

  Depsgraph *depsgraph = CTX_data_depsgraph_pointer(C);

  View2D *v2d = &region->v2d;
  v2d->flag |= V2D_PIXELOFS_X | V2D_PIXELOFS_Y;
  UI_view2d_view_ortho(v2d);

  /* TODO: Properly compute width and height. */
  UI_view2d_totRect_set(v2d, 5000, 5000);

  uiBlock *block = UI_block_begin(C, region, __func__, UI_EMBOSS_NONE);

  Object *object = CTX_data_active_object(C);
  if (object != nullptr && object->type == OB_MESH) {
    Object *object_eval = DEG_get_evaluated_object(depsgraph, object);
    const GeometrySet &geometry_set = *object_eval->runtime.geometry_set_eval;

    if (geometry_set.has<MeshComponent>()) {
      const MeshComponent &component = *geometry_set.get_component_for_read<MeshComponent>();
      spreadsheet_draw_readonly_table(block, component, ATTR_DOMAIN_POINT);
    }

    // for (const int i : IndexRange(mesh->totvert)) {
    //   const MVert &vert = mesh->mvert[i];
    //   uiBut *but = uiDefButF(block,
    //                          UI_BTYPE_NUM,
    //                          0,
    //                          "",
    //                          0,
    //                          -i * UI_UNIT_Y,
    //                          150,
    //                          UI_UNIT_Y,
    //                          (float *)vert.co,
    //                          -100.0f,
    //                          100.0f,
    //                          0,
    //                          0,
    //                          "My tip");
    //   UI_but_number_precision_set(but, 3);
    //   UI_but_disable(but, "cannot edit");
    // }
  }

  UI_block_end(C, block);
  UI_block_draw(C, block);
}

static void spreadsheet_main_region_listener(const wmRegionListenerParams *params)
{
  /* TODO: Do more precise check. */
  ED_region_tag_redraw(params->region);
}

static void spreadsheet_header_region_init(wmWindowManager *UNUSED(wm), ARegion *region)
{
  ED_region_header_init(region);
}

static void spreadsheet_header_region_draw(const bContext *C, ARegion *region)
{
  ED_region_header(C, region);
}

static void spreadsheet_header_region_free(ARegion *UNUSED(region))
{
}

void ED_spacetype_spreadsheet(void)
{
  SpaceType *st = (SpaceType *)MEM_callocN(sizeof(SpaceType), "spacetype spreadsheet");
  ARegionType *art;

  st->spaceid = SPACE_SPREADSHEET;
  strncpy(st->name, "Spreadsheet", BKE_ST_MAXNAME);

  st->create = spreadsheet_create;
  st->free = spreadsheet_free;
  st->init = spreadsheet_init;
  st->duplicate = spreadsheet_duplicate;
  st->operatortypes = spreadsheet_operatortypes;
  st->keymap = spreadsheet_keymap;

  /* regions: main window */
  art = (ARegionType *)MEM_callocN(sizeof(ARegionType), "spacetype spreadsheet region");
  art->regionid = RGN_TYPE_WINDOW;
  art->keymapflag = ED_KEYMAP_UI | ED_KEYMAP_VIEW2D;

  art->init = spreadsheet_main_region_init;
  art->draw = spreadsheet_main_region_draw;
  art->listener = spreadsheet_main_region_listener;
  BLI_addhead(&st->regiontypes, art);

  /* regions: header */
  art = (ARegionType *)MEM_callocN(sizeof(ARegionType), "spacetype spreadsheet header region");
  art->regionid = RGN_TYPE_HEADER;
  art->prefsizey = HEADERY;
  art->keymapflag = 0;
  art->keymapflag = ED_KEYMAP_UI | ED_KEYMAP_VIEW2D | ED_KEYMAP_HEADER;

  art->init = spreadsheet_header_region_init;
  art->draw = spreadsheet_header_region_draw;
  art->free = spreadsheet_header_region_free;
  BLI_addhead(&st->regiontypes, art);

  BKE_spacetype_register(st);
}
