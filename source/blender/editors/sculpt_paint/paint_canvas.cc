/* SPDX-License-Identifier: GPL-2.0-or-later */

#include <optional>

#include "ED_paint.h"

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "DNA_image_types.h"
#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_node_types.h"
#include "DNA_workspace_types.h"

#include "BKE_context.h"
#include "BKE_customdata.h"
#include "BKE_material.h"
#include "BKE_paint.h"

#include "NOD_shader.h"

#include "UI_resources.h"

#include "RNA_access.h"
#include "RNA_define.h"

#include "WM_toolsystem.h"

namespace blender::ed::sculpt_paint::canvas {

/**
 * @brief Store a material slot and index encoded as an int.
 */
struct MaterialCanvasIndex {
 private:
  int32_t encoded_;

 public:
  MaterialCanvasIndex(int32_t encoded) : encoded_(encoded)
  {
  }

  MaterialCanvasIndex(uint16_t material_slot, uint16_t resource_index)
      : encoded_(MaterialCanvasIndex::encode(material_slot, resource_index))
  {
  }

  /**
   * Decode the given encoded value into a material slot.
   */
  uint16_t material_slot() const
  {
    return encoded_ >> 16;
  }

  /**
   * Decode the given encoded value into a resource index.
   */
  uint16_t resource_index() const
  {
    return encoded_ & 65535;
  }

  /**
   * @brief Get the encoded value.
   *
   * @return int32_t
   */
  int32_t encoded() const
  {
    return encoded_;
  }

 private:
  /**
   * Encode the given material slot and resource index into a single int32_t.
   *
   * Result is used in rna enums.
   */
  int32_t encode(uint16_t material_slot, uint16_t resource_index)
  {
    int encoded = material_slot;
    encoded <<= 16;
    encoded |= resource_index;
    return encoded;
  }
};

struct MaterialCanvas {
  MaterialCanvasIndex index;

  bNode *node;
  EnumPropertyItem rna_enum_item;

  MaterialCanvas(uint16_t material_slot, uint16_t resource_index, bNode *node)
      : index(material_slot, resource_index), node(node)
  {
    init_rna_enum_item();
  }

  uint16_t resource_index() const
  {
    return index.resource_index();
  }

 private:
  void init_rna_enum_item()
  {
    switch (node->type) {
      case SH_NODE_TEX_IMAGE: {
        Image *image = static_cast<Image *>(static_cast<void *>(node->id));
        init_rna_enum_item_image(image);
        break;
      }

      case SH_NODE_ATTRIBUTE: {
        NodeShaderAttribute *attribute = static_cast<NodeShaderAttribute *>(node->storage);
        init_rna_enum_item_color_attribute(attribute);
      }
      default:
        break;
    }
  }

  void init_rna_enum_item_image(Image *image)
  {
    BLI_assert(image != nullptr);
    rna_enum_item.value = index.encoded();
    rna_enum_item.identifier = image->id.name + 2;
    rna_enum_item.icon = ICON_IMAGE;
    rna_enum_item.name = image->id.name + 2;
    rna_enum_item.description = image->id.name + 2;
  }

  void init_rna_enum_item_color_attribute(const NodeShaderAttribute *attribute)
  {
    rna_enum_item.value = index.encoded();
    rna_enum_item.identifier = attribute->name;
    rna_enum_item.icon = ICON_COLOR;
    rna_enum_item.name = attribute->name;
    rna_enum_item.description = attribute->name;
  }

 public:
  void activate(Object *ob)
  {
    switch (node->type) {
      case SH_NODE_TEX_IMAGE:
        break;

      case SH_NODE_ATTRIBUTE: {
        NodeShaderAttribute *attribute = static_cast<NodeShaderAttribute *>(node->storage);

        Mesh *mesh = (Mesh *)ob->data;
        int layer = CustomData_get_named_layer_index(&mesh->vdata, CD_PROP_COLOR, attribute->name);
        if (layer != -1) {
          CustomData_set_layer_active_index(&mesh->vdata, CD_PROP_COLOR, layer);
        }
        break;
      }

      default:
        BLI_assert_unreachable();
    }
  }

  eV3DShadingColorType shading_color_override() const
  {
    switch (node->type) {
      case SH_NODE_TEX_IMAGE:
        return V3D_SHADING_TEXTURE_COLOR;
      case SH_NODE_ATTRIBUTE:
        return V3D_SHADING_VERTEX_COLOR;
    }
    BLI_assert_unreachable();
    return V3D_SHADING_MATERIAL_COLOR;
  }

  Image *image() const
  {
    switch (node->type) {
      case SH_NODE_TEX_IMAGE:
        return static_cast<Image *>(static_cast<void *>(node->id));
    }
    return nullptr;
  }

  static bool supports(const Object *ob, bNode *node)
  {
    switch (node->type) {
      case SH_NODE_TEX_IMAGE:
        if (!U.experimental.use_sculpt_texture_paint) {
          return false;
        }
        return node->id != nullptr;

      case SH_NODE_ATTRIBUTE: {
        NodeShaderAttribute *attribute = static_cast<NodeShaderAttribute *>(node->storage);
        if (attribute->type != SHD_ATTRIBUTE_GEOMETRY) {
          return false;
        }
        if (ob->type != OB_MESH) {
          return false;
        }

        const struct Mesh *mesh = static_cast<Mesh *>(ob->data);
        int layer = CustomData_get_named_layer_index(&mesh->vdata, CD_PROP_COLOR, attribute->name);
        return layer != -1;
      }

      default:
        return false;
    }
  }

  bool operator<(const MaterialCanvas &rhs) const
  {
    if (node->type != rhs.node->type) {
      return node->type == SH_NODE_TEX_IMAGE;
    }
    return StringRef(rna_enum_item.name) < StringRef(rhs.rna_enum_item.name);
  }
};

struct MaterialCanvases {
  Vector<MaterialCanvas> items;

  const std::optional<MaterialCanvas> find(uint16_t resource_index) const
  {
    for (const MaterialCanvas &item : items) {
      if (item.resource_index() == resource_index) {
        return item;
      }
    }
    return std::nullopt;
  }
};

struct MaterialWrapper {
  /** Index of the material slot. */
  uint16_t material_slot;
  /** Material inside the material slot. */
  Material *ma;

  MaterialWrapper(uint16_t material_slot, Material *ma) : material_slot(material_slot), ma(ma)
  {
  }

  MaterialCanvases canvases(const Object *ob) const
  {
    MaterialCanvases result;
    if (!ma->use_nodes) {
      return result;
    }

    BLI_assert(ma->nodetree != nullptr);
    uint16_t resource_index = 0;
    LISTBASE_FOREACH (bNode *, node, &ma->nodetree->nodes) {
      if (!MaterialCanvas::supports(ob, node)) {
        continue;
      }
      result.items.append(MaterialCanvas(material_slot, resource_index, node));
      resource_index += 1;
    }
    std::sort(result.items.begin(), result.items.end());

    return result;
  }

  std::optional<MaterialCanvas> active_canvas(const Object *ob) const
  {
    if (!ma->use_nodes) {
      return std::nullopt;
    }

    uint16_t resource_index = 0;
    LISTBASE_FOREACH (bNode *, node, &ma->nodetree->nodes) {
      if (!MaterialCanvas::supports(ob, node)) {
        continue;
      }
      if ((node->flag & NODE_ACTIVE) == 0) {
        resource_index += 1;
        continue;
      }
      return MaterialCanvas(material_slot, resource_index, node);
    }

    return std::nullopt;
  }

  void append_rna_itemf(struct EnumPropertyItem **r_items, int *r_totitem)
  {
    EnumPropertyItem item;
    item.value = 0;
    item.identifier = "";
    item.icon = ICON_MATERIAL;
    item.name = ma->id.name + 2;
    item.description = ma->id.name + 2;
    RNA_enum_item_add(r_items, r_totitem, &item);
  }

  /** Activate the material slot on the given object. */
  void activate(Object *ob)
  {
    ob->actcol = material_slot + 1;
  }

  /** Activate a resource of this material. */
  void activate(Object *ob, uint16_t resource_index)
  {
    MaterialCanvases resources = canvases(ob);
    std::optional<MaterialCanvas> resource = resources.find(resource_index);
    if (!resource.has_value()) {
      return;
    }
    nodeSetActive(ma->nodetree, resource->node);
    resource->activate(ob);
  }
};

static Vector<MaterialWrapper> list_materials(Object *ob)
{
  Vector<MaterialWrapper> result;
  struct Material ***matarar = BKE_object_material_array_p(ob);
  short *tot_colp = BKE_object_material_len_p(ob);
  if (tot_colp == nullptr || matarar == nullptr) {
    return result;
  }

  for (int a = 0; a < *tot_colp; a++) {
    Material *ma = (*matarar)[a];
    if (ma == nullptr) {
      continue;
    }
    result.append(MaterialWrapper(a, ma));
  }

  return result;
}

static std::optional<MaterialWrapper> get_material_in_slot(Object *ob, uint16_t material_slot)
{
  uint16_t *tot_colp = (uint16_t *)BKE_object_material_len_p(ob);
  struct Material ***matarar = BKE_object_material_array_p(ob);
  if (tot_colp == nullptr || matarar == nullptr) {
    return std::nullopt;
  }
  if (material_slot >= *tot_colp) {
    return std::nullopt;
  }

  Material *mat = (*matarar)[material_slot];
  if (mat == nullptr) {
    return std::nullopt;
  }

  return MaterialWrapper(material_slot, mat);
}

static std::optional<MaterialWrapper> get_active_material(Object *ob)
{
  uint16_t material_slot = (uint16_t)ob->actcol - 1;
  return get_material_in_slot(ob, material_slot);
}

static std::optional<MaterialCanvas> get_active_canvas(Object *ob)
{
  std::optional<MaterialWrapper> material = get_active_material(ob);
  if (!material.has_value()) {
    return std::nullopt;
  }
  return material->active_canvas(ob);
}

}  // namespace blender::ed::sculpt_paint::canvas

extern "C" {

using namespace blender;
using namespace blender::ed::sculpt_paint::canvas;

int ED_paint_canvas_material_get(Object *ob)
{
  std::optional<MaterialCanvas> canvas = get_active_canvas(ob);
  if (!canvas.has_value()) {
    return 0;
  }
  return canvas->rna_enum_item.value;
}

void ED_paint_canvas_material_set(Object *ob, int new_value)
{
  const MaterialCanvasIndex material_resource(new_value);

  std::optional<MaterialWrapper> material = get_material_in_slot(
      ob, material_resource.material_slot());
  if (!material.has_value()) {
    return;
  }
  material->activate(ob);
  material->activate(ob, material_resource.resource_index());
}

void ED_paint_canvas_material_itemf(Object *ob, struct EnumPropertyItem **r_items, int *r_totitem)
{
  blender::Vector<MaterialWrapper> materials = list_materials(ob);
  for (MaterialWrapper &mat : materials) {
    MaterialCanvases canvases = mat.canvases(ob);
    if (canvases.items.is_empty()) {
      continue;
    }

    mat.append_rna_itemf(r_items, r_totitem);
    for (const MaterialCanvas &canvas : canvases.items) {
      RNA_enum_item_add(r_items, r_totitem, &canvas.rna_enum_item);
    }
  }
}

eV3DShadingColorType ED_paint_draw_color_override(bContext *C,
                                                  const PaintModeSettings *settings,
                                                  Object *ob,
                                                  eV3DShadingColorType orig_color_type)
{
  if (!ED_paint_tool_use_canvas(C, ob)) {
    return orig_color_type;
  }

  eV3DShadingColorType override = orig_color_type;
  switch (settings->canvas_source) {
    case PAINT_CANVAS_SOURCE_COLOR_ATTRIBUTE:
      override = V3D_SHADING_VERTEX_COLOR;
      break;
    case PAINT_CANVAS_SOURCE_IMAGE:
      override = V3D_SHADING_TEXTURE_COLOR;
      break;
    case PAINT_CANVAS_SOURCE_MATERIAL: {
      std::optional<MaterialCanvas> canvas = get_active_canvas(ob);
      if (!canvas.has_value()) {
        break;
      }

      override = canvas->shading_color_override();
      break;
    }
  }

  /* Reset to original color based on enabled experimental features */
  if (!U.experimental.use_sculpt_vertex_colors && override == V3D_SHADING_VERTEX_COLOR) {
    return orig_color_type;
  }
  if (!U.experimental.use_sculpt_texture_paint && override == V3D_SHADING_TEXTURE_COLOR) {
    return orig_color_type;
  }

  return override;
}

Image *ED_paint_canvas_image_get(const struct PaintModeSettings *settings, struct Object *ob)
{
  switch (settings->canvas_source) {
    case PAINT_CANVAS_SOURCE_COLOR_ATTRIBUTE:
      return nullptr;
    case PAINT_CANVAS_SOURCE_IMAGE:
      return settings->image;
    case PAINT_CANVAS_SOURCE_MATERIAL: {
      std::optional<MaterialCanvas> canvas = get_active_canvas(ob);
      if (!canvas.has_value()) {
        break;
      }
      return canvas->image();
    }
  }
  return nullptr;
}

int ED_paint_canvas_uvmap_layer_index_get(const struct PaintModeSettings *settings,
                                          struct Object *ob)
{
  switch (settings->canvas_source) {
    case PAINT_CANVAS_SOURCE_COLOR_ATTRIBUTE:
      return -1;
    case PAINT_CANVAS_SOURCE_IMAGE: {
      /* Use active uv map of the object. */
      if (ob->type != OB_MESH) {
        return -1;
      }

      const Mesh *mesh = static_cast<Mesh *>(ob->data);
      return CustomData_get_active_layer_index(&mesh->ldata, CD_MLOOPUV);
    }
    case PAINT_CANVAS_SOURCE_MATERIAL: {
      /* Use uv map of the canvas. */
      std::optional<MaterialCanvas> canvas = get_active_canvas(ob);
      if (!canvas.has_value()) {
        break;
      }

      if (ob->type != OB_MESH) {
        return -1;
      }

      /* TODO: when uv is directly linked with a uv map node we could that one. */
      const Mesh *mesh = static_cast<Mesh *>(ob->data);
      return CustomData_get_active_layer_index(&mesh->ldata, CD_MLOOPUV);
    }
  }
  return -1;
}

bool ED_paint_tool_use_canvas(struct bContext *C, struct Object *ob)
{
  /* Quick exit, only sculpt tools can use canvas. */
  if (ob->sculpt == nullptr) {
    return false;
  }

  bToolRef *tref = WM_toolsystem_ref_from_context(C);
  if (tref != nullptr) {
    if (STREQ(tref->idname, "builtin_brush.Paint")) {
      return true;
    }
    if (STREQ(tref->idname, "builtin.color_filter")) {
      return true;
    }
  }

  return false;
}
}