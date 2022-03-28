/* SPDX-License-Identifier: GPL-2.0-or-later */

#include <optional>

#include "ED_paint.h"

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "DNA_image_types.h"
#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_node_types.h"

#include "BKE_context.h"
#include "BKE_customdata.h"
#include "BKE_material.h"
#include "BKE_paint.h"

#include "NOD_shader.h"

#include "UI_resources.h"

#include "RNA_access.h"
#include "RNA_define.h"

namespace blender::ed::sculpt_paint::canvas {

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

/**
 * Decode the given encoded value into a material slot.
 */
uint16_t decode_material_slot(int32_t encoded_value)
{
  return encoded_value >> 16;
}

/**
 * Decide the given encoded value into a resource index.
 */
uint16_t decode_resource_index(int32_t encoded_value)
{
  return encoded_value & 65535;
}

struct MaterialCanvas {
  uint16_t material_slot;
  uint16_t resource_index;

  bNode *node;
  EnumPropertyItem rna_enum_item;

  MaterialCanvas(uint16_t material_slot, uint16_t resource_index, bNode *node)
      : material_slot(material_slot), resource_index(resource_index), node(node)
  {
    init_rna_enum_item();
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
    rna_enum_item.value = encode(material_slot, resource_index);
    rna_enum_item.identifier = image->id.name + 2;
    rna_enum_item.icon = ICON_IMAGE;
    rna_enum_item.name = image->id.name + 2;
    rna_enum_item.description = image->id.name + 2;
  }

  void init_rna_enum_item_color_attribute(const NodeShaderAttribute *attribute)
  {
    rna_enum_item.value = encode(material_slot, resource_index);
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
    }
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
      if (item.resource_index == resource_index) {
        return item;
      }
    }
    return std::nullopt;
  }

  std::optional<MaterialCanvas> active()
  {
    for (const MaterialCanvas &item : items) {
      if ((item.node->flag & NODE_ACTIVE) != 0) {
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

}  // namespace blender::ed::sculpt_paint::canvas

extern "C" {

using namespace blender;
using namespace blender::ed::sculpt_paint::canvas;

int ED_paint_canvas_material_get(Object *ob)
{
  std::optional<MaterialWrapper> material = get_active_material(ob);
  if (!material.has_value()) {
    return 0;
  }
  std::optional<MaterialCanvas> resource = material->canvases(ob).active();
  if (!resource.has_value()) {
    return 0;
  }
  return resource->rna_enum_item.value;
}

void ED_paint_canvas_material_set(Object *ob, int new_value)
{
  const uint16_t resource_index = decode_resource_index(new_value);
  const uint16_t material_slot = decode_material_slot(new_value);

  SculptSession *ss = ob->sculpt;
  if (ss != nullptr) {
    ss->mode.paint.resource_index = resource_index;
    ss->mode.paint.material_slot = material_slot;
  }

  std::optional<MaterialWrapper> material = get_material_in_slot(ob, material_slot);
  if (!material.has_value()) {
    return;
  }
  material->activate(ob);
  material->activate(ob, resource_index);
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
}