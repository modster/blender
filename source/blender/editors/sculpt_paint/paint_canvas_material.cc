/* SPDX-License-Identifier: GPL-2.0-or-later */

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

#include "NOD_shader.h"

#include "UI_resources.h"

#include "RNA_access.h"
#include "RNA_define.h"

namespace blender::ed::sculpt_paint::canvas {
struct CanvasDef {
  Material *ma;
  bNode *node;
  EnumPropertyItem rna_enum_item;

  CanvasDef(Material *ma, bNode *node) : ma(ma), node(node)
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
    rna_enum_item.value = 0;
    rna_enum_item.identifier = "IMAGE1";
    rna_enum_item.icon = ICON_IMAGE;
    rna_enum_item.name = image->id.name + 2;
    rna_enum_item.description = image->id.name + 2;
  }

  void init_rna_enum_item_color_attribute(const NodeShaderAttribute *attribute)
  {
    rna_enum_item.value = 0;
    rna_enum_item.identifier = "COLOR_1";
    rna_enum_item.icon = ICON_COLOR;
    rna_enum_item.name = attribute->name;
    rna_enum_item.description = attribute->name;
  }

 public:
  static bool supports(const Object *ob, bNode *node)
  {
    switch (node->type) {
      case SH_NODE_TEX_IMAGE:
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

  bool operator<(const CanvasDef &rhs) const
  {
    if (node->type != rhs.node->type) {
      return node->type == SH_NODE_TEX_IMAGE;
    }
    return StringRef(rna_enum_item.name) < StringRef(rhs.rna_enum_item.name);
  }
};

struct MaterialWrapper {
  Material *ma;

  MaterialWrapper(Material *ma) : ma(ma)
  {
  }

  Vector<CanvasDef> canvases(const Object *ob)
  {
    Vector<CanvasDef> result;
    if (!ma->use_nodes) {
      return result;
    }

    BLI_assert(ma->nodetree != nullptr);
    LISTBASE_FOREACH (bNode *, node, &ma->nodetree->nodes) {
      if (!CanvasDef::supports(ob, node)) {
        continue;
      }
      result.append(CanvasDef(ma, node));
    }
    std::sort(result.begin(), result.end());

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
    result.append(ma);
  }

  return result;
}

}  // namespace blender::ed::sculpt_paint::canvas

extern "C" {

using namespace blender;
using namespace blender::ed::sculpt_paint::canvas;

int ED_paint_canvas_material_get(const struct bContext *C,
                                 const struct PaintModeSettings *settings)
{
  return 0;
}

void ED_paint_canvas_material_set(struct bContext *C,
                                  const struct PaintModeSettings *settings,
                                  int new_value)
{
}

void ED_paint_canvas_material_itemf(const struct bContext *C,
                                    const struct PaintModeSettings *settings,
                                    struct EnumPropertyItem **r_items,
                                    int *r_totitem)
{
  Object *ob = CTX_data_active_object(C);
  blender::Vector<MaterialWrapper> materials = list_materials(ob);
  for (MaterialWrapper &mat : materials) {
    Vector<CanvasDef> canvases = mat.canvases(ob);
    if (canvases.is_empty()) {
      continue;
    }

    mat.append_rna_itemf(r_items, r_totitem);
    for (const CanvasDef &canvas : canvases) {
      RNA_enum_item_add(r_items, r_totitem, &canvas.rna_enum_item);
    }
  }
}
}