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

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_material.h"
#include "BKE_mesh.h"

#include "RNA_define.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "NOD_runtime_types.hh"

#include "node_geometry_util.hh"

namespace blender::nodes {

struct GeometryNodeMeshEllipse : public GeometryNodeDefinition<GeometryNodeMeshEllipse> {
  inline static const char *idname = "GeometryNodeMeshEllipse";
  inline static const char *ui_name = "Mesh Ellipse";
  inline static const char *ui_description = "Create an elliptical shape";

  enum FillType {
    FILL_NONE = 0,
    FILL_NGON = 1,
    FILL_TRIANGLE_FAN = 2,
  };

  static void define_rna(StructRNA *srna)
  {
    static EnumPropertyItem fill_type_items[] = {
        {FILL_NONE, "NONE", 0, "None", ""},
        {FILL_NGON, "NGON", 0, "N-Gon", ""},
        {FILL_TRIANGLE_FAN, "TRIANGLE_FAN", 0, "Triangles", ""},
        {0, NULL, 0, NULL, NULL},
    };

    RNA_def_enum(srna, "fill_type", fill_type_items, FILL_NONE, "Fill Type", "");
  }

  static void init(bNodeTree *ntree, bNode *node)
  {
    PointerRNA ptr;
    RNA_pointer_create(&ntree->id, &RNA_Node, node, &ptr);

    RNA_enum_set(&ptr, "fill_type", FILL_NONE);

    {
      bNodeSocketValueInt *dval =
          (bNodeSocketValueInt *)nodeAddSocket(
              ntree, node, SOCK_IN, "NodeSocketInt", "Vertices", "Vertices")
              ->default_value;
      dval->value = 32;
      dval->min = 3;
      dval->max = 4096;
    }
    {
      bNodeSocketValueFloat *dval =
          (bNodeSocketValueFloat *)nodeAddSocket(
              ntree, node, SOCK_IN, "NodeSocketFloat", "Radius A", "Radius A")
              ->default_value;
      dval->value = 1.0f;
      dval->min = 0.0f;
      dval->max = FLT_MAX;
      dval->subtype = PROP_DISTANCE;
    }
    {
      bNodeSocketValueFloat *dval =
          (bNodeSocketValueFloat *)nodeAddSocket(
              ntree, node, SOCK_IN, "NodeSocketFloat", "Radius B", "Radius B")
              ->default_value;
      dval->value = 1.0f;
      dval->min = 0.0f;
      dval->max = FLT_MAX;
      dval->subtype = PROP_DISTANCE;
    }

    nodeAddSocket(ntree, node, SOCK_OUT, "NodeSocketGeometry", "Geometry", "Geometry");
  }

  static void draw_buttons(uiLayout *layout, bContext *UNUSED(C), PointerRNA *ptr)
  {
    uiLayoutSetPropSep(layout, true);
    uiLayoutSetPropDecorate(layout, false);
    uiItemR(layout, ptr, "fill_type", 0, nullptr, ICON_NONE);
  }

  static int circle_vert_total(const FillType fill_type, const int verts_num)
  {
    switch (fill_type) {
      case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      case GEO_NODE_MESH_CIRCLE_FILL_NGON:
        return verts_num;
      case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
        return verts_num + 1;
    }
    BLI_assert_unreachable();
    return 0;
  }

  static int circle_edge_total(const FillType fill_type, const int verts_num)
  {
    switch (fill_type) {
      case GEO_NODE_MESH_CIRCLE_FILL_NONE:
      case GEO_NODE_MESH_CIRCLE_FILL_NGON:
        return verts_num;
      case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
        return verts_num * 2;
    }
    BLI_assert_unreachable();
    return 0;
  }

  static int circle_corner_total(const FillType fill_type, const int verts_num)
  {
    switch (fill_type) {
      case GEO_NODE_MESH_CIRCLE_FILL_NONE:
        return 0;
      case GEO_NODE_MESH_CIRCLE_FILL_NGON:
        return verts_num;
      case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
        return verts_num * 3;
    }
    BLI_assert_unreachable();
    return 0;
  }

  static int circle_face_total(const FillType fill_type, const int verts_num)
  {
    switch (fill_type) {
      case GEO_NODE_MESH_CIRCLE_FILL_NONE:
        return 0;
      case GEO_NODE_MESH_CIRCLE_FILL_NGON:
        return 1;
      case GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN:
        return verts_num;
    }
    BLI_assert_unreachable();
    return 0;
  }

  static Mesh *create_ellipse_mesh(const float radius_a,
                                   const float radius_b,
                                   const int verts_num,
                                   const FillType fill_type)
  {
    Mesh *mesh = BKE_mesh_new_nomain(circle_vert_total(fill_type, verts_num),
                                     circle_edge_total(fill_type, verts_num),
                                     0,
                                     circle_corner_total(fill_type, verts_num),
                                     circle_face_total(fill_type, verts_num));
    BKE_id_material_eval_ensure_default_slot(&mesh->id);
    MutableSpan<MVert> verts{mesh->mvert, mesh->totvert};
    MutableSpan<MLoop> loops{mesh->mloop, mesh->totloop};
    MutableSpan<MEdge> edges{mesh->medge, mesh->totedge};
    MutableSpan<MPoly> polys{mesh->mpoly, mesh->totpoly};

    /* Assign vertex coordinates. */
    const float angle_delta = 2.0f * (M_PI / static_cast<float>(verts_num));
    for (const int i : IndexRange(verts_num)) {
      const float angle = i * angle_delta;
      copy_v3_v3(verts[i].co,
                 float3(std::cos(angle) * radius_a, std::sin(angle) * radius_b, 0.0f));
    }
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      copy_v3_v3(verts.last().co, float3(0));
    }

    /* Point all vertex normals in the up direction. */
    const short up_normal[3] = {0, 0, SHRT_MAX};
    for (MVert &vert : verts) {
      copy_v3_v3_short(vert.no, up_normal);
    }

    /* Create outer edges. */
    for (const int i : IndexRange(verts_num)) {
      MEdge &edge = edges[i];
      edge.v1 = i;
      edge.v2 = (i + 1) % verts_num;
    }

    /* Set loose edge flags. */
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NONE) {
      for (const int i : IndexRange(verts_num)) {
        MEdge &edge = edges[i];
        edge.flag |= ME_LOOSEEDGE;
      }
    }

    /* Create triangle fan edges. */
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      for (const int i : IndexRange(verts_num)) {
        MEdge &edge = edges[verts_num + i];
        edge.v1 = verts_num;
        edge.v2 = i;
        edge.flag = ME_EDGEDRAW | ME_EDGERENDER;
      }
    }

    /* Create corners and faces. */
    if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_NGON) {
      MPoly &poly = polys[0];
      poly.loopstart = 0;
      poly.totloop = loops.size();

      for (const int i : IndexRange(verts_num)) {
        MLoop &loop = loops[i];
        loop.e = i;
        loop.v = i;
      }
    }
    else if (fill_type == GEO_NODE_MESH_CIRCLE_FILL_TRIANGLE_FAN) {
      for (const int i : IndexRange(verts_num)) {
        MPoly &poly = polys[i];
        poly.loopstart = 3 * i;
        poly.totloop = 3;

        MLoop &loop_a = loops[3 * i];
        loop_a.e = i;
        loop_a.v = i;
        MLoop &loop_b = loops[3 * i + 1];
        loop_b.e = verts_num + ((i + 1) % verts_num);
        loop_b.v = (i + 1) % verts_num;
        MLoop &loop_c = loops[3 * i + 2];
        loop_c.e = verts_num + i;
        loop_c.v = verts_num;
      }
    }

    return mesh;
  }

  static void geometry_node_execute(GeoNodeExecParams params)
  {
    PointerRNA ptr;
    RNA_pointer_create((ID *)&params.node_tree(), &RNA_Node, (void *)&params.node(), &ptr);

    const FillType fill_type = (FillType)RNA_enum_get(&ptr, "fill_type");

    const float radius_a = params.extract_input<float>("Radius A");
    const float radius_b = params.extract_input<float>("Radius B");
    const int verts_num = params.extract_input<int>("Vertices");
    if (verts_num < 3) {
      params.error_message_add(NodeWarningType::Info, TIP_("Vertices must be at least 3"));
      params.set_output("Geometry", GeometrySet());
      return;
    }

    Mesh *mesh = create_ellipse_mesh(radius_a, radius_b, verts_num, fill_type);

    BLI_assert(BKE_mesh_is_valid(mesh));

    params.set_output("Geometry", GeometrySet::create_with_mesh(mesh));
  }
};

}  // namespace blender::nodes

void register_node_type_geo_mesh_primitive_ellipse()
{
  blender::nodes::GeometryNodeMeshEllipse::register_type();
}
