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

#include "BKE_solidifiy.h"

#include "DNA_modifier_types.h"

#include "UI_interface.h"
#include "UI_resources.h"

#include "node_geometry_util.hh"

static bNodeSocketTemplate geo_node_solidify_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_FLOAT, N_("Thickness"), 0.1f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_FLOAT, N_("Offset"), -1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 1.0f},
    {SOCK_FLOAT, N_("Clamp Offset"), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f},
    {SOCK_BOOLEAN, N_("Fill"), true},
    {SOCK_BOOLEAN, N_("Rim"), true},

    {-1, ""},
};

static bNodeSocketTemplate geo_node_solidify_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes {

static void geo_node_solidify_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");
  bool add_rim = params.extract_input<bool>("Rim");
  bool add_fill = params.extract_input<bool>("Fill");

  char flag = MOD_SOLIDIFY_NORMAL_CALC;
  if(add_rim){
    flag |= MOD_SOLIDIFY_RIM;
  }
  if(!add_fill){
    flag |= MOD_SOLIDIFY_NOSHELL;
  }
  float thickness = params.extract_input<float>("Thickness");
  float offset = params.extract_input<float>("Offset");
  float offset_clamp = params.extract_input<float>("Clamp Offset");

  geometry_set = geometry_set_realize_instances(geometry_set);

  if (geometry_set.has<MeshComponent>()) {
    SolidifyData solidify_node_data = {
      /** Name of vertex group to use, MAX_VGROUP_NAME. */
      "char defgrp_name[64]",
      "shell_defgrp_name[64]",
      "rim_defgrp_name[64]",
      /** New surface offset level. */
      thickness,
      /** Midpoint of the offset. */
      offset,
      /**
       * Factor for the minimum weight to use when vertex-groups are used,
       * avoids 0.0 weights giving duplicate geometry.
       */
      0.0f,
      /** Clamp offset based on surrounding geometry. */
      offset_clamp,
      MOD_SOLIDIFY_MODE_EXTRUDE,

      /** Variables for #MOD_SOLIDIFY_MODE_NONMANIFOLD. */
      MOD_SOLIDIFY_NONMANIFOLD_OFFSET_MODE_FIXED,
      MOD_SOLIDIFY_NONMANIFOLD_BOUNDARY_MODE_NONE,

      0,
      0.0f,
      0.0f,
      0.0f,
      flag,
      0,
      0,
      0.01f,
      0.0f,
    };

    MeshComponent &meshComponent = geometry_set.get_component_for_write<MeshComponent>();
    Mesh *return_mesh = solidify_extrude(&solidify_node_data, meshComponent.get_for_write());
    geometry_set.replace_mesh(return_mesh);
  }

  params.set_output("Geometry", geometry_set);
}

}  // namespace blender::nodes

void register_node_type_geo_solidify()
{
  static bNodeType ntype;
  geo_node_type_base(&ntype, GEO_NODE_SOLIDIFY, "Solidify", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_solidify_in, geo_node_solidify_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_solidify_exec;
  nodeRegisterType(&ntype);
}
