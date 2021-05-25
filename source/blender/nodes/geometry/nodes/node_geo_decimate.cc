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

#include "UI_interface.h"
#include "UI_resources.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "BKE_mesh.h"
#include "BKE_mesh_types.h"
#include "BKE_deform.h"
#include "bmesh.h"
#include "bmesh_tools.h"

#include "node_geometry_util.hh"

extern "C" {
//Mesh *decimate_mesh(Mesh *mesh,
//                       const int quad_method,
//                       const int ngon_method,
//                       const int min_vertices,
//                       const int flag);
}

static bNodeSocketTemplate geo_node_decimate_in[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {SOCK_INT, N_("Minimum Vertices"), 4, 0, 0, 0, 4, 10000},
    {-1, ""},
};

static bNodeSocketTemplate geo_node_decimate_out[] = {
    {SOCK_GEOMETRY, N_("Geometry")},
    {-1, ""},
};

namespace blender::nodes{

typedef struct DecimateNodeData {
  /** (mode == MOD_DECIM_MODE_COLLAPSE). */
  float percent;
  /** (mode == MOD_DECIM_MODE_UNSUBDIV). */
  short iter;
  /** (mode == MOD_DECIM_MODE_DISSOLVE). */
  char delimit;
  /** (mode == MOD_DECIM_MODE_COLLAPSE). */
  char symmetry_axis;
  /** (mode == MOD_DECIM_MODE_DISSOLVE). */
  float angle;

  /** MAX_VGROUP_NAME. */
  char defgrp_name[64];
  float defgrp_factor;
  short flag, mode;

  /* runtime only */
  int face_count;
} DecimateNodeData;

static Mesh *decimateMesh(DecimateNodeData *dmd, Mesh *meshData)
{
  printf("RUNNING - A\n");

  Mesh *mesh = meshData, *result = NULL;
  BMesh *bm;
  bool calc_face_normal;
  float *vweights = NULL;

#ifdef USE_TIMEIT
  TIMEIT_START(decim);
#endif

  /* Set up front so we don't show invalid info in the UI. */
  //updateFaceCount(ctx, dmd, mesh->totpoly);

  switch (dmd->mode) {
    case MOD_DECIM_MODE_COLLAPSE:
      if (dmd->percent == 1.0f) {
        return mesh;
      }
      calc_face_normal = true;
      break;
    case MOD_DECIM_MODE_UNSUBDIV:
      if (dmd->iter == 0) {
        return mesh;
      }
      calc_face_normal = false;
      break;
    case MOD_DECIM_MODE_DISSOLVE:
      if (dmd->angle == 0.0f) {
        return mesh;
      }
      calc_face_normal = true;
      break;
    default:
      return mesh;
  }
  printf("RUNNING - B\n");
  if (dmd->face_count <= 3) {
    //BKE_modifier_set_error(ctx->object, md, "Modifier requires more than 3 input faces");
    return mesh;
  }
  printf("RUNNING - C\n");
  if (dmd->mode == MOD_DECIM_MODE_COLLAPSE) {
    if (dmd->defgrp_name[0] && (dmd->defgrp_factor > 0.0f)) {
      MDeformVert *dvert;
      int defgrp_index;

      //MOD_get_vgroup(ctx->object, mesh, dmd->defgrp_name, &dvert, &defgrp_index);

      if (dvert) {
        const uint vert_tot = mesh->totvert;
        uint i;

        vweights = (float*)MEM_malloc_arrayN(vert_tot, sizeof(float), __func__);

        if (dmd->flag & MOD_DECIM_FLAG_INVERT_VGROUP) {
          for (i = 0; i < vert_tot; i++) {
            vweights[i] = 1.0f - BKE_defvert_find_weight(&dvert[i], defgrp_index);
          }
        }
        else {
          for (i = 0; i < vert_tot; i++) {
            vweights[i] = BKE_defvert_find_weight(&dvert[i], defgrp_index);
          }
        }
      }
    }
  }
  printf("RUNNING - D\n");
  BMeshCreateParams bmesh_create_params = {0};
  BMeshFromMeshParams bmesh_from_mesh_params = {calc_face_normal,0,0,0,{CD_MASK_ORIGINDEX,CD_MASK_ORIGINDEX,CD_MASK_ORIGINDEX}};
  bm = BKE_mesh_to_bmesh_ex(mesh, &bmesh_create_params, &bmesh_from_mesh_params);

  switch (dmd->mode) {
    case MOD_DECIM_MODE_COLLAPSE: {
      const bool do_triangulate = (dmd->flag & MOD_DECIM_FLAG_TRIANGULATE) != 0;
      const int symmetry_axis = (dmd->flag & MOD_DECIM_FLAG_SYMMETRY) ? dmd->symmetry_axis : -1;
      const float symmetry_eps = 0.00002f;
      BM_mesh_decimate_collapse(bm,
                                dmd->percent,
                                vweights,
                                dmd->defgrp_factor,
                                do_triangulate,
                                symmetry_axis,
                                symmetry_eps);
      printf("RUNNING - E\n");
      break;
    }
    case MOD_DECIM_MODE_UNSUBDIV: {
      BM_mesh_decimate_unsubdivide(bm, dmd->iter);
      break;
    }
    case MOD_DECIM_MODE_DISSOLVE: {
      const bool do_dissolve_boundaries = (dmd->flag & MOD_DECIM_FLAG_ALL_BOUNDARY_VERTS) != 0;
      BM_mesh_decimate_dissolve(bm, dmd->angle, do_dissolve_boundaries, (BMO_Delimit)dmd->delimit);
      break;
    }
  }

  printf("RUNNING - F\n");
  if (vweights) {
    MEM_freeN(vweights);
  }

  //updateFaceCount(ctx, dmd, bm->totface);

  result = BKE_mesh_from_bmesh_for_eval_nomain(bm, NULL, mesh);
  /* make sure we never alloc'd these */
  BLI_assert(bm->vtoolflagpool == NULL && bm->etoolflagpool == NULL && bm->ftoolflagpool == NULL);
  BLI_assert(bm->vtable == NULL && bm->etable == NULL && bm->ftable == NULL);

  BM_mesh_free(bm);

#ifdef USE_TIMEIT
  TIMEIT_END(decim);
#endif

  result->runtime.cd_dirty_vert |= CD_MASK_NORMAL;
  printf("RUNNING - G\n");

  return result;
}

static void geo_node_decimate_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry");

  if(geometry_set.has_mesh()){
    Mesh *input_mesh = geometry_set.get_mesh_for_write();
    DecimateNodeData dmd = {
        /** (mode == MOD_DECIM_MODE_COLLAPSE). */
        0.5f,
        /** (mode == MOD_DECIM_MODE_UNSUBDIV). */
        2,
        /** (mode == MOD_DECIM_MODE_DISSOLVE). */
        5,
        /** (mode == MOD_DECIM_MODE_COLLAPSE). */
        0,
        /** (mode == MOD_DECIM_MODE_DISSOLVE). */
        30,

        /** MAX_VGROUP_NAME. */
        "",
        0.5f,
        0,
        MOD_DECIM_MODE_COLLAPSE,
        input_mesh->totpoly,
    };
    Mesh *result = decimateMesh(&dmd, input_mesh);
    geometry_set.replace_mesh(result);
    printf("RUNNING\n");
  }
  params.set_output("Geometry", std::move(geometry_set));
}
}  // namespace blender::nodes

void register_node_type_geo_decimate()
{
  static bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_DECIMATE, "decimate", NODE_CLASS_GEOMETRY, 0);
  node_type_socket_templates(&ntype, geo_node_decimate_in, geo_node_decimate_out);
  ntype.geometry_node_execute = blender::nodes::geo_node_decimate_exec;
  nodeRegisterType(&ntype);
}
