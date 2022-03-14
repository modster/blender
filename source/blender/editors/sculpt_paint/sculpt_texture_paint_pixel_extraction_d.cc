#include "DNA_material_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_scene_types.h"
#include "DNA_windowmanager_types.h"

#include "BKE_brush.h"
#include "BKE_context.h"
#include "BKE_customdata.h"
#include "BKE_image.h"
#include "BKE_material.h"
#include "BKE_mesh.h"
#include "BKE_mesh_mapping.h"
#include "BKE_pbvh.h"

#include "PIL_time_utildefines.h"

#include "BLI_task.h"
#include "BLI_vector.hh"

#include "IMB_rasterizer.hh"

#include "WM_types.h"

#include "bmesh.h"

#include "ED_uvedit.h"

#include "GPU_compute.h"
#include "GPU_shader.h"
#include "GPU_shader_shared.h"
#include "GPU_texture.h"
#include "GPU_uniform_buffer.h"
#include "GPU_vertex_buffer.h"
#include "GPU_vertex_format.h"

#include "sculpt_intern.h"
#include "sculpt_texture_paint_intern.hh"

namespace blender::ed::sculpt_paint::texture_paint::packed_pixels {

/* Express co as term of movement along 2 edges of a triangle. */
static float3 packed_edge_coordinate(const float2 v1,
                                     const float2 v2,
                                     const float2 v3,
                                     const float2 co)
{
#if 0
  float2 v13 = v3 - v1;
  float2 v12 = v2 - v1;
  float2 isect_point;
  float2 tmp;

  float2 point_1 = co - v13;
  isect_line_line_v2_point(v1, v2, co, point_1, isect_point);
  print_v2_id(v1);
  print_v2_id(v2);
  print_v2_id(co);
  print_v2_id(point_1);
  print_v2_id(isect_point);
  float d1 = closest_to_line_v2(tmp, isect_point, v1, v2);
  printf("d: %f\n")
  float2 point_2 = co - v12;
  isect_line_line_v2_point(v1, v3, co, point_2, isect_point);
  print_v2_id(v1);
  print_v2_id(v3);
  print_v2_id(co);
  print_v2_id(point_2);
  print_v2_id(isect_point);
  float d2 = closest_to_line_v2(tmp, isect_point, v1, v3);
  return float2(d1, d2);
#else
  float3 weights;
  barycentric_weights_v2(v1, v2, v3, co, weights);
  return weights;
#endif
}

static bool is_inside_triangle(const float3 co)
{
#if 0
  if (co.x < 0.0 || co.x > 1.0) {
    return false;
  }
  if (co.y < 0.0 || co.y > 1.0) {
    return false;
  }
  const float v = co.x + co.y;
  if (v > 1.0) {
    return false;
  }
  return true;
#else
  return barycentric_inside_triangle_v2(co);
#endif
}

static void init(Object *ob, int totnode, PBVHNode **nodes)
{
  Vector<PBVHNode *> nodes_to_initialize;
  for (int n = 0; n < totnode; n++) {
    PBVHNode *node = nodes[n];
    NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
    if (node_data != nullptr) {
      continue;
    }
    node_data = MEM_new<NodeData>(__func__);
    BKE_pbvh_node_texture_paint_data_set(node, node_data, NodeData::free_func);
    nodes_to_initialize.append(node);
  }
  if (nodes_to_initialize.size() == 0) {
    return;
  }
  printf("%ld nodes to initialize\n", nodes_to_initialize.size());

  Mesh *mesh = static_cast<Mesh *>(ob->data);
  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return;
  }
  SculptSession *ss = ob->sculpt;
  ImBuf *image_buffer = ss->mode.texture_paint.drawing_target;
  std::vector<bool> visited_polygons(mesh->totpoly);

  for (int n = 0; n < nodes_to_initialize.size(); n++) {
    printf("node [%d/%ld]\n", n + 1, nodes_to_initialize.size());

    Vector<TexturePaintPolygon> polygons;
    PBVHNode *node = nodes[n];
    NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
    PBVHVertexIter vd;
    int num_encoded_pixels = 0;

    BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
      MeshElemMap *vert_map = &ss->pmap[vd.index];
      for (int j = 0; j < ss->pmap[vd.index].count; j++) {
        const int poly_index = vert_map->indices[j];
        if (visited_polygons[poly_index]) {
          continue;
        }
        visited_polygons[poly_index] = true;

        const MPoly *p = &ss->mpoly[poly_index];
        const MLoop *loopstart = &ss->mloop[p->loopstart];
        for (int l = 0; l < p->totloop - 2; l++) {
          Triangle triangle;
          triangle.loop_indices = int3(p->loopstart, p->loopstart + l + 1, p->loopstart + l + 2);
          triangle.vert_indices = int3(loopstart[0].v, loopstart[l + 1].v, loopstart[l + 2].v);
          triangle.poly_index = poly_index;
          float2 uvs[3] = {
              ldata_uv[triangle.loop_indices[0]].uv,
              ldata_uv[triangle.loop_indices[1]].uv,
              ldata_uv[triangle.loop_indices[2]].uv,
          };

          // print_v2_id(uvs[0]);
          // print_v2_id(uvs[1]);
          // print_v2_id(uvs[2]);

          const float minv = min_fff(uvs[0].y, uvs[1].y, uvs[2].y);
          const int miny = floor(minv * image_buffer->y);
          const float maxv = max_fff(uvs[0].y, uvs[1].y, uvs[2].y);
          const int maxy = ceil(maxv * image_buffer->y);
          const float minu = min_fff(uvs[0].x, uvs[1].x, uvs[2].x);
          const int minx = floor(minu * image_buffer->x);
          const float maxu = max_fff(uvs[0].x, uvs[1].x, uvs[2].x);
          const int maxx = ceil(maxu * image_buffer->x);
          const float add_u = 1.0 / image_buffer->x;
          const float add_v = 1.0 / image_buffer->y;

          // printf("(%d, %d) - (%d, %d)\n", minx, miny, maxx, maxy);

          float2 min_uv(minu, minv);
          float3 start_edge_coord = packed_edge_coordinate(uvs[0], uvs[1], uvs[2], min_uv);
          float3 add_edge_coord_x = packed_edge_coordinate(
                                        uvs[0], uvs[1], uvs[2], min_uv + float2(add_u, 0.0)) -
                                    start_edge_coord;
          float3 add_edge_coord_y = packed_edge_coordinate(
                                        uvs[0], uvs[1], uvs[2], min_uv + float2(0.0, add_v)) -
                                    start_edge_coord;
          // print_v3_id(start_edge_coord);
          // print_v3_id(add_edge_coord_x);
          // print_v3_id(add_edge_coord_y);

          triangle.add_edge_coord_x = add_edge_coord_x;
          int triangle_index = node_data->triangles.size();
          node_data->triangles.append(triangle);

          int num_packages = 0;

          for (int y = miny; y < maxy; y++) {
            float3 start_y_edge_coord = start_edge_coord + add_edge_coord_y * (y - miny);
            float3 edge_coord = start_y_edge_coord;

            int start_x = -1;
            int end_x = -1;
            int x;
            for (x = minx; x < maxx; x++) {
              // printf("(%d, %d)", x, y);
              // print_v2_id(edge_coord);
              if (is_inside_triangle(edge_coord)) {
                start_x = x;
                break;
              }
              edge_coord += add_edge_coord_x;
            }
            edge_coord += add_edge_coord_x;
            x += 1;
            for (; x < maxx; x++) {
              // printf("(%d, %d)", x, y);
              // print_v2_id(edge_coord);
              if (!is_inside_triangle(edge_coord)) {
                break;
              }
              edge_coord += add_edge_coord_x;
            }
            end_x = x;

            if (start_x == -1 || end_x == -1) {
              continue;
            }

            int num_pixels = end_x - start_x;
            PixelsPackage package;
            package.start_image_coordinate = int2(start_x, y);
            package.start_edge_coord = start_y_edge_coord + add_edge_coord_x * (start_x - minx);
            package.triangle_index = triangle_index;
            package.num_pixels = num_pixels;
            node_data->encoded_pixels.append(package);
            num_encoded_pixels += num_pixels;
            // printf("x: %d y: %d, cx: %f, cy: %f, len: %d\n",
            //  package.start_image_coordinate.x,
            //  package.start_image_coordinate.y,
            //  package.start_edge_coord.x,
            //  package.start_edge_coord.y,
            //  package.num_pixels);
            num_packages += 1;
          }
          // printf("new pixel packages created: %d\n", num_packages);
        }
      }
    }
    BKE_pbvh_vertex_iter_end;

    // printf(" - encoded %d pixels into %lu bytes\n",
    //  num_encoded_pixels,
    //  node_data->encoded_pixels.size() * sizeof(PixelsPackage) +
    //  node_data->triangles.size() * sizeof(Triangle));
  }

  {
    int64_t compressed_data_len = 0;
    int64_t num_pixels = 0;
    for (int n = 0; n < totnode; n++) {
      PBVHNode *node = nodes[n];
      NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
      compressed_data_len += node_data->triangles.size() * sizeof(Triangle);
      compressed_data_len += node_data->encoded_pixels.size() * sizeof(PixelsPackage);

      for (PixelsPackage &encoded_pixels : node_data->encoded_pixels) {
        num_pixels += encoded_pixels.num_pixels;
      }
    }
    printf("Encoded %ld pixels in %ld bytes (%f bytes per pixel)\n",
           num_pixels,
           compressed_data_len,
           float(compressed_data_len) / num_pixels);
  }

// #define DO_WATERTIGHT_CHECK
#ifdef DO_WATERTIGHT_CHECK
  for (int n = 0; n < nodes_to_initialize.size(); n++) {
    PBVHNode *node = nodes[n];
    NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
    for (PixelsPackage &encoded_pixels : node_data->encoded_pixels) {
      int pixel_offset = encoded_pixels.start_image_coordinate.y * image_buffer->x +
                         encoded_pixels.start_image_coordinate.x;
      for (int x = 0; x < encoded_pixels.num_pixels; x++) {
        copy_v4_fl(&image_buffer->rect_float[pixel_offset * 4], 1.0);
        pixel_offset += 1;
      }
    }
  }

#endif
}

}  // namespace blender::ed::sculpt_paint::texture_paint::packed_pixels

extern "C" {
void SCULPT_extract_pixels(Object *ob, PBVHNode **nodes, int totnode)
{
  TIMEIT_START(extract_pixels);
  blender::ed::sculpt_paint::texture_paint::packed_pixels::init(ob, totnode, nodes);
  TIMEIT_END(extract_pixels);
}
}