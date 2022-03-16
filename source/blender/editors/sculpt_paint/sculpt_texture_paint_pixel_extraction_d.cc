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

enum class ExtractionMethod { BarycentricEdges, BarycentricPixels };

/* Express co as term of movement along 2 edges of a triangle. */
static float3 barycentric_weights(const float2 v1,
                                  const float2 v2,
                                  const float2 v3,
                                  const float2 co)
{
  float3 weights;
  barycentric_weights_v2(v1, v2, v3, co, weights);
  return weights;
}

static bool is_inside_triangle(const float3 barycentric_weights)
{
  return barycentric_inside_triangle_v2(barycentric_weights);
}

static bool has_been_visited(std::vector<bool> &visited_polygons, const int poly_index)
{
  bool visited = visited_polygons[poly_index];
  visited_polygons[poly_index] = true;
  return visited;
}

static void extract_barycentric_edges(NodeData *node_data,
                                      const ImBuf *image_buffer,
                                      Triangle &triangle,
                                      const int triangle_index,
                                      const float2 uvs[3],
                                      const int minx,
                                      const int miny,
                                      const int maxx,
                                      const int maxy,
                                      const float minu,
                                      const float minv)
{
  const float add_u = 1.0 / image_buffer->x;
  const float add_v = 1.0 / image_buffer->y;

  float2 min_uv(minu, minv);
  float3 start_barycentric_coord = barycentric_weights(uvs[0], uvs[1], uvs[2], min_uv);
  float3 add_barycentric_coord_x = barycentric_weights(
                                uvs[0], uvs[1], uvs[2], min_uv + float2(add_u, 0.0)) -
                            start_barycentric_coord;
  float3 add_edge_coord_y = barycentric_weights(
                                uvs[0], uvs[1], uvs[2], min_uv + float2(0.0, add_v)) -
                            start_barycentric_coord;

  triangle.add_barycentric_coord_x = add_barycentric_coord_x;

  for (int y = miny; y < maxy; y++) {
    float3 start_y_edge_coord = start_barycentric_coord + add_edge_coord_y * (y - miny);
    float3 edge_coord = start_y_edge_coord;

    int start_x = -1;
    int end_x = -1;
    int x;
    for (x = minx; x < maxx; x++) {
      if (is_inside_triangle(edge_coord)) {
        start_x = x;
        break;
      }
      edge_coord += add_barycentric_coord_x;
    }
    edge_coord += add_barycentric_coord_x;
    x += 1;
    for (; x < maxx; x++) {
      if (!is_inside_triangle(edge_coord)) {
        break;
      }
      edge_coord += add_barycentric_coord_x;
    }
    end_x = x;

    if (start_x == -1 || end_x == -1) {
      continue;
    }

    int num_pixels = end_x - start_x;

    PixelsPackage package;
    package.start_image_coordinate = int2(start_x, y);
    package.start_barycentric_coord = start_y_edge_coord + add_barycentric_coord_x * (start_x - minx);
    package.triangle_index = triangle_index;
    package.num_pixels = num_pixels;
    node_data->encoded_pixels.append(package);
  }
}

static void extract_barycentric_pixels(NodeData *node_data,
                                       const ImBuf *image_buffer,
                                       Triangle &triangle,
                                       const int triangle_index,
                                       const float2 uvs[3],
                                       const int minx,
                                       const int miny,
                                       const int maxx,
                                       const int maxy)
{
  int best_num_pixels = 0;
  for (int y = miny; y < maxy; y++) {
    bool start_detected = false;
    float3 barycentric;
    PixelsPackage package;
    package.triangle_index = triangle_index;
    package.num_pixels = 0;
    int x;

    for (x = minx; x < maxx; x++) {
      float2 uv(float(x) / image_buffer->x, float(y) / image_buffer->y);
      barycentric = barycentric_weights(uvs[0], uvs[1], uvs[2], uv);
      const bool is_inside = is_inside_triangle(barycentric);
      if (!start_detected && is_inside) {
        start_detected = true;
        package.start_image_coordinate = int2(x, y);
        package.start_barycentric_coord = barycentric;
      }
      else if (start_detected && !is_inside) {
        break;
      }
    }

    if (!start_detected) {
      continue;
    }
    package.num_pixels = x - package.start_image_coordinate.x;
    if (package.num_pixels > best_num_pixels) {
      triangle.add_barycentric_coord_x = (barycentric - package.start_barycentric_coord) / package.num_pixels;
      best_num_pixels = package.num_pixels;
    }
    node_data->encoded_pixels.append(package);
  }
}

static void init_triangles(SculptSession *ss,
                           PBVHNode *node,
                           NodeData *node_data,
                           std::vector<bool> &visited_polygons)
{
  PBVHVertexIter vd;

  BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
    MeshElemMap *vert_map = &ss->pmap[vd.index];
    for (int j = 0; j < ss->pmap[vd.index].count; j++) {
      const int poly_index = vert_map->indices[j];
      if (has_been_visited(visited_polygons, poly_index)) {
        continue;
      }

      const MPoly *p = &ss->mpoly[poly_index];
      const MLoop *loopstart = &ss->mloop[p->loopstart];
      for (int l = 0; l < p->totloop - 2; l++) {
        Triangle triangle;
        triangle.loop_indices = int3(p->loopstart, p->loopstart + l + 1, p->loopstart + l + 2);
        triangle.vert_indices = int3(loopstart[0].v, loopstart[l + 1].v, loopstart[l + 2].v);
        triangle.poly_index = poly_index;
        node_data->triangles.append(triangle);
      }
    }
  }
  BKE_pbvh_vertex_iter_end;
}

struct EncodePixelsUserData {
  ImBuf *image_buffer;
  Vector<PBVHNode *> *nodes;
  MLoopUV *ldata_uv;
  ExtractionMethod method;
};

static void do_encode_pixels(void *__restrict userdata,
                             const int n,
                             const TaskParallelTLS *__restrict UNUSED(tls))
{
  EncodePixelsUserData *data = static_cast<EncodePixelsUserData *>(userdata);

  PBVHNode *node = (*data->nodes)[n];
  NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
  int triangle_index = 0;
  for (Triangle &triangle : node_data->triangles) {
    float2 uvs[3] = {
        data->ldata_uv[triangle.loop_indices[0]].uv,
        data->ldata_uv[triangle.loop_indices[1]].uv,
        data->ldata_uv[triangle.loop_indices[2]].uv,
    };

    const float minv = min_fff(uvs[0].y, uvs[1].y, uvs[2].y);
    const int miny = floor(minv * data->image_buffer->y);
    const float maxv = max_fff(uvs[0].y, uvs[1].y, uvs[2].y);
    const int maxy = ceil(maxv * data->image_buffer->y);
    const float minu = min_fff(uvs[0].x, uvs[1].x, uvs[2].x);
    const int minx = floor(minu * data->image_buffer->x);
    const float maxu = max_fff(uvs[0].x, uvs[1].x, uvs[2].x);
    const int maxx = ceil(maxu * data->image_buffer->x);

    switch (data->method) {
      case ExtractionMethod::BarycentricEdges: {
        extract_barycentric_edges(node_data,
                                  data->image_buffer,
                                  triangle,
                                  triangle_index,
                                  uvs,
                                  minx,
                                  miny,
                                  maxx,
                                  maxy,
                                  minu,
                                  minv);
        break;
      }

      case ExtractionMethod::BarycentricPixels: {
        extract_barycentric_pixels(
            node_data, data->image_buffer, triangle, triangle_index, uvs, minx, miny, maxx, maxy);
        break;
      }
    }

    triangle_index += 1;
  }
}

static void init(const Object *ob, int totnode, PBVHNode **nodes, const ExtractionMethod method)
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
    PBVHNode *node = nodes_to_initialize[n];
    NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
    init_triangles(ss, node, node_data, visited_polygons);
  }

  EncodePixelsUserData user_data;
  user_data.image_buffer = image_buffer;
  user_data.ldata_uv = ldata_uv;
  user_data.method = method;
  user_data.nodes = &nodes_to_initialize;

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, nodes_to_initialize.size());
  BLI_task_parallel_range(0, nodes_to_initialize.size(), &user_data, do_encode_pixels, &settings);

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

//#define DO_WATERTIGHT_CHECK
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
  using namespace blender::ed::sculpt_paint::texture_paint::packed_pixels;
  TIMEIT_START(extract_pixels);
  init(ob, totnode, nodes, ExtractionMethod::BarycentricPixels);
  TIMEIT_END(extract_pixels);
}
}