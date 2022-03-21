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

static void extract_barycentric_pixels(TileData &tile_data,
                                       const ImBuf *image_buffer,
                                       TrianglePaintInput &triangle,
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
        package.start_image_coordinate = ushort2(x, y);
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
      triangle.add_barycentric_coord_x = (barycentric - package.start_barycentric_coord.decode()) /
                                         package.num_pixels;
      best_num_pixels = package.num_pixels;
    }
    tile_data.encoded_pixels.append(package);
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
  Image *image;
  ImageUser *image_user;
  Vector<PBVHNode *> *nodes;
  MLoopUV *ldata_uv;
};

static void do_encode_pixels(void *__restrict userdata,
                             const int n,
                             const TaskParallelTLS *__restrict UNUSED(tls))
{
  EncodePixelsUserData *data = static_cast<EncodePixelsUserData *>(userdata);
  Image *image = data->image;
  ImageUser image_user = *data->image_user;

  PBVHNode *node = (*data->nodes)[n];
  NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
  LISTBASE_FOREACH (ImageTile *, tile, &data->image->tiles) {
    imbuf::ImageTileWrapper image_tile(tile);
    image_user.tile = image_tile.get_tile_number();
    ImBuf *image_buffer = BKE_image_acquire_ibuf(image, &image_user, nullptr);
    if (image_buffer == nullptr) {
      continue;
    }

    float2 tile_offset = float2(image_tile.get_tile_offset());
    TileData tile_data;

    Triangles &triangles = node_data->triangles;
    for (int triangle_index = 0; triangle_index < triangles.size(); triangle_index++) {
      int3 loop_indices = triangles.get_loop_indices(triangle_index);
      float2 uvs[3] = {
          float2(data->ldata_uv[loop_indices[0]].uv) - tile_offset,
          float2(data->ldata_uv[loop_indices[1]].uv) - tile_offset,
          float2(data->ldata_uv[loop_indices[2]].uv) - tile_offset,
      };

      const float minv = clamp_f(min_fff(uvs[0].y, uvs[1].y, uvs[2].y), 0.0f, 1.0f);
      const int miny = floor(minv * image_buffer->y);
      const float maxv = clamp_f(max_fff(uvs[0].y, uvs[1].y, uvs[2].y), 0.0f, 1.0f);
      const int maxy = min_ii(ceil(maxv * image_buffer->y), image_buffer->y);
      const float minu = clamp_f(min_fff(uvs[0].x, uvs[1].x, uvs[2].x), 0.0f, 1.0f);
      const int minx = floor(minu * image_buffer->x);
      const float maxu = clamp_f(max_fff(uvs[0].x, uvs[1].x, uvs[2].x), 0.0f, 1.0f);
      const int maxx = min_ii(ceil(maxu * image_buffer->x), image_buffer->x);

      TrianglePaintInput &triangle = triangles.get_paint_input(triangle_index);
      extract_barycentric_pixels(
          tile_data, image_buffer, triangle, triangle_index, uvs, minx, miny, maxx, maxy);
    }

    BKE_image_release_ibuf(image, image_buffer, nullptr);

    if (tile_data.encoded_pixels.is_empty()) {
      continue;
    }

    tile_data.tile_number = image_tile.get_tile_number();
    node_data->tiles.append(tile_data);
  }
  node_data->triangles.cleanup_after_init();
}

static void init(const Object *ob, int totnode, PBVHNode **nodes)
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
  printf("%lld nodes to initialize\n", nodes_to_initialize.size());

  Mesh *mesh = static_cast<Mesh *>(ob->data);
  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return;
  }
  SculptSession *ss = ob->sculpt;
  std::vector<bool> visited_polygons(mesh->totpoly);

  for (int n = 0; n < nodes_to_initialize.size(); n++) {
    PBVHNode *node = nodes_to_initialize[n];
    NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
    init_triangles(ss, node, node_data, visited_polygons);
  }

  EncodePixelsUserData user_data;
  user_data.image = ss->mode.texture_paint.image;
  user_data.image_user = ss->mode.texture_paint.image_user;
  user_data.ldata_uv = ldata_uv;
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
      compressed_data_len += node_data->triangles.mem_size();
      for (const TileData &tile_data : node_data->tiles) {
        compressed_data_len += tile_data.encoded_pixels.size() * sizeof(PixelsPackage);
        for (const PixelsPackage &encoded_pixels : tile_data.encoded_pixels) {
          num_pixels += encoded_pixels.num_pixels;
        }
      }
    }
    printf("Encoded %lld pixels in %lld bytes (%f bytes per pixel)\n",
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
  init(ob, totnode, nodes);
  TIMEIT_END(extract_pixels);
}
}