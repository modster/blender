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

#include "sculpt_intern.h"
#include "sculpt_texture_paint_intern.hh"

namespace blender::ed::sculpt_paint::texture_paint::barycentric_extraction {

struct BucketEntry {
  PBVHNode *node;
  const MPoly *poly;
  rctf uv_bounds;
};
struct Bucket {
  static const int Size = 16;
  Vector<BucketEntry> entries;
  rctf bounds;
};

static bool init_using_intersection(SculptSession *ss,
                                    Bucket &bucket,
                                    ImBuf *image_buffer,
                                    MVert *mvert,
                                    MLoopUV *ldata_uv,
                                    float2 uv,
                                    int2 xy)
{
  const int pixel_offset = xy[1] * image_buffer->x + xy[0];
  for (BucketEntry &entry : bucket.entries) {
    if (!BLI_rctf_isect_pt_v(&entry.uv_bounds, uv)) {
      continue;
    }
    const MPoly *p = entry.poly;

    const MLoop *loopstart = &ss->mloop[p->loopstart];
    for (int triangle = 0; triangle < p->totloop - 2; triangle++) {
      const int v1_loop_index = p->loopstart;
      const int v2_loop_index = p->loopstart + triangle + 1;
      const int v3_loop_index = p->loopstart + triangle + 2;
      const float2 v1_uv = ldata_uv[v1_loop_index].uv;
      const float2 v2_uv = ldata_uv[v2_loop_index].uv;
      const float2 v3_uv = ldata_uv[v3_loop_index].uv;
      float3 weights;
      barycentric_weights_v2(v1_uv, v2_uv, v3_uv, uv, weights);
      if (weights[0] < 0.0 || weights[0] > 1.0 || weights[1] < 0.0 || weights[1] > 1.0 ||
          weights[2] < 0.0 || weights[2] > 1.0) {
        continue;
      }

      const int v1_index = loopstart[0].v;
      const int v2_index = loopstart[triangle + 1].v;
      const int v3_index = loopstart[triangle + 2].v;
      const float3 v1_pos = mvert[v1_index].co;
      const float3 v2_pos = mvert[v2_index].co;
      const float3 v3_pos = mvert[v3_index].co;
      float3 local_pos;
      interp_v3_v3v3v3(local_pos, v1_pos, v2_pos, v3_pos, weights);

      PixelData new_pixel;
      new_pixel.local_pos = local_pos;
      new_pixel.pixel_pos = xy;
      new_pixel.content = float4(&image_buffer->rect_float[pixel_offset * 4]);

      PBVHNode *node = entry.node;
      NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
      node_data->pixels.append(new_pixel);
      return true;
    }
  }
  return false;
}

static void init_using_intersection(Object *ob, int totnode, PBVHNode **nodes)
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
    BLI_rctf_init_minmax(&node_data->uv_region);
    nodes_to_initialize.append(node);
  }
  if (nodes_to_initialize.size() == 0) {
    return;
  }

  TIMEIT_START(extract_pixels);
  Mesh *mesh = static_cast<Mesh *>(ob->data);
  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return;
  }

  SculptSession *ss = ob->sculpt;
  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);
  ImBuf *image_buffer = ss->mode.texture_paint.drawing_target;
  int pixels_added = 0;
  Bucket bucket;
  for (int y_bucket = 0; y_bucket < image_buffer->y; y_bucket += Bucket::Size) {
    printf("%d: %d pixels added.\n", y_bucket, pixels_added);
    for (int x_bucket = 0; x_bucket < image_buffer->x; x_bucket += Bucket::Size) {
      bucket.entries.clear();
      BLI_rctf_init(&bucket.bounds,
                    float(x_bucket) / image_buffer->x,
                    float(x_bucket + Bucket::Size) / image_buffer->x,
                    float(y_bucket) / image_buffer->y,
                    float(y_bucket + Bucket::Size) / image_buffer->y);
      // print_rctf_id(&bucket.bounds);

      for (int n = 0; n < nodes_to_initialize.size(); n++) {
        PBVHNode *node = nodes_to_initialize[n];
        PBVHVertexIter vd;
        NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
        if (BLI_rctf_is_valid(&node_data->uv_region)) {
          if (!BLI_rctf_isect(&bucket.bounds, &node_data->uv_region, nullptr)) {
            continue;
          }
        }

        BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
          MeshElemMap *vert_map = &ss->pmap[vd.index];
          for (int j = 0; j < ss->pmap[vd.index].count; j++) {
            const MPoly *p = &ss->mpoly[vert_map->indices[j]];
            if (p->totloop < 3) {
              continue;
            }

            rctf poly_bound;
            BLI_rctf_init_minmax(&poly_bound);
            for (int l = 0; l < p->totloop; l++) {
              const int v_loop_index = p->loopstart + l;
              const float2 v_uv = ldata_uv[v_loop_index].uv;
              BLI_rctf_do_minmax_v(&poly_bound, v_uv);
              BLI_rctf_do_minmax_v(&node_data->uv_region, v_uv);
            }
            if (BLI_rctf_isect(&bucket.bounds, &poly_bound, nullptr)) {
              BucketEntry entry;
              entry.node = node;
              entry.poly = p;
              entry.uv_bounds = poly_bound;
              bucket.entries.append(entry);
            }
          }
        }
        BKE_pbvh_vertex_iter_end;
      }
      // printf("Loaded %ld entries in bucket\n", bucket.entries.size());
      if (bucket.entries.size() == 0) {
        continue;
      }

      for (int y = y_bucket; y < image_buffer->y && y < y_bucket + Bucket::Size; y++) {
        for (int x = x_bucket; x < image_buffer->x && x < x_bucket + Bucket::Size; x++) {
          float2 uv(float(x) / image_buffer->x, float(y) / image_buffer->y);
          if (init_using_intersection(ss, bucket, image_buffer, mvert, ldata_uv, uv, int2(x, y))) {
            pixels_added++;
          }
        }
      }
    }
  }
  TIMEIT_END(extract_pixels);
}
}  // namespace blender::ed::sculpt_paint::texture_paint::barycentric_extraction
extern "C" {
void SCULPT_extract_pixels(Object *ob, PBVHNode **nodes, int totnode)
{
  blender::ed::sculpt_paint::texture_paint::barycentric_extraction::init_using_intersection(
      ob, totnode, nodes);
}
}