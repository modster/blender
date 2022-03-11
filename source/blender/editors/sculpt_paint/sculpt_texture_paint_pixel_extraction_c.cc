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

namespace blender::ed::sculpt_paint::texture_paint::barycentric_extraction {

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

  static GPUVertFormat format = {0};
  GPU_vertformat_attr_add(&format, "uv1", GPU_COMP_F32, 2, GPU_FETCH_FLOAT);
  GPU_vertformat_attr_add(&format, "uv2", GPU_COMP_F32, 2, GPU_FETCH_FLOAT);
  GPU_vertformat_attr_add(&format, "uv3", GPU_COMP_F32, 2, GPU_FETCH_FLOAT);
  GPU_vertformat_attr_add(&format, "node_index", GPU_COMP_I32, 1, GPU_FETCH_INT);
  GPU_vertformat_attr_add(&format, "poly_index", GPU_COMP_I32, 1, GPU_FETCH_INT);

  GPUShader *shader = GPU_shader_get_builtin_shader(GPU_SHADER_SCULPT_PIXEL_EXTRACTION);
  GPU_shader_bind(shader);
  const int polygons_loc = GPU_shader_get_ssbo(shader, "polygons");
  BLI_assert(polygons_loc != -1);

  ImBuf *image_buffer = ss->mode.texture_paint.drawing_target;
  GPUTexture *pixels_tex = GPU_texture_create_2d(
      "gpu_shader_compute_2d", image_buffer->x, image_buffer->y, 1, GPU_RGBA32I, nullptr);
  GPU_texture_image_bind(pixels_tex, GPU_shader_get_texture_binding(shader, "pixels"));

  int pixels_added = 0;
  int4 clear_pixel(-1);
  GPU_texture_clear(pixels_tex, GPU_DATA_INT, clear_pixel);
  for (int n = 0; n < nodes_to_initialize.size(); n++) {
    printf("node [%d/%ld]\n", n + 1, nodes_to_initialize.size());

    Vector<TexturePaintPolygon> polygons;
    PBVHNode *node = nodes[n];
    PBVHVertexIter vd;
    BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
      MeshElemMap *vert_map = &ss->pmap[vd.index];
      for (int j = 0; j < ss->pmap[vd.index].count; j++) {
        const int poly_index = vert_map->indices[j];
        const MPoly *p = &ss->mpoly[poly_index];
        if (p->totloop < 3) {
          continue;
        }

        for (int l = 0; l < p->totloop - 2; l++) {
          const int v1_loop_index = p->loopstart;
          const int v2_loop_index = p->loopstart + l + 1;
          const int v3_loop_index = p->loopstart + l + 2;
          TexturePaintPolygon polygon;
          polygon.uv[0] = ldata_uv[v1_loop_index].uv;
          polygon.uv[1] = ldata_uv[v2_loop_index].uv;
          polygon.uv[2] = ldata_uv[v3_loop_index].uv;
          polygon.pbvh_node_index = n;
          polygon.poly_index = poly_index;
          polygons.append(polygon);
        }
      }
    }
    BKE_pbvh_vertex_iter_end;
    printf("%ld polygons loaded\n", polygons.size());

    GPUVertBuf *vbo = GPU_vertbuf_create_with_format(&format);
    GPU_vertbuf_data_alloc(vbo, polygons.size());
    for (int i = 0; i < polygons.size(); i++) {
      GPU_vertbuf_vert_set(vbo, i, &polygons[i]);
    }
    GPU_vertbuf_bind_as_ssbo(vbo, polygons_loc);

    int2 calc_size(image_buffer->x, image_buffer->y);

    const int batch_size = 10000;
    for (int batch = 0; batch * batch_size < polygons.size(); batch++) {
      printf("batch %d\n", batch);
      GPU_shader_uniform_1i(shader, "from_polygon", batch * batch_size);
      GPU_shader_uniform_1i(
          shader, "to_polygon", min_ii(batch * batch_size + batch_size, polygons.size()));
      GPU_compute_dispatch(shader, calc_size.x, calc_size.y, 1);
    }
    GPU_memory_barrier(GPU_BARRIER_TEXTURE_FETCH);
    TexturePaintPixel *pixels = static_cast<TexturePaintPixel *>(
        GPU_texture_read(pixels_tex, GPU_DATA_INT, 0));
    GPU_vertbuf_discard(vbo);

    MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);
    for (int y = 0; y < calc_size.y; y++) {
      for (int x = 0; x < calc_size.x; x++) {
        int pixel_offset = y * image_buffer->x + x;
        float2 uv(float(x) / image_buffer->x, float(y) / image_buffer->y);
        TexturePaintPixel *pixel = &pixels[pixel_offset];
        // printf("%d %d: %d %d\n", x, y, pixel->poly_index, pixel->pbvh_node_index);
        if (pixel->poly_index == -1 || pixel->pbvh_node_index != n) {
          /* No intersection detected.*/
          continue;
        }
        BLI_assert(pixel->pbvh_node_index < nodes_to_initialize.size());
        PBVHNode *node = nodes_to_initialize[pixel->pbvh_node_index];
        NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
        const MPoly *p = &ss->mpoly[pixel->poly_index];
        const MLoop *loopstart = &ss->mloop[p->loopstart];

        bool intersection_validated = false;
        for (int l = 0; l < p->totloop - 2; l++) {
          const int v1_loop_index = p->loopstart;
          const int v2_loop_index = p->loopstart + l + 1;
          const int v3_loop_index = p->loopstart + l + 2;
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
          const int v2_index = loopstart[l + 1].v;
          const int v3_index = loopstart[l + 2].v;
          const float3 v1_pos = mvert[v1_index].co;
          const float3 v2_pos = mvert[v2_index].co;
          const float3 v3_pos = mvert[v3_index].co;
          float3 local_pos;
          interp_v3_v3v3v3(local_pos, v1_pos, v2_pos, v3_pos, weights);

          PixelData new_pixel;
          new_pixel.local_pos = local_pos;
          new_pixel.pixel_pos = int2(x, y);
          new_pixel.content = float4(&image_buffer->rect_float[pixel_offset * 4]);
          node_data->pixels.append(new_pixel);
          pixels_added += 1;

          intersection_validated = true;
          break;
        }
        BLI_assert(intersection_validated);
      }
    }
    printf("%d pixels added\n", pixels_added);
    MEM_freeN(pixels);
  }

  GPU_shader_unbind();
  GPU_texture_free(pixels_tex);
}
}  // namespace blender::ed::sculpt_paint::texture_paint::barycentric_extraction
extern "C" {
void SCULPT_extract_pixels(Object *ob, PBVHNode **nodes, int totnode)
{
  TIMEIT_START(extract_pixels);
  blender::ed::sculpt_paint::texture_paint::barycentric_extraction::init_using_intersection(
      ob, totnode, nodes);
  TIMEIT_END(extract_pixels);
}
}