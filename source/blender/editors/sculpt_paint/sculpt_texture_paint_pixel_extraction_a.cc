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

namespace blender::ed::sculpt_paint::texture_paint {

namespace rasterization {

using namespace imbuf::rasterizer;

struct VertexInput {
  float3 pos;
  float2 uv;

  VertexInput(float3 pos, float2 uv) : pos(pos), uv(uv)
  {
  }
};

class VertexShader : public AbstractVertexShader<VertexInput, float3> {
 public:
  float2 image_size;
  void vertex(const VertexInputType &input, VertexOutputType *r_output) override
  {
    r_output->coord = input.uv * image_size;
    r_output->data = input.pos;
  }
};

struct FragmentOutput {
  float3 local_pos;
};

class FragmentShader : public AbstractFragmentShader<float3, FragmentOutput> {
 public:
  ImBuf *image_buffer;

 public:
  void fragment(const FragmentInputType &input, FragmentOutputType *r_output) override
  {
    r_output->local_pos = input;
  }
};

struct NodeDataPair {
  ImBuf *image_buffer;
  NodeData *node_data;

  struct {
    /* Rasterizer doesn't support glCoord yet, so for now we just store them in a runtime section.
     */
    int2 last_known_pixel_pos;
  } runtime;
};

class AddPixel : public AbstractBlendMode<FragmentOutput, NodeDataPair> {
 public:
  void blend(NodeDataPair *dest, const FragmentOutput &source) const override
  {
    PixelData new_pixel;
    new_pixel.local_pos = source.local_pos;
    new_pixel.pixel_pos = dest->runtime.last_known_pixel_pos;
    const int pixel_offset = new_pixel.pixel_pos[1] * dest->image_buffer->x +
                             new_pixel.pixel_pos[0];
    new_pixel.content = float4(dest->image_buffer->rect_float[pixel_offset * 4]);
    new_pixel.flags.dirty = false;

    dest->node_data->pixels.append(new_pixel);
    dest->runtime.last_known_pixel_pos[0] += 1;
  }
};

class NodeDataDrawingTarget : public AbstractDrawingTarget<NodeDataPair, NodeDataPair> {
 private:
  NodeDataPair *active_ = nullptr;

 public:
  uint64_t get_width() const
  {
    return active_->image_buffer->x;
  }
  uint64_t get_height() const
  {
    return active_->image_buffer->y;
  };
  NodeDataPair *get_pixel_ptr(uint64_t x, uint64_t y)
  {
    active_->runtime.last_known_pixel_pos = int2(x, y);
    return active_;
  };
  int64_t get_pixel_stride() const
  {
    return 0;
  };
  bool has_active_target() const
  {
    return active_ != nullptr;
  }
  void activate(NodeDataPair *instance)
  {
    active_ = instance;
  };
  void deactivate()
  {
    active_ = nullptr;
  }
};

using RasterizerType = Rasterizer<VertexShader, FragmentShader, AddPixel, NodeDataDrawingTarget>;

static void init_rasterization_task_cb_ex(void *__restrict userdata,
                                          const int n,
                                          const TaskParallelTLS *__restrict UNUSED(tls))
{
  TexturePaintingUserData *data = static_cast<TexturePaintingUserData *>(userdata);
  Object *ob = data->ob;
  SculptSession *ss = ob->sculpt;
  PBVHNode *node = data->nodes[n];

  NodeData *node_data = static_cast<NodeData *>(BKE_pbvh_node_texture_paint_data_get(node));
  // TODO: reinit when texturing on different image?
  if (node_data != nullptr) {
    return;
  }

  TIMEIT_START(init_texture_paint_for_node);
  node_data = MEM_new<NodeData>(__func__);
  node_data->init_pixels_rasterization(ob, node, ss->mode.texture_paint.drawing_target);
  BKE_pbvh_node_texture_paint_data_set(node, node_data, NodeData::free_func);
  TIMEIT_END(init_texture_paint_for_node);
}

static void init_using_rasterization(Object *ob, int totnode, PBVHNode **nodes)
{
  TIMEIT_START(init_using_rasterization);
  TexturePaintingUserData data = {nullptr};
  data.ob = ob;
  data.nodes = nodes;

  TaskParallelSettings settings;
  BKE_pbvh_parallel_range_settings(&settings, true, totnode);

  BLI_task_parallel_range(0, totnode, &data, init_rasterization_task_cb_ex, &settings);
  TIMEIT_END(init_using_rasterization);
}

}  // namespace rasterization
void NodeData::init_pixels_rasterization(Object *ob, PBVHNode *node, ImBuf *image_buffer)
{
  using namespace rasterization;
  Mesh *mesh = static_cast<Mesh *>(ob->data);
  MLoopUV *ldata_uv = static_cast<MLoopUV *>(CustomData_get_layer(&mesh->ldata, CD_MLOOPUV));
  if (ldata_uv == nullptr) {
    return;
  }

  RasterizerType rasterizer;
  NodeDataPair node_data_pair;
  rasterizer.vertex_shader().image_size = float2(image_buffer->x, image_buffer->y);
  rasterizer.fragment_shader().image_buffer = image_buffer;
  node_data_pair.node_data = this;
  node_data_pair.image_buffer = image_buffer;
  rasterizer.activate_drawing_target(&node_data_pair);

  SculptSession *ss = ob->sculpt;
  MVert *mvert = SCULPT_mesh_deformed_mverts_get(ss);

  PBVHVertexIter vd;
  BKE_pbvh_vertex_iter_begin (ss->pbvh, node, vd, PBVH_ITER_UNIQUE) {
    MeshElemMap *vert_map = &ss->pmap[vd.index];
    for (int j = 0; j < ss->pmap[vd.index].count; j++) {
      const MPoly *p = &ss->mpoly[vert_map->indices[j]];
      if (p->totloop < 3) {
        continue;
      }

      const MLoop *loopstart = &ss->mloop[p->loopstart];
      for (int triangle = 0; triangle < p->totloop - 2; triangle++) {
        const int v1_index = loopstart[0].v;
        const int v2_index = loopstart[triangle + 1].v;
        const int v3_index = loopstart[triangle + 2].v;
        const int v1_loop_index = p->loopstart;
        const int v2_loop_index = p->loopstart + triangle + 1;
        const int v3_loop_index = p->loopstart + triangle + 2;

        VertexInput v1(mvert[v1_index].co, ldata_uv[v1_loop_index].uv);
        VertexInput v2(mvert[v2_index].co, ldata_uv[v2_loop_index].uv);
        VertexInput v3(mvert[v3_index].co, ldata_uv[v3_loop_index].uv);
        rasterizer.draw_triangle(v1, v2, v3);
      }
    }
  }
  BKE_pbvh_vertex_iter_end;
  rasterizer.deactivate_drawing_target();
}
}  // namespace blender::ed::sculpt_paint::texture_paint

extern "C" {
void SCULPT_extract_pixels(Object *ob, PBVHNode **nodes, int totnode)
{
  TIMEIT_START(extract_pixels);
  blender::ed::sculpt_paint::texture_paint::rasterization::init_using_rasterization(
      ob, totnode, nodes);
  TIMEIT_END(extract_pixels);
}
}