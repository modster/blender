/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#include "MEM_guardedalloc.h"

#include "BLI_utildefines.h"

#include "BLI_bitmap.h"
#include "BLI_ghash.h"
#include "BLI_index_range.hh"
#include "BLI_math.h"
#include "BLI_rand.h"
#include "BLI_span.hh"
#include "BLI_task.h"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_attribute.h"
#include "BKE_ccg.h"
#include "BKE_mesh.h"
#include "BKE_mesh_mapping.h"
#include "BKE_paint.h"
#include "BKE_pbvh.h"
#include "BKE_subdiv_ccg.h"

#include "PIL_time.h"

#include "GPU_buffers.h"

#include "bmesh.h"

#include "atomic_ops.h"

#include "pbvh_intern.h"

#include <climits>

using blender::IndexRange;

namespace blender::bke {

template<typename T> inline void *get_color_pointer(PBVH, SculptVertRef vref)
{
  const size_t esize = pbvh->color_layer->type == CD_PROP_COLOR ? sizeof(MPropCol) :
                                                                  sizeof(MLoopCol);

  switch (pbvh->type) {
    case PBVH_FACES:
      return POINTER_OFFSET(pbvh->color_layer->data, (size_t)vref.i * esize);
    case PBVH_BMESH: {
      BMVert *v = reinterpret_cast<BMVert *>(vref.i);

      return BM_ELEM_CD_GET_VOID_P(v, pbvh->cd_vcol_offset);
    }
    default:
      return nullptr;
  }
}

template<typename Func>
inline void to_static_color_type(const CustomDataType type, const Func &func)
{
  switch (type) {
    case CD_PROP_COLOR:
      func(MPropCol());
      break;
    case CD_MLOOPCOL:
      func(MLoopCol());
      break;
    default:
      BLI_assert_unreachable();
      break;
  }
}

template<typename T> void to_float(const T &src, float dst[4]);

template<> void to_float(const MLoopCol &src, float dst[4])
{
  rgba_uchar_to_float(dst, reinterpret_cast<const unsigned char *>(&src));
  srgb_to_linearrgb_v3_v3(dst, dst);
}
template<> void to_float(const MPropCol &src, float dst[4])
{
  copy_v4_v4(dst, src.color);
}

template<typename T> void from_float(const float src[4], T &dst);

template<> void from_float(const float src[4], MLoopCol &dst)
{
  float temp[4];
  linearrgb_to_srgb_v3_v3(temp, src);
  temp[3] = src[3];
  rgba_float_to_uchar(reinterpret_cast<unsigned char *>(&dst), temp);
}
template<> void from_float(const float src[4], MPropCol &dst)
{
  copy_v4_v4(dst.color, src);
}

template<typename T>
static void pbvh_vertex_color_get_faces(const PBVH &pbvh, SculptVertRef vertex, float r_color[4])
{
  if (pbvh.color_domain == ATTR_DOMAIN_CORNER) {
    const MeshElemMap &melem = pbvh.pmap->pmap[vertex.i];

    int count = 0;
    zero_v4(r_color);
    for (const int i_poly : Span(melem.indices, melem.count)) {
      const MPoly &mp = pbvh.mpoly[i_poly];
      Span<T> colors{static_cast<const T *>(pbvh.color_layer->data) + mp.loopstart, mp.totloop};
      Span<MLoop> loops{pbvh.mloop + mp.loopstart, mp.totloop};

      for (const int i_loop : IndexRange(mp.totloop)) {
        if (loops[i_loop].v == vertex.i) {
          float temp[4];
          to_float(colors[i_loop], temp);

          add_v4_v4(r_color, temp);
          count++;
        }
      }
    }

    if (count) {
      mul_v4_fl(r_color, 1.0f / (float)count);
    }
  }
  else {
    to_float(static_cast<T *>(pbvh.color_layer->data)[vertex.i], r_color);
  }
}

template<typename T>
static void pbvh_vertex_color_get_bmesh(const PBVH &pbvh, SculptVertRef vertex, float r_color[4])
{
  BMVert *v = reinterpret_cast<BMVert *>(vertex.i);

  if (pbvh.color_domain == ATTR_DOMAIN_CORNER) {
    BMIter iter;
    BMLoop *l;

    int count = 0;
    zero_v4(r_color);

    BM_ITER_ELEM (l, &iter, v, BM_LOOPS_OF_VERT) {
      float temp[4];

      T *ptr = static_cast<T*>(BM_ELEM_CD_GET_VOID_P(l, pbvh.cd_vcol_offset));
      to_float(*ptr, temp);

      add_v4_v4(r_color, temp);
      count++;
    }

    if (count) {
      mul_v4_fl(r_color, 1.0f / (float)count);
    }
  }
  else {
    T *ptr = static_cast<T *>(BM_ELEM_CD_GET_VOID_P(v, pbvh.cd_vcol_offset));
    to_float(*ptr, r_color);
  }
}

template<typename T>
static void pbvh_vertex_color_get(const PBVH &pbvh, SculptVertRef vertex, float r_color[4])
{
  switch (pbvh.type) {
    case PBVH_FACES:
      pbvh_vertex_color_get_faces<T>(pbvh, vertex, r_color);
      break;
    case PBVH_BMESH:
      pbvh_vertex_color_get_bmesh<T>(pbvh, vertex, r_color);
      break;
    case PBVH_GRIDS:
      break;
  }
}

template<typename T>
static void pbvh_vertex_color_set_faces(PBVH &pbvh, SculptVertRef vertex, const float color[4])
{
  if (pbvh.color_domain == ATTR_DOMAIN_CORNER) {
    const MeshElemMap &melem = pbvh.pmap->pmap[vertex.i];

    for (const int i_poly : Span(melem.indices, melem.count)) {
      const MPoly &mp = pbvh.mpoly[i_poly];
      MutableSpan<T> colors{static_cast<T *>(pbvh.color_layer->data) + mp.loopstart, mp.totloop};
      Span<MLoop> loops{pbvh.mloop + mp.loopstart, mp.totloop};

      for (const int i_loop : IndexRange(mp.totloop)) {
        if (loops[i_loop].v == vertex.i) {
          from_float(color, colors[i_loop]);
        }
      }
    }
  }
  else {
    from_float(color, static_cast<T *>(pbvh.color_layer->data)[vertex.i]);
  }
}

template<typename T>
static void pbvh_vertex_color_set_bmesh(PBVH &pbvh, SculptVertRef vertex, const float color[4])
{
  BMVert *v = reinterpret_cast<BMVert *>(vertex.i);

  if (pbvh.color_domain == ATTR_DOMAIN_CORNER) {
    BMIter iter;
    BMLoop *l;

    BM_ITER_ELEM (l, &iter, v, BM_LOOPS_OF_VERT) {
      T *ptr = static_cast<T *>(BM_ELEM_CD_GET_VOID_P(l, pbvh.cd_vcol_offset));
      from_float(color, *ptr);
    }
  }
  else {
    T *ptr = static_cast<T *>(BM_ELEM_CD_GET_VOID_P(v, pbvh.cd_vcol_offset));
    from_float(color, *ptr);
  }
}

template<typename T>
static void pbvh_vertex_color_set(PBVH &pbvh, SculptVertRef vertex, const float color[4])
{
  switch (pbvh.type) {
    case PBVH_FACES:
      pbvh_vertex_color_set_faces<T>(pbvh, vertex, color);
      break;
    case PBVH_BMESH:
      pbvh_vertex_color_set_bmesh<T>(pbvh, vertex, color);
      break;
    case PBVH_GRIDS:
      break;
  }
}

}  // namespace blender::bke

extern "C" {
void BKE_pbvh_vertex_color_get(const PBVH *pbvh, SculptVertRef vertex, float r_color[4])
{
  blender::bke::to_static_color_type(CustomDataType(pbvh->color_type), [&](auto dummy) {
    using T = decltype(dummy);
    blender::bke::pbvh_vertex_color_get<T>(*pbvh, vertex, r_color);
  });
}

void BKE_pbvh_vertex_color_set(PBVH *pbvh, SculptVertRef vertex, const float color[4])
{
  blender::bke::to_static_color_type(CustomDataType(pbvh->color_type), [&](auto dummy) {
    using T = decltype(dummy);
    blender::bke::pbvh_vertex_color_set<T>(*pbvh, vertex, color);
  });
}

void BKE_pbvh_swap_colors(PBVH *pbvh,
                          const int *indices,
                          const int indices_num,
                          float (*r_colors)[4])
{
  blender::bke::to_static_color_type(CustomDataType(pbvh->color_layer->type), [&](auto dummy) {
    using T = decltype(dummy);
    T *pbvh_colors = static_cast<T *>(pbvh->color_layer->data);
    for (const int i : IndexRange(indices_num)) {
      T temp = pbvh_colors[indices[i]];
      blender::bke::from_float(r_colors[i], pbvh_colors[indices[i]]);
      blender::bke::to_float(temp, r_colors[i]);
    }
  });
}

ATTR_NO_OPT void BKE_pbvh_store_colors(PBVH *pbvh,
                           const int *indices,
                           const int indices_num,
                           float (*r_colors)[4])
{
  blender::bke::to_static_color_type(CustomDataType(pbvh->color_layer->type), [&](auto dummy) {
    using T = decltype(dummy);
    T *pbvh_colors = static_cast<T *>(pbvh->color_layer->data);
    for (const int i : IndexRange(indices_num)) {
      blender::bke::to_float(pbvh_colors[indices[i]], r_colors[i]);
    }
  });
}

void BKE_pbvh_store_colors_vertex(PBVH *pbvh,
                                  const int *indices,
                                  const int indices_num,
                                  float (*r_colors)[4])
{
  if (pbvh->color_domain == ATTR_DOMAIN_POINT) {
    BKE_pbvh_store_colors(pbvh, indices, indices_num, r_colors);
  }
  else {
    blender::bke::to_static_color_type(CustomDataType(pbvh->color_layer->type), [&](auto dummy) {
      using T = decltype(dummy);
      for (const int i : IndexRange(indices_num)) {
        SculptVertRef vertex = {(intptr_t)indices[i]};

        blender::bke::pbvh_vertex_color_get<T>(*pbvh, vertex, r_colors[i]);
      }
    });
  }
}
}
