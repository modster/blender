
#include "MOD_lineart.h"

#include "lineart_intern.h"

#include "BLI_enumerable_thread_specific.hh"
#include "BLI_task.h"
#include "BLI_utildefines.h"
#include "BLI_vector.hh"

using blender::Vector;
using blender::threading::EnumerableThreadSpecific;

typedef EnumerableThreadSpecific<Vector<LineartOcclusionPair>> _LineartThreadOcclusionData;
typedef struct LineartThreadOcclusionData LineartThreadOcclusionData;

typedef Vector<LineartOcclusionPair> _LineartThreadOcclusionDataCombined;
typedef struct LineartThreadOcclusionDataCombined LineartThreadOcclusionDataCombined;

LineartThreadOcclusionData *lineart_thread_init_occlusion_result()
{
  _LineartThreadOcclusionData *result = new _LineartThreadOcclusionData;
  return (LineartThreadOcclusionData *)result;
};

void lineart_thread_add_occlusion_pair(LineartThreadOcclusionData *data,
                                       LineartElementLinkNode *eln_edge,
                                       LineartElementLinkNode *eln_triangle,
                                       LineartEdge *e,
                                       LineartTriangle *t,
                                       double cut_l,
                                       double cut_r)
{
  LineartOcclusionPair op;
  op.e = e;
  op.t = t;
  op.eln_edge = eln_edge;
  op.eln_triangle = eln_triangle;
  op.cut_l = cut_l;
  op.cut_r = cut_r;
  ((_LineartThreadOcclusionData *)data)->local().append(op);
}

/* Memory returned by this needs to be freed manually. */
LineartOcclusionPair *lineart_thread_finalize_occlusion_result(
    LineartThreadOcclusionData *data,
    int *result_count,
    LineartThreadOcclusionDataCombined **r_combined_storage)
{
  Vector<LineartOcclusionPair> *result = new Vector<LineartOcclusionPair>;
  int count = 0;
  for (const Vector<LineartOcclusionPair> &local : (*(_LineartThreadOcclusionData *)data)) {
    count += local.size();
  }
  /* Reserve once so it's faster to extend. */
  result->reserve(count);
  for (const Vector<LineartOcclusionPair> &local : (*(_LineartThreadOcclusionData *)data)) {
    result->extend(local);
  }
  *result_count = count;
  *r_combined_storage = (LineartThreadOcclusionDataCombined *)result;
  return (LineartOcclusionPair *)result->data();
};

void lineart_thread_clear_occlusion_result(LineartThreadOcclusionData *data,
                                           LineartThreadOcclusionDataCombined *combined_storage)
{
  _LineartThreadOcclusionData *occlusion_data = (_LineartThreadOcclusionData *)data;
  _LineartThreadOcclusionDataCombined *combined = (_LineartThreadOcclusionDataCombined *)
      combined_storage;
  delete occlusion_data;
  delete combined;
};
