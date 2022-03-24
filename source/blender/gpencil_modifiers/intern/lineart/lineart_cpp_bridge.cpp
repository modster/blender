
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

LineartThreadOcclusionData *lineart_thread_init_occlusion_result()
{
  _LineartThreadOcclusionData *result = new _LineartThreadOcclusionData;
  return (LineartThreadOcclusionData *)result;
};

void lineart_thread_add_occlusion_pair(LineartThreadOcclusionData *data,
                                       LineartElementLinkNode *eln_edge,
                                       LineartElementLinkNode *eln_triangle,
                                       LineartEdge *e,
                                       LineartTriangle *t)
{
  LineartOcclusionPair op;
  op.e = e;
  op.t = t;
  op.eln_edge = eln_edge;
  op.eln_triangle = eln_triangle;
  ((_LineartThreadOcclusionData *)data)->local().append(op);
}

/* Memory returned by this needs to be freed manually. */
LineartOcclusionPair *lineart_thread_finalize_occlusion_result(LineartThreadOcclusionData *data,
                                                               int *result_count)
{
  Vector<LineartOcclusionPair> *result = new Vector<LineartOcclusionPair>;
  for (const Vector<LineartOcclusionPair> &local : (*(_LineartThreadOcclusionData *)data)) {
    result->extend(local);
  }
  *result_count = result->size();
  return (LineartOcclusionPair *)result->data();
};
