
#include "MOD_lineart.h"

#include "lineart_intern.h"

#include "BLI_enumerable_thread_specific.hh"
#include "BLI_task.h"
#include "BLI_utildefines.h"
#include "BLI_vector.hh"

#include "MEM_guardedalloc.h"

using blender::Vector;
using blender::threading::EnumerableThreadSpecific;

typedef EnumerableThreadSpecific<_LineartMemPool> _LineartStaticMemPoolThread;
typedef struct _LineartMemPool *LineartMemPool;

LineartMemPool lineart_mem_init()
{
  _LineartStaticMemPoolThread *result = new _LineartStaticMemPoolThread;
  return (LineartMemPool)result;
}

LineartMemPool lineart_mem_local(LineartMemPool smp)
{
  _LineartStaticMemPoolThread *smp_thread = (_LineartStaticMemPoolThread *)smp;
  return (LineartMemPool)&smp_thread->local();
}

void lineart_mem_destroy_internal(LineartMemPool *mp)
{
  if (!(*mp)) {
    return;
  }
  for (const _LineartMemPool &local : (*(_LineartStaticMemPoolThread *)(*mp))) {
    LineartStaticMemPoolNode *smpn;
    _LineartMemPool *smp = (_LineartMemPool *)(&local);
    while ((smpn = (LineartStaticMemPoolNode *)BLI_pophead(&smp->pools)) != NULL) {
      MEM_freeN(smpn);
    }
  }
  _LineartStaticMemPoolThread *smp_thread = (_LineartStaticMemPoolThread *)(*mp);
  delete smp_thread;
  (*mp) = NULL;
}
