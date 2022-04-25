#include "BLI_sort.hh"
#include "BLI_vector.hh"
#include "MOD_lineart.h"
#include "lineart_intern.h"

static int cmp_adjacent_items(const LineartAdjacentItem &p1, const LineartAdjacentItem &p2)
{
  int a = (int)p1.v1 - (int)p2.v1;
  int b = (int)p1.v2 - (int)p2.v2;
  return a ? a : b;
}

void lineart_sort_adjacent_items(LineartAdjacentItem *ai, int length)
{
  blender::Vector<LineartAdjacentItem> _ai;
  _ai.reserve(length);
  for (int i = 0; i < length; i++) {
    _ai[i] = ai[i];
    printf("(%d %d %d)", _ai[i].v1, _ai[i].v2, _ai[i].e);
  }
  _ai.resize(length);
  printf("\n");
  std::sort(ai, ai + length, cmp_adjacent_items);
  for (int i = 0; i < length; i++) {
    ai[i] = _ai[i];
    printf("(%d %d %d)", _ai[i].v1, _ai[i].v2, _ai[i].e);
  }
}
