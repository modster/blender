#include "BLI_sort.hh"
#include "BLI_vector.hh"
#include "MOD_lineart.h"
#include "lineart_intern.h"

static bool cmp_adjacent_items(const LineartAdjacentItem &p1, const LineartAdjacentItem &p2)
{
  int a = (int)p1.v1 - (int)p2.v1;
  int b = (int)p1.v2 - (int)p2.v2;
  return a < 0 ? true : (a == 0 ? b < 0 : false);
}

void lineart_sort_adjacent_items(LineartAdjacentItem *ai, int length)
{
  blender::parallel_sort(ai, ai + length - 1, cmp_adjacent_items);
}
