#include "BLI_generational_arena.hh"

#include "testing/testing.h"

namespace blender::tests {

using namespace blender::generational_arena;

TEST(generational_arena, DefaultConstructor)
{
  Arena<int> arena;
  EXPECT_EQ(arena.capacity(), 0);
}

TEST(generational_arena, SizeConstructor)
{
  Arena<int> arena(5);
  EXPECT_EQ(arena.capacity(), 5);
}

TEST(generational_arena, Insert)
{
  Arena<int> arena(3);
  EXPECT_EQ(arena.capacity(), 3);
  auto i1 = arena.insert(1);
  auto i2 = arena.insert(2);
  auto i3 = arena.insert(3);
  auto i4 = arena.insert(4);
  auto i5 = arena.insert(5);

  EXPECT_EQ(arena.capacity(), 6);
  EXPECT_EQ(arena.size(), 5);
  EXPECT_EQ(arena.get(i1), 1);
  EXPECT_EQ(arena.get(i2), 2);
  EXPECT_EQ(arena.get(i3), 3);
  EXPECT_EQ(arena.get(i4), 4);
  EXPECT_EQ(arena.get(i5), 5);
}

} /* namespace blender::tests */
