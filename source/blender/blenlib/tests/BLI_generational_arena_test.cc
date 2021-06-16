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

TEST(generational_arena, Insert2)
{
  Arena<int> arena;
  EXPECT_EQ(arena.capacity(), 0);
  auto i1 = arena.insert(1);
  auto i2 = arena.insert(2);
  auto i3 = arena.insert(3);
  auto i4 = arena.insert(4);
  auto i5 = arena.insert(5);

  EXPECT_EQ(arena.capacity(), 8);
  EXPECT_EQ(arena.size(), 5);
  EXPECT_EQ(arena.get(i1), 1);
  EXPECT_EQ(arena.get(i2), 2);
  EXPECT_EQ(arena.get(i3), 3);
  EXPECT_EQ(arena.get(i4), 4);
  EXPECT_EQ(arena.get(i5), 5);
}

TEST(generational_arena, Remove)
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
  auto value = arena.remove(i1);
  EXPECT_EQ(value, 1);
  EXPECT_EQ(arena.size(), 4);
  EXPECT_EQ(arena.get(i1), std::nullopt);
  arena.insert(9);
  EXPECT_EQ(arena.size(), 5);
  EXPECT_EQ(arena.get(i1), std::nullopt);
  EXPECT_EQ(arena.get_no_gen(0), 9);

  EXPECT_EQ(arena.get(i2), 2);
  EXPECT_EQ(arena.get(i3), 3);
  EXPECT_EQ(arena.get(i4), 4);
  EXPECT_EQ(arena.get(i5), 5);
}

TEST(generational_arena, Get)
{
  Arena<int> arena(3);
  EXPECT_EQ(arena.capacity(), 3);
  auto i1 = arena.insert(1);
  auto i2 = arena.insert(2);
  EXPECT_EQ(arena.capacity(), 3);
  EXPECT_EQ(arena.size(), 2);
  (*arena.get(i1)).get() = 5;
  EXPECT_EQ(arena.get(i1), 5);
  EXPECT_EQ(arena.get(i2), 2);
}

TEST(generational_arena, GetNoGen)
{
  Arena<int> arena(3);
  EXPECT_EQ(arena.capacity(), 3);
  auto i1 = arena.insert(1);
  auto i2 = arena.insert(2);
  EXPECT_EQ(arena.capacity(), 3);
  EXPECT_EQ(arena.size(), 2);
  (*arena.get_no_gen(0)).get() = 5;
  EXPECT_EQ(arena.get(i1), 5);
  EXPECT_EQ(arena.get(i2), 2);
  EXPECT_EQ(arena.get_no_gen(1), 2);
  EXPECT_EQ(arena.get_no_gen(2), std::nullopt);
}

TEST(generational_arena, GetNoGenIndex)
{
  Arena<int> arena(3);
  EXPECT_EQ(arena.capacity(), 3);
  auto i1 = arena.insert(1);
  auto i2 = arena.insert(2);
  EXPECT_EQ(arena.capacity(), 3);
  EXPECT_EQ(arena.size(), 2);
  EXPECT_EQ(arena.get_no_gen_index(0), i1);
  EXPECT_EQ(arena.get_no_gen_index(1), i2);
  EXPECT_EQ(arena.get_no_gen(2), std::nullopt);
}

} /* namespace blender::tests */

namespace blender::generational_arena {

TEST(generational_arena, GetNextFreeLocations)
{
  Arena<int> arena;
  auto i0 = arena.insert(0);
  auto i1 = arena.insert(1);
  auto i2 = arena.insert(2);
  auto i3 = arena.insert(3);

  arena.remove(i0);
  arena.remove(i1);
  arena.remove(i2);
  arena.remove(i3);

  auto locs = arena.get_next_free_locations();
  EXPECT_EQ(locs.size(), 4);
  EXPECT_EQ(locs[0], 3);
  EXPECT_EQ(locs[1], 2);
  EXPECT_EQ(locs[2], 1);
  EXPECT_EQ(locs[3], 0);

  i0 = arena.insert(0);
  i1 = arena.insert(1);
  i2 = arena.insert(2);
  i3 = arena.insert(3);

  locs = arena.get_next_free_locations();
  EXPECT_EQ(locs.size(), 0);

  auto i4 = arena.insert(4);
  arena.remove(i1);
  arena.remove(i4);
  locs = arena.get_next_free_locations();
  EXPECT_EQ(locs.size(), 5);
  EXPECT_EQ(locs[0], 4);
  EXPECT_EQ(locs[1], 2);
  EXPECT_EQ(locs[2], 5);
  EXPECT_EQ(locs[3], 6);
  EXPECT_EQ(locs[4], 7);

  auto i_0 = arena.insert(10);
  auto i_1 = arena.insert(11);
  auto i_2 = arena.insert(12);
  auto i_3 = arena.insert(13);
  auto i_4 = arena.insert(14);

  EXPECT_EQ(arena.size(), 8);
  EXPECT_EQ(arena.get(i0), 0);
  EXPECT_EQ(arena.get(i1), std::nullopt);
  EXPECT_EQ(arena.get(i2), 2);
  EXPECT_EQ(arena.get(i3), 3);
  EXPECT_EQ(arena.get(i4), std::nullopt);
  EXPECT_EQ(arena.get(i_0), 10);
  EXPECT_EQ(arena.get(i_1), 11);
  EXPECT_EQ(arena.get(i_2), 12);
  EXPECT_EQ(arena.get(i_3), 13);
  EXPECT_EQ(arena.get(i_4), 14);
  EXPECT_EQ(arena.get_no_gen(0), 3);
  EXPECT_EQ(arena.get_no_gen(1), 2);
  EXPECT_EQ(arena.get_no_gen(2), 11);
  EXPECT_EQ(arena.get_no_gen(3), 0);
  EXPECT_EQ(arena.get_no_gen(4), 10);
  EXPECT_EQ(arena.get_no_gen(5), 12);
  EXPECT_EQ(arena.get_no_gen(6), 13);
  EXPECT_EQ(arena.get_no_gen(7), 14);
}

} /* namespace blender::generational_arena */
