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
