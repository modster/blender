/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#include <algorithm>

#include "BKE_curves.hh"

#include "BLI_index_mask_ops.hh"
#include "BLI_math_vec_types.hh"

#include "gpencil_new_proposal.hh"
#include "testing/testing.h"

namespace blender::bke {

class GPLayerGroup : ::GPLayerGroup {
 public:
  GPLayerGroup()
  {
    /* TODO */
  }

  ~GPLayerGroup()
  {
    /* Recursivly free the children of this layer group first. */
    for (int i = 0; i < this->children_size; i++) {
      MEM_delete(&this->children[i]);
    }
    /* Then free its data. */
    MEM_SAFE_FREE(this->children);
    MEM_SAFE_FREE(this->layer_indices);
  }
};

class GPDataRuntime {
 public:
  /* mutable void *sbuffer */

  /**
   * Cache that maps the index of a layer to the index mask of the frames in that layer.
   */
  mutable Map<int, Vector<int64_t>> cached_frame_index_masks;

  IndexMask get_cached_frame_index_mask(int layer_index)
  {
    return cached_frame_index_masks.lookup(layer_index).as_span();
  }
};

class GPStroke {
 public:
  GPStroke(int num_points, int offset_index)
      : num_points_(num_points), offset_index_(offset_index){};

  ~GPStroke() = default;

  int points_num() const
  {
    return num_points_;
  }

 private:
  int num_points_;
  int offset_index_;
};

class GPFrame : public ::GPFrame {
 public:
  GPFrame() : GPFrame(-1, -1)
  {
  }

  GPFrame(int layer_index) : GPFrame(layer_index, -1)
  {
  }

  GPFrame(int layer_index, int start_frame)
  {
    this->layer_index = layer_index;
    this->start = start_frame;

    /* Unused for now. */
    this->end = -1;

    this->strokes = MEM_new<CurvesGeometry>(__func__);
  }

  GPFrame(const GPFrame &other) : GPFrame(other.layer_index, other.start)
  {
    if (other.strokes != nullptr) {
      this->strokes_as_curves() = CurvesGeometry::wrap(*other.strokes);
    }
  }

  GPFrame &operator=(const GPFrame &other)
  {
    if (this != &other && other.strokes != nullptr) {
      this->strokes_as_curves() = CurvesGeometry::wrap(*other.strokes);
    }
    this->layer_index = other.layer_index;
    this->start = other.start;
    return *this;
  }

  GPFrame(GPFrame &&other) : GPFrame(other.layer_index, other.start)
  {
    if (this != &other) {
      std::swap(this->strokes, other.strokes);
    }
  }

  GPFrame &operator=(GPFrame &&other)
  {
    if (this != &other) {
      std::swap(this->strokes, other.strokes);
    }
    this->layer_index = other.layer_index;
    this->start = other.start;
    return *this;
  }

  ~GPFrame()
  {
    MEM_delete(reinterpret_cast<CurvesGeometry *>(this->strokes));
    this->strokes = nullptr;
  }

  bool operator<(const GPFrame &other) const
  {
    if (this->start == other.start) {
      return this->layer_index < other.layer_index;
    }
    return this->start < other.start;
  }

  /* Assumes that elem.first is the layer index and elem.second is the frame start. */
  bool operator<(const std::pair<int, int> elem) const
  {
    if (this->start == elem.second) {
      return this->layer_index < elem.first;
    }
    return this->start < elem.second;
  }

  bool operator==(const GPFrame &other) const
  {
    return this->layer_index == other.layer_index && this->start == other.start;
  }

  CurvesGeometry &strokes_as_curves()
  {
    return CurvesGeometry::wrap(*this->strokes);
  }

  int strokes_num() const
  {
    return this->strokes->curve_size;
  }

  GPStroke add_new_stroke(int num_points)
  {
    CurvesGeometry &strokes = this->strokes_as_curves();
    strokes.resize(strokes.points_num() + num_points, strokes.curves_num() + 1);
    return {num_points, strokes.offsets().last()};
  }
};

class GPLayer : public ::GPLayer {
 public:
  GPLayer() : GPLayer("GP_Layer")
  {
  }

  GPLayer(const StringRefNull name)
  {
    strcpy(this->name, name.c_str());
  }

  ~GPLayer() = default;

  bool operator==(const GPLayer &other) const
  {
    return STREQ(this->name, other.name);
  }
};

class GPData : public ::GPData {
 public:
  GPData() : GPData(0, 0)
  {
  }

  GPData(const int layers_size, const int frame_size)
  {
    BLI_assert(layers_size >= 0);
    BLI_assert(frame_size >= 0);

    this->frames_size = frame_size;
    this->layers_size = layers_size;

    if (this->frames_size > 0) {
      this->frames_array = reinterpret_cast<::GPFrame *>(
          MEM_calloc_arrayN(this->frames_size, sizeof(::GPFrame), __func__));
      default_construct_n(reinterpret_cast<GPFrame *>(this->frames_array), this->frames_size);
    }
    else {
      this->frames_array = nullptr;
    }
    CustomData_reset(&this->frame_data);

    if (this->layers_size > 0) {
      this->layers_array = reinterpret_cast<::GPLayer *>(
          MEM_calloc_arrayN(this->layers_size, sizeof(::GPLayer), __func__));
      this->active_layer_index = 0;
    }
    else {
      this->layers_array = nullptr;
      this->active_layer_index = -1;
    }

    this->default_group = MEM_new<::GPLayerGroup>(__func__);

    this->runtime = MEM_new<GPDataRuntime>(__func__);
  }

  ~GPData()
  {
    /* Free frames and frame custom data. */
    destruct_n(reinterpret_cast<GPFrame *>(this->frames_array), this->frames_size);
    MEM_SAFE_FREE(this->frames_array);
    CustomData_free(&this->frame_data, this->frames_size);

    /* Free layer and layer groups. */
    destruct_n(reinterpret_cast<GPLayer *>(this->layers_array), this->layers_size);
    MEM_SAFE_FREE(this->layers_array);
    MEM_delete(reinterpret_cast<GPLayerGroup *>(this->default_group));
    this->default_group = nullptr;

    MEM_delete(this->runtime);
    this->runtime = nullptr;
  }

  Span<GPFrame> frames() const
  {
    return {(const GPFrame *)this->frames_array, this->frames_size};
  }

  MutableSpan<GPFrame> frames_for_write()
  {
    return {(GPFrame *)this->frames_array, this->frames_size};
  }

  IndexMask frames_on_layer(int layer_index) const
  {
    if (layer_index < 0 || layer_index > this->layers_size) {
      return {};
    }

    /* If the indices are cached for this layer, use the cache. */
    if (this->runtime->cached_frame_index_masks.contains(layer_index)) {
      return this->runtime->get_cached_frame_index_mask(layer_index);
    }

    Vector<int64_t> indices;
    const IndexMask mask = index_mask_ops::find_indices_based_on_predicate(
        IndexMask(this->frames_size), 1024, indices, [&](const int index) {
          return this->frames()[index].layer_index == layer_index;
        });

    /* Cache the resulting index mask. */
    this->runtime->cached_frame_index_masks.add(layer_index, std::move(indices));
    return mask;
  }

  IndexMask frames_on_active_layer() const
  {
    return frames_on_layer(this->active_layer_index);
  }

  Span<GPLayer> layers() const
  {
    return {(const GPLayer *)this->layers_array, this->layers_size};
  }

  MutableSpan<GPLayer> layers_for_write()
  {
    return {(GPLayer *)this->layers_array, this->layers_size};
  }

  const int add_layer(GPLayer &new_layer)
  {
    /* Ensure that the layer array has enough space. */
    if (!ensure_layer_array_has_size_at_least(this->layers_size + 1)) {
      return -1;
    }

    /* Move new_layer to the end in the array. */
    this->layers_for_write().last() = new_layer;
    return this->layers_size - 1;
  }

  int frame_index_at(const int layer_idx, const int start_frame_number)
  {
    auto it = std::lower_bound(this->frames().begin(),
                               this->frames().end(),
                               std::pair<int, int>(layer_idx, start_frame_number));
    if (it == this->frames().end() || it->start != start_frame_number) {
      return -1;
    }
    return it - this->frames().begin();
  }

  /**
   * Find a GPFrame in the frame_array given by the layer index and the frame number in logarithmic
   * time. If the frame is found, returns a pointer/iterator to it, otherwise nullptr.
   *
   * Note: This assumes that the array is sorted.
   */
  const GPFrame *frame_at(const int layer_idx, const int start_frame_number)
  {
    int index = frame_index_at(layer_idx, start_frame_number);
    if (index == -1) {
      return nullptr;
    }
    return &this->frames()[index];
  }

  void create_new_frame_on_layer(const int layer_index, const int start_frame_number)
  {
    BLI_assert(layer_index >= 0 && layer_index < this->layers_size);
    /* Allocate new space for the frame. */
    if (!ensure_frame_array_has_size_at_least(this->frames_size + 1)) {
      /* TODO: handle this properly. */
      BLI_assert(false);
      return;
    }

    /* Create a new frame and append it at the end. */
    this->frames_for_write().last().layer_index = layer_index;
    this->frames_for_write().last().start = start_frame_number;

    /* Sort the frame array. */
    update_frames_array();
  }

  const GPFrame &get_new_frame_on_layer(const int layer_index, const int start_frame_number)
  {
    create_new_frame_on_layer(layer_index, start_frame_number);
    /* Find the frame in the array and return it. */
    const GPFrame *gpf = this->frame_at(layer_index, start_frame_number);
    BLI_assert(gpf != nullptr);
    return *gpf;
  }

 private:
  const bool ensure_layer_array_has_size_at_least(int64_t size)
  {
    if (this->layers_size > size) {
      return true;
    }

    int old_size = this->layers_size;
    this->layers_size = size;

    ::GPLayer *new_array = reinterpret_cast<::GPLayer *>(
        MEM_calloc_arrayN(this->layers_size, sizeof(::GPLayer), __func__));
    if (new_array == nullptr) {
      return false;
    }

    if (this->layers_array != nullptr) {
      memcpy(new_array, this->layers_array, old_size * sizeof(::GPLayer));
      MEM_SAFE_FREE(this->layers_array);
    }
    this->layers_array = new_array;

    return true;
  }

  const bool ensure_frame_array_has_size_at_least(int64_t size)
  {
    if (this->frames_size > size) {
      return true;
    }

    int old_size = this->frames_size;
    this->frames_size = size;

    ::GPFrame *new_array = reinterpret_cast<::GPFrame *>(
        MEM_calloc_arrayN(this->frames_size, sizeof(::GPFrame), __func__));
    if (new_array == nullptr) {
      return false;
    }

    if (this->frames_array != nullptr) {
      uninitialized_relocate_n(reinterpret_cast<GPFrame *>(this->frames_array),
                               old_size,
                               reinterpret_cast<GPFrame *>(new_array));
      MEM_SAFE_FREE(this->frames_array);
      this->frames_array = new_array;
    }
    else {
      this->frames_array = new_array;
      default_construct_n(reinterpret_cast<GPFrame *>(this->frames_array), this->frames_size);
    }
    return true;
  }

  void update_frames_array()
  {
    /* Make sure frames are ordered chronologically and by layer order. */
    std::sort(this->frames_for_write().begin(), this->frames_for_write().end());

    /* Clear the cached indices since they are probably no longer valid. */
    this->runtime->cached_frame_index_masks = {};
  }
};

}  // namespace blender::bke

namespace blender::bke::gpencil::tests {

TEST(gpencil_proposal, EmptyGPData)
{
  GPData data;
  EXPECT_EQ(data.layers_size, 0);
  EXPECT_EQ(data.frames_size, 0);
}

TEST(gpencil_proposal, OneLayer)
{
  GPData data(1, 0);
  EXPECT_EQ(data.layers_size, 1);
  EXPECT_EQ(data.frames_size, 0);
}

TEST(gpencil_proposal, LayerName)
{
  GPLayer layer1;
  EXPECT_STREQ(layer1.name, "GP_Layer");

  GPLayer layer2("FooLayer");
  EXPECT_STREQ(layer2.name, "FooLayer");
}

TEST(gpencil_proposal, AddOneLayer)
{
  GPData data;
  GPLayer layer("FooLayer");

  data.add_layer(layer);
  EXPECT_EQ(data.layers_size, 1);
  EXPECT_STREQ(data.layers().last().name, layer.name);
}

TEST(gpencil_proposal, AddLayers)
{
  GPData data;
  GPLayer layers[3] = {GPLayer("TestLayer1"), GPLayer("TestLayer2"), GPLayer("TestLayer3")};

  for (int i : IndexRange(3)) {
    data.add_layer(layers[i]);
  }
  EXPECT_EQ(data.layers_size, 3);

  for (int i : IndexRange(3)) {
    EXPECT_STREQ(data.layers()[i].name, layers[i].name);
  }
}

TEST(gpencil_proposal, ChangeLayerName)
{
  GPData data;
  GPLayer layer("FooLayer");

  data.add_layer(layer);
  EXPECT_EQ(data.layers_size, 1);
  EXPECT_STREQ(data.layers().last().name, layer.name);

  strcpy(data.layers_for_write().last().name, "BarLayer");

  EXPECT_EQ(data.layers_size, 1);
  EXPECT_STREQ(data.layers().last().name, "BarLayer");
}

TEST(gpencil_proposal, AddFrameToLayer)
{
  GPData data;
  GPLayer layer1("TestLayer1");
  GPLayer layer2("TestLayer2");

  data.add_layer(layer1);
  const int layer2_idx = data.add_layer(layer2);
  data.create_new_frame_on_layer(layer2_idx, 0);
  const GPFrame *frame = data.frame_at(layer2_idx, 0);
  EXPECT_EQ(frame->layer_index, 1);
}

TEST(gpencil_proposal, CheckFramesSorted1)
{
  GPData data;
  GPLayer layer1("TestLayer1");

  const int frame_numbers1[5] = {10, 5, 6, 1, 3};
  const int frame_numbers_sorted1[5] = {1, 3, 5, 6, 10};

  const int layer1_idx = data.add_layer(layer1);
  for (int i : IndexRange(5)) {
    const GPFrame &frame = data.get_new_frame_on_layer(layer1_idx, frame_numbers1[i]);
    int idx = data.frames().first_index(frame);
    EXPECT_EQ(data.frames()[idx].start, frame_numbers1[i]);
  }

  for (const int i : data.frames().index_range()) {
    EXPECT_EQ(data.frames()[i].start, frame_numbers_sorted1[i]);
  }
}

TEST(gpencil_proposal, CheckFramesSorted2)
{
  GPData data;
  GPLayer layer1("TestLayer1");
  GPLayer layer2("TestLayer2");
  const int frame_numbers_layer1[5] = {10, 5, 6, 1, 3};
  const int frame_numbers_layer2[5] = {8, 5, 7, 1, 4};
  const int frame_numbers_sorted2[10][2] = {
      {0, 1}, {1, 1}, {0, 3}, {1, 4}, {0, 5}, {1, 5}, {0, 6}, {1, 7}, {1, 8}, {0, 10}};

  const int layer1_idx = data.add_layer(layer1);
  const int layer2_idx = data.add_layer(layer2);
  for (int i : IndexRange(5)) {
    data.create_new_frame_on_layer(layer1_idx, frame_numbers_layer1[i]);
    data.create_new_frame_on_layer(layer2_idx, frame_numbers_layer2[i]);
  }

  for (const int i : data.frames().index_range()) {
    EXPECT_EQ(data.frames()[i].layer_index, frame_numbers_sorted2[i][0]);
    EXPECT_EQ(data.frames()[i].start, frame_numbers_sorted2[i][1]);
  }
}

TEST(gpencil_proposal, IterateOverFramesOnLayer)
{
  GPData data;
  GPLayer layer1("TestLayer1");
  GPLayer layer2("TestLayer2");

  const int frame_numbers_layer1[5] = {10, 5, 6, 1, 3};
  const int frame_numbers_layer2[5] = {8, 5, 7, 1, 4};

  const int frame_numbers_sorted1[5] = {1, 3, 5, 6, 10};
  const int frame_numbers_sorted2[5] = {1, 4, 5, 7, 8};

  const int layer1_idx = data.add_layer(layer1);
  const int layer2_idx = data.add_layer(layer2);
  for (int i : IndexRange(5)) {
    data.create_new_frame_on_layer(layer1_idx, frame_numbers_layer1[i]);
    data.create_new_frame_on_layer(layer2_idx, frame_numbers_layer2[i]);
  }

  IndexMask indices_frames_layer1 = data.frames_on_layer(layer1_idx);
  EXPECT_TRUE(data.runtime->cached_frame_index_masks.contains(0));
  for (const int i : indices_frames_layer1.index_range()) {
    EXPECT_EQ(data.frames()[indices_frames_layer1[i]].start, frame_numbers_sorted1[i]);
  }

  IndexMask indices_frames_layer2 = data.frames_on_layer(layer2_idx);
  EXPECT_TRUE(data.runtime->cached_frame_index_masks.contains(1));
  for (const int i : indices_frames_layer2.index_range()) {
    EXPECT_EQ(data.frames()[indices_frames_layer2[i]].start, frame_numbers_sorted2[i]);
  }
}

TEST(gpencil_proposal, AddSingleStroke)
{
  GPData data;
  GPLayer layer1("TestLayer1");

  const int layer1_idx = data.add_layer(layer1);
  GPFrame frame = data.get_new_frame_on_layer(layer1_idx, 0);
  GPStroke stroke = frame.add_new_stroke(100);

  EXPECT_EQ(frame.strokes_num(), 1);
  EXPECT_EQ(stroke.points_num(), 100);
}

}  // namespace blender::bke::gpencil::tests