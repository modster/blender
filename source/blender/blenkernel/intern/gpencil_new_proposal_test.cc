/* SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#include "testing/testing.h"
#include <algorithm>

#include "BKE_curves.hh"
#include "BKE_gpencil.h"

#include "BLI_index_mask_ops.hh"
#include "BLI_math_vec_types.hh"

#include "DNA_gpencil_types.h"

#include "gpencil_new_proposal.hh"

#include "PIL_time_utildefines.h"

namespace blender::bke {

class GPLayerGroup : ::GPLayerGroup { /* Unused for now. Placeholder class. */
 public:
  GPLayerGroup()
  {
    this->children = nullptr;
    this->children_size = 0;
    this->layer_indices = nullptr;
    this->layer_indices_size = 0;
  }

  GPLayerGroup(const StringRefNull name) : GPLayerGroup()
  {
    BLI_assert(name.size() < 128);
    strcpy(this->name, name.c_str());
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

  IndexMask layers_index_mask()
  {
    return {reinterpret_cast<int64_t>(this->layer_indices), this->layer_indices_size};
  }
};

class GPDataRuntime {
 public:
  /* mutable void *sbuffer */

  /**
   * Cache that maps the index of a layer to the index mask of the frames in that layer.
   */
  mutable Map<int, Vector<int64_t>> frame_index_masks_cache;
  mutable std::mutex frame_index_masks_cache_mutex;

  IndexMask frame_index_masks_cache_for_layer(int layer_index)
  {
    return frame_index_masks_cache.lookup(layer_index).as_span();
  }
};

/**
 * A wrapper class around a single curve in GPFrame.strokes (CurvesGeometry). It holds the offset
 * of where to find the stroke in the frame and it's size.
 * This class is only meant to facilitate the handling of individual strokes.
 */
class GPStroke : NonCopyable, NonMovable {
 public:
  GPStroke(CurvesGeometry *geometry, int num_points, int offset)
      : geometry_(geometry), points_num_(num_points), offset_(offset){};

  ~GPStroke() = default;

  int points_num() const
  {
    return points_num_;
  }

  /**
   * Start index of this stroke in the points array of geometry_.
   */
  int points_offset() const
  {
    return offset_;
  }

  Span<float3> points_positions() const
  {
    return {geometry_->positions().begin() + offset_, points_num_};
  }

  MutableSpan<float3> points_positions_for_write() const
  {
    return {geometry_->positions_for_write().begin() + offset_, points_num_};
  }

 private:
  CurvesGeometry *geometry_ = nullptr;
  int points_num_ = 0;
  int offset_;
};

class GPFrame : public ::GPFrame {
 public:
  GPFrame() : GPFrame(-1, -1)
  {
  }

  GPFrame(int start_frame) : GPFrame(start_frame, -1)
  {
  }

  GPFrame(int start_frame, int end_frame)
  {
    this->start_time = start_frame;
    this->end_time = end_frame;
    this->strokes = nullptr;
  }

  GPFrame(const GPFrame &other) : GPFrame(other.start_time, other.end_time)
  {
    if (other.strokes != nullptr) {
      /* Make sure old strokes are freed before copying. */
      MEM_SAFE_FREE(this->strokes);
      this->strokes = MEM_new<CurvesGeometry>(__func__);

      *reinterpret_cast<CurvesGeometry *>(this->strokes) = CurvesGeometry::wrap(*other.strokes);
    }
    this->layer_index = other.layer_index;
  }

  GPFrame &operator=(const GPFrame &other)
  {
    if (this != &other && other.strokes != nullptr) {
      /* Make sure old strokes are freed before copying. */
      MEM_SAFE_FREE(this->strokes);
      this->strokes = MEM_new<CurvesGeometry>(__func__);

      *reinterpret_cast<CurvesGeometry *>(this->strokes) = CurvesGeometry::wrap(*other.strokes);
    }
    this->layer_index = other.layer_index;
    this->start_time = other.start_time;
    this->end_time = other.end_time;
    return *this;
  }

  GPFrame(GPFrame &&other) : GPFrame(other.start_time, other.end_time)
  {
    if (this != &other) {
      std::swap(this->strokes, other.strokes);
      other.strokes = nullptr;
    }
    this->layer_index = other.layer_index;
  }

  GPFrame &operator=(GPFrame &&other)
  {
    if (this != &other) {
      std::swap(this->strokes, other.strokes);
      other.strokes = nullptr;
    }
    this->layer_index = other.layer_index;
    this->start_time = other.start_time;
    this->end_time = other.end_time;
    return *this;
  }

  ~GPFrame()
  {
    MEM_delete(reinterpret_cast<CurvesGeometry *>(this->strokes));
    this->strokes = nullptr;
  }

  bool operator<(const GPFrame &other) const
  {
    if (this->start_time == other.start_time) {
      return this->layer_index < other.layer_index;
    }
    return this->start_time < other.start_time;
  }

  /* Assumes that elem.first is the layer index and elem.second is the start time. */
  bool operator<(const std::pair<int, int> elem) const
  {
    if (this->start_time == elem.second) {
      return this->layer_index < elem.first;
    }
    return this->start_time < elem.second;
  }

  bool operator==(const GPFrame &other) const
  {
    return this->layer_index == other.layer_index && this->start_time == other.start_time;
  }

  CurvesGeometry &strokes_as_curves()
  {
    return CurvesGeometry::wrap(*this->strokes);
  }

  int strokes_num() const
  {
    if (this->strokes == nullptr) {
      return 0;
    }
    return this->strokes->curve_num;
  }

  int points_num() const
  {
    if (this->strokes == nullptr) {
      return 0;
    }
    return this->strokes->point_num;
  }

  GPStroke add_new_stroke(int new_points_num)
  {
    if (this->strokes == nullptr) {
      this->strokes = MEM_new<CurvesGeometry>(__func__);
    }
    CurvesGeometry &strokes = this->strokes_as_curves();
    int orig_last_offset = strokes.offsets().last();

    strokes.resize(strokes.points_num() + new_points_num, strokes.curves_num() + 1);
    strokes.offsets_for_write().last() = strokes.points_num();

    /* Use poly type by default. */
    strokes.curve_types_for_write().last() = CURVE_TYPE_POLY;

    strokes.tag_topology_changed();
    return {reinterpret_cast<CurvesGeometry *>(this->strokes), new_points_num, orig_last_offset};
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
          MEM_malloc_arrayN(this->frames_size, sizeof(::GPFrame), __func__));
      default_construct_n(reinterpret_cast<GPFrame *>(this->frames_array), this->frames_size);
    }
    else {
      this->frames_array = nullptr;
    }
    CustomData_reset(&this->frame_data);

    if (this->layers_size > 0) {
      this->layers_array = reinterpret_cast<::GPLayer *>(
          MEM_malloc_arrayN(this->layers_size, sizeof(::GPLayer), __func__));
      default_construct_n(reinterpret_cast<GPLayer *>(this->layers_array), this->layers_size);
      this->active_layer_index = 0;
    }
    else {
      this->layers_array = nullptr;
      this->active_layer_index = -1;
    }

    this->default_group = MEM_new<::GPLayerGroup>(__func__);

    this->runtime = MEM_new<GPDataRuntime>(__func__);
  }

  GPData(const GPData &other) : GPData(other.layers_size, other.frames_size)
  {
    copy_gpdata(*this, other);
  }

  GPData &operator=(const GPData &other)
  {
    if (this != &other) {
      copy_gpdata(*this, other);
    }
    return *this;
  }

  GPData(GPData &&other) : GPData(other.layers_size, other.frames_size)
  {
    move_gpdata(*this, other);
  }

  GPData &operator=(GPData &&other)
  {
    if (this != &other) {
      move_gpdata(*this, other);
    }
    return *this;
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

    /* Free the runtime structure. */
    MEM_delete(this->runtime);
    this->runtime = nullptr;
  }

  Span<GPFrame> frames() const
  {
    return {reinterpret_cast<const GPFrame *>(this->frames_array), this->frames_size};
  }

  const GPFrame &frames(int index) const
  {
    return this->frames()[index];
  }

  MutableSpan<GPFrame> frames_for_write()
  {
    return {reinterpret_cast<GPFrame *>(this->frames_array), this->frames_size};
  }

  GPFrame &frames_for_write(int index)
  {
    return this->frames_for_write()[index];
  }

  IndexMask frames_on_layer(int layer_index) const
  {
    if (layer_index < 0 || layer_index > this->layers_size) {
      return IndexMask();
    }

    /* If the indices are cached for this layer, use the cache. */
    if (this->runtime->frame_index_masks_cache.contains(layer_index)) {
      return this->runtime->frame_index_masks_cache_for_layer(layer_index);
    }

    /* A double checked lock. */
    std::scoped_lock{this->runtime->frame_index_masks_cache_mutex};
    if (this->runtime->frame_index_masks_cache.contains(layer_index)) {
      return this->runtime->frame_index_masks_cache_for_layer(layer_index);
    }

    Vector<int64_t> indices;
    const IndexMask mask = index_mask_ops::find_indices_based_on_predicate(
        IndexMask(this->frames_size), 1024, indices, [&](const int index) {
          return this->frames()[index].layer_index == layer_index;
        });

    /* Cache the resulting index mask. */
    this->runtime->frame_index_masks_cache.add(layer_index, std::move(indices));
    return mask;
  }

  IndexMask frames_on_layer(GPLayer &layer) const
  {
    int index = this->layers().first_index_try(layer);
    if (index == -1) {
      return IndexMask();
    }
    return frames_on_layer(index);
  }

  IndexMask frames_on_active_layer() const
  {
    return frames_on_layer(this->active_layer_index);
  }

  Span<GPLayer> layers() const
  {
    return {reinterpret_cast<const GPLayer *>(this->layers_array), this->layers_size};
  }

  const GPLayer &layers(int index) const
  {
    return layers()[index];
  }

  MutableSpan<GPLayer> layers_for_write()
  {
    return {reinterpret_cast<GPLayer *>(this->layers_array), this->layers_size};
  }

  GPLayer &layers_for_write(int index)
  {
    return layers_for_write()[index];
  }

  const GPLayer &active_layer()
  {
    return this->layers()[this->active_layer_index];
  }

  GPLayer &active_layer_for_write()
  {
    return this->layers_for_write()[this->active_layer_index];
  }

  int add_layer(StringRefNull name)
  {
    /* Ensure that the layer array has enough space. */
    if (!ensure_layers_array_has_size_at_least(this->layers_size + 1)) {
      return -1;
    }

    GPLayer new_layer(name);
    /* Move new_layer to the end in the array. */
    this->layers_for_write().last() = std::move(new_layer);
    return this->layers_size - 1;
  }

  void add_layers(Array<StringRefNull> names)
  {
    for (StringRefNull name : names) {
      this->add_layer(name);
    }
  }

  int add_frame_on_layer(int layer_index, int frame_start)
  {
    /* TODO: Check for collisions. */

    if (!ensure_frames_array_has_size_at_least(this->frames_size + 1)) {
      return -1;
    }

    return add_frame_on_layer_initialized(layer_index, frame_start, 1);
  }

  int add_frame_on_layer(GPLayer &layer, int frame_start)
  {
    int index = this->layers().first_index_try(layer);
    if (index == -1) {
      return -1;
    }
    return add_frame_on_layer(index, frame_start);
  }

  int add_frame_on_active_layer(int frame_start)
  {
    return add_frame_on_layer(active_layer_index, frame_start);
  }

  void add_frames_on_layer(int layer_index, Array<int> start_frames)
  {
    int new_frames_size = start_frames.size();
    /* TODO: Check for collisions before resizing the array. */
    if (!ensure_frames_array_has_size_at_least(this->frames_size + new_frames_size)) {
      return;
    }

    int reserved = new_frames_size;
    for (int start_frame : start_frames) {
      add_frame_on_layer_initialized(layer_index, start_frame, reserved);
      reserved--;
    }
  }

  int strokes_num() const
  {
    /* TODO: could be done with parallel_for */
    int count = 0;
    for (const GPFrame &gpf : this->frames()) {
      count += gpf.strokes_num();
    }
    return count;
  }

  int points_num() const
  {
    /* TODO: could be done with parallel_for */
    int count = 0;
    for (const GPFrame &gpf : this->frames()) {
      count += gpf.points_num();
    }
    return count;
  }

  void set_active_layer(int layer_index)
  {
    if (layer_index < 0 || layer_index >= this->layers_size) {
      return;
    }
    this->active_layer_index = layer_index;
  }

 private:
  const void copy_gpdata(GPData &dst, const GPData &src)
  {
    /* Make sure previous frame data is freed. */
    MEM_SAFE_FREE(dst.frames_array);
    CustomData_free(&dst.frame_data, dst.frames_size);

    /* Copy frame data. */
    dst.frames_size = src.frames_size;
    dst.frames_array = reinterpret_cast<::GPFrame *>(
        MEM_malloc_arrayN(dst.frames_size, sizeof(::GPFrame), __func__));
    uninitialized_copy_n(reinterpret_cast<GPFrame *>(src.frames_array),
                         src.frames_size,
                         reinterpret_cast<GPFrame *>(dst.frames_array));
    CustomData_copy(&src.frame_data, &dst.frame_data, CD_MASK_ALL, CD_DUPLICATE, dst.frames_size);

    /* Make sure layer data is freed then copy it over. */
    MEM_SAFE_FREE(dst.layers_array);
    dst.layers_size = src.layers_size;
    dst.layers_array = reinterpret_cast<::GPLayer *>(
        MEM_malloc_arrayN(dst.layers_size, sizeof(::GPLayer), __func__));
    uninitialized_copy_n(reinterpret_cast<GPLayer *>(src.layers_array),
                         src.layers_size,
                         reinterpret_cast<GPLayer *>(dst.layers_array));
    dst.active_layer_index = src.active_layer_index;

    /* Copy layer default group. */
    *dst.default_group = *src.default_group;
  }

  const void move_gpdata(GPData &dst, GPData &src)
  {
    /* Move frame data. */
    dst.frames_size = src.frames_size;
    std::swap(dst.frames_array, src.frames_array);
    std::swap(dst.frame_data, src.frame_data);
    MEM_SAFE_FREE(src.frames_array);
    CustomData_free(&src.frame_data, src.frames_size);
    src.frames_size = 0;

    /* Move layer data. */
    dst.layers_size = src.layers_size;
    std::swap(dst.layers_array, src.layers_array);
    dst.active_layer_index = src.active_layer_index;
    MEM_SAFE_FREE(src.layers_array);
    src.layers_size = 0;
    src.active_layer_index = -1;

    /* Move layer group and runtime pointers. */
    std::swap(dst.default_group, src.default_group);
    std::swap(dst.runtime, src.runtime);
  }

  const bool ensure_layers_array_has_size_at_least(int64_t size)
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
      /* Since the layers have default move constructors, we just use memcpy here. */
      memcpy(new_array, this->layers_array, old_size * sizeof(::GPLayer));
      MEM_SAFE_FREE(this->layers_array);
    }
    this->layers_array = new_array;

    return true;
  }

  const bool ensure_frames_array_has_size_at_least(int64_t size)
  {
    if (this->frames_size > size) {
      return true;
    }

    int old_size = this->frames_size;
    this->frames_size = size;

    ::GPFrame *new_array = reinterpret_cast<::GPFrame *>(
        MEM_malloc_arrayN(this->frames_size, sizeof(::GPFrame), __func__));
    if (new_array == nullptr) {
      return false;
    }

    if (this->frames_array != nullptr) {
      uninitialized_relocate_n(reinterpret_cast<GPFrame *>(this->frames_array),
                               old_size,
                               reinterpret_cast<GPFrame *>(new_array));
      default_construct_n(reinterpret_cast<GPFrame *>(new_array + old_size),
                          this->frames_size - old_size);
      MEM_SAFE_FREE(this->frames_array);
      this->frames_array = new_array;
    }
    else {
      this->frames_array = new_array;
      default_construct_n(reinterpret_cast<GPFrame *>(this->frames_array), this->frames_size);
    }
    return true;
  }

  /**
   * Creates a new frame and inserts it into the  \a frames_array so that the ordering is kept.
   * Assumes that \a frames_array is sorted and that the array has been reallocated + expaned by \a
   * reserved.
   */
  int add_frame_on_layer_initialized(int layer_index, int frame_start, int reserved)
  {
    /* Create the new frame. */
    GPFrame new_frame(frame_start);
    new_frame.layer_index = layer_index;

    int last_index = this->frames_size - reserved - 1;

    /* Check if the frame can be appended at the end. */
    if (this->frames_size == 0 || this->frames_size == reserved ||
        this->frames(last_index) < std::pair<int, int>(layer_index, frame_start)) {
      this->frames_for_write(last_index + 1) = std::move(new_frame);
      return last_index + 1;
    }

    /* Look for the first frame that is equal or greater than the new frame. */
    auto it = std::lower_bound(this->frames().begin(),
                               this->frames().drop_back(reserved).end(),
                               std::pair<int, int>(layer_index, frame_start));
    /* Get the index of the frame. */
    int index = std::distance(this->frames().begin(), it);
    /* Move all the frames and make space at index. */
    initialized_reversed_move_n(reinterpret_cast<GPFrame *>(this->frames_array + index),
                                this->frames_size - index - 1,
                                reinterpret_cast<GPFrame *>(this->frames_array + index + 1));
    /* Move the new frame into the space at index. */
    this->frames_for_write(index) = std::move(new_frame);

    return index;
  }

  void update_frames_array()
  {
    /* Make sure frames are ordered chronologically and by layer order. */
    std::sort(this->frames_for_write().begin(), this->frames_for_write().end());

    /* Clear the cached indices since they are (probably) no longer valid. */
    this->runtime->frame_index_masks_cache.clear();
  }
};

}  // namespace blender::bke

namespace blender::bke::gpencil::tests {

static GPData build_gpencil_data(int num_layers,
                                 int frames_per_layer,
                                 int strokes_per_frame,
                                 int points_per_stroke)
{
  GPData gpd;

  Vector<std::string> test_names;
  for (const int i : IndexRange(num_layers)) {
    test_names.append(std::string("GPLayer") + std::to_string(i));
  }
  gpd.add_layers(test_names.as_span());

  Array<int> test_start_frames(IndexRange(frames_per_layer).as_span());
  for (const int i : gpd.layers().index_range()) {
    gpd.add_frames_on_layer(i, test_start_frames);
  }

  for (const int i : gpd.frames().index_range()) {
    for (const int j : IndexRange(strokes_per_frame)) {
      GPStroke stroke = gpd.frames_for_write(i).add_new_stroke(points_per_stroke);
      for (const int k : stroke.points_positions_for_write().index_range()) {
        stroke.points_positions_for_write()[k] = {
            float(k), float((k * j) % stroke.points_num()), float(k + j)};
      }
    }
  }

  return gpd;
}

static bGPdata *build_old_gpencil_data(int num_layers,
                                       int frames_per_layer,
                                       int strokes_per_frame,
                                       int points_per_stroke)
{
  bGPdata *gpd = reinterpret_cast<bGPdata *>(MEM_mallocN(sizeof(bGPdata), __func__));
  BLI_listbase_clear(&gpd->layers);
  for (int i = 0; i < num_layers; i++) {
    bGPDlayer *gpl = reinterpret_cast<bGPDlayer *>(MEM_mallocN(sizeof(bGPDlayer), __func__));
    sprintf(gpl->info, "%s%d", "GPLayer", i);
    gpl->flag = 0;

    BLI_listbase_clear(&gpl->mask_layers);
    BLI_listbase_clear(&gpl->frames);
    for (int j = 0; j < frames_per_layer; j++) {
      bGPDframe *gpf = reinterpret_cast<bGPDframe *>(MEM_mallocN(sizeof(bGPDframe), __func__));
      gpf->framenum = j;

      BLI_listbase_clear(&gpf->strokes);
      for (int k = 0; k < strokes_per_frame; k++) {
        bGPDstroke *gps = reinterpret_cast<bGPDstroke *>(
            MEM_mallocN(sizeof(bGPDstroke), __func__));
        gps->totpoints = points_per_stroke;
        gps->points = reinterpret_cast<bGPDspoint *>(
            MEM_calloc_arrayN(points_per_stroke, sizeof(bGPDspoint), __func__));
        gps->triangles = nullptr;
        gps->editcurve = nullptr;
        gps->dvert = nullptr;

        for (int l = 0; l < points_per_stroke; l++) {
          float pos[3] = {(float)l, (float)((l * k) % points_per_stroke), (float)(l + k)};
          bGPDspoint *pt = &gps->points[l];
          copy_v3_v3(&pt->x, pos);
        }

        BLI_addtail(&gpf->strokes, gps);
      }
      BLI_addtail(&gpl->frames, gpf);
    }
    BLI_addtail(&gpd->layers, gpl);
  }

  return gpd;
}

static bGPdata *copy_old_gpencil_data(bGPdata *gpd_src)
{
  bGPdata *gpd_dst = reinterpret_cast<bGPdata *>(MEM_mallocN(sizeof(bGPdata), __func__));
  BLI_listbase_clear(&gpd_dst->layers);
  LISTBASE_FOREACH (bGPDlayer *, gpl_src, &gpd_src->layers) {
    bGPDlayer *gpl_dst = BKE_gpencil_layer_duplicate(gpl_src, true, true);
    BLI_addtail(&gpd_dst->layers, gpl_dst);
  }

  return gpd_dst;
}

static void insert_new_frame_old_gpencil_data(bGPdata *gpd, int frame_num)
{
  bGPDlayer *gpl_active = BKE_gpencil_layer_active_get(gpd);
  BKE_gpencil_frame_addnew(gpl_active, frame_num);
}

static void free_old_gpencil_data(bGPdata *gpd)
{
  BKE_gpencil_free_layers(&gpd->layers);
  MEM_SAFE_FREE(gpd);
}

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

  const int layer_index = data.add_layer("FooLayer");
  EXPECT_EQ(data.layers_size, 1);
  EXPECT_STREQ(data.layers(layer_index).name, "FooLayer");
}

TEST(gpencil_proposal, AddLayers)
{
  GPData data;
  StringRefNull layer_names[3] = {"TestLayer1", "TestLayer2", "TestLayer3"};

  for (int i : IndexRange(3)) {
    data.add_layer(layer_names[i]);
  }
  EXPECT_EQ(data.layers_size, 3);

  for (int i : IndexRange(3)) {
    EXPECT_STREQ(data.layers(i).name, layer_names[i].c_str());
  }
}

TEST(gpencil_proposal, ChangeLayerName)
{
  GPData data;

  const int layer_index = data.add_layer("FooLayer");
  EXPECT_EQ(data.layers_size, 1);
  EXPECT_STREQ(data.layers(layer_index).name, "FooLayer");

  strcpy(data.layers_for_write(layer_index).name, "BarLayer");

  EXPECT_EQ(data.layers_size, 1);
  EXPECT_STREQ(data.layers(layer_index).name, "BarLayer");
}

TEST(gpencil_proposal, AddFrameToLayer)
{
  GPData data;

  data.add_layer("TestLayer1");
  const int layer2_index = data.add_layer("TestLayer2");

  const int frame_index = data.add_frame_on_layer(layer2_index, 0);
  EXPECT_NE(frame_index, -1);

  EXPECT_EQ(data.frames_size, 1);
  EXPECT_EQ(data.frames().last().layer_index, 1);
  EXPECT_EQ(data.frames(frame_index).layer_index, 1);

  data.frames_for_write(frame_index).start_time = 20;
  EXPECT_EQ(data.frames(frame_index).start_time, 20);
}

TEST(gpencil_proposal, CheckFramesSorted1)
{
  GPData data;

  const int frame_numbers1[5] = {10, 5, 6, 1, 3};
  const int frame_numbers_sorted1[5] = {1, 3, 5, 6, 10};

  int layer1_index = data.add_layer("TestLayer1");
  for (int i : IndexRange(5)) {
    const int frame_index = data.add_frame_on_layer(layer1_index, frame_numbers1[i]);
    EXPECT_NE(frame_index, -1);
    EXPECT_EQ(data.frames(frame_index).start_time, frame_numbers1[i]);
  }

  for (const int i : data.frames().index_range()) {
    EXPECT_EQ(data.frames(i).start_time, frame_numbers_sorted1[i]);
  }
}

TEST(gpencil_proposal, CheckFramesSorted2)
{
  GPData data;

  const int frame_numbers_layer1[5] = {10, 5, 6, 1, 3};
  const int frame_numbers_layer2[5] = {8, 5, 7, 1, 4};
  const int frame_numbers_sorted2[10][2] = {
      {0, 1}, {1, 1}, {0, 3}, {1, 4}, {0, 5}, {1, 5}, {0, 6}, {1, 7}, {1, 8}, {0, 10}};

  const int layer1_index = data.add_layer("TestLayer1");
  const int layer2_index = data.add_layer("TestLayer2");
  for (int i : IndexRange(5)) {
    data.add_frame_on_layer(layer1_index, frame_numbers_layer1[i]);
    data.add_frame_on_layer(layer2_index, frame_numbers_layer2[i]);
  }

  for (const int i : data.frames().index_range()) {
    EXPECT_EQ(data.frames(i).layer_index, frame_numbers_sorted2[i][0]);
    EXPECT_EQ(data.frames(i).start_time, frame_numbers_sorted2[i][1]);
  }
}

TEST(gpencil_proposal, IterateOverFramesOnLayer)
{
  GPData data;

  const int frame_numbers_layer1[5] = {10, 5, 6, 1, 3};
  const int frame_numbers_layer2[5] = {8, 5, 7, 1, 4};

  const int frame_numbers_sorted1[5] = {1, 3, 5, 6, 10};
  const int frame_numbers_sorted2[5] = {1, 4, 5, 7, 8};

  const int layer1_index = data.add_layer("TestLayer1");
  const int layer2_index = data.add_layer("TestLayer2");
  for (int i : IndexRange(5)) {
    data.add_frame_on_layer(layer1_index, frame_numbers_layer1[i]);
    data.add_frame_on_layer(layer2_index, frame_numbers_layer2[i]);
  }

  IndexMask indices_frames_layer1 = data.frames_on_layer(layer1_index);
  EXPECT_TRUE(data.runtime->frame_index_masks_cache.contains(layer1_index));
  for (const int i : indices_frames_layer1.index_range()) {
    EXPECT_EQ(data.frames(indices_frames_layer1[i]).start_time, frame_numbers_sorted1[i]);
  }

  IndexMask indices_frames_layer2 = data.frames_on_layer(layer2_index);
  EXPECT_TRUE(data.runtime->frame_index_masks_cache.contains(layer2_index));
  for (const int i : indices_frames_layer2.index_range()) {
    EXPECT_EQ(data.frames(indices_frames_layer2[i]).start_time, frame_numbers_sorted2[i]);
  }
}

TEST(gpencil_proposal, AddSingleStroke)
{
  GPData data;
  const int layer1_index = data.add_layer("TestLayer1");

  const int frame_index = data.add_frame_on_layer(layer1_index, 0);
  EXPECT_NE(frame_index, -1);
  GPStroke stroke = data.frames_for_write(frame_index).add_new_stroke(100);

  EXPECT_EQ(data.strokes_num(), 1);
  EXPECT_EQ(data.frames(frame_index).strokes_num(), 1);
  EXPECT_EQ(stroke.points_num(), 100);
}

TEST(gpencil_proposal, ChangeStrokePoints)
{
  GPData data;
  const int layer1_index = data.add_layer("TestLayer1");

  static const Array<float3> test_positions{{
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
      {7.0f, 8.0f, 9.0f},
  }};

  const int frame_index = data.add_frame_on_layer(layer1_index, 0);
  EXPECT_NE(frame_index, -1);
  GPStroke stroke = data.frames_for_write(frame_index).add_new_stroke(test_positions.size());

  for (const int i : stroke.points_positions_for_write().index_range()) {
    stroke.points_positions_for_write()[i] = test_positions[i];
  }

  for (const int i : stroke.points_positions().index_range()) {
    EXPECT_V3_NEAR(stroke.points_positions()[i], test_positions[i], 1e-5f);
  }
}

TEST(gpencil_proposal, BigGPData)
{
  GPData data = build_gpencil_data(5, 500, 100, 100);

  EXPECT_EQ(data.strokes_num(), 250e3);
  EXPECT_EQ(data.points_num(), 25e6);
}

TEST(gpencil_proposal, TimeBigGPDataCopy)
{
  int layers_num = 10, frames_num = 500, strokes_num = 100, points_num = 100;

  GPData data = build_gpencil_data(layers_num, frames_num, strokes_num, points_num);
  GPData data_copy;

  TIMEIT_START(BigGPDataCopy);
  data_copy = data;
  TIMEIT_END(BigGPDataCopy);

  bGPdata *old_data = build_old_gpencil_data(layers_num, frames_num, strokes_num, points_num);
  bGPdata *old_data_copy;

  TIMEIT_START(BigGPDataCopyOld);
  old_data_copy = copy_old_gpencil_data(old_data);
  TIMEIT_END(BigGPDataCopyOld);

  free_old_gpencil_data(old_data);
  free_old_gpencil_data(old_data_copy);
}

TEST(gpencil_proposal, TimeBigGPDataInsertFrame)
{
  int layers_num = 100, frames_num = 1000, strokes_num = 10, points_num = 10;
  GPData data = build_gpencil_data(layers_num, frames_num, strokes_num, points_num);
  data.set_active_layer(7);

  TIMEIT_START(TimeBigGPDataInsertFrame);
  data.add_frame_on_active_layer(347);
  TIMEIT_END(TimeBigGPDataInsertFrame);

  EXPECT_EQ(data.frames_on_active_layer().size(), 1001);

  bGPdata *old_data = build_old_gpencil_data(layers_num, frames_num, strokes_num, points_num);
  int i = 0;
  bGPDlayer *gpl_active = NULL;
  LISTBASE_FOREACH_INDEX (bGPDlayer *, gpl, &old_data->layers, i) {
    if (i == 7) {
      BKE_gpencil_layer_active_set(old_data, gpl);
      gpl_active = gpl;
      break;
    }
  }
  /* Remove the frame so we can insert it again. */
  LISTBASE_FOREACH (bGPDframe *, gpf, &gpl_active->frames) {
    if (gpf->framenum == 347) {
      BKE_gpencil_layer_frame_delete(gpl_active, gpf);
      break;
    }
  }

  TIMEIT_START(TimeBigGPDataOldInsertFrame);
  insert_new_frame_old_gpencil_data(old_data, 347);
  TIMEIT_END(TimeBigGPDataOldInsertFrame);

  EXPECT_EQ(BLI_listbase_count(&gpl_active->frames), 1000);

  free_old_gpencil_data(old_data);
}

}  // namespace blender::bke::gpencil::tests
