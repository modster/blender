/* Apache License, Version 2.0 */

#include "testing/testing.h"

#include "draw_testing.hh"
#include "intern/draw_manager_testing.h"

#include "DRW_engine.h"
#include "draw_cache_impl.h"

#include "BKE_editmesh.h"
#include "BKE_idtype.h"
#include "BKE_mesh.h"
#include "BKE_mesh_wrapper.h"
#include "BKE_scene.h"

#include "BLI_rand.hh"
#include "BLI_task.h"

#include "ED_mesh.h"

#include "DNA_mesh_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "GPU_context.h"
#include "GPU_init_exit.h"
#include "GPU_shader.h"

#include "engines/eevee/eevee_private.h"
#include "engines/gpencil/gpencil_engine.h"
#include "engines/image/image_private.h"
#include "engines/overlay/overlay_private.h"
#include "engines/workbench/workbench_private.h"

namespace blender::draw {

/**
 * During investigation or executing in a profiler it is handly to disable multithreading. This can
 * be done by setting RUN_SINGLE_THREADED to true.
 *
 * Default(false) => run multithreaded
 */
constexpr bool RUN_SINGLE_THREADED = false;

class DrawCacheTest : public DrawTest {
 protected:
  TaskGraph *task_graph;

 public:
  void SetUp() override
  {
    DrawTest::SetUp();
    if (RUN_SINGLE_THREADED) {
      BLI_system_num_threads_override_set(1);
    }
    task_graph = BLI_task_graph_create();
  }

  void TearDown() override
  {
    BLI_task_graph_free(task_graph);
    if (RUN_SINGLE_THREADED) {
      BLI_system_num_threads_override_set(0);
    }
    DrawTest::TearDown();
  }
};

class DrawCachePerformanceTest : public DrawCacheTest {
 protected:
  Scene scene = {{nullptr}};
  Object ob_mesh = {{nullptr}};
  Mesh mesh = {{nullptr}};
  BMesh *bm = nullptr;
  RandomNumberGenerator rng;

 public:
  void SetUp() override
  {
    DrawCacheTest::SetUp();
    IDType_ID_SCE.init_data(&scene.id);
    IDType_ID_OB.init_data(&ob_mesh.id);
    IDType_ID_ME.init_data(&mesh.id);
    ob_mesh.type = OB_MESH;
    ob_mesh.data = &mesh;
    EDBM_mesh_make(&ob_mesh, SCE_SELECT_VERTEX, false);
    bm = mesh.edit_mesh->bm;

    /* Ensure batch cache is available. */
    DRW_mesh_batch_cache_validate(&mesh);
  }

  void TearDown() override
  {
    EDBM_mesh_free(mesh.edit_mesh);
    bm = nullptr;
    IDType_ID_ME.free_data(&mesh.id);
    IDType_ID_OB.free_data(&ob_mesh.id);
    IDType_ID_SCE.free_data(&scene.id);
    DrawCacheTest::TearDown();
  }

 protected:
  /**
   * Build a test mesh with given number of polygons.
   * Each polygon is created from 3 random generated verts.
   */
  void build_mesh(size_t num_polygons)
  {
    add_polygons_to_bm(num_polygons);

    /* Make sure mesh_eval_final is up to date (inline depsgraph evaluation). See
     * `editbmesh_calc_modifiers`. */
    mesh.edit_mesh->mesh_eval_final = BKE_mesh_from_bmesh_for_eval_nomain(
        mesh.edit_mesh->bm, nullptr, &mesh);
    mesh.edit_mesh->mesh_eval_cage = BKE_mesh_wrapper_from_editmesh_with_coords(
        mesh.edit_mesh, nullptr, nullptr, &mesh);

    BKE_editmesh_looptri_calc(mesh.edit_mesh);
  }

  /**
   * Check if the given GPUBatch is filled.
   */
  void expect_filled(GPUBatch *batch)
  {
    EXPECT_NE(batch->elem, nullptr);
  }

  /**
   * Check if the given GPUBatch is filled.
   */
  void expect_empty(GPUBatch *batch)
  {
    EXPECT_EQ(batch->elem, nullptr);
    for (int i = 0; i < GPU_BATCH_VBO_MAX_LEN; i++) {
      EXPECT_EQ(batch->verts[i], nullptr);
    }
  }

 private:
  /**
   * Create a new random vert in BMesh.
   */
  BMVert *add_random_vert_to_bm()
  {
    float co[3] = {(rng.get_float() - 0.5f) * 10.0f,
                   (rng.get_float() - 0.5f) * 10.0f,
                   (rng.get_float() - 0.5f) * 10.0f};
    BMVert *result = BM_vert_create(bm, co, nullptr, BM_CREATE_NOP);
    return result;
  }

  /**
   * Add `num_polygons` polygons to the BMesh.
   */
  void add_polygons_to_bm(size_t num_polygons)
  {
    /* Use 3 verts per face to skip triangulation. */
    const int verts_per_face = 3;

    for (int i = 0; i < num_polygons; i++) {
      BMVert *verts[verts_per_face];
      for (int j = 0; j < verts_per_face; j++) {
        verts[j] = add_random_vert_to_bm();
      }
      BM_face_create_verts(bm, verts, verts_per_face, nullptr, BM_CREATE_NOP, true);
    }
  }
};

/**
 * Base line benchmark simulating edit mesh vertice transform of a large mesh.
 *
 * In Blender 2.93 the whole cache would be freed so the baseline is to recalculate
 * all needed caches.
 */
TEST_F(DrawCachePerformanceTest, edit_mesh_performance_baseline_293)
{
  /* Approximates a subdivided cube 7 times in faces. */
  const int num_polygons = 100000;
  const int num_benchmark_loops = 32;
  /* Create a bmesh object in edit mode. */
  build_mesh(num_polygons);

  for (int i = 0; i < num_benchmark_loops; i++) {
    /* Invalidate caches.
     * In reality the mesh gets a copy on write signal and frees the cache (mesh_data_free). */
    BKE_mesh_batch_cache_dirty_tag_cb(&mesh, BKE_MESH_BATCH_DIRTY_ALL);
    DRW_mesh_batch_cache_validate(&mesh);

    /* Request caches. */
    GPUBatch *batch_triangles = DRW_mesh_batch_cache_get_edit_triangles(&mesh);
    GPUBatch *batch_vertices = DRW_mesh_batch_cache_get_edit_vertices(&mesh);
    GPUBatch *batch_edges = DRW_mesh_batch_cache_get_edit_edges(&mesh);
    GPUBatch *batch_vnors = DRW_mesh_batch_cache_get_edit_vnors(&mesh);
    GPUBatch *batch_lnors = DRW_mesh_batch_cache_get_edit_lnors(&mesh);
    GPUBatch *batch_facedots = DRW_mesh_batch_cache_get_edit_facedots(&mesh);

    /* Check if caches are empty. */
    expect_empty(batch_triangles);
    expect_empty(batch_vertices);
    expect_empty(batch_edges);
    expect_empty(batch_vnors);
    expect_empty(batch_lnors);
    expect_empty(batch_facedots);

    /* Update caches. */
    DRW_mesh_batch_cache_create_requested(task_graph, &ob_mesh, &mesh, &scene, false, true);
    BLI_task_graph_work_and_wait(task_graph);

    /* Check if caches are filled. */
    expect_filled(batch_triangles);
    expect_filled(batch_vertices);
    expect_filled(batch_edges);
    expect_filled(batch_vnors);
    expect_filled(batch_lnors);
    expect_filled(batch_facedots);
  }
}

}  // namespace blender::draw
