/* Apache License, Version 2.0 */

#include <gtest/gtest.h>
#include <ios>

#include "testing/testing.h"
#include "tests/blendfile_loading_base_test.h"

#include "BLI_float3.hh"
#include "BLI_string_utf8.h"
#include "BLI_vector.hh"

#include "DEG_depsgraph.h"

#include "IO_wavefront_obj.h"
#include "obj_export_mesh.hh"
#include "obj_export_nurbs.hh"
#include "obj_exporter.hh"

namespace blender::io::obj {

class Export_OBJ : public BlendfileLoadingBaseTest {
 public:
  /**
   * \param filepath: relative to "tests" directory.
   */
  bool load_file_and_depsgraph(const std::string &filepath,
                               const eEvaluationMode eval_mode = DAG_EVAL_VIEWPORT)
  {
    if (!blendfile_load(filepath.c_str())) {
      return false;
    }
    depsgraph_create(eval_mode);
    return true;
  }
};

struct OBJExportParamsDefault {
  OBJExportParams params;
  OBJExportParamsDefault()
  {
    params.filepath[0] = '\0';
    params.export_animation = false;
    params.start_frame = 0;
    params.end_frame = 1;

    params.forward_axis = OBJ_AXIS_NEGATIVE_Z_FORWARD;
    params.up_axis = OBJ_AXIS_Y_UP;
    params.scaling_factor = 1.f;

    params.export_eval_mode = DAG_EVAL_VIEWPORT;
    params.export_selected_objects = false;
    params.export_uv = true;
    params.export_normals = true;
    params.export_materials = true;
    params.export_triangulated_mesh = false;
    params.export_curves_as_nurbs = false;

    params.export_object_groups = false;
    params.export_material_groups = false;
    params.export_vertex_groups = false;
    params.export_smooth_groups = true;
    params.smooth_groups_bitflags = false;
  }
};

const std::string all_objects_file = "io_tests/blend_scene/all_objects_2_92.blend";

TEST_F(Export_OBJ, filter_objects_as_mesh)
{
  OBJExportParamsDefault _export;
  if (!load_file_and_depsgraph(all_objects_file)) {
    return;
  }

  auto [objmeshes, objcurves]{find_exportable_objects(depsgraph, _export.params)};
  EXPECT_EQ(objmeshes.size(), 22);
  EXPECT_EQ(objcurves.size(), 0);
}

TEST_F(Export_OBJ, filter_objects_as_curves)
{
  OBJExportParamsDefault _export;
  if (!load_file_and_depsgraph(all_objects_file)) {
    return;
  }
  _export.params.export_curves_as_nurbs = true;
  auto [objmeshes, objcurves]{find_exportable_objects(depsgraph, _export.params)};
  EXPECT_EQ(objmeshes.size(), 18);
  EXPECT_EQ(objcurves.size(), 4);
}

TEST_F(Export_OBJ, filter_objects_selected)
{
  OBJExportParamsDefault _export;
  if (!load_file_and_depsgraph(all_objects_file)) {
    return;
  }
  _export.params.export_selected_objects = true;
  _export.params.export_curves_as_nurbs = true;
  auto [objmeshes, objcurves]{find_exportable_objects(depsgraph, _export.params)};
  EXPECT_EQ(objmeshes.size(), 8);
  EXPECT_EQ(objcurves.size(), 2);
}

TEST(Export_OBJ_utils, append_negative_frame_to_filename)
{
  const char path_original[FILE_MAX] = "/my_file.obj";
  const char path_expected[FILE_MAX] = "/my_file-123.obj";
  const int frame = -123;
  char path_with_frame[FILE_MAX] = {0};
  const bool ok = append_frame_to_filename(path_original, frame, path_with_frame);
  EXPECT_TRUE(ok);
  EXPECT_EQ_ARRAY(path_with_frame, path_expected, BLI_strlen_utf8(path_expected));
}

TEST(Export_OBJ_utils, append_positive_frame_to_filename)
{
  const char path_original[FILE_MAX] = "/my_file.obj";
  const char path_expected[FILE_MAX] = "/my_file123.obj";
  const int frame = 123;
  char path_with_frame[FILE_MAX] = {0};
  const bool ok = append_frame_to_filename(path_original, frame, path_with_frame);
  EXPECT_TRUE(ok);
  EXPECT_EQ_ARRAY(path_with_frame, path_expected, BLI_strlen_utf8(path_expected));
}
}  // namespace blender::io::obj
