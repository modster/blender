/* Apache License, Version 2.0 */

#include <fstream>
#include <gtest/gtest.h>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>

#include "testing/testing.h"
#include "tests/blendfile_loading_base_test.h"

#include "BKE_blender_version.h"

#include "BLI_index_range.hh"
#include "BLI_string_utf8.h"
#include "BLI_vector.hh"

#include "DEG_depsgraph.h"

#include "obj_export_file_writer.hh"
#include "obj_export_mesh.hh"
#include "obj_export_nurbs.hh"
#include "obj_exporter.hh"

#include "obj_exporter_tests.hh"

namespace blender::io::obj {

/* This is also the test name. */
class obj_exporter_test : public BlendfileLoadingBaseTest {
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

// https://developer.blender.org/F9260238
const std::string all_objects_file = "io_tests/blend_scene/all_objects_2_92.blend";
// https://developer.blender.org/F9278970
const std::string all_curve_objects_file = "io_tests/blend_scene/all_curves_2_92.blend";

const std::string some_meshes_file = "io_tests/blend_scene/all_meshes.blend";

TEST_F(obj_exporter_test, filter_objects_curves_as_mesh)
{
  OBJExportParamsDefault _export;
  if (!load_file_and_depsgraph(all_objects_file)) {
    ADD_FAILURE();
    return;
  }

  auto [objmeshes, objcurves]{filter_supported_objects(depsgraph, _export.params)};
  EXPECT_EQ(objmeshes.size(), 22);
  EXPECT_EQ(objcurves.size(), 0);
}

TEST_F(obj_exporter_test, filter_objects_curves_as_nurbs)
{
  OBJExportParamsDefault _export;
  if (!load_file_and_depsgraph(all_objects_file)) {
    ADD_FAILURE();
    return;
  }
  _export.params.export_curves_as_nurbs = true;
  auto [objmeshes, objcurves]{filter_supported_objects(depsgraph, _export.params)};
  EXPECT_EQ(objmeshes.size(), 18);
  EXPECT_EQ(objcurves.size(), 4);
}

TEST_F(obj_exporter_test, filter_objects_selected)
{
  OBJExportParamsDefault _export;
  if (!load_file_and_depsgraph(all_objects_file)) {
    ADD_FAILURE();
    return;
  }
  _export.params.export_selected_objects = true;
  _export.params.export_curves_as_nurbs = true;
  auto [objmeshes, objcurves]{filter_supported_objects(depsgraph, _export.params)};
  EXPECT_EQ(objmeshes.size(), 8);
  EXPECT_EQ(objcurves.size(), 2);
}

TEST(obj_exporter_utils, append_negative_frame_to_filename)
{
  const char path_original[FILE_MAX] = "/my_file.obj";
  const char path_truth[FILE_MAX] = "/my_file-123.obj";
  const int frame = -123;
  char path_with_frame[FILE_MAX] = {0};
  const bool ok = append_frame_to_filename(path_original, frame, path_with_frame);
  EXPECT_TRUE(ok);
  EXPECT_EQ_ARRAY(path_with_frame, path_truth, BLI_strlen_utf8(path_truth));
}

TEST(obj_exporter_utils, append_positive_frame_to_filename)
{
  const char path_original[FILE_MAX] = "/my_file.obj";
  const char path_truth[FILE_MAX] = "/my_file123.obj";
  const int frame = 123;
  char path_with_frame[FILE_MAX] = {0};
  const bool ok = append_frame_to_filename(path_original, frame, path_with_frame);
  EXPECT_TRUE(ok);
  EXPECT_EQ_ARRAY(path_with_frame, path_truth, BLI_strlen_utf8(path_truth));
}

TEST_F(obj_exporter_test, curve_nurbs_points)
{
  if (!load_file_and_depsgraph(all_curve_objects_file)) {
    ADD_FAILURE();
    return;
  }

  OBJExportParamsDefault _export;
  _export.params.export_curves_as_nurbs = true;
  auto [objmeshes_unused, objcurves]{filter_supported_objects(depsgraph, _export.params)};

  for (StealUniquePtr<OBJCurve> objcurve : objcurves) {
    if (all_nurbs_truth.count(objcurve->get_curve_name()) != 1) {
      ADD_FAILURE();
      return;
    }
    const NurbsObject *const nurbs_truth = all_nurbs_truth.at(objcurve->get_curve_name()).get();
    EXPECT_EQ(objcurve->total_splines(), nurbs_truth->total_splines());
    for (int spline_index : IndexRange(objcurve->total_splines())) {
      EXPECT_EQ(objcurve->total_spline_vertices(spline_index),
                nurbs_truth->total_spline_vertices(spline_index));
      EXPECT_EQ(objcurve->get_nurbs_degree(spline_index),
                nurbs_truth->get_nurbs_degree(spline_index));
      EXPECT_EQ(objcurve->total_spline_control_points(spline_index),
                nurbs_truth->total_spline_control_points(spline_index));
    }
  }
}

TEST_F(obj_exporter_test, curve_coordinates)
{
  if (!load_file_and_depsgraph(all_curve_objects_file)) {
    ADD_FAILURE();
    return;
  }

  OBJExportParamsDefault _export;
  _export.params.export_curves_as_nurbs = true;
  auto [objmeshes_unused, objcurves]{filter_supported_objects(depsgraph, _export.params)};

  for (StealUniquePtr<OBJCurve> objcurve : objcurves) {
    if (all_nurbs_truth.count(objcurve->get_curve_name()) != 1) {
      ADD_FAILURE();
      return;
    }
    const NurbsObject *const nurbs_truth = all_nurbs_truth.at(objcurve->get_curve_name()).get();
    EXPECT_EQ(objcurve->total_splines(), nurbs_truth->total_splines());
    for (int spline_index : IndexRange(objcurve->total_splines())) {
      for (int vertex_index : IndexRange(objcurve->total_spline_vertices(spline_index))) {
        EXPECT_V3_NEAR(objcurve->vertex_coordinates(
                           spline_index, vertex_index, _export.params.scaling_factor),
                       nurbs_truth->vertex_coordinates(spline_index, vertex_index),
                       0.000001f);
      }
    }
  }
}

static std::unique_ptr<OBJWriter> init_writer(const OBJExportParams &params,
                                              const std::string out_filepath)
{
  try {
    auto writer = std::make_unique<OBJWriter>(out_filepath.c_str(), params);
    return writer;
  }
  catch (const std::system_error &ex) {
    std::cerr << ex.code().category().name() << ": " << ex.what() << ": " << ex.code().message()
              << std::endl;
    return nullptr;
  }
}

const char *const temp_file_path =
    "/Users/ankitkumar/blender-build/build_darwin_lite/Testing/output.OBJ";

static std::string read_temp_file_in_string(const std::string &file_path)
{
  std::ifstream temp_stream(file_path);
  std::ostringstream input_ss;
  input_ss << temp_stream.rdbuf();
  return input_ss.str();
}

TEST(obj_exporter_writer, header)
{
  {
    OBJExportParamsDefault _export;
    std::unique_ptr<OBJWriter> writer = init_writer(_export.params, temp_file_path);
    if (!writer) {
      ADD_FAILURE();
      return;
    }
    writer->write_header();
  }
  const std::string result = read_temp_file_in_string(temp_file_path);
  using namespace std::string_literals;
  ASSERT_EQ(result, "# Blender "s + BKE_blender_version_string() + "\n" + "# www.blender.org\n");
}

TEST(obj_exporter_writer, mtllib)
{
  {
    OBJExportParamsDefault _export;
    std::unique_ptr<OBJWriter> writer = init_writer(_export.params, temp_file_path);
    if (!writer) {
      ADD_FAILURE();
      return;
    }
    writer->write_mtllib_name("/Users/blah.mtl");
    writer->write_mtllib_name("\\C:\\blah.mtl");
  }
  const std::string result = read_temp_file_in_string(temp_file_path);
  ASSERT_EQ(result, "mtllib blah.mtl\nmtllib blah.mtl\n");
}

}  // namespace blender::io::obj
