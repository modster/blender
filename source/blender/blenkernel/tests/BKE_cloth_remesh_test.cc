#include "BKE_cloth_remesh.hh"

#include "testing/testing.h"
#include <gtest/gtest.h>
#include <sstream>

namespace blender::bke::tests {

using namespace internal;

static const char *cube_pos_uv_normal =
    "# Blender v3.0.0 Alpha OBJ File: ''\n"
    "# www.blender.org\n"
    "mtllib untitled.mtl\n"
    "o Cube\n"
    "v 1.000000 1.000000 -1.000000\n"
    "v 1.000000 -1.000000 -1.000000\n"
    "v 1.000000 1.000000 1.000000\n"
    "v 1.000000 -1.000000 1.000000\n"
    "v -1.000000 1.000000 -1.000000\n"
    "v -1.000000 -1.000000 -1.000000\n"
    "v -1.000000 1.000000 1.000000\n"
    "v -1.000000 -1.000000 1.000000\n"
    "vt 0.625000 0.500000\n"
    "vt 0.875000 0.500000\n"
    "vt 0.875000 0.750000\n"
    "vt 0.625000 0.750000\n"
    "vt 0.375000 0.750000\n"
    "vt 0.625000 1.000000\n"
    "vt 0.375000 1.000000\n"
    "vt 0.375000 0.000000\n"
    "vt 0.625000 0.000000\n"
    "vt 0.625000 0.250000\n"
    "vt 0.375000 0.250000\n"
    "vt 0.125000 0.500000\n"
    "vt 0.375000 0.500000\n"
    "vt 0.125000 0.750000\n"
    "vn 0.0000 1.0000 0.0000\n"
    "vn 0.0000 0.0000 1.0000\n"
    "vn -1.0000 0.0000 0.0000\n"
    "vn 0.0000 -1.0000 0.0000\n"
    "vn 1.0000 0.0000 0.0000\n"
    "vn 0.0000 0.0000 -1.0000\n"
    "usemtl Material\n"
    "s off\n"
    "f 1/1/1 5/2/1 7/3/1 3/4/1\n"
    "f 4/5/2 3/4/2 7/6/2 8/7/2\n"
    "f 8/8/3 7/9/3 5/10/3 6/11/3\n"
    "f 6/12/4 2/13/4 4/5/4 8/14/4\n"
    "f 2/13/5 1/1/5 3/4/5 4/5/5\n"
    "f 6/11/6 5/10/6 1/1/6 2/13/6\n";

TEST(cloth_remesh, MeshReader_ReadObj)
{
  MeshReader reader;
  std::istringstream stream(cube_pos_uv_normal);
  auto res = reader.read(std::move(stream), MeshReader::FILETYPE_OBJ);

  EXPECT_TRUE(res);

  const auto positions = reader.get_positions();
  const auto uvs = reader.get_uvs();
  const auto normals = reader.get_normals();
  const auto face_indices = reader.get_face_indices();
  const auto line_indices = reader.get_line_indices();

  EXPECT_EQ(positions.size(), 8);
  EXPECT_EQ(uvs.size(), 14);
  EXPECT_EQ(normals.size(), 6);
  EXPECT_EQ(face_indices.size(), 6);
  EXPECT_EQ(line_indices.size(), 0);
}

TEST(cloth_remesh, Mesh_Read)
{
  MeshReader reader;
  std::istringstream stream(cube_pos_uv_normal);
  reader.read(std::move(stream), MeshReader::FILETYPE_OBJ);

  Mesh<bool, bool, bool, bool> mesh;
  mesh.read(reader);

  const auto &nodes = mesh.get_nodes();
  const auto &verts = mesh.get_verts();
  const auto &edges = mesh.get_edges();
  const auto &faces = mesh.get_faces();

  EXPECT_EQ(nodes.size(), 8);
  EXPECT_EQ(verts.size(), 14);
  EXPECT_EQ(edges.size(), 19);
  EXPECT_EQ(faces.size(), 6);

  for (const auto &face : faces) {
    EXPECT_EQ(face.get_verts().size(), 4);
  }

  for (const auto &edge : edges) {
    auto num_faces = edge.get_faces().size();
    EXPECT_TRUE(num_faces == 1 || num_faces == 2);
  }

  for (const auto &vert : verts) {
    auto num_edges = vert.get_edges().size();
    EXPECT_TRUE(num_edges >= 2 && num_edges <= 4);
    EXPECT_NE(vert.get_node(), std::nullopt);
  }

  for (const auto &node : nodes) {
    auto num_verts = node.get_verts().size();
    EXPECT_TRUE(num_verts >= 1 && num_verts <= 3);
  }
}

} /* namespace blender::bke::tests */
