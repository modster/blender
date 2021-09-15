/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#pragma once

#include "NOD_node_declaration.hh"

#include "RNA_types.h"

#include "BLI_color.hh"
#include "BLI_float3.hh"

namespace blender::nodes::decl {

class Float : public SocketDeclaration {
 private:
  float default_value_ = 0.0f;
  float soft_min_value_ = -FLT_MAX;
  float soft_max_value_ = FLT_MAX;
  PropertySubType subtype_ = PROP_NONE;

  friend class FloatBuilder;

 public:
  using Builder = class FloatBuilder;

  bNodeSocket &build(bNodeTree &ntree, bNode &node, eNodeSocketInOut in_out) const override;
  bool matches(const bNodeSocket &socket) const override;
  bNodeSocket &update_or_build(bNodeTree &ntree, bNode &node, bNodeSocket &socket) const override;
};

class FloatBuilder : public SocketDeclarationBuilder<Float> {
 public:
  FloatBuilder &min(const float value)
  {
    decl_->soft_min_value_ = value;
    return *this;
  }

  FloatBuilder &max(const float value)
  {
    decl_->soft_max_value_ = value;
    return *this;
  }

  FloatBuilder &default_value(const float value)
  {
    decl_->default_value_ = value;
    return *this;
  }

  FloatBuilder &subtype(PropertySubType subtype)
  {
    decl_->subtype_ = subtype;
    return *this;
  }
};

class Int : public SocketDeclaration {
 private:
  int default_value_ = 0;
  int soft_min_value_ = INT32_MIN;
  int soft_max_value_ = INT32_MAX;
  PropertySubType subtype_ = PROP_NONE;

  friend class IntBuilder;

 public:
  using Builder = class IntBuilder;

  bNodeSocket &build(bNodeTree &ntree, bNode &node, eNodeSocketInOut in_out) const override;
  bool matches(const bNodeSocket &socket) const override;
  bNodeSocket &update_or_build(bNodeTree &ntree, bNode &node, bNodeSocket &socket) const override;
};

class IntBuilder : public SocketDeclarationBuilder<Int> {
 public:
  IntBuilder &min(const int value)
  {
    decl_->soft_min_value_ = value;
    return *this;
  }

  IntBuilder &max(const int value)
  {
    decl_->soft_max_value_ = value;
    return *this;
  }

  IntBuilder &default_value(const int value)
  {
    decl_->default_value_ = value;
    return *this;
  }

  IntBuilder &subtype(PropertySubType subtype)
  {
    decl_->subtype_ = subtype;
    return *this;
  }
};

class Vector : public SocketDeclaration {
 private:
  float3 default_value_ = {0, 0, 0};
  float soft_min_value_ = -FLT_MAX;
  float soft_max_value_ = FLT_MAX;
  PropertySubType subtype_ = PROP_NONE;

  friend class VectorBuilder;

 public:
  using Builder = class VectorBuilder;

  bNodeSocket &build(bNodeTree &ntree, bNode &node, eNodeSocketInOut in_out) const override;
  bool matches(const bNodeSocket &socket) const override;
  bNodeSocket &update_or_build(bNodeTree &ntree, bNode &node, bNodeSocket &socket) const override;
};

class VectorBuilder : public SocketDeclarationBuilder<Vector> {
 public:
  VectorBuilder &default_value(const float3 value)
  {
    decl_->default_value_ = value;
    return *this;
  }

  VectorBuilder &subtype(PropertySubType subtype)
  {
    decl_->subtype_ = subtype;
    return *this;
  }

  VectorBuilder &min(const float min)
  {
    decl_->soft_min_value_ = min;
    return *this;
  }

  VectorBuilder &max(const float max)
  {
    decl_->soft_max_value_ = max;
    return *this;
  }
};

class Bool : public SocketDeclaration {
 private:
  bool default_value_ = false;
  friend class BoolBuilder;

 public:
  using Builder = class BoolBuilder;

  bNodeSocket &build(bNodeTree &ntree, bNode &node, eNodeSocketInOut in_out) const override;
  bool matches(const bNodeSocket &socket) const override;
};

class BoolBuilder : public SocketDeclarationBuilder<Bool> {
 public:
  BoolBuilder &default_value(const bool value)
  {
    decl_->default_value_ = value;
    return *this;
  }
};

class Color : public SocketDeclaration {
 private:
  ColorGeometry4f default_value_;

  friend class ColorBuilder;

 public:
  using Builder = class ColorBuilder;

  bNodeSocket &build(bNodeTree &ntree, bNode &node, eNodeSocketInOut in_out) const override;
  bool matches(const bNodeSocket &socket) const override;
};

class ColorBuilder : public SocketDeclarationBuilder<Color> {
 public:
  ColorBuilder &default_value(const ColorGeometry4f value)
  {
    decl_->default_value_ = value;
    return *this;
  }
};

class String : public SocketDeclaration {
 public:
  using Builder = class StringBuilder;

  bNodeSocket &build(bNodeTree &ntree, bNode &node, eNodeSocketInOut in_out) const override;
  bool matches(const bNodeSocket &socket) const override;
};

class StringBuilder : public SocketDeclarationBuilder<String> {
};

namespace detail {
struct CommonIDSocketData {
  const char *idname;
};

bNodeSocket &build_id_socket(const SocketDeclaration &decl,
                             bNodeTree &ntree,
                             bNode &node,
                             eNodeSocketInOut in_out,
                             const CommonIDSocketData &data);
bool matches_id_socket(const SocketDeclaration &decl,
                       const bNodeSocket &socket,
                       const CommonIDSocketData &data);

class IDSocketDeclaration : public SocketDeclaration {
 private:
  CommonIDSocketData data_;

 public:
  IDSocketDeclaration(const char *idname) : data_({idname})
  {
  }

  bNodeSocket &build(bNodeTree &ntree, bNode &node, eNodeSocketInOut in_out) const override
  {
    return build_id_socket(*this, ntree, node, in_out, data_);
  }

  bool matches(const bNodeSocket &socket) const override
  {
    return matches_id_socket(*this, socket, data_);
  }

  bNodeSocket &update_or_build(bNodeTree &ntree, bNode &node, bNodeSocket &socket) const override
  {
    if (StringRef(socket.idname) != data_.idname) {
      return this->build(ntree, node, (eNodeSocketInOut)socket.in_out);
    }
    return socket;
  }
};
}  // namespace detail

class Object : public detail::IDSocketDeclaration {
 public:
  using Builder = class ObjectBuilder;

  Object() : detail::IDSocketDeclaration("NodeSocketObject")
  {
  }
};

class ObjectBuilder : public SocketDeclarationBuilder<Object> {
};

class Material : public detail::IDSocketDeclaration {
 public:
  using Builder = class MaterialBuilder;

  Material() : detail::IDSocketDeclaration("NodeSocketMaterial")
  {
  }
};

class MaterialBuilder : public SocketDeclarationBuilder<Material> {
};

class Collection : public detail::IDSocketDeclaration {
 public:
  using Builder = class CollectionBuilder;

  Collection() : detail::IDSocketDeclaration("NodeSocketCollection")
  {
  }
};

class CollectionBuilder : public SocketDeclarationBuilder<Collection> {
};

class Texture : public detail::IDSocketDeclaration {
 public:
  using Builder = class TextureBuilder;

  Texture() : detail::IDSocketDeclaration("NodeSocketTexture")
  {
  }
};

class TextureBuilder : public SocketDeclarationBuilder<Texture> {
};

class Geometry : public SocketDeclaration {
 public:
  using Builder = class GeometryBuilder;

  bNodeSocket &build(bNodeTree &ntree, bNode &node, eNodeSocketInOut in_out) const override;
  bool matches(const bNodeSocket &socket) const override;
};

class GeometryBuilder : public SocketDeclarationBuilder<Geometry> {
};

}  // namespace blender::nodes::decl
