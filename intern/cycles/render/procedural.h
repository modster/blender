/*
 * Copyright 2011-2018 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "graph/node.h"

CCL_NAMESPACE_BEGIN

class Progress;
class Scene;

class Procedural : public Node, public NodeOwner {
 public:
  NODE_ABSTRACT_DECLARE
  explicit Procedural(const NodeType *type);
  virtual ~Procedural();
  virtual void generate(Scene *scene, Progress &progress) = 0;

  virtual bool is_procedural() const;
};

class ProceduralManager {
  bool need_update_;

 public:
  ProceduralManager();
  ~ProceduralManager();

  void update(Scene *scene, Progress &progress);

  void tag_update()
  {
    need_update_ = true;
  }

  bool need_update() const
  {
    return need_update_;
  }
};

CCL_NAMESPACE_END
