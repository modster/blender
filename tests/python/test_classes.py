import math
import os
import sys
from random import shuffle, seed

import bpy

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from modules.base_mesh_test import SpecTest, BlendFileTest
from modules.mesh_test import RunTest, ModifierSpec, MeshTest

# modifier_test = SpecTest("test_name_demo", "test_demo", "exp_demo", [])

geo_node_test = BlendFileTest("test_object", "expected_object")
geo_node_test.run_test()
