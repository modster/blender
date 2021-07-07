
import math
import os
import sys
from random import shuffle, seed

import bpy

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from modules.spec_classes import ModifierSpec, DeformModifierSpec, OperatorSpecEditMode, OperatorSpecObjectMode, ParticleSystemSpec
from modules.base_mesh_test import SpecMeshTest, BlendFileTest

# modifier_test = SpecTest("test_name_demo", "test_demo", "exp_demo", [])

# geo_node_test = BlendFileTest("test_object", "expected_object")
# geo_node_test.run_test()

modifier_test = SpecMeshTest("CubeSubsurfPass", "test_object", "subsurf_obj",
                             [ModifierSpec('subsurf', 'SUBSURF', {})])

modifier_test.run_test()

modifier_test_2 = SpecMeshTest("CubeSubsurffail", "test_object", "no_subsurf",
                             [ModifierSpec('subsurf', 'SUBSURF', {})])

modifier_test_2.run_test()