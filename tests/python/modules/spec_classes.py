# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

# A framework to run regression tests on mesh modifiers and operators based on howardt's mesh_ops_test.py
#
# General idea:
# A test is:
#    Object mode
#    Select <test_object>
#    Duplicate the object
#    Select the object
#    Apply operation for each operation in <operations_stack> with given parameters
#    (an operation is either a modifier or an operator)
#    test_mesh = <test_object>.data
#    run test_mesh.unit_test_compare(<expected object>.data)
#    delete the duplicate object
#
# The words in angle brackets are parameters of the test, and are specified in
# the main class MeshTest.
#
# If the environment variable BLENDER_TEST_UPDATE is set to 1, the <expected_object>
# is updated with the new test result.
# Tests are verbose when the environment variable BLENDER_VERBOSE is set.


import bpy
import functools
import inspect
import os

# Output from this module and from blender itself will occur during tests.
# We need to flush python so that the output is properly interleaved, otherwise
# blender's output for one test will end up showing in the middle of another test...
import sys

print = functools.partial(print, flush=True)
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class ModifierSpec:
    """
    Holds a Generate or Deform or Physics modifier type and its parameters.
    """

    def __init__(self, modifier_name: str, modifier_type: str, modifier_parameters: dict, frame_end=0):
        """
        Constructs a modifier spec.
        :param modifier_name: str - name of object modifier, e.g. "myFirstSubsurfModif"
        :param modifier_type: str - type of object modifier, e.g. "SUBSURF"
        :param modifier_parameters: dict - {name : val} dictionary giving modifier parameters, e.g. {"quality" : 4}
        :param frame_end: int - frame at which simulation needs to be baked or modifier needs to be applied.
        """
        self.modifier_name = modifier_name
        self.modifier_type = modifier_type
        self.modifier_parameters = modifier_parameters
        self.frame_end = frame_end

    def __str__(self):
        return "Modifier: " + self.modifier_name + " of type " + self.modifier_type + \
               " with parameters: " + str(self.modifier_parameters)


class ParticleSystemSpec:
    """
    Holds a Particle System modifier and its parameters.
    """

    def __init__(self, modifier_name: str, modifier_type: str, modifier_parameters: dict, frame_end: int):
        """
        Constructs a particle system spec.
        :param modifier_name: str - name of object modifier, e.g. "Particles"
        :param modifier_type: str - type of object modifier, e.g. "PARTICLE_SYSTEM"
        :param modifier_parameters: dict - {name : val} dictionary giving modifier parameters, e.g. {"seed" : 1}
        :param frame_end: int - the last frame of the simulation at which the modifier is applied
        """
        self.modifier_name = modifier_name
        self.modifier_type = modifier_type
        self.modifier_parameters = modifier_parameters
        self.frame_end = frame_end

    def __str__(self):
        return "Physics Modifier: " + self.modifier_name + " of type " + self.modifier_type + \
               " with parameters: " + \
            str(self.modifier_parameters) + \
            " with frame end: " + str(self.frame_end)


class OperatorSpecEditMode:
    """
    Holds one operator and its parameters.
    """

    def __init__(self, operator_name: str, operator_parameters: dict, select_mode: str, selection: set):
        """
        Constructs an OperatorSpecEditMode. Raises ValueError if selec_mode is invalid.
        :param operator_name: str - name of mesh operator from bpy.ops.mesh, e.g. "bevel" or "fill"
        :param operator_parameters: dict - {name : val} dictionary containing operator parameters.
        :param select_mode: str - mesh selection mode, must be either 'VERT', 'EDGE' or 'FACE'
        :param selection: set - set of vertices/edges/faces indices to select, e.g. [0, 9, 10].
        """
        self.operator_name = operator_name
        self.operator_parameters = operator_parameters
        if select_mode not in ['VERT', 'EDGE', 'FACE']:
            raise ValueError("select_mode must be either {}, {} or {}".format(
                'VERT', 'EDGE', 'FACE'))
        self.select_mode = select_mode
        self.selection = selection

    def __str__(self):
        return "Operator: " + self.operator_name + " with parameters: " + str(self.operator_parameters) + \
               " in selection mode: " + self.select_mode + \
            ", selecting " + str(self.selection)


class OperatorSpecObjectMode:
    """
    Holds an object operator and its parameters. Helper class for DeformModifierSpec.
    Needed to support operations in Object Mode and not Edit Mode which is supported by OperatorSpecEditMode.
    """

    def __init__(self, operator_name: str, operator_parameters: dict):
        """
        :param operator_name: str - name of the object operator from bpy.ops.object, e.g. "shade_smooth" or "shape_keys"
        :param operator_parameters: dict - contains operator parameters.
        """
        self.operator_name = operator_name
        self.operator_parameters = operator_parameters

    def __str__(self):
        return "Operator: " + self.operator_name + " with parameters: " + str(self.operator_parameters)


class DeformModifierSpec:
    """
    Holds a list of deform modifier and OperatorSpecObjectMode.
    For deform modifiers which have an object operator
    """

    def __init__(self, frame_number: int, modifier_list: list, object_operator_spec: OperatorSpecObjectMode = None):
        """
        Constructs a Deform Modifier spec (for user input)
        :param frame_number: int - the frame at which animated keyframe is inserted
        :param modifier_list: ModifierSpec - contains modifiers
        :param object_operator_spec: OperatorSpecObjectMode - contains object operators
        """
        self.frame_number = frame_number
        self.modifier_list = modifier_list
        self.object_operator_spec = object_operator_spec

    def __str__(self):
        return "Modifier: " + str(self.modifier_list) + " with object operator " + str(self.object_operator_spec)
