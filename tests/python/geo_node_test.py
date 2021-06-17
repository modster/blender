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


# Script for running tests pre-loaded from a blend file.
#
# General Idea
#
# Load a blend file.
# Select the object.
# Apply the GN modifier on a duplicated object.
# Compare the result.
# If test pass, print("SUCCESS").
# If test fail, print("FAIL").
#    Update tests if BLENDER_TEST_UPDATE flag is set.
# Display result of failed tests.
#
# Set the BLENDER_TEST_UPDATE env variable to update the expected object.
#
### RUN TEST COMMAND ###
# blender -b path_to_blend_file --python path/to/geo_node_test.py

import bpy
import os
import sys


def get_test_object():
    """
    Get test object or raise Exception.
    """
    try:
        test_object = bpy.data.objects["test_object"]
        return test_object
    except KeyError:
        raise Exception("No test object found!")


def get_expected_object():
    """
    Get expected object or raise Exception.
    """
    try:
        expected_object = bpy.data.objects["expected_object"]
        return expected_object
    except KeyError:
        raise Exception("No expected object found!")


def get_evaluated_object(test_object):
    evaluated_object = duplicate_test_object(test_object)
    apply_modifier(evaluated_object)
    return evaluated_object


def select_only_object(any_object):
    """
    Select the given object and make it active object.
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = any_object
    any_object.select_set(True)


def create_expected_object(test_object):
    """
    Create expected object when run with BLENDER_TEST_UPDATE
    """
    try:
        expected_object = bpy.data.objects["expected_object"]

    except KeyError:
        print("Creating an expected object...")
        evaluated_object = get_evaluated_object(test_object)

        # Setting evaluted_object to expected_object to create
        # a new expected_object.
        expected_object = evaluated_object
        expected_object.name = "expected_object"

        # Be careful with tests dealing with global coordinates.
        expected_object.location = (0, 10, 0)

        # Save file with the expected object.
        bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)


def apply_modifier(evaluated_object):
    """
    Apply all modifiers (Geometry Nodes for now) added to the current object [Discuss]
    """
    select_only_object(evaluated_object)

    modifiers_list = evaluated_object.modifiers

    if modifiers_list[0].type == "NODES":
        bpy.ops.object.modifier_apply(modifier=modifiers_list[0].name)
    else:
        raise Exception("Modifier not of Geometry Nodes type")


def compare_meshes(evaluated_object, expected_object):
    """
    Compares the meshes of evaluated and expected objects.
    """
    evaluated_data = evaluated_object.data
    exp_data = expected_object.data
    result = evaluated_data.unit_test_compare(mesh=exp_data)
    if result == "Same":
        print("\nPASSED")
    else:
        failed_test(evaluated_object, expected_object, result)


def failed_test(evaluated_object, expected_object, result):
    """
    Prints the failed test.
    Updates the expected object on failure if BLENDER_TEST_UPDATE
    environment variable is set.
    """
    print("\nFAILED with {}".format(result))
    update_test_flag = os.getenv('BLENDER_TEST_UPDATE') is not None
    if not update_test_flag:
        sys.exit(1)

    update_expected_object(evaluated_object, expected_object)
    print("Re-running the test...")
    run_test()


def update_expected_object(evaluated_object, expected_object):
    print("Updating the test...")
    evaluated_object.location = expected_object.location
    expected_object_name = expected_object.name
    bpy.data.objects.remove(expected_object, do_unlink=True)
    evaluated_object.name = expected_object_name

    # Save file.
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

    print("The test file was updated with new expected object")
    print("The blend file {} was updated.".format(
        bpy.path.display_name_from_filepath(bpy.data.filepath)))


def duplicate_test_object(test_object):
    """
    Duplicate test object.
    """
    select_only_object(test_object)
    bpy.ops.object.duplicate()
    evaluated_object = bpy.context.active_object
    evaluated_object.name = "evaluated_object"
    return evaluated_object


def run_test():
    test_object = get_test_object()
    expected_object = get_expected_object()
    evaluated_object = get_evaluated_object(test_object)
    compare_meshes(evaluated_object, expected_object)


def main():
    """
    Main function controlling the workflow and running the tests.
    """
    test_object = get_test_object()
    update_test_flag = os.getenv('BLENDER_TEST_UPDATE') is not None
    if update_test_flag:
        create_expected_object(test_object)
    run_test()


if __name__ == "__main__":
    main()
