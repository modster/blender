# Load a blend file 
# Select the object
# Apply the GN modifier on a duplicated object
# Compare the result
# If test pass, print("SUCESS")
# If test fail, print("FAIL")
    # Update tests if BLENDER_TEST_UPDATE flag is set.
# Display result of failed tests [?]

# Code to be re-used from Mesh Test
# Depending on what all we want to use
 ## the mesh comparison code
 ## -- run-test code
# Code to be re-used from a Compositor
 ## Edit Cmake to iterate over directories.

### Questions ###
# Usage of __slots__ (only getting save memory, optimize) [Can Skip specific to Compositor]
# How to keep track of failed tests.
# Every blend file will run the test script and no memory.
# For compositor, it only tells which directory has a failed test, not the exact file name
# Pre-decide on the name of the test object and expected object ? Default name of the modifier?
# Should I make it generic for any modifier or just geometry nodes?


### RUN TEST COMMAND ###
# blender -b path_to_blend_file --python path/to/geo_node_test.py

import bpy
import os

def get_objects():
    try:
        test_object = bpy.data.objects["test_obj"]
    except:
        raise Exception("No test object found!")
    try:
        expected_object = bpy.data.objects["expected_obj"]
    except:
        raise Exception("No expected object found!")
    return [test_object, expected_object]

def apply_modifier(evaluated_object):
    """
    Apply all modifiers added to the current object [Discuss]
    """
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = evaluated_object
    
    modifiers_list = evaluated_object.modifiers

    if modifiers_list[0].type == "NODES":
        bpy.ops.object.modifier_apply(modifier=modifiers_list[0].name)
    else:
        raise Exception("Modifier not of Geometry Nodes type")
    return evaluated_object


def compare_mesh(evaluated_object, expected_object):
    evaluated_data = evaluated_object.data
    exp_data = expected_object.data
    result = evaluated_data.unit_test_compare(mesh=exp_data)
    if result == "Same":
        print("PASS")    
    else:
        failed_test(evaluated_object, expected_object, result)

def passed_test():
    pass

def failed_test(evaluated_object, expected_object, result):
    """
    [Need discussion]
    """
    update_test_flag = os.getenv('BLENDER_TEST_UPDATE') is not None
    if not update_test_flag:
        print("Test failed with {}".format(result))
        return
    
    evaluated_object.location = expected_object.location
    expected_object_name = expected_object.name
    bpy.data.objects.remove(expected_object, do_unlink=True)
    evaluated_object.name = expected_object_name

    # Save file.
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

    print("The test file was updated with new expected object")
    print("The blend file was saved.")

def duplicate_test_object(test_object):
    # Duplicate test object.
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = test_object

    test_object.select_set(True)
    bpy.ops.object.duplicate()
    evaluated_object = bpy.context.active_object
    evaluated_object.name = "evaluated_object"
    return evaluated_object

def main():
    test_object, expected_object = get_objects()
    evaluated_object = duplicate_test_object(test_object)
    evaluated_object = apply_modifier(evaluated_object)
    compare_mesh(evaluated_object, expected_object)



if __name__ == "__main__":
    main()