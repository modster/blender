from abc import ABC, abstractmethod
import bpy
import os
import sys

from modules.spec_classes import ModifierSpec, DeformModifierSpec, OperatorSpecEditMode, OperatorSpecObjectMode


class MeshTest(ABC):
    def __init__(self, test_object_name, exp_object_name, threshold=None):
        self.test_object_name = test_object_name
        self.exp_object_name = exp_object_name
        self.threshold = threshold
        self.update = os.getenv("BLENDER_TEST_UPDATE") is not None
        # self.eval_object_name = "evaluated_object"

        # Private flag to indicate whether the blend file was updated after the test.
        self._test_updated = False
        objects = bpy.data.objects
        self.test_object = objects[self.test_object_name]

        # TODO - create exp object, in case doesn't exist.
        self.expected_object = objects[self.exp_object_name]

    def create_evaluated_object(self):
        bpy.context.view_layer.objects.active = self.test_object

        # Duplicate test object.
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = self.test_object

        self.test_object.select_set(True)
        bpy.ops.object.duplicate()
        evaluated_test_object = bpy.context.active_object
        evaluated_test_object.name = "evaluated_object"
        # Real implementation.
        return evaluated_test_object

    @staticmethod
    def _print_result(result):
        comparison_result, selection_result, validation_result = result
        print("Mesh Compariosn: {}".format(comparison_result))
        print("Selection Result: {}".format(selection_result))
        print("Mesh Validation: {}".format(validation_result))

    def run_test(self):
        evaluated_test_object = self.create_evaluated_object()
        self.apply_operations(evaluated_test_object)
        result = self.compare_meshes(evaluated_test_object)

        comparison_result, selection_result, validation_result = result

        if comparison_result == "Same" and selection_result == "Same" and validation_result == "Valid":
            self.passed_test(result)

        elif self.update:
            self.failed_test(result)
            self.update_failed_test(evaluated_test_object)

        else:
            self.failed_test(result)

        # Real implementation.
        pass

    def failed_test(self, result):
        print("The test failed with the following: ")
        self._print_result(result)
        # Real implementation.
        pass

    def passed_test(self, result):
        print("The tests passed successfully.")
        self._print_result(result)
        # Real implementation
        pass

    def do_selection(self, mesh: bpy.types.Mesh, select_mode: str, selection: set):
        """
        Do selection on a mesh
        :param mesh: bpy.types.Mesh - input mesh
        :param: select_mode: str - selection mode. Must be 'VERT', 'EDGE' or 'FACE'
        :param: selection: set - indices of selection.

        Example: select_mode='VERT' and selection={1,2,3} selects veritces 1, 2 and 3 of input mesh
        """
        # deselect all
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        bpy.context.tool_settings.mesh_select_mode = (select_mode == 'VERT',
                                                      select_mode == 'EDGE',
                                                      select_mode == 'FACE')

        items = (mesh.vertices if select_mode == 'VERT'
                 else mesh.edges if select_mode == 'EDGE'
                 else mesh.polygons if select_mode == 'FACE'
                 else None)
        if items is None:
            raise ValueError("Invalid selection mode")
        for index in selection:
            items[index].select = True

    def update_failed_test(self, evaluated_test_object):
        # Update expected object.
        evaluated_test_object.location = self.expected_object.location
        expected_object_name = self.expected_object.name
        evaluated_selection = {
            v.index for v in evaluated_test_object.data.vertices if v.select}

        bpy.data.objects.remove(self.expected_object, do_unlink=True)
        evaluated_test_object.name = expected_object_name
        self.do_selection(evaluated_test_object.data,
                          "VERT", evaluated_selection)

        # Save file.
        bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
        self._test_updated = True
        self.expected_object = evaluated_test_object

        # Real implementation.

    def compare_meshes(self, evaluated_object):
        objects = bpy.data.objects
        evaluated_test_mesh = objects[evaluated_object.name].data
        expected_mesh = objects[self.exp_object_name].data
        result_codes = []

        # Mesh Comparison.
        if self.threshold:
            result_mesh = expected_mesh.unit_test_compare(
                mesh=evaluated_test_mesh, threshold=self.threshold)
        else:
            result_mesh = expected_mesh.unit_test_compare(
                mesh=evaluated_test_mesh)
        result_codes.append(result_mesh)

        # Selection comparison.

        selected_evaluatated_verts = [
            v.index for v in evaluated_test_mesh.vertices if v.select]
        selected_expected_verts = [
            v.index for v in expected_mesh.vertices if v.select]

        if selected_evaluatated_verts == selected_expected_verts:
            result_selection = "Same"
        else:
            result_selection = "Selection doesn't match."
        result_codes.append(result_selection)

        # Validation check.
        result_validation = evaluated_test_mesh.validate(verbose=True)
        if result_validation:
            result_validation = "Invalid Mesh"
        else:
            result_validation = "Valid"
        result_codes.append(result_validation)

        return result_codes

    @abstractmethod
    def apply_operations(self, evaluated_test_object):
        pass

    def __repr__(self):
        return "MeshTest({}, {} )".format(self.test_object_name, self.exp_object_name)


class SpecMeshTest(MeshTest):
    def __init__(self, test_name, test_object_name, exp_object_name, operation_stack=None, threshold=None):
        self.test_name = test_name
        super().__init__(test_object_name, exp_object_name, threshold)
        pass

    def apply_operations(self):
        pass


class BlendFileTest(MeshTest):
    def apply_operations(self, evaluated_test_object):
        """
        Apply all modifiers (Geometry Nodes for now) added to the current object [Discuss]
        """
        # select_only_object(evaluated_test_object)

        modifiers_list = evaluated_test_object.modifiers

        for modifier in modifiers_list:
            bpy.ops.object.modifier_apply(modifier=modifier.name)


geometry_nodes_test = BlendFileTest("test_object", "expected_object")
geometry_nodes_test.run_test()

modifier_test = SpecMeshTest("test_name", "test_array",
                             "exp_array", operation_stack=[])
modifier_test.run_test()
