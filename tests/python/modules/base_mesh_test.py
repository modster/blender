from abc import ABC, abstractmethod
import bpy
import os
import sys


class MeshTest(ABC):
    def __init__(self, test_object_name, exp_object_name, threshold=None):
        self.test_object_name = test_object_name
        self.exp_object_name = exp_object_name
        self.threshold = threshold
        self.update = os.getenv("BLENDER_TEST_UPDATE") is not None
        self.eval_object_name = "evaluated_object"

    def create_evaluated_object(self):
        # Real implementation.
        pass

    def run_test(self):
        # Real implementation.
        pass

    def failed_test(self):
        # Real implementation.
        pass

    def passed_test(self):
        # Real implementation
        pass

    def update_failed_test(self):
        if self.failed_test() and self.update:
            self.create_evaluated_object()
        # Real implementation
        pass

    def compare_meshes(self, evaluated_object, with_selection=True):
        objects = bpy.data.objects
        evaluated_test_mesh = objects[evaluated_object.name].data
        expected_mesh = objects[self.exp_object_name].data
        result_codes = []

        # Mesh Comparison
        if self.threshold:
            result_mesh = expected_mesh.unit_test_compare(
                mesh=evaluated_test_mesh, threshold=self.threshold)
        else:
            result_mesh = expected_mesh.unit_test_compare(
                mesh=evaluated_test_mesh)
        result_codes.append(result_mesh)

        # Selection comparison.
        if with_selection:
            selected_evaluatated_verts = [
                v.index for v in evaluated_test_mesh.vertices if v.select]
            selected_expected_verts = [
                v.index for v in expected_mesh.vertices if v.select]

            if selected_evaluatated_verts == selected_expected_verts:
                result_selection = "Same"
            else:
                result_selection = "Selection doesn't match."
        else:
            result_selection = "NA"

        result_codes.append(result_selection)

        # Validation check.
        # TODO

        return result_codes

    @abstractmethod
    def apply_operations(self):
        pass

    def __repr__(self):
        return "MeshTest({}, {} )".format(self.test_object_name, self.exp_object_name)


class SpecTest(MeshTest):
    def __init__(self, test_object_name, exp_object_name, threshold=None, operation_stack=None):
        super.__init__(test_object_name, exp_object_name, threshold=None)
        pass

    def apply_operations(self):
        pass


class BlendFileTest(MeshTest):
    def apply_operations(self):
        pass


geometry_nodes_test = BlendFileTest("test_object", "expected_object")
geometry_nodes_test.run_test()

modifier_test = SpecTest("test_array", "exp_array", operation_stack=[])
modifier_test.run_test()
