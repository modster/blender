from abc import ABC, ABCMeta, abstractmethod


class MeshTest(ABC):
    def __init__(self, test_object_name, exp_object_name, threshold=None):
        self.test_object_name = test_object_name
        self.exp_object_name = exp_object_name
        self.threshold = threshold
        self.update = ENV_VAR

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
        if self.failed_test() == True and self.update:
            self.create_evaluated_object()
        # Real implementation
        pass

    def compare_mesh(self, with_selection=True):
        # Real implementation.
        pass

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


def compare_meshes():
    # do_some_comparison
    return result


def compare_meshes():
    if pass:
        return True
    elif fail:
        return False


###
"""
Workflow:
1. compare_meshes
"""
