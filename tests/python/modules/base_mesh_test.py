from abc import ABC, abstractmethod
import bpy
import os
import sys

from modules.spec_classes import ModifierSpec, DeformModifierSpec, OperatorSpecEditMode, OperatorSpecObjectMode, ParticleSystemSpec


class MeshTest(ABC):
    def __init__(self, test_object_name, exp_object_name, test_name=None, threshold=None):
        self.test_object_name = test_object_name
        self.exp_object_name = exp_object_name
        self.test_name = test_name
        self.threshold = threshold
        self.update = os.getenv("BLENDER_TEST_UPDATE") is not None
        self.verbose = os.getenv("BLENDER_VERBOSE") is not None
        # self.eval_object_name = "evaluated_object"


        self.test_updated_counter = 0
        objects = bpy.data.objects
        self.test_object = objects[self.test_object_name]
        # self.expected_object = objects[self.exp_object_name]

        if self.update:
            if objects.find(exp_object_name) > -1:
                self.expected_object = objects[self.exp_object_name]
            else:
                self.create_expected_object()
        else:
            self.expected_object = objects[self.exp_object_name]

    
    def create_expected_object(self):
        print("Creating expected object...")
        evaluated_object = self.create_evaluated_object()
        self.expected_object = evaluated_object
        self.expected_object.name = self.exp_object_name
        x, y, z = self.test_object.location
        self.expected_object.location = (x, y+10, z)
        bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

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
        print("Mesh Comparison: {}".format(comparison_result))
        print("Selection Result: {}".format(selection_result))
        print("Mesh Validation: {}".format(validation_result))
        print()

    def run_test(self):
        # self.test_updated_counter = 0
        evaluated_test_object = self.create_evaluated_object()
        self.apply_operations(evaluated_test_object)
        result = self.compare_meshes(evaluated_test_object)

        comparison_result, selection_result, validation_result = result

        if comparison_result == "Same" and selection_result == "Same" and validation_result == "Valid":
            self.passed_test(result)
            # Clean up.
            if self.verbose:
                print("Cleaning up...")
            # Delete evaluated_test_object.
            bpy.ops.object.delete()
            # bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
            
            return True

        elif self.update:
            self.failed_test(result)
            self.update_failed_test(evaluated_test_object)
            if self.test_updated_counter == 1:
                self.run_test()
            else:
                print("The test fails consistently. Exiting...")

        else:
            self.failed_test(result)


    def failed_test(self, result):
        print("\nFAILED {} test with the following: ".format(self.test_name))
        self._print_result(result)
        # Real implementation.
        pass

    def passed_test(self, result):
        print("\nPASSED {} test successfully.".format(self.test_name))
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
        self.test_updated_counter += 1
        self.expected_object = evaluated_test_object

        # Real implementation.

    def compare_meshes(self, evaluated_object):
        objects = bpy.data.objects
        evaluated_test_mesh = objects[evaluated_object.name].data
        expected_mesh = self.expected_object.data
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

        selected_evaluated_verts = [
            v.index for v in evaluated_test_mesh.vertices if v.select]
        selected_expected_verts = [
            v.index for v in expected_mesh.vertices if v.select]

        if selected_evaluated_verts == selected_expected_verts:
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
    def __init__(self, test_name, test_object_name, exp_object_name, operations_stack=None, apply_modifier=True,
        do_compare=True, threshold=None):
        
        super().__init__(test_object_name, exp_object_name, test_name, threshold)
        self.test_name = test_name
        if operations_stack is None:
            self.operations_stack = []
        else:
            self.operations_stack = operations_stack 
        self.apply_modifier = apply_modifier
        self.do_compare = do_compare

    def apply_operations(self, evaluated_test_object):
        # Add modifiers and operators.
        print("Applying operations...")
        for operation in self.operations_stack:
            if isinstance(operation, ModifierSpec):
                self._add_modifier(evaluated_test_object, operation)
                if self.apply_modifier:
                    self._apply_modifier(
                        evaluated_test_object, operation.modifier_name)

            elif isinstance(operation, OperatorSpecEditMode):
                self._apply_operator_edit_mode(
                    evaluated_test_object, operation)

            elif isinstance(operation, OperatorSpecObjectMode):
                self._apply_operator_object_mode(operation)

            elif isinstance(operation, DeformModifierSpec):
                self._apply_deform_modifier(evaluated_test_object, operation)

            elif isinstance(operation, ParticleSystemSpec):
                self._apply_particle_system(evaluated_test_object, operation)

            else:
                raise ValueError("Expected operation of type {} or {} or {} or {}. Got {}".
                                 format(type(ModifierSpec), type(OperatorSpecEditMode),
                                        type(OperatorSpecObjectMode), type(ParticleSystemSpec), type(operation)))

    def _set_parameters_impl(self, modifier, modifier_parameters, nested_settings_path, modifier_name):
        """
        Doing a depth first traversal of the modifier parameters and setting their values.
        :param: modifier: Of type modifier, its altered to become a setting in recursion.
        :param: modifier_parameters : dict or sequence, a simple/nested dictionary of modifier parameters.
        :param: nested_settings_path : list(stack): helps in tracing path to each node.
        """
        if not isinstance(modifier_parameters, dict):
            param_setting = None
            for i, setting in enumerate(nested_settings_path):

                # We want to set the attribute only when we have reached the last setting.
                # Applying of intermediate settings is meaningless.
                if i == len(nested_settings_path) - 1:
                    setattr(modifier, setting, modifier_parameters)

                elif hasattr(modifier, setting):
                    param_setting = getattr(modifier, setting)
                    # getattr doesn't accept canvas_surfaces["Surface"], but we need to pass it to setattr.
                    if setting == "canvas_surfaces":
                        modifier = param_setting.active
                    else:
                        modifier = param_setting
                else:
                    # Clean up first
                    bpy.ops.object.delete()
                    raise Exception("Modifier '{}' has no parameter named '{}'".
                                    format(modifier_name, setting))

            # It pops the current node before moving on to its sibling.
            nested_settings_path.pop()
            return

        for key in modifier_parameters:
            nested_settings_path.append(key)
            self._set_parameters_impl(
                modifier, modifier_parameters[key], nested_settings_path, modifier_name)

        if nested_settings_path:
            nested_settings_path.pop()

    def set_parameters(self, modifier, modifier_parameters):
        """
        Wrapper for _set_parameters_util
        """
        settings = []
        modifier_name = modifier.name
        self._set_parameters_impl(
            modifier, modifier_parameters, settings, modifier_name)

    def _add_modifier(self, test_object, modifier_spec: ModifierSpec):
        """
        Add modifier to object.
        :param test_object: bpy.types.Object - Blender object to apply modifier on.
        :param modifier_spec: ModifierSpec - ModifierSpec object with parameters
        """
        bakers_list = ['CLOTH', 'SOFT_BODY', 'DYNAMIC_PAINT', 'FLUID']
        scene = bpy.context.scene
        scene.frame_set(1)
        modifier = test_object.modifiers.new(modifier_spec.modifier_name,
                                             modifier_spec.modifier_type)

        if modifier is None:
            raise Exception(
                "This modifier type is already added on the Test Object, please remove it and try again.")

        if self.verbose:
            print("Created modifier '{}' of type '{}'.".
                  format(modifier_spec.modifier_name, modifier_spec.modifier_type))

        # Special case for Dynamic Paint, need to toggle Canvas on.
        if modifier.type == "DYNAMIC_PAINT":
            bpy.ops.dpaint.type_toggle(type='CANVAS')

        self.set_parameters(modifier, modifier_spec.modifier_parameters)

        if modifier.type in bakers_list:
            self._bake_current_simulation(
                test_object, modifier.name, modifier_spec.frame_end)

        scene.frame_set(modifier_spec.frame_end)

    def _apply_modifier(self, test_object, modifier_name):
        # Modifier automatically gets applied when converting from Curve to Mesh.
        if test_object.type == 'CURVE':
            bpy.ops.object.convert(target='MESH')
        elif test_object.type == 'MESH':
            bpy.ops.object.modifier_apply(modifier=modifier_name)
        else:
            raise Exception("This object type is not yet supported!")

    def _bake_current_simulation(self, test_object, test_modifier_name, frame_end):
        """
        FLUID: Bakes the simulation
        SOFT BODY, CLOTH, DYNAMIC PAINT: Overrides the point_cache context and then bakes.
        """

        for scene in bpy.data.scenes:
            for modifier in test_object.modifiers:
                if modifier.type == 'FLUID':
                    bpy.ops.fluid.bake_all()
                    break

                elif modifier.type == 'CLOTH' or modifier.type == 'SOFT_BODY':
                    test_object.modifiers[test_modifier_name].point_cache.frame_end = frame_end
                    override_setting = modifier.point_cache
                    override = {
                        'scene': scene, 'active_object': test_object, 'point_cache': override_setting}
                    bpy.ops.ptcache.bake(override, bake=True)
                    break

                elif modifier.type == 'DYNAMIC_PAINT':
                    dynamic_paint_setting = modifier.canvas_settings.canvas_surfaces.active
                    override_setting = dynamic_paint_setting.point_cache
                    override = {
                        'scene': scene, 'active_object': test_object, 'point_cache': override_setting}
                    bpy.ops.ptcache.bake(override, bake=True)
                    break

    def _apply_particle_system(self, test_object, particle_sys_spec: ParticleSystemSpec):
        """
        Applies Particle System settings to test objects
        """
        bpy.context.scene.frame_set(1)
        bpy.ops.object.select_all(action='DESELECT')

        test_object.modifiers.new(
            particle_sys_spec.modifier_name, particle_sys_spec.modifier_type)

        settings_name = test_object.particle_systems.active.settings.name
        particle_setting = bpy.data.particles[settings_name]
        if self.verbose:
            print("Created modifier '{}' of type '{}'.".
                  format(particle_sys_spec.modifier_name, particle_sys_spec.modifier_type))

        for param_name in particle_sys_spec.modifier_parameters:
            try:
                if param_name == "seed":
                    system_setting = test_object.particle_systems[particle_sys_spec.modifier_name]
                    setattr(system_setting, param_name,
                            particle_sys_spec.modifier_parameters[param_name])
                else:
                    setattr(particle_setting, param_name,
                            particle_sys_spec.modifier_parameters[param_name])

                if self.verbose:
                    print("\t set parameter '{}' with value '{}'".
                          format(param_name, particle_sys_spec.modifier_parameters[param_name]))
            except AttributeError:
                # Clean up first
                bpy.ops.object.delete()
                raise AttributeError("Modifier '{}' has no parameter named '{}'".
                                     format(particle_sys_spec.modifier_type, param_name))

        bpy.context.scene.frame_set(particle_sys_spec.frame_end)
        test_object.select_set(True)
        bpy.ops.object.duplicates_make_real()
        test_object.select_set(True)
        bpy.ops.object.join()
        if self.apply_modifier:
            self._apply_modifier(test_object, particle_sys_spec.modifier_name)

    def _do_selection(self, mesh: bpy.types.Mesh, select_mode: str, selection: set):
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

    def _apply_operator_edit_mode(self, test_object, operator: OperatorSpecEditMode):
        """
        Apply operator on test object.
        :param test_object: bpy.types.Object - Blender object to apply operator on.
        :param operator: OperatorSpecEditMode - OperatorSpecEditMode object with parameters.
        """
        self._do_selection(
            test_object.data, operator.select_mode, operator.selection)

        # Apply operator in edit mode.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type=operator.select_mode)
        mesh_operator = getattr(bpy.ops.mesh, operator.operator_name)

        try:
            retval = mesh_operator(**operator.operator_parameters)
        except AttributeError:
            raise AttributeError(
                "bpy.ops.mesh has no attribute {}".format(operator.operator_name))
        except TypeError as ex:
            raise TypeError("Incorrect operator parameters {!r} raised {!r}".format(
                operator.operator_parameters, ex))

        if retval != {'FINISHED'}:
            raise RuntimeError(
                "Unexpected operator return value: {}".format(retval))
        if self.verbose:
            print("Applied {}".format(operator))

        bpy.ops.object.mode_set(mode='OBJECT')

    def _apply_operator_object_mode(self, operator: OperatorSpecObjectMode):
        """
        Applies the object operator.
        """
        bpy.ops.object.mode_set(mode='OBJECT')
        object_operator = getattr(bpy.ops.object, operator.operator_name)

        try:
            retval = object_operator(**operator.operator_parameters)
        except AttributeError:
            raise AttributeError(
                "bpy.ops.object has no attribute {}".format(operator.operator_name))
        except TypeError as ex:
            raise TypeError("Incorrect operator parameters {!r} raised {!r}".format(
                operator.operator_parameters, ex))

        if retval != {'FINISHED'}:
            raise RuntimeError(
                "Unexpected operator return value: {}".format(retval))
        if self.verbose:
            print("Applied operator {}".format(operator))

    def _apply_deform_modifier(self, test_object, operation: list):
        """
        param: operation: list: List of modifiers or combination of modifier and object operator.
        """

        scene = bpy.context.scene
        scene.frame_set(1)
        bpy.ops.object.mode_set(mode='OBJECT')
        modifier_operations_list = operation.modifier_list
        modifier_names = []
        object_operations = operation.object_operator_spec
        for modifier_operations in modifier_operations_list:
            if isinstance(modifier_operations, ModifierSpec):
                self._add_modifier(test_object, modifier_operations)
                modifier_names.append(modifier_operations.modifier_name)

        if isinstance(object_operations, OperatorSpecObjectMode):
            self._apply_operator_object_mode(object_operations)

        scene.frame_set(operation.frame_number)

        if self.apply_modifier:
            for mod_name in modifier_names:
                self._apply_modifier(test_object, mod_name)


class BlendFileTest(MeshTest):
    def apply_operations(self, evaluated_test_object):
        """
        Apply all modifiers (Geometry Nodes for now) added to the current object [Discuss]
        """

        modifiers_list = evaluated_test_object.modifiers

        for modifier in modifiers_list:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
