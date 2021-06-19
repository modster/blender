# Instructions for converting nodes from strings to attribute sockets

1. Change `SOCK_STRING` sockets to `SOCK_ATTRIBUTE`

    (Except where strings are not actually attribute names)

1. Move output attributes over to the output socket template list.

1. Add default value and limits to the attribute socket template as well as the single value socket.
    This is needed because the attribute socket also has default values for basic data types.
    Output sockets don't need default value and limits.

    ```c
    {SOCK_ATTRIBUTE, N_("A"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    {SOCK_VECTOR, N_("A"), 0.0f, 0.0f, 0.0f, 0.0f, -FLT_MAX, FLT_MAX},
    ```

    The single-value socket may not be needed in future because of broadcasting, but for now we keep it.

1. Update attribute socket data types where types are known.

    Use the `set_attribute_socket_data_type` function to update the type, like so:

    ```cpp
    blender::nodes::set_attribute_socket_data_type(*node, "A", SOCK_VECTOR);
    ```

    - If the attribute data type is fixed, change it in the _init_ callback.
    - If the attribute data type depends on runtime options (e.g. math operation), change it in the _update_ callback.

1. In the _execute_ callback, create an `AttributeRef` for each output attribute. The output value must be set __exactly__ once, so watch out for early-exits and repeated function calls.

    _TODO output attribute names will be auto-generated_

    ```cpp
    AttributeRef result("DummyName this will be auto generated", CD_PROP_FLOAT);

    do_the_actual_work(component, params, result);

    params.set_output("Result", result);
    ```

1. Remove old lookups for attribute name strings and use the result `AttributeRef` instead.

    Old:

    ```cpp
    static void do_the_actual_work(GeometryComponent &component,
                                   const GeoNodeExecParams &params)
    {
        const std::string result_name = params.get_input<std::string>("Result");

        [...]
    }
    ```

    New:

    ```cpp
    static void do_the_actual_work(GeometryComponent &component,
                                   const GeoNodeExecParams &params,
                                   const AttributeRef &result)
    {
        [...]
    }
    ```

1. Many nodes have a `get_result_domain` function to determine the domain for output attributes. This currently uses the output attribute name to re-use existing attributes. Attribute outputs are now always unique, so this name lookup is deprecated and should be removed. Result domains should always be determined from inputs only.

    Old:

    ```cpp
    static AttributeDomain get_result_domain(const GeometryComponent &component,
                                             StringRef input_name,
                                             StringRef result_name)
    {
        /* Use the domain of the result attribute if it already exists. */
        ReadAttributeLookup result_attribute = component.attribute_try_get_for_read(result_name);
        if (result_attribute) {
            return result_attribute.domain;
        }

        /* Otherwise use the input attribute's domain. */
        [...]
    }
    ```

    New:

    ```cpp
    static AttributeDomain get_result_domain(const GeometryComponent &component,
                                             AttributeRef input_attribute)
    {
        /* Use the input attribute's domain. */
        [...]
    }
    ```

1. Use the generated output attribute name to create actual attributes and data arrays:

    ```cpp
    static void do_the_actual_work(GeometryComponent &component,
                                    const GeoNodeExecParams &params,
                                    const AttributeRef &result)
    {
        [...]
        OutputAttribute_Typed<float> attribute_result =
            component.attribute_try_get_for_output_only<float>(result.name(), result_domain);

        [...]
    }
    ```