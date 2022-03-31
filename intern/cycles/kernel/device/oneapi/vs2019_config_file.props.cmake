<Project DefaultTargets="Build" ToolsVersion="16.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ImportGroup Label="PropertySheets" />
  <PropertyGroup Label="UserMacros" />
  <PropertyGroup />
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <Link>
      <SpecifyDevCmplAdditionalOptions>-Xsycl-target-backend=spir64;@CYCLES_ONEAPI_GPU_COMPILATION_OPTIONS@;%(SpecifyDevCmplAdditionalOptions)</SpecifyDevCmplAdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <Link>
      <SpecifyDevCmplAdditionalOptions>-Xsycl-target-backend=spir64;@CYCLES_ONEAPI_GPU_COMPILATION_OPTIONS@;%(SpecifyDevCmplAdditionalOptions)</SpecifyDevCmplAdditionalOptions>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup />
</Project>