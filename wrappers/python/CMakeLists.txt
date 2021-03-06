check_python_and_openmm()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(SWIG REQUIRED COMPONENTS python)

set_python_module_path()

# Policies for UseSWIG
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
include(UseSWIG)
include(${PROJECT_MODULE_PATH}/Fetchpybind11.cmake)

install(DIRECTORY openmm_dlext/
    DESTINATION ${OpenMMDLExt_Python_PATH}
)

# We build just a minimal SWIG wrapper for the DLExt::Force class
# to make it compatible with the SWIG wrapper of OpenMM::Force
# The name of the module in the SWIG headers should be "_${PySWIG_MODULE_NAME}"
set(PySWIG_MODULE_NAME "dlpack_extension_swig")

set_property(SOURCE DLExtForce.i PROPERTY CPLUSPLUS ON)
swig_add_library(${PySWIG_MODULE_NAME} TYPE MODULE LANGUAGE python SOURCES DLExtForce.i)

target_include_directories(${PySWIG_MODULE_NAME} SYSTEM PRIVATE ${Python_INCLUDE_DIRS})
target_compile_features(${PySWIG_MODULE_NAME} PRIVATE cxx_std_11)
target_link_libraries(${PySWIG_MODULE_NAME} PRIVATE ${PROJECT_NAME})

if(APPLE)
    target_link_options(${PySWIG_MODULE_NAME} PRIVATE "LINKER:-undefined,dynamic_lookup")
endif()

install(TARGETS ${PySWIG_MODULE_NAME}
    DESTINATION ${OpenMMDLExt_Python_PATH}
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PySWIG_MODULE_NAME}.py"
    DESTINATION ${OpenMMDLExt_Python_PATH}
)

# We then build a Python API for the rest of the main library with pybind11
# The name of the module should match the one declared within the sources
set(pybind11_MODULE_NAME "dlpack_extension")

pybind11_add_module(${pybind11_MODULE_NAME} MODULE "")

target_sources(${pybind11_MODULE_NAME} PRIVATE PyDLExt.cpp)
target_compile_features(${pybind11_MODULE_NAME} PRIVATE cxx_std_11)
target_link_libraries(${pybind11_MODULE_NAME} PRIVATE ${PROJECT_NAME})

install(TARGETS ${pybind11_MODULE_NAME}
    DESTINATION ${OpenMMDLExt_Python_PATH}
)
