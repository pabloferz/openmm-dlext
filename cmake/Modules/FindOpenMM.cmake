#  FindOpenMM
#  ----------
#
#  OpenMM_INCLUDE_DIR   --  Path to the OpenMM headers
#  OpenMM_LIBRARY       --  Path to the main OpenMM library
#  OpenMM_CUDA_LIBRARY  --  Path to the OpenMMCUDA library (if available)
#  OpenMM_FOUND
#

include("${PROJECT_MODULE_PATH}/OpenMMTools.cmake")

if(NOT OpenMM_ROOT AND DEFINED ENV{OpenMM_ROOT})
    set(OpenMM_ROOT $ENV{OpenMM_ROOT})
endif()

if(NOT OpenMM_ROOT)
    if(DEFINED ENV{CONDA_PREFIX})
        set(OpenMM_ROOT $ENV{CONDA_PREFIX})
    elseif(DEFINED ENV{CONDA_EXE})
        get_filename_component(CONDA_BIN_DIR $ENV{CONDA_EXE} DIRECTORY)
        get_filename_component(OpenMM_ROOT ${CONDA_BIN_DIR} DIRECTORY)
    endif()
endif()

find_openmm_with_python()

find_path(OpenMM_INCLUDE_DIR
    NAMES OpenMM.h
    HINTS "${OpenMM_ROOT}/include"
)
find_library(OpenMM_LIBRARY
    NAMES OpenMM
    HINTS "${OpenMM_ROOT}/lib"
)
find_library(OpenMM_CPU_LIBRARY
    NAMES OpenMMCPU
    HINTS "${OpenMM_ROOT}/lib/plugins"
)
find_library(OpenMM_CUDA_LIBRARY
    NAMES OpenMMCUDA
    HINTS "${OpenMM_ROOT}/lib/plugins"
)
mark_as_advanced(
    OpenMM_FOUND OpenMM_INCLUDE_DIR OpenMM_LIBRARY OpenMM_CPU_LIBRARY
    OpenMM_CUDA_LIBRARY
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMM
    REQUIRED_VARS OpenMM_INCLUDE_DIR OpenMM_LIBRARY OpenMM_CPU_LIBRARY
)

if(OpenMM_FOUND AND NOT TARGET OpenMM::OpenMM)
    add_library(OpenMM::OpenMM UNKNOWN IMPORTED)
    set_target_properties(OpenMM::OpenMM PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION ${OpenMM_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${OpenMM_INCLUDE_DIR}
    )
    target_include_directories(OpenMM::OpenMM INTERFACE
        "${OpenMM_INCLUDE_DIR}/openmm"
        "${OpenMM_INCLUDE_DIR}/openmm/cpu"
        "${OpenMM_INCLUDE_DIR}/openmm/internal"
        "${OpenMM_INCLUDE_DIR}/openmm/reference"
    )
    target_link_libraries(OpenMM::OpenMM INTERFACE
        ${OpenMM_LIBRARY} ${OpenMM_CPU_LIBRARY}
    )

    if(OpenMM_CUDA_LIBRARY)
        if(${CMAKE_VERSION} VERSION_LESS 3.17)
            find_package(CUDA)
        else()
            find_package(CUDAToolkit)
        endif()

        if(CUDA_FOUND OR CUDAToolkit_FOUND)
            target_compile_definitions(OpenMM::OpenMM INTERFACE OPENMM_BUILD_CUDA_LIB)
            target_include_directories(OpenMM::OpenMM INTERFACE
                "${OpenMM_INCLUDE_DIR}/openmm/cuda"
            )
            target_link_libraries(OpenMM::OpenMM INTERFACE
                ${OpenMM_CUDA_LIBRARY}
            )
        endif()
    endif(OpenMM_CUDA_LIBRARY)
endif()
