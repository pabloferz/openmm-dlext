function(fetch_pybind11 ver)
    CPMAddPackage(NAME pybind11
        VERSION         ${ver}
        GIT_REPOSITORY  https://github.com/pybind/pybind11.git
        GIT_TAG         v${ver}
        GIT_SHALLOW     TRUE
        DOWNLOAD_ONLY   TRUE
    )
    set(pybind11_SOURCE_DIR ${pybind11_SOURCE_DIR} PARENT_SCOPE)
endfunction()

if(NOT pybind11_VERSION)
    set(pybind11_VERSION 2.6.2)
endif()

fetch_pybind11(${pybind11_VERSION})
add_subdirectory(${pybind11_SOURCE_DIR} "${PROJECT_BINARY_DIR}/extern/pybind11")
