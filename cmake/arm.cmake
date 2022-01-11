file(GLOB_RECURSE PPLNN_ARM_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/*.cc ${CMAKE_CURRENT_SOURCE_DIR}/src/ppl/nn/engines/arm/*.S)
list(APPEND PPLNN_SOURCES ${PPLNN_ARM_SRC})

add_subdirectory(src/ppl/nn/engines/arm/impls)
list(APPEND PPLNN_LINK_LIBRARIES PPLKernelArmServer)

set(PPLNN_USE_AARCH64 ON)
list(APPEND PPLNN_COMPILE_DEFINITIONS PPLNN_USE_ARM)

