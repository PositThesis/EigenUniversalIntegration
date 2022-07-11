cmake_minimum_required(VERSION 3.22)

set( CMAKE_VERBOSE_MAKEFILE on )
project(EigenUniversalIntegration VERSION 0.1.0)
include(CTest)

option(TEST "Run Tests. This requires access to Eigen and Universal." OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts -std=c++2a")


find_package(universal CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# This HAS to be after the find-packages as it appears that they set the variable aswell
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Based on/taken from this article https://dominikberner.ch/cmake-interface-lib/
include(GNUInstallDirs)
add_library(EigenUniversalIntegration INTERFACE)
target_include_directories(EigenUniversalIntegration INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                                                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(EigenUniversalIntegration INTERFACE cxx_std_20)
install(TARGETS EigenUniversalIntegration
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)


install(DIRECTORY "${PROJECT_SOURCE_DIR}/EigenIntegration" DESTINATION include)

if(TEST)
    list(APPEND CORRECTNESS_TEST_DECLARES LAZY GEMM_COL GEMM_ROW GEMV_COL GEMV_ROW _DOT _DOT_T)
    foreach(td IN LISTS CORRECTNESS_TEST_DECLARES)
        add_executable(Real${td} test/test_eigen_correctness_and_usage.cxx)
        target_link_libraries(Real${td} Eigen3::Eigen OpenMP::OpenMP_CXX)
        target_include_directories(Real${td} PUBLIC "${PROJECT_BINARY_DIR}" "EigenIntegration" universal)
        target_compile_definitions(Real${td} PUBLIC ${td} REAL TEST_EIGEN_USAGE)
        target_compile_features(Real${td} INTERFACE cxx_std_20)
        add_test(NAME Real${td} COMMAND Real${td})

        add_executable(Complex${td} test/test_eigen_correctness_and_usage.cxx)
        target_link_libraries(Complex${td} Eigen3::Eigen OpenMP::OpenMP_CXX)
        target_include_directories(Complex${td} PUBLIC "${PROJECT_BINARY_DIR}" "EigenIntegration" universal)
        target_compile_definitions(Complex${td} PUBLIC ${td} COMPLEX TEST_EIGEN_USAGE USE_AVX2)
        target_compile_features(Complex${td} INTERFACE cxx_std_20)
        add_test(NAME Complex${td} COMMAND Complex${td})
    endforeach()

    add_executable(TestIO test/test_matrix_io.cxx)
    target_link_libraries(TestIO Eigen3::Eigen OpenMP::OpenMP_CXX)
    target_include_directories(TestIO PUBLIC "${PROJECT_BINARY_DIR}" "EigenIntegration")
    target_compile_features(TestIO INTERFACE cxx_std_20)
    add_test(NAME TestIO COMMAND TestIO)

    add_executable(TestPatch test/test_patched_eigen_svd.cxx)
    target_link_libraries(TestPatch Eigen3::Eigen OpenMP::OpenMP_CXX)
    target_include_directories(TestPatch PUBLIC "${PROJECT_BINARY_DIR}" "EigenIntegration")
    target_compile_features(TestPatch INTERFACE cxx_std_20)
    add_test(NAME TestPatch COMMAND TestPatch)

    add_executable(UnitTests test/test_eigen_unit.cxx)
    target_link_libraries(UnitTests Eigen3::Eigen OpenMP::OpenMP_CXX)
    target_include_directories(UnitTests PUBLIC "${PROJECT_BINARY_DIR}" "EigenIntegration" universal)
    target_compile_definitions(UnitTests PUBLIC COMPLEX TEST_EIGEN_USAGE)
    target_compile_features(UnitTests INTERFACE cxx_std_20)
    add_test(NAME UnitTests COMMAND UnitTests)
endif(TEST)