cmake_minimum_required(VERSION 3.22)

project(eigen_universal_integration)

option(TEST "Run Tests. This requires access to Eigen and Universal." OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(universal CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)


add_library(EigenUniversalIntegration INTERFACE)
target_include_directories(EigenUniversalIntegration INTERFACE "./include")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/EigenIntegration" DESTINATION include)

if(TEST)
    list(APPEND CORRECTNESS_TEST_DECLARES LAZY GEMM_COL GEMM_ROW GEMV_COL GEMV_ROW _DOT _DOT_T)
    # list(APPEND USE_Type_Flags USE_Float USE_Double USE_LongDouble USE_Posit162 USE_Posit322 USE_Posit644)

    foreach(td IN LISTS CORRECTNESS_TEST_DECLARES)
        add_executable(Real${td} test/test_eigen_correctness_and_usage.cxx)
        target_link_libraries(Real${td} Eigen3::Eigen OpenMP::OpenMP_CXX)
        target_include_directories(Real${td} PUBLIC "${PROJECT_BINARY_DIR}" "EigenIntegration" universal)
        target_compile_definitions(Real${td} PUBLIC ${td} REAL TEST_EIGEN_USAGE)
        add_test(NAME Real${td} COMMAND Real${td})

        add_executable(Complex${td} test/test_eigen_correctness_and_usage.cxx)
        target_link_libraries(Complex${td} Eigen3::Eigen OpenMP::OpenMP_CXX)
        target_include_directories(Complex${td} PUBLIC "${PROJECT_BINARY_DIR}" "EigenIntegration" universal)
        target_compile_definitions(Complex${td} PUBLIC ${td} COMPLEX TEST_EIGEN_USAGE)
        add_test(NAME Complex${td} COMMAND Complex${td})
    endforeach()
endif(TEST)