cmake_minimum_required(VERSION 3.25)
project(DecisionTree)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_library(DecisionTree SHARED decision_tree/libDecisionTree.cpp pyWrapp.cpp decision_tree/DecisionTree.cpp gpt.cpp pb11.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})