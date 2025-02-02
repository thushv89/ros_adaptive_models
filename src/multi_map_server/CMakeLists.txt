cmake_minimum_required(VERSION 2.8.3)
project(multi_map_server)

find_package(catkin REQUIRED
        COMPONENTS
            roscpp
            tf
            nav_msgs
        )

find_package(Boost REQUIRED COMPONENTS system)

find_package(PkgConfig)
pkg_check_modules(NEW_YAMLCPP yaml-cpp>=0.5)
if(NEW_YAMLCPP_FOUND)
add_definitions(-DHAVE_NEW_YAMLCPP)
endif(NEW_YAMLCPP_FOUND)

catkin_package(
    INCLUDE_DIRS
        include
    LIBRARIES
        map_server_image_loader
    CATKIN_DEPENDS
        roscpp
        tf
        nav_msgs
)

include_directories( include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} )
add_library(map_server_image_loader src/image_loader.cpp)
target_link_libraries(map_server_image_loader SDL SDL_image ${Boost_LIBRARIES})

add_executable(multi_map_server src/main.cpp)
target_link_libraries(multi_map_server
    map_server_image_loader
    yaml-cpp
    ${catkin_LIBRARIES}
)

add_executable(map_server-map_saver src/map_saver.cpp)
set_target_properties(map_server-map_saver PROPERTIES OUTPUT_NAME map_saver)
target_link_libraries(map_server-map_saver
    ${catkin_LIBRARIES}
    )

## Install executables and/or libraries
install(TARGETS map_server-map_saver multi_map_server map_server_image_loader
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

## Install project namespaced headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE)

