project(lab01 C)

add_executable(lab01 lab01/main.c)

set( CMAKE_C_FLAGS "-O3 -g0" )

target_link_libraries(lab01 m)