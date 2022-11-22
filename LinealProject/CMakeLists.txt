cmake_minimum_required(VERSION 3.5)

project(LinealProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#SE crea una variable para las fuentes
set(project_sources
    main.cpp
    ClassExtraction/extractiondata.cpp
    Regresion/linearregression.cpp)

#SE crea una variable para las interfaces
set(project_headers
    ClassExtraction/extractiondata.h
    Regresion/linearregression.h)

add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers})

install(TARGETS LinealProject
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
