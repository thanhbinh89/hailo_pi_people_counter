cmake_minimum_required(VERSION 3.16)
project(people_counter LANGUAGES C CXX)

# ============================================================
# Settings
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HAILO_USE_SYSTEM_DEPS "Use system-installed yaml-cpp/libcurl instead of submodules" OFF)

# ============================================================
# Core dependencies
# ============================================================
find_package(Threads REQUIRED)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV includes: ${OpenCV_INCLUDE_DIRS}")

# ============================================================
# Third-party directory
# ============================================================
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external")

# ============================================================
# yaml-cpp
# ============================================================
if(HAILO_USE_SYSTEM_DEPS)
    find_package(yaml-cpp REQUIRED)
else()
    if(NOT EXISTS "${THIRD_PARTY_DIR}/yaml-cpp/CMakeLists.txt")
        message(FATAL_ERROR
            "Missing submodule: yaml-cpp\n"
            "Run: git submodule update --init --recursive")
    endif()
    set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${THIRD_PARTY_DIR}/yaml-cpp"
        "${CMAKE_BINARY_DIR}/_deps/yaml-cpp")
endif()

if(NOT TARGET yaml-cpp)
    message(FATAL_ERROR "yaml-cpp target was not created.")
endif()

# ============================================================
# libcurl (required by resources_manager.cpp in common)
# ============================================================
if(HAILO_USE_SYSTEM_DEPS)
    find_package(CURL REQUIRED)
else()
    if(NOT EXISTS "${THIRD_PARTY_DIR}/curl/CMakeLists.txt")
        message(FATAL_ERROR
            "Missing submodule: curl\n"
            "Run: git submodule update --init --recursive")
    endif()
    set(BUILD_CURL_EXE       OFF CACHE BOOL "" FORCE)
    set(CURL_DISABLE_TESTS   ON  CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
    set(CURL_USE_LIBPSL      OFF CACHE BOOL "" FORCE)
    set(CURL_BROTLI          OFF CACHE BOOL "" FORCE)
    set(CURL_ZSTD            OFF CACHE BOOL "" FORCE)
    set(USE_NGHTTP2          OFF CACHE BOOL "" FORCE)
    set(CURL_DISABLE_LDAP    ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_LDAPS   ON  CACHE BOOL "" FORCE)
    if(WIN32)
        set(CURL_USE_SCHANNEL ON  CACHE BOOL "" FORCE)
        set(CURL_USE_OPENSSL  OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(
        "${THIRD_PARTY_DIR}/curl"
        "${CMAKE_BINARY_DIR}/_deps/curl")
    if(TARGET libcurl AND NOT TARGET CURL::libcurl)
        add_library(CURL::libcurl ALIAS libcurl)
    endif()
endif()

if(NOT TARGET CURL::libcurl)
    message(FATAL_ERROR "libcurl target missing (CURL::libcurl).")
endif()

# ============================================================
# Sources
# ============================================================
set(APP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/people_counter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/zone_tracker.cpp"
)
list(APPEND APP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/toolbox.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/hailo_infer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/resources_manager.cpp"
)

add_executable(${PROJECT_NAME} ${APP_SOURCES})

# ============================================================
# Copy config.yaml next to the built executable
# ============================================================
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/config.yaml"
    COMMENT "Copying config.yaml next to executable")

# Copy repo-level resources_config.yaml (needed by resources_manager)
set(HAILO_APPS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(RESOURCES_YAML_SRC "${HAILO_APPS_ROOT}/config/resources_config.yaml")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/config"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RESOURCES_YAML_SRC}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/config/resources_config.yaml"
    COMMENT "Copying resources_config.yaml next to executable")

# ============================================================
# Include directories
# ============================================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common"
)

# ============================================================
# Compile options
# ============================================================
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -O3
        -fconcepts
        -Wno-ignored-qualifiers
        -Wno-extra
        -Wno-stringop-truncation
        -Wno-reorder
    )
endif()

# ============================================================
# Link libraries
# ============================================================
target_link_libraries(${PROJECT_NAME} PRIVATE
    HailoRT::libhailort
    Threads::Threads
    ${OpenCV_LIBS}
    yaml-cpp
    CURL::libcurl
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
    endif()
endif()
