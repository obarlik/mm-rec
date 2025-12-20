# Create a local static library to hold the embedded assets
# This keeps the generation logic self-contained and avoids cross-directory scoping issues with GENERATED properties.
add_library(dashboard_assets STATIC)
set_target_properties(dashboard_assets PROPERTIES LINKER_LANGUAGE CXX)

# Embed assets into the local library
# usage: embed_resource(target input_file)
embed_resource(dashboard_assets ${CMAKE_CURRENT_SOURCE_DIR}/index.html)
embed_resource(dashboard_assets ${CMAKE_CURRENT_SOURCE_DIR}/style.css)
embed_resource(dashboard_assets ${CMAKE_CURRENT_SOURCE_DIR}/app.js)

# Allow consumers to find embedded_assets.h
target_include_directories(dashboard_assets INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
