cmake_minimum_required(VERSION 3.10)

# Function to embed file as C++ string
function(embed_file input_file output_var)
    file(READ ${input_file} file_content)
    # Escape special characters for C++ string
    string(REPLACE "\\" "\\\\" file_content "${file_content}")
    string(REPLACE "\"" "\\\"" file_content "${file_content}")
    string(REPLACE "\n" "\\n" file_content "${file_content}")
    set(${output_var} "${file_content}" PARENT_SCOPE)
endfunction()

# Read and embed all assets
embed_file(${CMAKE_CURRENT_SOURCE_DIR}/index.html EMBEDDED_HTML)
embed_file(${CMAKE_CURRENT_SOURCE_DIR}/style.css EMBEDDED_CSS)
embed_file(${CMAKE_CURRENT_SOURCE_DIR}/app.js EMBEDDED_JS)

# Configure the header file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/embedded_assets.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/embedded_assets.h
    @ONLY
)

# Make the generated header available
include_directories(${CMAKE_CURRENT_BINARY_DIR})
