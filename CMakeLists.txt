cmake_minimum_required(VERSION 2.8.9)

project(CaseHub)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/pieper/CaseHub")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Steve Pieper (Isomics, Inc.)")
set(EXTENSION_DESCRIPTION "The CaseHub project is part of the Neuroimage Analysis Center (NAC), a Biomedical Technology Resource Center supported by the National Institute of Biomedical Imaging and Bioengineering (NIBIB) (P41 EB015902).")
set(EXTENSION_ICONURL "https://raw.github.com/pieper/CaseHub/master/CaseHub.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/CaseHub/Screenshots/1.png")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BenchtopNeuro)
add_subdirectory(ParameterExplorer)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
