set(NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR "" CACHE PATH
  "Path to NEP test data containing nep.txt and train.xyz for cpu_nep3 tests")

if(NOT NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR)
  foreach(_candidate IN ITEMS
      "${PROJECT_SOURCE_DIR}/tests/fixtures/cpu_nep3_baseline"
      "${PROJECT_SOURCE_DIR}/../NepTrainKit/tests/data/nep")
    if(EXISTS "${_candidate}/nep.txt" AND EXISTS "${_candidate}/train.xyz")
      set(NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR "${_candidate}" CACHE PATH
        "Path to NEP test data containing nep.txt and train.xyz for cpu_nep3 tests" FORCE)
      break()
    endif()
  endforeach()
endif()

set(NEP_ADAPTERS_NEP89_MODEL_PATH "" CACHE FILEPATH
  "Path to a nep89 model file for large-model tests and benchmarks")
set(NEP_ADAPTERS_NEP89_XYZ_PATH "" CACHE FILEPATH
  "Path to an extxyz file compatible with the nep89 model")

if(NOT NEP_ADAPTERS_NEP89_MODEL_PATH)
  foreach(_candidate IN ITEMS
      "${PROJECT_SOURCE_DIR}/../NepTrainKit/src/NepTrainKit/Config/nep89.txt")
    if(EXISTS "${_candidate}")
      set(NEP_ADAPTERS_NEP89_MODEL_PATH "${_candidate}" CACHE FILEPATH
        "Path to a nep89 model file for large-model tests and benchmarks" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT NEP_ADAPTERS_NEP89_XYZ_PATH)
  foreach(_candidate IN ITEMS
      "${PROJECT_SOURCE_DIR}/../NepTrainKit/tests/data/nep/train.xyz")
    if(EXISTS "${_candidate}")
      set(NEP_ADAPTERS_NEP89_XYZ_PATH "${_candidate}" CACHE FILEPATH
        "Path to an extxyz file compatible with the nep89 model" FORCE)
      break()
    endif()
  endforeach()
endif()
