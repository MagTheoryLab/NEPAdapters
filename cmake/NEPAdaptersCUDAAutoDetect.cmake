function(nep_adapters_resolve_cuda_default output_variable)
  if(DEFINED NEP_ADAPTERS_ENABLE_CUDA)
    set(${output_variable} "${NEP_ADAPTERS_ENABLE_CUDA}" PARENT_SCOPE)
    return()
  endif()

  set(_nep_cuda_mode AUTO)
  if(DEFINED ENV{NEP_CUDA})
    string(TOUPPER "$ENV{NEP_CUDA}" _nep_cuda_env)
    if(_nep_cuda_env MATCHES "^(1|ON|TRUE|YES)$")
      set(_nep_cuda_mode ON)
    elseif(_nep_cuda_env MATCHES "^(0|OFF|FALSE|NO)$")
      set(_nep_cuda_mode OFF)
    else()
      message(FATAL_ERROR
        "NEP_CUDA must be one of 1/0, ON/OFF, TRUE/FALSE, or YES/NO.")
    endif()
  endif()

  if(_nep_cuda_mode STREQUAL "AUTO" AND
     (NOT DEFINED NEP_ADAPTERS_ENABLE_PYTHON OR
      NOT NEP_ADAPTERS_ENABLE_PYTHON))
    set(${output_variable} OFF PARENT_SCOPE)
    message(STATUS
      "NEPAdapters CUDA auto-detect is inactive for non-Python builds; "
      "set NEP_ADAPTERS_ENABLE_CUDA=ON to enable CUDA.")
    return()
  endif()

  unset(_nep_nvcc)
  if(NOT _nep_cuda_mode STREQUAL "OFF")
    if(DEFINED CMAKE_CUDA_COMPILER AND
       NOT CMAKE_CUDA_COMPILER STREQUAL "")
      set(_nep_nvcc "${CMAKE_CUDA_COMPILER}")
    elseif(DEFINED ENV{CUDACXX} AND EXISTS "$ENV{CUDACXX}")
      set(_nep_nvcc "$ENV{CUDACXX}")
    else()
      set(_nep_cuda_hints "")
      foreach(_nep_cuda_root
          CUDAToolkit_ROOT
          CUDA_TOOLKIT_ROOT_DIR
          CUDA_PATH
          CUDA_HOME)
        if(DEFINED ${_nep_cuda_root} AND
           NOT "${${_nep_cuda_root}}" STREQUAL "")
          list(APPEND _nep_cuda_hints "${${_nep_cuda_root}}/bin")
        endif()
        if(DEFINED ENV{${_nep_cuda_root}} AND
           NOT "$ENV{${_nep_cuda_root}}" STREQUAL "")
          list(APPEND _nep_cuda_hints "$ENV{${_nep_cuda_root}}/bin")
        endif()
      endforeach()
      find_program(
        _nep_nvcc
        NAMES nvcc nvcc.exe
        HINTS ${_nep_cuda_hints}
        NO_CACHE)
    endif()
  endif()

  if(_nep_cuda_mode STREQUAL "AUTO")
    if(_nep_nvcc)
      set(_nep_cuda_default ON)
      message(STATUS
        "NEPAdapters CUDA auto-detect: found ${_nep_nvcc}; enabling CUDA.")
    else()
      set(_nep_cuda_default OFF)
      message(STATUS
        "NEPAdapters CUDA auto-detect: nvcc not found; building CPU only.")
    endif()
  else()
    set(_nep_cuda_default "${_nep_cuda_mode}")
    message(STATUS
      "NEPAdapters CUDA selection forced by NEP_CUDA=${_nep_cuda_mode}.")
  endif()

  if(_nep_cuda_default AND _nep_nvcc AND
     NOT DEFINED CMAKE_CUDA_COMPILER)
    set(
      CMAKE_CUDA_COMPILER
      "${_nep_nvcc}"
      CACHE FILEPATH
      "CUDA compiler detected by NEPAdapters")
  endif()

  set(${output_variable} "${_nep_cuda_default}" PARENT_SCOPE)
endfunction()
