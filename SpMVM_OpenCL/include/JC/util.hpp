#ifndef __JC_UTIL_HPP__
#define __JC_UTIL_HPP__

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

namespace jc {

bool
is_prefix(const std::string& string1, const std::string& string2)
{
  if (string1.size() > string2.size())
    return false;
  return std::equal(string1.cbegin(), string1.cend(), string2.cbegin());
}

//-------------------------- implementation details ---------------------------#
namespace detail {

template<typename Predicate>
cl::Device
get_device(const Predicate& pred)
{
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for (const auto& platform : platforms) {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    auto it = std::find_if(devices.cbegin(), devices.cend(), pred);
    if (it != devices.cend())
      return *it;
  }
  throw std::runtime_error("No appropriate device found");
}

struct DeviceOfType
{
  cl_device_info type;

  bool operator()(const cl::Device& device)
  {
    cl_device_type t;
    device.getInfo(CL_DEVICE_TYPE, &t);
    return type == t;
  }
};

struct DeviceCalled
{
  std::string name;

  bool operator()(const cl::Device& device)
  {
    std::string n;
    device.getInfo(CL_DEVICE_NAME, &n);
    return is_prefix(name, n);
  }
};

const std::map<cl_int, const char*> error_codes = {
  { CL_SUCCESS, "CL_SUCCESS" },
  { CL_DEVICE_NOT_FOUND, "CL_DEVICE_NOT_FOUND" },
  { CL_DEVICE_NOT_AVAILABLE, "CL_DEVICE_NOT_AVAILABLE" },
  { CL_COMPILER_NOT_AVAILABLE, "CL_COMPILER_NOT_AVAILABLE" },
  { CL_MEM_OBJECT_ALLOCATION_FAILURE, "CL_MEM_OBJECT_ALLOCATION_FAILURE" },
  { CL_OUT_OF_RESOURCES, "CL_OUT_OF_RESOURCES" },
  { CL_OUT_OF_HOST_MEMORY, "CL_OUT_OF_HOST_MEMORY" },
  { CL_PROFILING_INFO_NOT_AVAILABLE, "CL_PROFILING_INFO_NOT_AVAILABLE" },
  { CL_MEM_COPY_OVERLAP, "CL_MEM_COPY_OVERLAP" },
  { CL_IMAGE_FORMAT_MISMATCH, "CL_IMAGE_FORMAT_MISMATCH" },
  { CL_IMAGE_FORMAT_NOT_SUPPORTED, "CL_IMAGE_FORMAT_NOT_SUPPORTED" },
  { CL_BUILD_PROGRAM_FAILURE, "CL_BUILD_PROGRAM_FAILURE" },
  { CL_MAP_FAILURE, "CL_MAP_FAILURE" },
  { CL_MISALIGNED_SUB_BUFFER_OFFSET, "CL_MISALIGNED_SUB_BUFFER_OFFSET" },
  { CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
    "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" },
#ifdef CL_VERSION_1_2
  { CL_COMPILE_PROGRAM_FAILURE, "CL_COMPILE_PROGRAM_FAILURE" },
  { CL_LINKER_NOT_AVAILABLE, "CL_LINKER_NOT_AVAILABLE" },
  { CL_LINK_PROGRAM_FAILURE, "CL_LINK_PROGRAM_FAILURE" },
  { CL_DEVICE_PARTITION_FAILED, "CL_DEVICE_PARTITION_FAILED" },
  { CL_KERNEL_ARG_INFO_NOT_AVAILABLE, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE" },
#endif
  { CL_INVALID_VALUE, "CL_INVALID_VALUE" },
  { CL_INVALID_DEVICE_TYPE, "CL_INVALID_DEVICE_TYPE" },
  { CL_INVALID_PLATFORM, "CL_INVALID_PLATFORM" },
  { CL_INVALID_DEVICE, "CL_INVALID_DEVICE" },
  { CL_INVALID_CONTEXT, "CL_INVALID_CONTEXT" },
  { CL_INVALID_QUEUE_PROPERTIES, "CL_INVALID_QUEUE_PROPERTIES" },
  { CL_INVALID_COMMAND_QUEUE, "CL_INVALID_COMMAND_QUEUE" },
  { CL_INVALID_HOST_PTR, "CL_INVALID_HOST_PTR" },
  { CL_INVALID_MEM_OBJECT, "CL_INVALID_MEM_OBJECT" },
  { CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR" },
  { CL_INVALID_IMAGE_SIZE, "CL_INVALID_IMAGE_SIZE" },
  { CL_INVALID_SAMPLER, "CL_INVALID_SAMPLER" },
  { CL_INVALID_BINARY, "CL_INVALID_BINARY" },
  { CL_INVALID_BUILD_OPTIONS, "CL_INVALID_BUILD_OPTIONS" },
  { CL_INVALID_PROGRAM, "CL_INVALID_PROGRAM" },
  { CL_INVALID_PROGRAM_EXECUTABLE, "CL_INVALID_PROGRAM_EXECUTABLE" },
  { CL_INVALID_KERNEL_NAME, "CL_INVALID_KERNEL_NAME" },
  { CL_INVALID_KERNEL_DEFINITION, "CL_INVALID_KERNEL_DEFINITION" },
  { CL_INVALID_KERNEL, "CL_INVALID_KERNEL" },
  { CL_INVALID_ARG_INDEX, "CL_INVALID_ARG_INDEX" },
  { CL_INVALID_ARG_VALUE, "CL_INVALID_ARG_VALUE" },
  { CL_INVALID_ARG_SIZE, "CL_INVALID_ARG_SIZE" },
  { CL_INVALID_KERNEL_ARGS, "CL_INVALID_KERNEL_ARGS" },
  { CL_INVALID_WORK_DIMENSION, "CL_INVALID_WORK_DIMENSION" },
  { CL_INVALID_WORK_GROUP_SIZE, "CL_INVALID_WORK_GROUP_SIZE" },
  { CL_INVALID_WORK_ITEM_SIZE, "CL_INVALID_WORK_ITEM_SIZE" },
  { CL_INVALID_GLOBAL_OFFSET, "CL_INVALID_GLOBAL_OFFSET" },
  { CL_INVALID_EVENT_WAIT_LIST, "CL_INVALID_EVENT_WAIT_LIST" },
  { CL_INVALID_EVENT, "CL_INVALID_EVENT" },
  { CL_INVALID_OPERATION, "CL_INVALID_OPERATION" },
  { CL_INVALID_GL_OBJECT, "CL_INVALID_GL_OBJECT" },
  { CL_INVALID_BUFFER_SIZE, "CL_INVALID_BUFFER_SIZE" },
  { CL_INVALID_MIP_LEVEL, "CL_INVALID_MIP_LEVEL" },
  { CL_INVALID_GLOBAL_WORK_SIZE, "CL_INVALID_GLOBAL_WORK_SIZE" },
#ifdef CL_VERSION_1_2
  { CL_INVALID_PROPERTY, "CL_INVALID_PROPERTY" },
  { CL_INVALID_IMAGE_DESCRIPTOR, "CL_INVALID_IMAGE_DESCRIPTOR" },
  { CL_INVALID_COMPILER_OPTIONS, "CL_INVALID_COMPILER_OPTIONS" },
  { CL_INVALID_LINKER_OPTIONS, "CL_INVALID_LINKER_OPTIONS" },
  { CL_INVALID_DEVICE_PARTITION_COUNT, "CL_INVALID_DEVICE_PARTITION_COUNT" }
#endif
};
}

//----------------------------- useful functions ------------------------------#

std::string
file_to_string(const std::string& file_path)
{
  std::ifstream filestream{ file_path };
  if (!filestream) {
    std::ostringstream oss;
    oss << "Could not open a file " << file_path;
    throw std::runtime_error{ oss.str() };
  }
  std::string file_string;
  file_string.assign(std::istreambuf_iterator<char>(filestream),
                     std::istreambuf_iterator<char>());
  return file_string;
}

cl::Program
build_program_from_string(const std::string& source,
                          const cl::Context& context,
                          const cl::Device& device,
                          const char* options = nullptr)
{
  cl::Program program{ context, source };

  try {
    program.build({ device }, options);
  } catch (cl::Error&) {
    std::string build_log;
    std::ostringstream oss;
    program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
    oss << "Your program failed to compile: \n";
    oss << "--------------------------------\n";
    oss << build_log << std::endl;
    oss << "--------------------------------\n";
    throw std::runtime_error{ oss.str() };
  }

  return program;
}

cl::Program
build_program_from_file(const std::string& source_file,
                        const cl::Context& context,
                        const cl::Device& device,
                        const char* options = nullptr)
{
  return build_program_from_string(
    file_to_string(source_file), context, device, options);
}

cl::Device
get_device(cl_device_info type)
{
  return detail::get_device(detail::DeviceOfType{ type });
}

cl::Device
get_device(const std::string& name)
{
  return detail::get_device(detail::DeviceCalled{ name });
}

cl_ulong
run_and_time_kernel(const cl::Kernel& kernel,
                    const cl::CommandQueue& queue,
                    cl::NDRange global,
                    cl::NDRange local = cl::NullRange)
{
  cl_ulong t1, t2;
  cl::Event event;

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &event);
  event.wait();
  event.getProfilingInfo(CL_PROFILING_COMMAND_START, &t1);
  event.getProfilingInfo(CL_PROFILING_COMMAND_END, &t2);

  return t2 - t1;
}

size_t
best_fit(size_t global, size_t local)
{
  size_t times = global / local;
  if (local * times != global)
    ++times;
  return times * local;
}

const char*
readable_error(cl_int e)
{
  if (detail::error_codes.count(e) == 0)
    return "UNKNOWN ERROR CODE";

  return detail::error_codes.at(e);
}

template<typename T>
void
show_matrix(const std::vector<T>& matrix, int height, int width)
{
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col)
      std::cout << std::setw(6) << matrix[width * row + col] << " ";
    std::cout << "\n";
  }
}
}

#endif
