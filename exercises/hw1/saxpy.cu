// single precision z=ax+y
// from https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/

#include <boost/program_options.hpp>
#include <iostream>

// type for program_options
struct compute_location {
  compute_location(std::string const& val): value(val) {}
  std::string value;
};
// validator from compute_location type for program_options
void validate(boost::any& v, 
              std::vector<std::string> const& values,
              compute_location * /* target_type */,
              int) {
  using namespace boost::program_options;

  // Make sure no previous assignment to 'v' was made.
  validators::check_first_occurrence(v);

  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  std::string const& s = validators::get_single_string(values);

  if (s == "cpu" || s == "gpu") {
    v = boost::any(compute_location(s));
  } else {
    throw validation_error(validation_error::invalid_option_value);
  }
}

void cpu_saxpy(const float a, const float *x, float *y, const int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = a * x[i] + y[i];
  }
}

__global__ void gpu_saxpy(const float a, const float *x, float *y, const int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = a * x[idx] + y[idx];
  }
}

int main(int argc, const char *argv[]) {
  bool run_on_gpu = true;
  try {
    boost::program_options::options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Print options")   
      ("compute", boost::program_options::value<compute_location>()->default_value(compute_location("gpu"), "gpu"), "Compute location: cpu|gpu");
   
    boost::program_options::variables_map vm;
    boost::program_options::store(parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
  
    if (vm.count("help")) {
      std::cout << desc << '\n';
      return 0;
    }
    
    run_on_gpu = (vm["compute"].as<compute_location>().value == "gpu");
  }
  catch (std::exception const& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }

  const int N = 1 << 20; // 1M elements in a vector
  const int block_size = 256; // threads per block

  // declare and allocate host and device ptr
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  // initialize host ptr with data
  for (int i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  if (run_on_gpu) {
    std::cout << "Running on gpu.." << std::endl;

    // copy to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    gpu_saxpy<<<(N + block_size - 1) / block_size, block_size>>>(3.0f, d_x, d_y, N);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  }
  else {
    std::cout << "Running on cpu.." << std::endl;
    cpu_saxpy(3.0f, x, y, N);
  }

  float max_err = 0.0f;
  for (int i = 0; i < N; ++i) {
    max_err = std::max(max_err, std::abs(y[i] - 5.0f));
  }
  std::cout << "max_err=" << max_err << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}

