#include <cstdlib>
#include <memory>

#include <ATK/Core/Utilities.h>
#include <ATK/Modelling/ModellerFilter.h>
#include <ATK/Modelling/StaticComponent/StaticCapacitor.h>
#include <ATK/Modelling/StaticComponent/StaticCoil.h>
#include <ATK/Modelling/StaticComponent/StaticCurrent.h>
#include <ATK/Modelling/StaticComponent/StaticDiode.h>
#include <ATK/Modelling/StaticComponent/StaticEbersMollTransistor.h>
#include <ATK/Modelling/StaticComponent/StaticMOSFETTransistor.h>
#include <ATK/Modelling/StaticComponent/StaticResistor.h>
#include <ATK/Modelling/StaticComponent/StaticResistorCapacitor.h>

#include <Eigen/Eigen>

namespace
{
constexpr gsl::index MAX_ITERATION = 10;
constexpr gsl::index MAX_ITERATION_STEADY_STATE{200};

constexpr gsl::index INIT_WARMUP{10};
constexpr double EPS{1e-8};
constexpr double MAX_DELTA{1e-1};

class StaticFilter final: public ATK::ModellerFilter<double>
{
  using typename ATK::TypedBaseFilter<double>::DataType;
  bool initialized{false};

  Eigen::Matrix<DataType, 1, 1> static_state{Eigen::Matrix<DataType, 1, 1>::Zero()};
  mutable Eigen::Matrix<DataType, 1, 1> input_state{Eigen::Matrix<DataType, 1, 1>::Zero()};
  mutable Eigen::Matrix<DataType, 4, 1> dynamic_state{Eigen::Matrix<DataType, 4, 1>::Zero()};
  DataType pgain{250000};
  DataType pgain_trimmer{0};
  ATK::StaticCapacitor<DataType> c3{1.2e-10};
  ATK::StaticResistorCapacitor<DataType> r7c4{820, 8.2e-08};
  DataType pbassr8{51500};
  DataType pbassr8_trimmer{0};
  ATK::StaticCapacitor<DataType> c5{2.2e-06};
  ATK::StaticDiode<DataType, 1, 1> d1d2{2.52e-09, 1.752, 0.026};
  ATK::StaticResistor<DataType> r6{1800};

public:
  StaticFilter(): ModellerFilter<DataType>(4, 1)
  {
    static_state << 0.000000;
  }

  ~StaticFilter() override = default;

  gsl::index get_nb_dynamic_pins() const override
  {
    return 4;
  }

  gsl::index get_nb_input_pins() const override
  {
    return 1;
  }

  gsl::index get_nb_static_pins() const override
  {
    return 1;
  }

  Eigen::Matrix<DataType, Eigen::Dynamic, 1> get_static_state() const override
  {
    return static_state;
  }

  gsl::index get_nb_components() const override
  {
    return 8;
  }

  std::string get_dynamic_pin_name(gsl::index identifier) const override
  {
    switch(identifier)
    {
    case 3:
      return "5";
    case 2:
      return "vout";
    case 1:
      return "2";
    case 0:
      return "1";
    default:
      throw ATK::RuntimeError("No such pin");
    }
  }

  std::string get_input_pin_name(gsl::index identifier) const override
  {
    switch(identifier)
    {
    case 0:
      return "vin";
    default:
      throw ATK::RuntimeError("No such pin");
    }
  }

  std::string get_static_pin_name(gsl::index identifier) const override
  {
    switch(identifier)
    {
    case 0:
      return "0";
    default:
      throw ATK::RuntimeError("No such pin");
    }
  }

  gsl::index get_number_parameters() const override
  {
    return 2;
  }

  std::string get_parameter_name(gsl::index identifier) const override
  {
    switch(identifier)
    {
    case 0:
    {
      return "pgain";
    }
    case 1:
    {
      return "pbassr8";
    }
    default:
      throw ATK::RuntimeError("No such pin");
    }
  }

  DataType get_parameter(gsl::index identifier) const override
  {
    switch(identifier)
    {
    case 0:
    {
      return pgain_trimmer;
    }
    case 1:
    {
      return pbassr8_trimmer;
    }
    default:
      throw ATK::RuntimeError("No such pin");
    }
  }

  void set_parameter(gsl::index identifier, DataType value) override
  {
    switch(identifier)
    {
    case 0:
    {
      pgain_trimmer = value;
      break;
    }
    case 1:
    {
      pbassr8_trimmer = value;
      break;
    }
    default:
      throw ATK::RuntimeError("No such pin");
    }
  }

  /// Setup the inner state of the filter, slowly incrementing the static state
  void setup() override
  {
    assert(input_sampling_rate == output_sampling_rate);

    if(!initialized)
    {
      setup_inverse<true>();
      auto target_static_state = static_state;

      for(gsl::index i = 0; i < INIT_WARMUP; ++i)
      {
        static_state = target_static_state * ((i + 1.) / INIT_WARMUP);
        init();
      }
      static_state = target_static_state;
    }
    setup_inverse<false>();
  }

  template <bool steady_state>
  void setup_inverse()
  {
  }

  void init()
  {
    // update_steady_state
    c3.update_steady_state(1. / input_sampling_rate, dynamic_state[0], dynamic_state[2]);
    r7c4.update_steady_state(1. / input_sampling_rate, static_state[0], dynamic_state[0]);
    c5.update_steady_state(1. / input_sampling_rate, static_state[0], dynamic_state[3]);

    solve<true>();

    // update_steady_state
    c3.update_steady_state(1. / input_sampling_rate, dynamic_state[0], dynamic_state[2]);
    r7c4.update_steady_state(1. / input_sampling_rate, static_state[0], dynamic_state[0]);
    c5.update_steady_state(1. / input_sampling_rate, static_state[0], dynamic_state[3]);

    initialized = true;
  }

  void process_impl(gsl::index size) const override
  {
    for(gsl::index i = 0; i < size; ++i)
    {
      for(gsl::index j = 0; j < nb_input_ports; ++j)
      {
        input_state[j] = converted_inputs[j][i];
      }

      solve<false>();

      // Update state
      c3.update_state(dynamic_state[0], dynamic_state[2]);
      r7c4.update_state(static_state[0], dynamic_state[0]);
      c5.update_state(static_state[0], dynamic_state[3]);
      for(gsl::index j = 0; j < nb_output_ports; ++j)
      {
        outputs[j][i] = dynamic_state[j];
      }
    }
  }

  /// Solve for steady state and non steady state the system
  template <bool steady_state>
  void solve() const
  {
    gsl::index iteration = 0;

    constexpr int current_max_iter = steady_state ? MAX_ITERATION_STEADY_STATE : MAX_ITERATION;

    while(iteration < current_max_iter && !iterate<steady_state>())
    {
      ++iteration;
    }
  }

  template <bool steady_state>
  bool iterate() const
  {
    // Static states
    auto s0_ = static_state[0];

    // Input states
    auto i0_ = input_state[0];

    // Dynamic states
    auto d0_ = dynamic_state[0];
    auto d1_ = dynamic_state[1];
    auto d2_ = dynamic_state[2];
    auto d3_ = dynamic_state[3];

    // Precomputes
    d1d2.precompute(dynamic_state[0], dynamic_state[2]);

    Eigen::Matrix<DataType, 4, 1> eqs(Eigen::Matrix<DataType, 4, 1>::Zero());
    auto eq0 = +(pgain_trimmer != 0 ? (d1_ - d0_) / (pgain_trimmer * pgain) : 0)
             + (steady_state ? 0 : c3.get_current(d0_, d2_)) - (steady_state ? 0 : r7c4.get_current(s0_, d0_))
             + (pbassr8_trimmer != 1 ? (d3_ - d0_) / ((1 - pbassr8_trimmer) * pbassr8) : 0) + d1d2.get_current();
    auto eq1 = +(pgain_trimmer != 0 ? (d0_ - d1_) / (pgain_trimmer * pgain) : 0)
             + (pgain_trimmer != 1 ? (d2_ - d1_) / ((1 - pgain_trimmer) * pgain) : 0) - r6.get_current(d2_, d1_);
    auto eq2 = input_state[0] - dynamic_state[0];
    auto eq3 = +(pbassr8_trimmer != 1 ? (d0_ - d3_) / ((1 - pbassr8_trimmer) * pbassr8) : 0)
             - (steady_state ? 0 : c5.get_current(s0_, d3_));
    eqs << eq0, eq1, eq2, eq3;

    // Check if the equations have converged
    if((eqs.array().abs() < EPS).all())
    {
      return true;
    }

    auto jac0_0 = 0 + (pgain_trimmer != 0 ? -1 / (pgain_trimmer * pgain) : 0) - (steady_state ? 0 : c3.get_gradient())
                - (steady_state ? 0 : r7c4.get_gradient())
                + (pbassr8_trimmer != 1 ? -1 / ((1 - pbassr8_trimmer) * pbassr8) : 0) - d1d2.get_gradient();
    auto jac0_1 = 0 + (pgain_trimmer != 0 ? 1 / (pgain_trimmer * pgain) : 0);
    auto jac0_2 = 0 + (steady_state ? 0 : c3.get_gradient()) + d1d2.get_gradient();
    auto jac0_3 = 0 + (pbassr8_trimmer != 1 ? 1 / ((1 - pbassr8_trimmer) * pbassr8) : 0);
    auto jac1_0 = 0 + (pgain_trimmer != 0 ? 1 / (pgain_trimmer * pgain) : 0);
    auto jac1_1 = 0 + (pgain_trimmer != 0 ? -1 / (pgain_trimmer * pgain) : 0)
                + (pgain_trimmer != 1 ? -1 / ((1 - pgain_trimmer) * pgain) : 0) - r6.get_gradient();
    auto jac1_2 = 0 + (pgain_trimmer != 1 ? 1 / ((1 - pgain_trimmer) * pgain) : 0) + r6.get_gradient();
    auto jac1_3 = 0;
    auto jac2_0 = 0 + -1;
    auto jac2_1 = 0;
    auto jac2_2 = 0;
    auto jac2_3 = 0;
    auto jac3_0 = 0 + (pbassr8_trimmer != 1 ? 1 / ((1 - pbassr8_trimmer) * pbassr8) : 0);
    auto jac3_1 = 0;
    auto jac3_2 = 0;
    auto jac3_3 = 0 + (pbassr8_trimmer != 1 ? -1 / ((1 - pbassr8_trimmer) * pbassr8) : 0)
                - (steady_state ? 0 : c5.get_gradient());
    auto det
        = (-1 * jac0_1 * (-1 * jac1_2 * (1 * jac2_0 * jac3_3)) + 1 * jac0_2 * (-1 * jac1_1 * (1 * jac2_0 * jac3_3)));
    auto invdet = 1 / det;
    auto com0_0 = 0;
    auto com1_0 = -1 * (-1 * jac1_2 * (1 * jac2_0 * jac3_3));
    auto com2_0 = (-1 * jac1_1 * (1 * jac2_0 * jac3_3));
    auto com3_0 = -1 * 0;
    auto com0_1 = -1 * 0;
    auto com1_1 = (-1 * jac0_2 * (1 * jac2_0 * jac3_3));
    auto com2_1 = -1 * (-1 * jac0_1 * (1 * jac2_0 * jac3_3));
    auto com3_1 = 0;
    auto com0_2 = (1 * jac0_1 * (1 * jac1_2 * jac3_3) + -1 * jac0_2 * (1 * jac1_1 * jac3_3));
    auto com1_2 = -1
                * (1 * jac0_0 * (1 * jac1_2 * jac3_3) + -1 * jac0_2 * (1 * jac1_0 * jac3_3)
                    + 1 * jac0_3 * (-1 * jac1_2 * jac3_0));
    auto com2_2 = (1 * jac0_0 * (1 * jac1_1 * jac3_3) + -1 * jac0_1 * (1 * jac1_0 * jac3_3)
                   + 1 * jac0_3 * (-1 * jac1_1 * jac3_0));
    auto com3_2 = -1 * (-1 * jac0_1 * (-1 * jac1_2 * jac3_0) + 1 * jac0_2 * (-1 * jac1_1 * jac3_0));
    auto com0_3 = -1 * 0;
    auto com1_3 = (1 * jac0_3 * (-1 * jac1_2 * jac2_0));
    auto com2_3 = -1 * (1 * jac0_3 * (-1 * jac1_1 * jac2_0));
    auto com3_3 = (-1 * jac0_1 * (-1 * jac1_2 * jac2_0) + 1 * jac0_2 * (-1 * jac1_1 * jac2_0));
    Eigen::Matrix<DataType, 4, 4> cojacobian(Eigen::Matrix<DataType, 4, 4>::Zero());

    cojacobian << com0_0, com0_1, com0_2, com0_3, com1_0, com1_1, com1_2, com1_3, com2_0, com2_1, com2_2, com2_3,
        com3_0, com3_1, com3_2, com3_3;
    Eigen::Matrix<DataType, 4, 1> delta = cojacobian * eqs * invdet;

    // Check if the update is big enough
    if(delta.hasNaN() || (delta.array().abs() < EPS).all())
    {
      return true;
    }

    // Big variations are only in steady state mode
    if constexpr(steady_state)
    {
      auto max_delta = delta.array().abs().maxCoeff();
      if(max_delta > MAX_DELTA)
      {
        delta *= MAX_DELTA / max_delta;
      }
    }

    dynamic_state -= delta;

    return false;
  }
};
} // namespace

extern "C"
{
  std::unique_ptr<ATK::ModellerFilter<double>> createStaticFilter()
  {
    return std::make_unique<StaticFilter>();
  }
} // namespace
