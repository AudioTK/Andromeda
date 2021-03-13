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

  Eigen::Matrix<DataType, 3, 1> static_state{Eigen::Matrix<DataType, 3, 1>::Zero()};
  mutable Eigen::Matrix<DataType, 1, 1> input_state{Eigen::Matrix<DataType, 1, 1>::Zero()};
  mutable Eigen::Matrix<DataType, 11, 1> dynamic_state{Eigen::Matrix<DataType, 11, 1>::Zero()};
  DataType pspectrumr17{26200};
  DataType pspectrumr17_trimmer{0};
  ATK::StaticEBNPN<DataType> q2{
      1e-12,
      0.026,
      1,
      1,
      100,
  };
  ATK::StaticResistor<DataType> r16{3300};
  ATK::StaticCapacitor<DataType> c13{8.2e-09};
  ATK::StaticCapacitor<DataType> c12{1e-07};
  ATK::StaticResistor<DataType> r14{2200};
  ATK::StaticResistor<DataType> r19{10000};
  ATK::StaticResistor<DataType> r13{5100};
  ATK::StaticCapacitor<DataType> c14{5.8e-10};
  ATK::StaticCapacitor<DataType> c11{2.7e-08};
  ATK::StaticResistor<DataType> r20{20000};
  ATK::StaticCapacitor<DataType> c10{2.2e-08};
  ATK::StaticResistor<DataType> r11{10000};
  ATK::StaticResistor<DataType> r12{12000};
  ATK::StaticCapacitor<DataType> c9{1e-09};
  ATK::StaticResistor<DataType> r15{150000};
  ATK::StaticDiode<DataType, 1, 1> d5d4{2.52e-09, 1.752, 0.026};
  ATK::StaticResistor<DataType> r10{39000};
  ATK::StaticResistorCapacitor<DataType> r9c6{12000, 2.2e-06};
  ATK::StaticResistor<DataType> r18{43000};
  ATK::StaticCapacitor<DataType> c8{8.2e-08};

public:
  StaticFilter(): ModellerFilter<DataType>(11, 1)
  {
    static_state << 0.000000, -4.500000, 4.500000;
  }

  ~StaticFilter() override = default;

  gsl::index get_nb_dynamic_pins() const override
  {
    return 11;
  }

  gsl::index get_nb_input_pins() const override
  {
    return 1;
  }

  gsl::index get_nb_static_pins() const override
  {
    return 3;
  }

  Eigen::Matrix<DataType, Eigen::Dynamic, 1> get_static_state() const override
  {
    return static_state;
  }

  gsl::index get_nb_components() const override
  {
    return 22;
  }

  std::string get_dynamic_pin_name(gsl::index identifier) const override
  {
    switch(identifier)
    {
    case 10:
      return "2";
    case 7:
      return "vout";
    case 0:
      return "6";
    case 3:
      return "9";
    case 8:
      return "3";
    case 2:
      return "12";
    case 9:
      return "4";
    case 5:
      return "8";
    case 4:
      return "10";
    case 6:
      return "5";
    case 1:
      return "7";
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
    case 1:
      return "vdd";
    case 2:
      return "vcc";
    case 0:
      return "0";
    default:
      throw ATK::RuntimeError("No such pin");
    }
  }

  gsl::index get_number_parameters() const override
  {
    return 1;
  }

  std::string get_parameter_name(gsl::index identifier) const override
  {
    switch(identifier)
    {
    case 0:
    {
      return "pspectrumr17";
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
      return pspectrumr17_trimmer;
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
      pspectrumr17_trimmer = value;
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
    c13.update_steady_state(1. / input_sampling_rate, dynamic_state[5], dynamic_state[3]);
    c12.update_steady_state(1. / input_sampling_rate, dynamic_state[1], dynamic_state[5]);
    c14.update_steady_state(1. / input_sampling_rate, dynamic_state[2], dynamic_state[7]);
    c11.update_steady_state(1. / input_sampling_rate, dynamic_state[1], static_state[0]);
    c10.update_steady_state(1. / input_sampling_rate, dynamic_state[6], dynamic_state[0]);
    c9.update_steady_state(1. / input_sampling_rate, dynamic_state[9], static_state[0]);
    r9c6.update_steady_state(1. / input_sampling_rate, input_state[0], dynamic_state[10]);
    c8.update_steady_state(1. / input_sampling_rate, dynamic_state[8], dynamic_state[9]);

    solve<true>();

    // update_steady_state
    c13.update_steady_state(1. / input_sampling_rate, dynamic_state[5], dynamic_state[3]);
    c12.update_steady_state(1. / input_sampling_rate, dynamic_state[1], dynamic_state[5]);
    c14.update_steady_state(1. / input_sampling_rate, dynamic_state[2], dynamic_state[7]);
    c11.update_steady_state(1. / input_sampling_rate, dynamic_state[1], static_state[0]);
    c10.update_steady_state(1. / input_sampling_rate, dynamic_state[6], dynamic_state[0]);
    c9.update_steady_state(1. / input_sampling_rate, dynamic_state[9], static_state[0]);
    r9c6.update_steady_state(1. / input_sampling_rate, input_state[0], dynamic_state[10]);
    c8.update_steady_state(1. / input_sampling_rate, dynamic_state[8], dynamic_state[9]);

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
      c13.update_state(dynamic_state[5], dynamic_state[3]);
      c12.update_state(dynamic_state[1], dynamic_state[5]);
      c14.update_state(dynamic_state[2], dynamic_state[7]);
      c11.update_state(dynamic_state[1], static_state[0]);
      c10.update_state(dynamic_state[6], dynamic_state[0]);
      c9.update_state(dynamic_state[9], static_state[0]);
      r9c6.update_state(input_state[0], dynamic_state[10]);
      c8.update_state(dynamic_state[8], dynamic_state[9]);
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
    auto s1_ = static_state[1];
    auto s2_ = static_state[2];

    // Input states
    auto i0_ = input_state[0];

    // Dynamic states
    auto d0_ = dynamic_state[0];
    auto d1_ = dynamic_state[1];
    auto d2_ = dynamic_state[2];
    auto d3_ = dynamic_state[3];
    auto d4_ = dynamic_state[4];
    auto d5_ = dynamic_state[5];
    auto d6_ = dynamic_state[6];
    auto d7_ = dynamic_state[7];
    auto d8_ = dynamic_state[8];
    auto d9_ = dynamic_state[9];
    auto d10_ = dynamic_state[10];

    // Precomputes
    q2.precompute(dynamic_state[3], static_state[2], dynamic_state[4]);
    d5d4.precompute(static_state[0], dynamic_state[10]);

    Eigen::Matrix<DataType, 11, 1> eqs(Eigen::Matrix<DataType, 11, 1>::Zero());
    auto eq0 = +(pspectrumr17_trimmer != 0 ? (d1_ - d0_) / (pspectrumr17_trimmer * pspectrumr17) : 0)
             - r13.get_current(d6_, d0_) - (steady_state ? 0 : c10.get_current(d6_, d0_));
    auto eq1 = +(pspectrumr17_trimmer != 0 ? (d0_ - d1_) / (pspectrumr17_trimmer * pspectrumr17) : 0)
             + (pspectrumr17_trimmer != 1 ? (d2_ - d1_) / ((1 - pspectrumr17_trimmer) * pspectrumr17) : 0)
             + (steady_state ? 0 : c12.get_current(d1_, d5_)) + (steady_state ? 0 : c11.get_current(d1_, s0_));
    auto eq2 = +(pspectrumr17_trimmer != 1 ? (d1_ - d2_) / ((1 - pspectrumr17_trimmer) * pspectrumr17) : 0)
             + r19.get_current(d2_, s0_) + (steady_state ? 0 : c14.get_current(d2_, d7_)) + r20.get_current(d2_, d7_);
    auto eq3 = -q2.ib() + r16.get_current(d3_, s1_) - (steady_state ? 0 : c13.get_current(d5_, d3_))
             + r15.get_current(d3_, s0_);
    auto eq4 = +q2.ib() + q2.ic() + r14.get_current(d4_, d5_);
    auto eq5 = +(steady_state ? 0 : c13.get_current(d5_, d3_)) - (steady_state ? 0 : c12.get_current(d1_, d5_))
             - r14.get_current(d4_, d5_);
    auto eq6 = +r13.get_current(d6_, d0_) + (steady_state ? 0 : c10.get_current(d6_, d0_)) - r11.get_current(d8_, d6_)
             + r18.get_current(d6_, s0_);
    auto eq7 = dynamic_state[6] - dynamic_state[2];
    auto eq8 = +r11.get_current(d8_, d6_) - r10.get_current(d10_, d8_) + (steady_state ? 0 : c8.get_current(d8_, d9_));
    auto eq9 = +r12.get_current(d9_, s0_) + (steady_state ? 0 : c9.get_current(d9_, s0_))
             - (steady_state ? 0 : c8.get_current(d8_, d9_));
    auto eq10 = -d5d4.get_current() + r10.get_current(d10_, d8_) - (steady_state ? 0 : r9c6.get_current(i0_, d10_));
    eqs << eq0, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10;

    // Check if the equations have converged
    if((eqs.array().abs() < EPS).all())
    {
      return true;
    }

    auto jac0_0 = 0 + (pspectrumr17_trimmer != 0 ? -1 / (pspectrumr17_trimmer * pspectrumr17) : 0) - r13.get_gradient()
                - (steady_state ? 0 : c10.get_gradient());
    auto jac0_1 = 0 + (pspectrumr17_trimmer != 0 ? 1 / (pspectrumr17_trimmer * pspectrumr17) : 0);
    auto jac0_2 = 0;
    auto jac0_3 = 0;
    auto jac0_4 = 0;
    auto jac0_5 = 0;
    auto jac0_6 = 0 + r13.get_gradient() + (steady_state ? 0 : c10.get_gradient());
    auto jac0_7 = 0;
    auto jac0_8 = 0;
    auto jac0_9 = 0;
    auto jac0_10 = 0;
    auto jac1_0 = 0 + (pspectrumr17_trimmer != 0 ? 1 / (pspectrumr17_trimmer * pspectrumr17) : 0);
    auto jac1_1 = 0 + (pspectrumr17_trimmer != 0 ? -1 / (pspectrumr17_trimmer * pspectrumr17) : 0)
                + (pspectrumr17_trimmer != 1 ? -1 / ((1 - pspectrumr17_trimmer) * pspectrumr17) : 0)
                - (steady_state ? 0 : c12.get_gradient()) - (steady_state ? 0 : c11.get_gradient());
    auto jac1_2 = 0 + (pspectrumr17_trimmer != 1 ? 1 / ((1 - pspectrumr17_trimmer) * pspectrumr17) : 0);
    auto jac1_3 = 0;
    auto jac1_4 = 0;
    auto jac1_5 = 0 + (steady_state ? 0 : c12.get_gradient());
    auto jac1_6 = 0;
    auto jac1_7 = 0;
    auto jac1_8 = 0;
    auto jac1_9 = 0;
    auto jac1_10 = 0;
    auto jac2_0 = 0;
    auto jac2_1 = 0 + (pspectrumr17_trimmer != 1 ? 1 / ((1 - pspectrumr17_trimmer) * pspectrumr17) : 0);
    auto jac2_2 = 0 + (pspectrumr17_trimmer != 1 ? -1 / ((1 - pspectrumr17_trimmer) * pspectrumr17) : 0)
                - r19.get_gradient() - (steady_state ? 0 : c14.get_gradient()) - r20.get_gradient();
    auto jac2_3 = 0;
    auto jac2_4 = 0;
    auto jac2_5 = 0;
    auto jac2_6 = 0;
    auto jac2_7 = 0 + (steady_state ? 0 : c14.get_gradient()) + r20.get_gradient();
    auto jac2_8 = 0;
    auto jac2_9 = 0;
    auto jac2_10 = 0;
    auto jac3_0 = 0;
    auto jac3_1 = 0;
    auto jac3_2 = 0;
    auto jac3_3 = 0 - q2.ib_Vbc() - q2.ib_Vbe() - r16.get_gradient() - (steady_state ? 0 : c13.get_gradient())
                - r15.get_gradient();
    auto jac3_4 = 0 + q2.ib_Vbe();
    auto jac3_5 = 0 + (steady_state ? 0 : c13.get_gradient());
    auto jac3_6 = 0;
    auto jac3_7 = 0;
    auto jac3_8 = 0;
    auto jac3_9 = 0;
    auto jac3_10 = 0;
    auto jac4_0 = 0;
    auto jac4_1 = 0;
    auto jac4_2 = 0;
    auto jac4_3 = 0 + q2.ib_Vbc() + q2.ib_Vbe() + q2.ic_Vbc() + q2.ic_Vbe();
    auto jac4_4 = 0 - q2.ib_Vbe() - q2.ic_Vbe() - r14.get_gradient();
    auto jac4_5 = 0 + r14.get_gradient();
    auto jac4_6 = 0;
    auto jac4_7 = 0;
    auto jac4_8 = 0;
    auto jac4_9 = 0;
    auto jac4_10 = 0;
    auto jac5_0 = 0;
    auto jac5_1 = 0 + (steady_state ? 0 : c12.get_gradient());
    auto jac5_2 = 0;
    auto jac5_3 = 0 + (steady_state ? 0 : c13.get_gradient());
    auto jac5_4 = 0 + r14.get_gradient();
    auto jac5_5
        = 0 - (steady_state ? 0 : c13.get_gradient()) - (steady_state ? 0 : c12.get_gradient()) - r14.get_gradient();
    auto jac5_6 = 0;
    auto jac5_7 = 0;
    auto jac5_8 = 0;
    auto jac5_9 = 0;
    auto jac5_10 = 0;
    auto jac6_0 = 0 + r13.get_gradient() + (steady_state ? 0 : c10.get_gradient());
    auto jac6_1 = 0;
    auto jac6_2 = 0;
    auto jac6_3 = 0;
    auto jac6_4 = 0;
    auto jac6_5 = 0;
    auto jac6_6
        = 0 - r13.get_gradient() - (steady_state ? 0 : c10.get_gradient()) - r11.get_gradient() - r18.get_gradient();
    auto jac6_7 = 0;
    auto jac6_8 = 0 + r11.get_gradient();
    auto jac6_9 = 0;
    auto jac6_10 = 0;
    auto jac7_0 = 0;
    auto jac7_1 = 0;
    auto jac7_2 = 0 + -1;
    auto jac7_3 = 0;
    auto jac7_4 = 0;
    auto jac7_5 = 0;
    auto jac7_6 = 0 + 1;
    auto jac7_7 = 0;
    auto jac7_8 = 0;
    auto jac7_9 = 0;
    auto jac7_10 = 0;
    auto jac8_0 = 0;
    auto jac8_1 = 0;
    auto jac8_2 = 0;
    auto jac8_3 = 0;
    auto jac8_4 = 0;
    auto jac8_5 = 0;
    auto jac8_6 = 0 + r11.get_gradient();
    auto jac8_7 = 0;
    auto jac8_8 = 0 - r11.get_gradient() - r10.get_gradient() - (steady_state ? 0 : c8.get_gradient());
    auto jac8_9 = 0 + (steady_state ? 0 : c8.get_gradient());
    auto jac8_10 = 0 + r10.get_gradient();
    auto jac9_0 = 0;
    auto jac9_1 = 0;
    auto jac9_2 = 0;
    auto jac9_3 = 0;
    auto jac9_4 = 0;
    auto jac9_5 = 0;
    auto jac9_6 = 0;
    auto jac9_7 = 0;
    auto jac9_8 = 0 + (steady_state ? 0 : c8.get_gradient());
    auto jac9_9
        = 0 - r12.get_gradient() - (steady_state ? 0 : c9.get_gradient()) - (steady_state ? 0 : c8.get_gradient());
    auto jac9_10 = 0;
    auto jac10_0 = 0;
    auto jac10_1 = 0;
    auto jac10_2 = 0;
    auto jac10_3 = 0;
    auto jac10_4 = 0;
    auto jac10_5 = 0;
    auto jac10_6 = 0;
    auto jac10_7 = 0;
    auto jac10_8 = 0 + r10.get_gradient();
    auto jac10_9 = 0;
    auto jac10_10 = 0 - d5d4.get_gradient() - r10.get_gradient() - (steady_state ? 0 : r9c6.get_gradient());
    auto det
        = (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                    + 1 * jac6_8
                                                          * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                      + 1 * jac6_8
                                                            * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                      + 1 * jac6_8
                                                            * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                        + 1 * jac6_8
                                                              * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                      + 1 * jac6_8
                                                            * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                        + 1 * jac6_8
                                                              * (1 * jac7_2
                                                                  * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1
                                              * (-1 * jac6_6
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                  + 1 * jac6_8
                                                        * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                    + 1 * jac6_8
                                                          * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                      + 1 * jac6_8
                                                            * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                        + 1 * jac6_8
                                                              * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                        + 1 * jac6_8
                                                              * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6
                                                              * (1 * jac7_2
                                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                          + 1 * jac6_8
                                                                * (1 * jac7_2
                                                                    * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                        + 1 * jac6_8
                                                              * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6
                                                              * (1 * jac7_2
                                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                          + 1 * jac6_8
                                                                * (1 * jac7_2
                                                                    * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5
                                                  * (1 * jac6_0
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1
                                                * (1 * jac6_0
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))));
    auto invdet = 1 / det;
    auto com0_0
        = (1 * jac1_1
                * (-1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac4_4
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac4_5
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + 1 * jac3_4
                              * (-1 * jac4_3
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (-1 * jac4_3
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8
                                                      * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + 1 * jac1_5
                  * (-1 * jac2_7
                      * (1 * jac3_3
                              * (1 * jac4_4
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                          + -1 * jac3_4
                                * (1 * jac4_3
                                    * (1 * jac5_1
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com1_0
        = -1
        * (1 * jac1_0
                * (-1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac4_4
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac4_5
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + 1 * jac3_4
                              * (-1 * jac4_3
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (-1 * jac4_3
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8
                                                      * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac1_2
                  * (-1 * jac2_7
                      * (-1 * jac3_3
                              * (-1 * jac4_4
                                      * (-1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_6
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_4
                                * (-1 * jac4_3
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_5
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_5
                                * (-1 * jac4_3
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_4
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com2_0 = (-1 * jac1_1
                       * (-1 * jac2_7
                           * (-1 * jac3_3
                                   * (-1 * jac4_4
                                           * (-1 * jac5_5
                                               * (1 * jac6_0
                                                   * (1 * jac7_6
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                       + 1 * jac4_5
                                             * (-1 * jac5_4
                                                 * (1 * jac6_0
                                                     * (1 * jac7_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                               + 1 * jac3_4
                                     * (-1 * jac4_3
                                             * (-1 * jac5_5
                                                 * (1 * jac6_0
                                                     * (1 * jac7_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                         + 1 * jac4_5
                                               * (-1 * jac5_3
                                                   * (1 * jac6_0
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                               + -1 * jac3_5
                                     * (-1 * jac4_3
                                             * (-1 * jac5_4
                                                 * (1 * jac6_0
                                                     * (1 * jac7_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                         + 1 * jac4_4
                                               * (-1 * jac5_3
                                                   * (1 * jac6_0
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                   + 1 * jac1_5
                         * (-1 * jac2_7
                             * (1 * jac3_3
                                     * (1 * jac4_4
                                         * (-1 * jac5_1
                                             * (1 * jac6_0
                                                 * (1 * jac7_6
                                                     * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                         + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                         + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                 + -1 * jac3_4
                                       * (1 * jac4_3
                                           * (-1 * jac5_1
                                               * (1 * jac6_0
                                                   * (1 * jac7_6
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com3_0
        = -1
        * (1 * jac1_0
                * (-1 * jac2_7
                    * (1 * jac3_4
                            * (1 * jac4_5
                                * (1 * jac5_1
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (1 * jac4_4
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + 1 * jac1_2
                  * (-1 * jac2_7
                      * (1 * jac3_4
                              * (1 * jac4_5
                                  * (-1 * jac5_1
                                      * (1 * jac6_0
                                          * (1 * jac7_6
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_5
                                * (1 * jac4_4
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_6
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com4_0
        = (1 * jac1_0
                * (-1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac4_5
                                * (1 * jac5_1
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (1 * jac4_3
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + 1 * jac1_2
                  * (-1 * jac2_7
                      * (1 * jac3_3
                              * (1 * jac4_5
                                  * (-1 * jac5_1
                                      * (1 * jac6_0
                                          * (1 * jac7_6
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_5
                                * (1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_6
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com5_0
        = -1
        * (1 * jac1_0
                * (-1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac4_4
                                * (1 * jac5_1
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_4
                              * (1 * jac4_3
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + 1 * jac1_2
                  * (-1 * jac2_7
                      * (1 * jac3_3
                              * (1 * jac4_4
                                  * (-1 * jac5_1
                                      * (1 * jac6_0
                                          * (1 * jac7_6
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_4
                                * (1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_6
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com6_0 = (-1 * jac1_1
                       * (-1 * jac2_7
                           * (1 * jac3_3
                                   * (1 * jac4_4
                                           * (1 * jac5_5
                                               * (1 * jac6_0
                                                   * (1 * jac7_2
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                       + -1 * jac4_5
                                             * (1 * jac5_4
                                                 * (1 * jac6_0
                                                     * (1 * jac7_2
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                               + -1 * jac3_4
                                     * (1 * jac4_3
                                             * (1 * jac5_5
                                                 * (1 * jac6_0
                                                     * (1 * jac7_2
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                         + -1 * jac4_5
                                               * (1 * jac5_3
                                                   * (1 * jac6_0
                                                       * (1 * jac7_2
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                               + 1 * jac3_5
                                     * (1 * jac4_3
                                             * (1 * jac5_4
                                                 * (1 * jac6_0
                                                     * (1 * jac7_2
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                         + -1 * jac4_4
                                               * (1 * jac5_3
                                                   * (1 * jac6_0
                                                       * (1 * jac7_2
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                   + -1 * jac1_5
                         * (-1 * jac2_7
                             * (-1 * jac3_3
                                     * (-1 * jac4_4
                                         * (-1 * jac5_1
                                             * (1 * jac6_0
                                                 * (1 * jac7_2
                                                     * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                         + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                         + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                 + 1 * jac3_4
                                       * (-1 * jac4_3
                                           * (-1 * jac5_1
                                               * (1 * jac6_0
                                                   * (1 * jac7_2
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com7_0
        = -1
        * (1 * jac1_0
                * (1 * jac2_1
                    * (-1 * jac3_3
                            * (-1 * jac4_4
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac4_5
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + 1 * jac3_4
                              * (-1 * jac4_3
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (-1 * jac4_3
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8
                                                      * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac1_1
                  * (-1 * jac2_2
                      * (-1 * jac3_3
                              * (-1 * jac4_4
                                      * (-1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_6
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_4
                                * (-1 * jac4_3
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_5
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_5
                                * (-1 * jac4_3
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_4
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + 1 * jac1_2
                  * (-1 * jac2_1
                      * (-1 * jac3_3
                              * (-1 * jac4_4
                                      * (-1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_6
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_4
                                * (-1 * jac4_3
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_5
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_5
                                * (-1 * jac4_3
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_4
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac1_5
                  * (1 * jac2_2
                      * (1 * jac3_3
                              * (1 * jac4_4
                                  * (-1 * jac5_1
                                      * (1 * jac6_0
                                          * (1 * jac7_6
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_4
                                * (1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_6
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com8_0
        = (-1 * jac1_1
                * (1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac4_4
                                    * (1 * jac5_5
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac4_5
                                      * (1 * jac5_4
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_4
                              * (1 * jac4_3
                                      * (1 * jac5_5
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac4_5
                                        * (1 * jac5_3
                                            * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + 1 * jac3_5
                              * (1 * jac4_3
                                      * (1 * jac5_4
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac4_4
                                        * (1 * jac5_3
                                            * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac1_5
                  * (1 * jac2_7
                      * (-1 * jac3_3
                              * (-1 * jac4_4
                                  * (-1 * jac5_1
                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                          + 1 * jac3_4
                                * (-1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com9_0
        = -1
        * (-1 * jac1_1
                * (1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac4_4
                                    * (1 * jac5_5
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                                + -1 * jac4_5
                                      * (1 * jac5_4
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))))
                        + -1 * jac3_4
                              * (1 * jac4_3
                                      * (1 * jac5_5
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                                  + -1 * jac4_5
                                        * (1 * jac5_3
                                            * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))))
                        + 1 * jac3_5
                              * (1 * jac4_3
                                      * (1 * jac5_4
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                                  + -1 * jac4_4
                                        * (1 * jac5_3
                                            * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))))))
            + -1 * jac1_5
                  * (1 * jac2_7
                      * (-1 * jac3_3
                              * (-1 * jac4_4
                                  * (-1 * jac5_1
                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))))
                          + 1 * jac3_4
                                * (-1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))))));
    auto com10_0
        = (-1 * jac1_1
                * (1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac4_4
                                    * (1 * jac5_5
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                                + -1 * jac4_5
                                      * (1 * jac5_4
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))))
                        + -1 * jac3_4
                              * (1 * jac4_3
                                      * (1 * jac5_5
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac4_5
                                        * (1 * jac5_3
                                            * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))))
                        + 1 * jac3_5
                              * (1 * jac4_3
                                      * (1 * jac5_4
                                          * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac4_4
                                        * (1 * jac5_3
                                            * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac1_5
                  * (1 * jac2_7
                      * (-1 * jac3_3
                              * (-1 * jac4_4
                                  * (-1 * jac5_1
                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_4
                                * (-1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))))));
    auto com0_1
        = -1
        * (1 * jac0_1
            * (-1 * jac2_7
                * (-1 * jac3_3
                        * (-1 * jac4_4
                                * (-1 * jac5_5
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac4_5
                                  * (-1 * jac5_4
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                    + 1 * jac3_4
                          * (-1 * jac4_3
                                  * (-1 * jac5_5
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac4_5
                                    * (-1 * jac5_3
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                    + -1 * jac3_5
                          * (-1 * jac4_3
                                  * (-1 * jac5_4
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac4_4
                                    * (-1 * jac5_3
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com1_1
        = (1 * jac0_0
                * (-1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac4_4
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac4_5
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + 1 * jac3_4
                              * (-1 * jac4_3
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (-1 * jac4_3
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8
                                                      * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (-1 * jac2_7
                      * (1 * jac3_3
                              * (1 * jac4_4
                                      * (1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac4_5
                                        * (1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_4
                                * (1 * jac4_3
                                        * (1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac4_5
                                          * (1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_5
                                * (1 * jac4_3
                                        * (1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac4_4
                                          * (1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com2_1 = -1
                * (-1 * jac0_1
                    * (-1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_4
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_5
                                          * (-1 * jac5_4
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                            + 1 * jac3_4
                                  * (-1 * jac4_3
                                          * (-1 * jac5_5
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_3
                                                * (1 * jac6_0
                                                    * (1 * jac7_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                            + -1 * jac3_5
                                  * (-1 * jac4_3
                                          * (-1 * jac5_4
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + 1 * jac4_4
                                            * (-1 * jac5_3
                                                * (1 * jac6_0
                                                    * (1 * jac7_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com3_1
        = (1 * jac0_0
                * (-1 * jac2_7
                    * (1 * jac3_4
                            * (1 * jac4_5
                                * (1 * jac5_1
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (1 * jac4_4
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (-1 * jac2_7
                      * (-1 * jac3_4
                              * (-1 * jac4_5
                                  * (-1 * jac5_1
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_5
                                * (-1 * jac4_4
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com4_1
        = -1
        * (1 * jac0_0
                * (-1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac4_5
                                * (1 * jac5_1
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (1 * jac4_3
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (-1 * jac2_7
                      * (-1 * jac3_3
                              * (-1 * jac4_5
                                  * (-1 * jac5_1
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_5
                                * (-1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com5_1
        = (1 * jac0_0
                * (-1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac4_4
                                * (1 * jac5_1
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_4
                              * (1 * jac4_3
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (-1 * jac2_7
                      * (-1 * jac3_3
                              * (-1 * jac4_4
                                  * (-1 * jac5_1
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_4
                                * (-1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com6_1 = -1
                * (-1 * jac0_1
                    * (-1 * jac2_7
                        * (1 * jac3_3
                                * (1 * jac4_4
                                        * (1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac4_5
                                          * (1 * jac5_4
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                            + -1 * jac3_4
                                  * (1 * jac4_3
                                          * (1 * jac5_5
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac4_5
                                            * (1 * jac5_3
                                                * (1 * jac6_0
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                            + 1 * jac3_5
                                  * (1 * jac4_3
                                          * (1 * jac5_4
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac4_4
                                            * (1 * jac5_3
                                                * (1 * jac6_0
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com7_1
        = (1 * jac0_0
                * (1 * jac2_1
                    * (-1 * jac3_3
                            * (-1 * jac4_4
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac4_5
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + 1 * jac3_4
                              * (-1 * jac4_3
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                        + -1 * jac3_5
                              * (-1 * jac4_3
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8
                                                      * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (-1 * jac2_2
                      * (-1 * jac3_3
                              * (-1 * jac4_4
                                      * (-1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_6
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_4
                                * (-1 * jac4_3
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_5
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_5
                                * (-1 * jac4_3
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_4
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + 1 * jac0_6
                  * (-1 * jac2_1
                      * (1 * jac3_3
                              * (1 * jac4_4
                                      * (1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac4_5
                                        * (1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + -1 * jac3_4
                                * (1 * jac4_3
                                        * (1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac4_5
                                          * (1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                          + 1 * jac3_5
                                * (1 * jac4_3
                                        * (1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac4_4
                                          * (1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com8_1
        = -1
        * (-1 * jac0_1
            * (1 * jac2_7
                * (1 * jac3_3
                        * (1 * jac4_4
                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + -1 * jac4_5
                                  * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                    + -1 * jac3_4
                          * (1 * jac4_3
                                  * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac4_5
                                    * (1 * jac5_3
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                    + 1 * jac3_5
                          * (1 * jac4_3
                                  * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac4_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com9_1
        = (-1 * jac0_1
            * (1 * jac2_7
                * (1 * jac3_3
                        * (1 * jac4_4
                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                            + -1 * jac4_5
                                  * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))))
                    + -1 * jac3_4
                          * (1 * jac4_3
                                  * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                              + -1 * jac4_5
                                    * (1 * jac5_3
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))))
                    + 1 * jac3_5
                          * (1 * jac4_3
                                  * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                              + -1 * jac4_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))))));
    auto com10_1
        = -1
        * (-1 * jac0_1
            * (1 * jac2_7
                * (1 * jac3_3
                        * (1 * jac4_4
                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                            + -1 * jac4_5
                                  * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))))
                    + -1 * jac3_4
                          * (1 * jac4_3
                                  * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac4_5
                                    * (1 * jac5_3
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))))
                    + 1 * jac3_5
                          * (1 * jac4_3
                                  * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac4_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))))));
    auto com0_2 = 0;
    auto com1_2 = -1 * 0;
    auto com2_2 = 0;
    auto com3_2 = -1 * 0;
    auto com4_2 = 0;
    auto com5_2 = -1 * 0;
    auto com6_2 = 0;
    auto com7_2
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac3_3
                                * (-1 * jac4_4
                                        * (-1 * jac5_5
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                    + 1 * jac4_5
                                          * (-1 * jac5_4
                                              * (-1 * jac6_6
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                  + 1 * jac6_8
                                                        * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                            + 1 * jac3_4
                                  * (-1 * jac4_3
                                          * (-1 * jac5_5
                                              * (-1 * jac6_6
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                  + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_3
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                    + 1 * jac6_8
                                                          * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                            + -1 * jac3_5
                                  * (-1 * jac4_3
                                          * (-1 * jac5_4
                                              * (-1 * jac6_6
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                  + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                      + 1 * jac4_4
                                            * (-1 * jac5_3
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                    + 1 * jac6_8
                                                          * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (1 * jac3_3
                                  * (1 * jac4_4
                                      * (1 * jac5_1
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                        * (1 * jac5_1
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8
                                                      * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5
                                              * (-1 * jac6_6
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                  + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                    + 1 * jac6_8
                                                          * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                    + 1 * jac6_8
                                                          * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                      + 1 * jac6_8
                                                            * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                    + 1 * jac6_8
                                                          * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                      + 1 * jac6_8
                                                            * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5
                                                * (1 * jac6_0
                                                    * (1 * jac7_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4
                                                  * (1 * jac6_0
                                                      * (1 * jac7_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5
                                                  * (1 * jac6_0
                                                      * (1 * jac7_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (1 * jac6_0
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4
                                                  * (1 * jac6_0
                                                      * (1 * jac7_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (1 * jac6_0
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (1 * jac3_3
                                  * (1 * jac4_4
                                          * (1 * jac5_5
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4
                                                * (1 * jac6_0
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5
                                                * (1 * jac6_0
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3
                                                  * (1 * jac6_0
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4
                                                * (1 * jac6_0
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (1 * jac6_0
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                      + -1 * jac1_5
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                        * (-1 * jac5_1
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                          * (-1 * jac5_1
                                              * (1 * jac6_0
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com8_2 = 0;
    auto com9_2 = -1 * 0;
    auto com10_2 = 0;
    auto com0_3
        = -1
        * (1 * jac0_1
            * (-1 * jac1_5
                * (1 * jac2_7
                    * (-1 * jac4_3
                            * (-1 * jac5_4
                                * (-1 * jac6_6
                                        * (1 * jac7_2
                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                    + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                        + 1 * jac4_4
                              * (-1 * jac5_3
                                  * (-1 * jac6_6
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com1_3
        = (1 * jac0_0
                * (-1 * jac1_5
                    * (1 * jac2_7
                        * (-1 * jac4_3
                                * (-1 * jac5_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac4_4
                                  * (-1 * jac5_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_5
                      * (1 * jac2_7
                          * (1 * jac4_3
                                  * (1 * jac5_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac4_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com2_3
        = -1
        * (-1 * jac0_1
            * (-1 * jac1_5
                * (1 * jac2_7
                    * (-1 * jac4_3
                            * (-1 * jac5_4
                                * (1 * jac6_0
                                    * (1 * jac7_6
                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10) + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                        + 1 * jac4_4
                              * (-1 * jac5_3
                                  * (1 * jac6_0
                                      * (1 * jac7_6
                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com3_3
        = (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (-1 * jac4_4
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac4_5
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + -1 * jac1_5
                          * (1 * jac2_7
                              * (1 * jac4_4
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac4_4
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_4
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac4_4
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_5
                                          * (-1 * jac5_4
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                          * (1 * jac2_7
                              * (1 * jac4_4
                                      * (1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac4_5
                                        * (1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                      + 1 * jac1_5
                            * (1 * jac2_7
                                * (-1 * jac4_4
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com4_3
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (-1 * jac4_3
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac4_5
                                      * (-1 * jac5_3
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + -1 * jac1_5
                          * (1 * jac2_7
                              * (1 * jac4_3
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac4_3
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_5
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac4_3
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_5
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                          * (1 * jac2_7
                              * (1 * jac4_3
                                      * (1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac4_5
                                        * (1 * jac5_3
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                      + 1 * jac1_5
                            * (1 * jac2_7
                                * (-1 * jac4_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com5_3
        = (1 * jac0_0
                * (1 * jac1_1
                    * (1 * jac2_7
                        * (-1 * jac4_3
                                * (-1 * jac5_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac4_4
                                  * (-1 * jac5_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac4_3
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac4_3
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac4_4
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                      * (1 * jac2_7
                          * (1 * jac4_3
                                  * (1 * jac5_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac4_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com6_3
        = -1
        * (-1 * jac0_1
            * (1 * jac1_5
                * (1 * jac2_7
                    * (1 * jac4_3
                            * (1 * jac5_4
                                * (1 * jac6_0
                                    * (1 * jac7_2
                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10) + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                        + -1 * jac4_4
                              * (1 * jac5_3
                                  * (1 * jac6_0
                                      * (1 * jac7_2
                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com7_3
        = (1 * jac0_0
                * (1 * jac1_5
                    * (1 * jac2_1
                        * (-1 * jac4_3
                                * (-1 * jac5_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac4_4
                                  * (-1 * jac5_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_5
                      * (-1 * jac2_2
                          * (-1 * jac4_3
                                  * (-1 * jac5_4
                                      * (1 * jac6_0
                                          * (1 * jac7_6
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + 1 * jac4_4
                                    * (-1 * jac5_3
                                        * (1 * jac6_0
                                            * (1 * jac7_6
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_5
                      * (-1 * jac2_1
                          * (1 * jac4_3
                                  * (1 * jac5_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac4_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com8_3
        = -1
        * (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac4_3 * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                        + -1 * jac4_4
                              * (1 * jac5_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com9_3
        = (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac4_3 * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                        + -1 * jac4_4
                              * (1 * jac5_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))))));
    auto com10_3
        = -1
        * (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac4_3 * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                        + -1 * jac4_4
                              * (1 * jac5_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))))));
    auto com0_4 = (1 * jac0_1
                   * (-1 * jac1_5
                       * (1 * jac2_7
                           * (-1 * jac3_3
                                   * (-1 * jac5_4
                                       * (-1 * jac6_6
                                               * (1 * jac7_2
                                                   * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                       + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                       + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                           + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                               + 1 * jac3_4
                                     * (-1 * jac5_3
                                         * (-1 * jac6_6
                                                 * (1 * jac7_2
                                                     * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                         + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                         + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                             + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com1_4
        = -1
        * (1 * jac0_0
                * (-1 * jac1_5
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac5_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_4
                                  * (-1 * jac5_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_5
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac5_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com2_4
        = (-1 * jac0_1
            * (-1 * jac1_5
                * (1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac5_4
                                * (1 * jac6_0
                                    * (1 * jac7_6
                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10) + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                        + 1 * jac3_4
                              * (-1 * jac5_3
                                  * (1 * jac6_0
                                      * (1 * jac7_6
                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com3_4
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (-1 * jac3_4
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + -1 * jac1_5
                          * (1 * jac2_7
                              * (1 * jac3_4
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac3_4
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (-1 * jac5_4
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_4
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_5
                                          * (-1 * jac5_4
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                          * (1 * jac2_7
                              * (1 * jac3_4
                                      * (1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac3_5
                                        * (1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                      + 1 * jac1_5
                            * (1 * jac2_7
                                * (-1 * jac3_4
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com4_4
        = (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac5_5
                                        * (-1 * jac6_6
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                            + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (-1 * jac5_3
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + -1 * jac1_5
                          * (1 * jac2_7
                              * (1 * jac3_3
                                  * (1 * jac5_1
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac5_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac5_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_5
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                          * (1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac5_5
                                          * (1 * jac6_0
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac3_5
                                        * (1 * jac5_3
                                            * (1 * jac6_0
                                                * (1 * jac7_2
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                      + 1 * jac1_5
                            * (1 * jac2_7
                                * (-1 * jac3_3
                                    * (-1 * jac5_1
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com5_4
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac5_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_4
                                  * (-1 * jac5_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac5_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac5_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac5_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac5_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac5_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com6_4
        = (-1 * jac0_1
            * (1 * jac1_5
                * (1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac5_4
                                * (1 * jac6_0
                                    * (1 * jac7_2
                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10) + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                        + -1 * jac3_4
                              * (1 * jac5_3
                                  * (1 * jac6_0
                                      * (1 * jac7_2
                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com7_4
        = -1
        * (1 * jac0_0
                * (1 * jac1_5
                    * (1 * jac2_1
                        * (-1 * jac3_3
                                * (-1 * jac5_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_4
                                  * (-1 * jac5_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_5
                      * (-1 * jac2_2
                          * (-1 * jac3_3
                                  * (-1 * jac5_4
                                      * (1 * jac6_0
                                          * (1 * jac7_6
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + 1 * jac3_4
                                    * (-1 * jac5_3
                                        * (1 * jac6_0
                                            * (1 * jac7_6
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_5
                      * (-1 * jac2_1
                          * (1 * jac3_3
                                  * (1 * jac5_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_4
                                    * (1 * jac5_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com8_4
        = (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac3_3 * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                        + -1 * jac3_4
                              * (1 * jac5_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com9_4
        = -1
        * (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac3_3 * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                        + -1 * jac3_4
                              * (1 * jac5_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))))));
    auto com10_4
        = (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac3_3 * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                        + -1 * jac3_4
                              * (1 * jac5_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))))));
    auto com0_5
        = -1
        * (1 * jac0_1
            * (-1 * jac1_5
                * (1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac4_4
                                * (-1 * jac6_6
                                        * (1 * jac7_2
                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                    + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                        + 1 * jac3_4
                              * (-1 * jac4_3
                                  * (-1 * jac6_6
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com1_5
        = (1 * jac0_0
                * (-1 * jac1_5
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_4
                                  * (-1 * jac4_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_5
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com2_5
        = -1
        * (-1 * jac0_1
            * (-1 * jac1_5
                * (1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac4_4
                                * (1 * jac6_0
                                    * (1 * jac7_6
                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10) + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                        + 1 * jac3_4
                              * (-1 * jac4_3
                                  * (1 * jac6_0
                                      * (1 * jac7_6
                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com3_5
        = (1 * jac0_0
                * (1 * jac1_1
                    * (1 * jac2_7
                        * (-1 * jac3_4
                                * (-1 * jac4_5
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (-1 * jac4_4
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac3_4
                                      * (-1 * jac4_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (-1 * jac4_4
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_4
                                        * (-1 * jac4_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_5
                                          * (-1 * jac4_4
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                      * (1 * jac2_7
                          * (1 * jac3_4
                                  * (1 * jac4_5
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_5
                                    * (1 * jac4_4
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com4_5
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_5
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (-1 * jac4_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_5
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (-1 * jac4_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_5
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_5
                                          * (-1 * jac4_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_5
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_5
                                    * (1 * jac4_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com5_5
        = (1 * jac0_0
                * (1 * jac1_1
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_4
                                  * (-1 * jac4_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (-1 * jac6_6
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                              + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (-1 * jac6_6
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                                + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (1 * jac6_0
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (1 * jac6_0
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com6_5
        = -1
        * (-1 * jac0_1
            * (1 * jac1_5
                * (1 * jac2_7
                    * (1 * jac3_3
                            * (1 * jac4_4
                                * (1 * jac6_0
                                    * (1 * jac7_2
                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10) + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                        + -1 * jac3_4
                              * (1 * jac4_3
                                  * (1 * jac6_0
                                      * (1 * jac7_2
                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com7_5
        = (1 * jac0_0
                * (1 * jac1_5
                    * (1 * jac2_1
                        * (-1 * jac3_3
                                * (-1 * jac4_4
                                    * (-1 * jac6_6
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_4
                                  * (-1 * jac4_3
                                      * (-1 * jac6_6
                                              * (1 * jac7_2
                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))
                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_5
                      * (-1 * jac2_2
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                      * (1 * jac6_0
                                          * (1 * jac7_6
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                        * (1 * jac6_0
                                            * (1 * jac7_6
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_5
                      * (-1 * jac2_1
                          * (1 * jac3_3
                                  * (1 * jac4_4
                                      * (1 * jac6_0
                                          * (1 * jac7_2
                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                        * (1 * jac6_0
                                            * (1 * jac7_2
                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com8_5
        = -1
        * (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac3_3 * (1 * jac4_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                        + -1 * jac3_4
                              * (1 * jac4_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com9_5
        = (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac3_3 * (1 * jac4_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                        + -1 * jac3_4
                              * (1 * jac4_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))))));
    auto com10_5
        = -1
        * (-1 * jac0_1
            * (1 * jac1_5
                * (-1 * jac2_7
                    * (1 * jac3_3 * (1 * jac4_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                        + -1 * jac3_4
                              * (1 * jac4_3 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))))));
    auto com0_6 = (1 * jac0_1
                       * (1 * jac1_2
                           * (1 * jac2_7
                               * (1 * jac3_3
                                       * (1 * jac4_4
                                               * (1 * jac5_5
                                                   * (1 * jac7_6
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                           + -1 * jac4_5
                                                 * (1 * jac5_4
                                                     * (1 * jac7_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                   + -1 * jac3_4
                                         * (1 * jac4_3
                                                 * (1 * jac5_5
                                                     * (1 * jac7_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                             + -1 * jac4_5
                                                   * (1 * jac5_3
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                   + 1 * jac3_5
                                         * (1 * jac4_3
                                                 * (1 * jac5_4
                                                     * (1 * jac7_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                             + -1 * jac4_4
                                                   * (1 * jac5_3
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                   + -1 * jac0_6
                         * (1 * jac1_1
                                 * (1 * jac2_7
                                     * (-1 * jac3_3
                                             * (-1 * jac4_4
                                                     * (-1 * jac5_5
                                                         * (1 * jac7_2
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                 + 1 * jac4_5
                                                       * (-1 * jac5_4
                                                           * (1 * jac7_2
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                         + 1 * jac3_4
                                               * (-1 * jac4_3
                                                       * (-1 * jac5_5
                                                           * (1 * jac7_2
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                   + 1 * jac4_5
                                                         * (-1 * jac5_3
                                                             * (1 * jac7_2
                                                                 * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                     + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                     + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                         + -1 * jac3_5
                                               * (-1 * jac4_3
                                                       * (-1 * jac5_4
                                                           * (1 * jac7_2
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                   + 1 * jac4_4
                                                         * (-1 * jac5_3
                                                             * (1 * jac7_2
                                                                 * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                     + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                     + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                             + 1 * jac1_5
                                   * (1 * jac2_7
                                       * (1 * jac3_3
                                               * (1 * jac4_4
                                                   * (1 * jac5_1
                                                       * (1 * jac7_2
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                           + -1 * jac3_4
                                                 * (1 * jac4_3
                                                     * (1 * jac5_1
                                                         * (1 * jac7_2
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com1_6 = -1
                * (1 * jac0_0
                        * (1 * jac1_2
                            * (1 * jac2_7
                                * (1 * jac3_3
                                        * (1 * jac4_4
                                                * (1 * jac5_5
                                                    * (1 * jac7_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_4
                                                      * (1 * jac7_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac3_4
                                          * (1 * jac4_3
                                                  * (1 * jac5_5
                                                      * (1 * jac7_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                              + -1 * jac4_5
                                                    * (1 * jac5_3
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_5
                                          * (1 * jac4_3
                                                  * (1 * jac5_4
                                                      * (1 * jac7_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                              + -1 * jac4_4
                                                    * (1 * jac5_3
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                    + -1 * jac0_6
                          * (1 * jac1_0
                              * (1 * jac2_7
                                  * (-1 * jac3_3
                                          * (-1 * jac4_4
                                                  * (-1 * jac5_5
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_4
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + 1 * jac3_4
                                            * (-1 * jac4_3
                                                    * (-1 * jac5_5
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                + 1 * jac4_5
                                                      * (-1 * jac5_3
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac3_5
                                            * (-1 * jac4_3
                                                    * (-1 * jac5_4
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                + 1 * jac4_4
                                                      * (-1 * jac5_3
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com2_6 = (1 * jac0_0
                       * (1 * jac1_1
                               * (1 * jac2_7
                                   * (1 * jac3_3
                                           * (1 * jac4_4
                                                   * (1 * jac5_5
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                               + -1 * jac4_5
                                                     * (1 * jac5_4
                                                         * (1 * jac7_6
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                       + -1 * jac3_4
                                             * (1 * jac4_3
                                                     * (1 * jac5_5
                                                         * (1 * jac7_6
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                 + -1 * jac4_5
                                                       * (1 * jac5_3
                                                           * (1 * jac7_6
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                       + 1 * jac3_5
                                             * (1 * jac4_3
                                                     * (1 * jac5_4
                                                         * (1 * jac7_6
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                 + -1 * jac4_4
                                                       * (1 * jac5_3
                                                           * (1 * jac7_6
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                           + -1 * jac1_5
                                 * (1 * jac2_7
                                     * (-1 * jac3_3
                                             * (-1 * jac4_4
                                                 * (1 * jac5_1
                                                     * (1 * jac7_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                         + 1 * jac3_4
                                               * (-1 * jac4_3
                                                   * (1 * jac5_1
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                   + -1 * jac0_1
                         * (1 * jac1_0
                             * (1 * jac2_7
                                 * (1 * jac3_3
                                         * (1 * jac4_4
                                                 * (1 * jac5_5
                                                     * (1 * jac7_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                             + -1 * jac4_5
                                                   * (1 * jac5_4
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + -1 * jac3_4
                                           * (1 * jac4_3
                                                   * (1 * jac5_5
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                               + -1 * jac4_5
                                                     * (1 * jac5_3
                                                         * (1 * jac7_6
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + 1 * jac3_5
                                           * (1 * jac4_3
                                                   * (1 * jac5_4
                                                       * (1 * jac7_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                               + -1 * jac4_4
                                                     * (1 * jac5_3
                                                         * (1 * jac7_6
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com3_6 = -1
                * (1 * jac0_0
                        * (-1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_4
                                        * (-1 * jac4_5
                                            * (1 * jac5_1
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_5
                                          * (-1 * jac4_4
                                              * (1 * jac5_1
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                    + -1 * jac0_6
                          * (1 * jac1_0
                              * (1 * jac2_7
                                  * (1 * jac3_4
                                          * (1 * jac4_5
                                              * (1 * jac5_1
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac3_5
                                            * (1 * jac4_4
                                                * (1 * jac5_1
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com4_6 = (1 * jac0_0
                       * (-1 * jac1_2
                           * (1 * jac2_7
                               * (-1 * jac3_3
                                       * (-1 * jac4_5
                                           * (1 * jac5_1
                                               * (1 * jac7_6
                                                   * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                       + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                       + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                   + 1 * jac3_5
                                         * (-1 * jac4_3
                                             * (1 * jac5_1
                                                 * (1 * jac7_6
                                                     * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                         + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                         + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                   + -1 * jac0_6
                         * (1 * jac1_0
                             * (1 * jac2_7
                                 * (1 * jac3_3
                                         * (1 * jac4_5
                                             * (1 * jac5_1
                                                 * (1 * jac7_2
                                                     * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                         + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                         + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + -1 * jac3_5
                                           * (1 * jac4_3
                                               * (1 * jac5_1
                                                   * (1 * jac7_2
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com5_6 = -1
                * (1 * jac0_0
                        * (-1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (1 * jac5_1
                                                * (1 * jac7_6
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (1 * jac5_1
                                                  * (1 * jac7_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                    + -1 * jac0_6
                          * (1 * jac1_0
                              * (1 * jac2_7
                                  * (1 * jac3_3
                                          * (1 * jac4_4
                                              * (1 * jac5_1
                                                  * (1 * jac7_2
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac3_4
                                            * (1 * jac4_3
                                                * (1 * jac5_1
                                                    * (1 * jac7_2
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com6_6 = (1 * jac0_0
                       * (1 * jac1_1
                               * (1 * jac2_7
                                   * (-1 * jac3_3
                                           * (-1 * jac4_4
                                                   * (-1 * jac5_5
                                                       * (1 * jac7_2
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                               + 1 * jac4_5
                                                     * (-1 * jac5_4
                                                         * (1 * jac7_2
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                       + 1 * jac3_4
                                             * (-1 * jac4_3
                                                     * (-1 * jac5_5
                                                         * (1 * jac7_2
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                 + 1 * jac4_5
                                                       * (-1 * jac5_3
                                                           * (1 * jac7_2
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                       + -1 * jac3_5
                                             * (-1 * jac4_3
                                                     * (-1 * jac5_4
                                                         * (1 * jac7_2
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                 + 1 * jac4_4
                                                       * (-1 * jac5_3
                                                           * (1 * jac7_2
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                           + 1 * jac1_5
                                 * (1 * jac2_7
                                     * (1 * jac3_3
                                             * (1 * jac4_4
                                                 * (1 * jac5_1
                                                     * (1 * jac7_2
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                         + -1 * jac3_4
                                               * (1 * jac4_3
                                                   * (1 * jac5_1
                                                       * (1 * jac7_2
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                   + -1 * jac0_1
                         * (1 * jac1_0
                             * (1 * jac2_7
                                 * (-1 * jac3_3
                                         * (-1 * jac4_4
                                                 * (-1 * jac5_5
                                                     * (1 * jac7_2
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                             + 1 * jac4_5
                                                   * (-1 * jac5_4
                                                       * (1 * jac7_2
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + 1 * jac3_4
                                           * (-1 * jac4_3
                                                   * (-1 * jac5_5
                                                       * (1 * jac7_2
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                               + 1 * jac4_5
                                                     * (-1 * jac5_3
                                                         * (1 * jac7_2
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + -1 * jac3_5
                                           * (-1 * jac4_3
                                                   * (-1 * jac5_4
                                                       * (1 * jac7_2
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                               + 1 * jac4_4
                                                     * (-1 * jac5_3
                                                         * (1 * jac7_2
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com7_6 = -1
                * (1 * jac0_0
                        * (1 * jac1_1
                                * (1 * jac2_2
                                    * (1 * jac3_3
                                            * (1 * jac4_4
                                                    * (1 * jac5_5
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                + -1 * jac4_5
                                                      * (1 * jac5_4
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                        + -1 * jac3_4
                                              * (1 * jac4_3
                                                      * (1 * jac5_5
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                  + -1 * jac4_5
                                                        * (1 * jac5_3
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                        + 1 * jac3_5
                                              * (1 * jac4_3
                                                      * (1 * jac5_4
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                  + -1 * jac4_4
                                                        * (1 * jac5_3
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                            + -1 * jac1_2
                                  * (1 * jac2_1
                                      * (1 * jac3_3
                                              * (1 * jac4_4
                                                      * (1 * jac5_5
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                  + -1 * jac4_5
                                                        * (1 * jac5_4
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                          + -1 * jac3_4
                                                * (1 * jac4_3
                                                        * (1 * jac5_5
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                    + -1 * jac4_5
                                                          * (1 * jac5_3
                                                              * (1 * jac7_6
                                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                          + 1 * jac3_5
                                                * (1 * jac4_3
                                                        * (1 * jac5_4
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                    + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                    + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                    + -1 * jac4_4
                                                          * (1 * jac5_3
                                                              * (1 * jac7_6
                                                                  * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                      + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                      + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                            + 1 * jac1_5
                                  * (-1 * jac2_2
                                      * (-1 * jac3_3
                                              * (-1 * jac4_4
                                                  * (1 * jac5_1
                                                      * (1 * jac7_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                          + 1 * jac3_4
                                                * (-1 * jac4_3
                                                    * (1 * jac5_1
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                    + -1 * jac0_1
                          * (1 * jac1_0
                              * (1 * jac2_2
                                  * (1 * jac3_3
                                          * (1 * jac4_4
                                                  * (1 * jac5_5
                                                      * (1 * jac7_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                              + -1 * jac4_5
                                                    * (1 * jac5_4
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac3_4
                                            * (1 * jac4_3
                                                    * (1 * jac5_5
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                + -1 * jac4_5
                                                      * (1 * jac5_3
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + 1 * jac3_5
                                            * (1 * jac4_3
                                                    * (1 * jac5_4
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                + -1 * jac4_4
                                                      * (1 * jac5_3
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))))))
                    + 1 * jac0_6
                          * (1 * jac1_0
                              * (1 * jac2_1
                                  * (-1 * jac3_3
                                          * (-1 * jac4_4
                                                  * (-1 * jac5_5
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_4
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + 1 * jac3_4
                                            * (-1 * jac4_3
                                                    * (-1 * jac5_5
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                + 1 * jac4_5
                                                      * (-1 * jac5_3
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac3_5
                                            * (-1 * jac4_3
                                                    * (-1 * jac5_4
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                                + 1 * jac4_4
                                                      * (-1 * jac5_3
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com8_6
        = (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (-1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4 * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com9_6
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (-1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4 * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (1 * jac7_2 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))))));
    auto com10_6
        = (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (-1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4 * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (1 * jac7_2 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))))));
    auto com0_7 = -1
                * (1 * jac0_1
                    * (1 * jac1_2
                        * (1 * jac2_7
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com1_7 = (1 * jac0_0
                       * (1 * jac1_2
                           * (1 * jac2_7
                               * (1 * jac3_3
                                       * (1 * jac4_4
                                               * (1 * jac5_5
                                                   * (1 * jac6_6
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                       + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                           + -1 * jac4_5
                                                 * (1 * jac5_4
                                                     * (1 * jac6_6
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                         + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                   + -1 * jac3_4
                                         * (1 * jac4_3
                                                 * (1 * jac5_5
                                                     * (1 * jac6_6
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                         + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                             + -1 * jac4_5
                                                   * (1 * jac5_3
                                                       * (1 * jac6_6
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                           + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                   + 1 * jac3_5
                                         * (1 * jac4_3
                                                 * (1 * jac5_4
                                                     * (1 * jac6_6
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                         + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                             + -1 * jac4_4
                                                   * (1 * jac5_3
                                                       * (1 * jac6_6
                                                               * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                   + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                   + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                           + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
                   + -1 * jac0_6
                         * (-1 * jac1_2
                             * (1 * jac2_7
                                 * (-1 * jac3_3
                                         * (-1 * jac4_4
                                                 * (-1 * jac5_5
                                                     * (1 * jac6_0
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                             + 1 * jac4_5
                                                   * (-1 * jac5_4
                                                       * (1 * jac6_0
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + 1 * jac3_4
                                           * (-1 * jac4_3
                                                   * (-1 * jac5_5
                                                       * (1 * jac6_0
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                               + 1 * jac4_5
                                                     * (-1 * jac5_3
                                                         * (1 * jac6_0
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + -1 * jac3_5
                                           * (-1 * jac4_3
                                                   * (-1 * jac5_4
                                                       * (1 * jac6_0
                                                           * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                               + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                               + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                               + 1 * jac4_4
                                                     * (-1 * jac5_3
                                                         * (1 * jac6_0
                                                             * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                 + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                 + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com2_7
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + -1 * jac1_5
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (1 * jac5_1
                                              * (1 * jac6_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                  + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (1 * jac5_1
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4
                                          * (1 * jac5_5
                                              * (1 * jac6_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                  + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (-1 * jac1_1
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5
                                                  * (1 * jac6_0
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                      + 1 * jac1_5
                            * (1 * jac2_7
                                * (1 * jac3_3
                                        * (1 * jac4_4
                                            * (-1 * jac5_1
                                                * (1 * jac6_0
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac3_4
                                          * (1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com3_7 = (1 * jac0_0
                       * (-1 * jac1_2
                           * (1 * jac2_7
                               * (-1 * jac3_4
                                       * (-1 * jac4_5
                                           * (1 * jac5_1
                                               * (1 * jac6_6
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                   + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                   + 1 * jac3_5
                                         * (-1 * jac4_4
                                             * (1 * jac5_1
                                                 * (1 * jac6_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                     + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
                   + -1 * jac0_6
                         * (1 * jac1_2
                             * (1 * jac2_7
                                 * (1 * jac3_4
                                         * (1 * jac4_5
                                             * (-1 * jac5_1
                                                 * (1 * jac6_0
                                                     * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                         + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                         + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + -1 * jac3_5
                                           * (1 * jac4_4
                                               * (-1 * jac5_1
                                                   * (1 * jac6_0
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com4_7 = -1
                * (1 * jac0_0
                        * (-1 * jac1_2
                            * (1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_5
                                            * (1 * jac5_1
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                    + 1 * jac3_5
                                          * (-1 * jac4_3
                                              * (1 * jac5_1
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
                    + -1 * jac0_6
                          * (1 * jac1_2
                              * (1 * jac2_7
                                  * (1 * jac3_3
                                          * (1 * jac4_5
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                      + -1 * jac3_5
                                            * (1 * jac4_3
                                                * (-1 * jac5_1
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com5_7 = (1 * jac0_0
                       * (-1 * jac1_2
                           * (1 * jac2_7
                               * (-1 * jac3_3
                                       * (-1 * jac4_4
                                           * (1 * jac5_1
                                               * (1 * jac6_6
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                   + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                   + 1 * jac3_4
                                         * (-1 * jac4_3
                                             * (1 * jac5_1
                                                 * (1 * jac6_6
                                                         * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                             + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                             + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                     + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
                   + -1 * jac0_6
                         * (1 * jac1_2
                             * (1 * jac2_7
                                 * (1 * jac3_3
                                         * (1 * jac4_4
                                             * (-1 * jac5_1
                                                 * (1 * jac6_0
                                                     * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                         + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                         + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                     + -1 * jac3_4
                                           * (1 * jac4_3
                                               * (-1 * jac5_1
                                                   * (1 * jac6_0
                                                       * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                           + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                           + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com6_7 = -1
                * (-1 * jac0_1
                    * (-1 * jac1_2
                        * (1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5
                                                * (1 * jac6_0
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4
                                                  * (1 * jac6_0
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5
                                                  * (1 * jac6_0
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4
                                                  * (1 * jac6_0
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com7_7
        = (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_2
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + -1 * jac1_2
                          * (1 * jac2_1
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                          + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4
                                                    * (1 * jac6_6
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                        + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_6
                                                              * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                  + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                  + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                          + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_2
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (1 * jac5_1
                                              * (1 * jac6_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                  + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (1 * jac5_1
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_2
                          * (1 * jac3_3
                                  * (1 * jac4_4
                                          * (1 * jac5_5
                                              * (1 * jac6_6
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                  + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4
                                                * (1 * jac6_6
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                    + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (1 * jac6_6
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))
                                                      + -1 * jac6_8 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_2
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5
                                                  * (1 * jac6_0
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                      + 1 * jac1_2
                            * (-1 * jac2_1
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                            + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                            + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0
                                                          * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                              + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                              + 1 * jac8_10 * (-1 * jac9_9 * jac10_8))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0
                                                            * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                                + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                                + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))
                      + -1 * jac1_5
                            * (1 * jac2_2
                                * (1 * jac3_3
                                        * (1 * jac4_4
                                            * (-1 * jac5_1
                                                * (1 * jac6_0
                                                    * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                        + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                        + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac3_4
                                          * (1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0
                                                      * (1 * jac8_8 * (1 * jac9_9 * jac10_10)
                                                          + -1 * jac8_9 * (1 * jac9_8 * jac10_10)
                                                          + 1 * jac8_10 * (-1 * jac9_9 * jac10_8)))))))));
    auto com8_7
        = -1
        * (-1 * jac0_1
            * (-1 * jac1_2
                * (-1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac4_4 * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                + 1 * jac4_5 * (-1 * jac5_4 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                        + 1 * jac3_4
                              * (-1 * jac4_3 * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                  + 1 * jac4_5 * (-1 * jac5_3 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))
                        + -1 * jac3_5
                              * (-1 * jac4_3 * (-1 * jac5_4 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_9 * jac10_10))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com9_7
        = (-1 * jac0_1
            * (-1 * jac1_2
                * (-1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac4_4 * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                + 1 * jac4_5 * (-1 * jac5_4 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                        + 1 * jac3_4
                              * (-1 * jac4_3 * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                  + 1 * jac4_5 * (-1 * jac5_3 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))
                        + -1 * jac3_5
                              * (-1 * jac4_3 * (-1 * jac5_4 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_8 * jac10_10))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3 * (1 * jac6_0 * (1 * jac8_6 * (1 * jac9_8 * jac10_10)))))))));
    auto com10_7
        = -1
        * (-1 * jac0_1
            * (-1 * jac1_2
                * (-1 * jac2_7
                    * (-1 * jac3_3
                            * (-1 * jac4_4 * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                + 1 * jac4_5 * (-1 * jac5_4 * (1 * jac6_0 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                        + 1 * jac3_4
                              * (-1 * jac4_3 * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                  + 1 * jac4_5 * (-1 * jac5_3 * (1 * jac6_0 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))
                        + -1 * jac3_5
                              * (-1 * jac4_3 * (-1 * jac5_4 * (1 * jac6_0 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8))))
                                  + 1 * jac4_4
                                        * (-1 * jac5_3 * (1 * jac6_0 * (1 * jac8_6 * (-1 * jac9_9 * jac10_8)))))))));
    auto com0_8
        = (1 * jac0_1
                * (1 * jac1_2
                    * (1 * jac2_7
                        * (1 * jac3_3
                                * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                    + -1 * jac4_5
                                          * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                            + -1 * jac3_4
                                  * (1 * jac4_3 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (1 * jac4_3 * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                      + -1 * jac4_4
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_1
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))
                      + 1 * jac1_5
                            * (1 * jac2_7
                                * (1 * jac3_3
                                        * (1 * jac4_4
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                    + -1 * jac3_4
                                          * (1 * jac4_3
                                              * (1 * jac5_1
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))));
    auto com1_8
        = -1
        * (1 * jac0_0
                * (1 * jac1_2
                    * (1 * jac2_7
                        * (1 * jac3_3
                                * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                    + -1 * jac4_5
                                          * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                            + -1 * jac3_4
                                  * (1 * jac4_3 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (1 * jac4_3 * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                      + -1 * jac4_4
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))));
    auto com2_8
        = (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))))
                    + -1 * jac1_5
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))))));
    auto com3_8
        = -1
        * (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_4
                                * (-1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (-1 * jac4_4
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_4
                                  * (1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (1 * jac4_4
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))));
    auto com4_8
        = (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (-1 * jac4_3
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (1 * jac4_3
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))));
    auto com5_8
        = -1
        * (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_4 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                            + 1 * jac3_4
                                  * (-1 * jac4_3
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))));
    auto com6_8
        = (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))));
    auto com7_8
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_2
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))))
                    + -1 * jac1_2
                          * (1 * jac2_1
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_2
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_2
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + 1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_1
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))));
    auto com8_8
        = (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_9 * jac10_10)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_9 * jac10_10))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_9 * jac10_10))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_9 * jac10_10)))))))));
    auto com9_8
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_8 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_8 * jac10_10)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_8 * jac10_10))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_8 * jac10_10)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_8 * jac10_10))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (1 * jac9_8 * jac10_10))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_8 * jac10_10))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0 * (1 * jac7_2 * (1 * jac9_8 * jac10_10)))))))));
    auto com10_8
        = (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac9_9 * jac10_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac9_9 * jac10_8))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac9_9 * jac10_8)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac9_9 * jac10_8))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac9_9 * jac10_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac9_9 * jac10_8)))))))));
    auto com0_9
        = -1
        * (1 * jac0_1
                * (1 * jac1_2
                    * (1 * jac2_7
                        * (1 * jac3_3
                                * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                    + -1 * jac4_5
                                          * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                            + -1 * jac3_4
                                  * (1 * jac4_3 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (1 * jac4_3 * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                      + -1 * jac4_4
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_1
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))
                      + 1 * jac1_5
                            * (1 * jac2_7
                                * (1 * jac3_3
                                        * (1 * jac4_4
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                    + -1 * jac3_4
                                          * (1 * jac4_3
                                              * (1 * jac5_1
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))));
    auto com1_9
        = (1 * jac0_0
                * (1 * jac1_2
                    * (1 * jac2_7
                        * (1 * jac3_3
                                * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                    + -1 * jac4_5
                                          * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                            + -1 * jac3_4
                                  * (1 * jac4_3 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (1 * jac4_3 * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                      + -1 * jac4_4
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))));
    auto com2_9
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))))
                    + -1 * jac1_5
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))))));
    auto com3_9
        = (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_4
                                * (-1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (-1 * jac4_4
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_4
                                  * (1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (1 * jac4_4
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))));
    auto com4_9
        = -1
        * (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                            + 1 * jac3_5
                                  * (-1 * jac4_3
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (1 * jac4_3
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))));
    auto com5_9
        = (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_4 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                            + 1 * jac3_4
                                  * (-1 * jac4_3
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))));
    auto com6_9
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))));
    auto com7_9
        = (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_2
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))))
                    + -1 * jac1_2
                          * (1 * jac2_1
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_2
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_2
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + 1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_1
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))));
    auto com8_9
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (1 * jac8_9 * jac10_10)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (1 * jac8_9 * jac10_10))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_9 * jac10_10))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1 * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0 * (1 * jac7_2 * (1 * jac8_9 * jac10_10)))))))));
    auto com9_9
        = (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                    + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1
                                              * (-1 * jac6_6
                                                      * (1 * jac7_2 * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                  + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                    + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6
                                                              * (1 * jac7_2
                                                                  * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6
                                                              * (1 * jac7_2
                                                                  * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))
                                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac10_10)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac7_6
                                                            * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * jac10_10
                                                                    + -1 * jac8_10 * jac10_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5
                                                  * (1 * jac6_0
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2
                                                            * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1
                                                * (1 * jac6_0
                                                    * (1 * jac7_2 * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * jac10_10 + -1 * jac8_10 * jac10_8)))))))));
    auto com10_9
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_9 * jac10_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_9 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_9 * jac10_8))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_9 * jac10_8)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_9 * jac10_8))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_9 * jac10_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_9 * jac10_8)))))))));
    auto com0_10
        = (1 * jac0_1
                * (1 * jac1_2
                    * (1 * jac2_7
                        * (1 * jac3_3
                                * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                    + -1 * jac4_5
                                          * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                            + -1 * jac3_4
                                  * (1 * jac4_3 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                      + -1 * jac4_5
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                            + 1 * jac3_5
                                  * (1 * jac4_3 * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                      + -1 * jac4_4
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_6
                  * (1 * jac1_1
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))
                      + 1 * jac1_5
                            * (1 * jac2_7
                                * (1 * jac3_3
                                        * (1 * jac4_4
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                    + -1 * jac3_4
                                          * (1 * jac4_3
                                              * (1 * jac5_1
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))));
    auto com1_10
        = -1
        * (1 * jac0_0
                * (1 * jac1_2
                    * (1 * jac2_7
                        * (1 * jac3_3
                                * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                    + -1 * jac4_5
                                          * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                            + -1 * jac3_4
                                  * (1 * jac4_3 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                      + -1 * jac4_5
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                            + 1 * jac3_5
                                  * (1 * jac4_3 * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                      + -1 * jac4_4
                                            * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))));
    auto com2_10
        = (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))))
                    + -1 * jac1_5
                          * (1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))))));
    auto com3_10
        = -1
        * (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_4
                                * (-1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                            + 1 * jac3_5
                                  * (-1 * jac4_4
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_4
                                  * (1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + -1 * jac3_5
                                    * (1 * jac4_4
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))));
    auto com4_10
        = (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                            + 1 * jac3_5
                                  * (-1 * jac4_3
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_5 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + -1 * jac3_5
                                    * (1 * jac4_3
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))));
    auto com5_10
        = -1
        * (1 * jac0_0
                * (-1 * jac1_2
                    * (1 * jac2_7
                        * (-1 * jac3_3
                                * (-1 * jac4_4 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                            + 1 * jac3_4
                                  * (-1 * jac4_3
                                      * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                        * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))));
    auto com6_10
        = (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))
                    + 1 * jac1_5
                          * (1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_7
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))));
    auto com7_10
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (1 * jac2_2
                            * (1 * jac3_3
                                    * (1 * jac4_4
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                        + -1 * jac4_5
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                + -1 * jac3_4
                                      * (1 * jac4_3
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                          + -1 * jac4_5
                                                * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                + 1 * jac3_5
                                      * (1 * jac4_3
                                              * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                          + -1 * jac4_4
                                                * (1 * jac5_3
                                                    * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))))
                    + -1 * jac1_2
                          * (1 * jac2_1
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_2
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                      * (1 * jac2_2
                          * (1 * jac3_3
                                  * (1 * jac4_4 * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                      + -1 * jac4_5
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                              + -1 * jac3_4
                                    * (1 * jac4_3
                                            * (1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                        + -1 * jac4_5
                                              * (1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                              + 1 * jac3_5
                                    * (1 * jac4_3
                                            * (1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                        + -1 * jac4_4
                                              * (1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + 1 * jac0_6
                  * (1 * jac1_0
                      * (1 * jac2_1
                          * (-1 * jac3_3
                                  * (-1 * jac4_4
                                          * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                      + 1 * jac4_5
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + 1 * jac3_4
                                    * (-1 * jac4_3
                                            * (-1 * jac5_5 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_3 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                              + -1 * jac3_5
                                    * (-1 * jac4_3
                                            * (-1 * jac5_4 * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                        + 1 * jac4_4
                                              * (-1 * jac5_3
                                                  * (-1 * jac6_8 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))));
    auto com8_10
        = (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_9))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_9)))))))));
    auto com9_10
        = -1
        * (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4 * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5 * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_8))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_8)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_8))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0 * (1 * jac7_6 * (-1 * jac8_10 * jac9_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1 * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0 * (1 * jac7_2 * (-1 * jac8_10 * jac9_8)))))))));
    auto com10_10
        = (1 * jac0_0
                * (1 * jac1_1
                        * (-1 * jac2_7
                            * (-1 * jac3_3
                                    * (-1 * jac4_4
                                            * (-1 * jac5_5
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                    + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9))))
                                        + 1 * jac4_5
                                              * (-1 * jac5_4
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9)))))
                                + 1 * jac3_4
                                      * (-1 * jac4_3
                                              * (-1 * jac5_5
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9)))))
                                + -1 * jac3_5
                                      * (-1 * jac4_3
                                              * (-1 * jac5_4
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9))))
                                          + 1 * jac4_4
                                                * (-1 * jac5_3
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9)))))))
                    + 1 * jac1_5
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                          * (1 * jac5_1
                                              * (-1 * jac6_6
                                                      * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                  + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                            * (1 * jac5_1
                                                * (-1 * jac6_6
                                                        * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                    + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9))))))))
            + -1 * jac0_1
                  * (1 * jac1_0
                          * (-1 * jac2_7
                              * (-1 * jac3_3
                                      * (-1 * jac4_4
                                              * (-1 * jac5_5
                                                  * (-1 * jac6_6
                                                          * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                      + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9))))
                                          + 1 * jac4_5
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9)))))
                                  + 1 * jac3_4
                                        * (-1 * jac4_3
                                                * (-1 * jac5_5
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6
                                                              * (1 * jac7_2
                                                                  * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9)))))
                                  + -1 * jac3_5
                                        * (-1 * jac4_3
                                                * (-1 * jac5_4
                                                    * (-1 * jac6_6
                                                            * (1 * jac7_2
                                                                * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                        + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9))))
                                            + 1 * jac4_4
                                                  * (-1 * jac5_3
                                                      * (-1 * jac6_6
                                                              * (1 * jac7_2
                                                                  * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))
                                                          + 1 * jac6_8 * (1 * jac7_2 * (1 * jac8_6 * jac9_9)))))))
                      + -1 * jac1_2
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                                * (-1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac7_6 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))))
                                            + 1 * jac4_5
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_5
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))))
                                              + 1 * jac4_5
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8)))))
                                    + -1 * jac3_5
                                          * (-1 * jac4_3
                                                  * (-1 * jac5_4
                                                      * (1 * jac6_0
                                                          * (1 * jac7_6
                                                              * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))))
                                              + 1 * jac4_4
                                                    * (-1 * jac5_3
                                                        * (1 * jac6_0
                                                            * (1 * jac7_6
                                                                * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))))))))
            + 1 * jac0_6
                  * (-1 * jac1_1
                          * (-1 * jac2_7
                              * (1 * jac3_3
                                      * (1 * jac4_4
                                              * (1 * jac5_5
                                                  * (1 * jac6_0
                                                      * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))))
                                          + -1 * jac4_5
                                                * (1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8)))))
                                  + -1 * jac3_4
                                        * (1 * jac4_3
                                                * (1 * jac5_5
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))))
                                            + -1 * jac4_5
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8)))))
                                  + 1 * jac3_5
                                        * (1 * jac4_3
                                                * (1 * jac5_4
                                                    * (1 * jac6_0
                                                        * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8))))
                                            + -1 * jac4_4
                                                  * (1 * jac5_3
                                                      * (1 * jac6_0
                                                          * (1 * jac7_2
                                                              * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8)))))))
                      + -1 * jac1_5
                            * (-1 * jac2_7
                                * (-1 * jac3_3
                                        * (-1 * jac4_4
                                            * (-1 * jac5_1
                                                * (1 * jac6_0
                                                    * (1 * jac7_2 * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8)))))
                                    + 1 * jac3_4
                                          * (-1 * jac4_3
                                              * (-1 * jac5_1
                                                  * (1 * jac6_0
                                                      * (1 * jac7_2
                                                          * (1 * jac8_8 * jac9_9 + -1 * jac8_9 * jac9_8)))))))));
    Eigen::Matrix<DataType, 11, 11> cojacobian(Eigen::Matrix<DataType, 11, 11>::Zero());

    cojacobian << com0_0, com0_1, com0_2, com0_3, com0_4, com0_5, com0_6, com0_7, com0_8, com0_9, com0_10, com1_0,
        com1_1, com1_2, com1_3, com1_4, com1_5, com1_6, com1_7, com1_8, com1_9, com1_10, com2_0, com2_1, com2_2, com2_3,
        com2_4, com2_5, com2_6, com2_7, com2_8, com2_9, com2_10, com3_0, com3_1, com3_2, com3_3, com3_4, com3_5, com3_6,
        com3_7, com3_8, com3_9, com3_10, com4_0, com4_1, com4_2, com4_3, com4_4, com4_5, com4_6, com4_7, com4_8, com4_9,
        com4_10, com5_0, com5_1, com5_2, com5_3, com5_4, com5_5, com5_6, com5_7, com5_8, com5_9, com5_10, com6_0,
        com6_1, com6_2, com6_3, com6_4, com6_5, com6_6, com6_7, com6_8, com6_9, com6_10, com7_0, com7_1, com7_2, com7_3,
        com7_4, com7_5, com7_6, com7_7, com7_8, com7_9, com7_10, com8_0, com8_1, com8_2, com8_3, com8_4, com8_5, com8_6,
        com8_7, com8_8, com8_9, com8_10, com9_0, com9_1, com9_2, com9_3, com9_4, com9_5, com9_6, com9_7, com9_8, com9_9,
        com9_10, com10_0, com10_1, com10_2, com10_3, com10_4, com10_5, com10_6, com10_7, com10_8, com10_9, com10_10;
    Eigen::Matrix<DataType, 11, 1> delta = cojacobian * eqs * invdet;

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
