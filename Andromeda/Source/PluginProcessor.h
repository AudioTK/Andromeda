/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include <atk_core/atk_core.h>
#include <atk_eq/atk_eq.h>
#include <atk_tools/atk_tools.h>

#include <memory>

//==============================================================================
/**
 */

class AndromedaAudioProcessor: public juce::AudioProcessor
{
public:
  //==============================================================================
  AndromedaAudioProcessor();
  ~AndromedaAudioProcessor() override;

  //==============================================================================
  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
  bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

  void processBlock(juce::AudioSampleBuffer&, juce::MidiBuffer&) override;

  //==============================================================================
  juce::AudioProcessorEditor* createEditor() override;
  bool hasEditor() const override;

  //==============================================================================
  const juce::String getName() const override;

  bool isMidiEffect() const override;
  bool acceptsMidi() const override;
  bool producesMidi() const override;
  double getTailLengthSeconds() const override;

  //==============================================================================
  int getNumPrograms() override;
  int getCurrentProgram() override;
  void setCurrentProgram(int index) override;
  const juce::String getProgramName(int index) override;
  void changeProgramName(int index, const juce::String& newName) override;

  //==============================================================================
  void getStateInformation(juce::MemoryBlock& destData) override;
  void setStateInformation(const void* data, int sizeInBytes) override;

private:
  static constexpr int OVERSAMPLING = 8;
  //==============================================================================
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AndromedaAudioProcessor)

  ATK::InPointerFilter<float> inFilter;
  std::unique_ptr<ATK::ModellerFilter<double>> bandPassFilter;
  ATK::OversamplingFilter<double, ATK::Oversampling6points5order_8<double>> oversamplingFilter;
  std::unique_ptr<ATK::ModellerFilter<double>> distortionFilter;
  std::unique_ptr<ATK::ModellerFilter<double>> toneShapingOverdriveFilter;
  ATK::IIRFilter<ATK::ButterworthLowPassCoefficients<double>> lowpassFilter;
  ATK::DecimationFilter<double> decimationFilter;
  std::unique_ptr<ATK::ModellerFilter<double>> lowPass2Filter;
  std::unique_ptr<ATK::ModellerFilter<double>> bandPass2Filter;
  ATK::OutPointerFilter<float> outFilter;

  juce::AudioProcessorValueTreeState parameters;
  long sampleRate;
  int lastParameterSet;

  float old_distLevel{1000};
  float old_tone{1000};
};
