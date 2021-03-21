/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include "JuceHeader.h"
#include "PluginProcessor.h"

#include <ATKJUCEComponents/JUCE/ImageLookAndFeel.h>
#include <ATKJUCEComponents/JUCE/Slider.h>

//==============================================================================
/**
 */
class AndromedaAudioProcessorEditor: public juce::AudioProcessorEditor
{
public:
  AndromedaAudioProcessorEditor(AndromedaAudioProcessor&, juce::AudioProcessorValueTreeState& paramState);
  ~AndromedaAudioProcessorEditor() override;

  //==============================================================================
  void paint(juce::Graphics&) override;
  void resized() override;

private:
  // This reference is provided as a quick way for your editor to
  // access the processor object that created it.
  AndromedaAudioProcessor& processor;
  juce::AudioProcessorValueTreeState& paramState;

  ATK::juce::ImageLookAndFeel knob;
  juce::Image bckgndImage;

  ATK::juce::SliderComponent distLevel;
  ATK::juce::SliderComponent tone;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AndromedaAudioProcessorEditor)
};
