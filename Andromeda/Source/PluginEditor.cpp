/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginEditor.h"
#include "PluginProcessor.h"

//==============================================================================
AndromedaAudioProcessorEditor::AndromedaAudioProcessorEditor(AndromedaAudioProcessor& p,
    juce::AudioProcessorValueTreeState& paramState)
  : AudioProcessorEditor(&p)
  , processor(p)
  , paramState(paramState)
  , knob(juce::ImageFileFormat::loadFrom(BinaryData::KNB_Pitt_L_png, BinaryData::KNB_Pitt_L_pngSize), 55, 55, 101)
  , distLevel(paramState, "distLevel", "Distortion", &knob)
  , tone(paramState, "tone", "Tone", &knob)

{
  addAndMakeVisible(distLevel);
  addAndMakeVisible(tone);

  bckgndImage = juce::ImageFileFormat::loadFrom(BinaryData::background_jpg, BinaryData::background_jpgSize);

  // Make sure that before the constructor has finished, you've set the
  // editor's size to whatever you need it to be.
  setSize(200, 133);
}

AndromedaAudioProcessorEditor::~AndromedaAudioProcessorEditor() = default;

void AndromedaAudioProcessorEditor::paint(juce::Graphics& g)
{
  g.drawImageAt(bckgndImage, 0, 0);
  g.setFont(juce::Font("Times New Roman", 30.0f, juce::Font::bold | juce::Font::italic));
  g.setColour(juce::Colours::whitesmoke);
  g.drawText("Andromeda", 20, 10, 200, 30, juce::Justification::verticallyCentred);
  g.setFont(juce::Font("Times New Roman", 12.0f, juce::Font::bold));
  g.drawText(
      "Dist", 20, 100, 50, 30, juce::Justification::horizontallyCentred | juce::Justification::verticallyCentred);
  g.drawText(
      "Tone", 120, 100, 50, 30, juce::Justification::horizontallyCentred | juce::Justification::verticallyCentred);
}

void AndromedaAudioProcessorEditor::resized()
{
  distLevel.setBounds(20, 50, 55, 55);
  tone.setBounds(120, 50, 55, 55);
}
