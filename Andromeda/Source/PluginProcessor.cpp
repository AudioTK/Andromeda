/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "static_elements.h"

//==============================================================================
AndromedaAudioProcessor::AndromedaAudioProcessor()
  :
#ifndef JucePlugin_PreferredChannelConfigurations
  juce::AudioProcessor(BusesProperties()
#  if !JucePlugin_IsMidiEffect
#    if !JucePlugin_IsSynth
                           .withInput("Input", juce::AudioChannelSet::mono(), true)
#    endif
                           .withOutput("Output", juce::AudioChannelSet::mono(), true)
#  endif
          )
  ,
#endif
  inFilter(nullptr, 1, 0, false)
  , bandPassFilter(Andromeda::createStaticFilter_stage1())
  , oversamplingFilter(1)
  , distortionFilter(Andromeda::createStaticFilter_stage2())
  , toneShapingOverdriveFilter(Andromeda::createStaticFilter_stage3())
  , lowpassFilter(1)
  , decimationFilter(1)
  , lowPass2Filter(Andromeda::createStaticFilter_stage4())
  , bandPass2Filter(Andromeda::createStaticFilter_stage5())
  , outFilter(nullptr, 1, 0, false)
  , parameters(*this,
        nullptr,
        juce::Identifier("ATKAndromeda"),
        {std::make_unique<juce::AudioParameterFloat>("distLevel", "Distortion Level", 0.f, 100.f, 50.f),
            std::make_unique<juce::AudioParameterFloat>("tone", "Tone", 0.f, 100.f, 50.f)})
{
  bandPassFilter->set_input_port(bandPassFilter->find_input_pin("vin"), &inFilter, 0);
  oversamplingFilter.set_input_port(0, bandPassFilter.get(), bandPassFilter->find_dynamic_pin("vout"));
  distortionFilter->set_input_port(distortionFilter->find_input_pin("vin"), &oversamplingFilter, 0);
  toneShapingOverdriveFilter->set_input_port(toneShapingOverdriveFilter->find_input_pin("vin"),
      distortionFilter.get(),
      distortionFilter->find_dynamic_pin("vout"));
  lowpassFilter.set_input_port(
      0, toneShapingOverdriveFilter.get(), toneShapingOverdriveFilter->find_dynamic_pin("vout"));
  decimationFilter.set_input_port(0, &lowpassFilter, 0);
  lowPass2Filter->set_input_port(lowPass2Filter->find_input_pin("vin"), decimationFilter, 0);
  bandPass2Filter->set_input_port(
      bandPass2Filter->find_input_pin("vin"), lowPass2Filter.get(), lowPass2Filter->find_dynamic_pin("vout"));
  outFilter.set_input_port(0, bandPass2Filter.get(), bandPass2Filter->find_dynamic_pin("vout"));

  lowpassFilter.set_cut_frequency(20000);
  lowpassFilter.set_order(6);
}

AndromedaAudioProcessor::~AndromedaAudioProcessor()
{
}

//==============================================================================
const juce::String AndromedaAudioProcessor::getName() const
{
  return JucePlugin_Name;
}

bool AndromedaAudioProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
  return true;
#else
  return false;
#endif
}

bool AndromedaAudioProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
  return true;
#else
  return false;
#endif
}

bool AndromedaAudioProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
  return true;
#else
  return false;
#endif
}

double AndromedaAudioProcessor::getTailLengthSeconds() const
{
  return 0.0;
}

int AndromedaAudioProcessor::getNumPrograms()
{
  return 2;
}

int AndromedaAudioProcessor::getCurrentProgram()
{
  return lastParameterSet;
}

void AndromedaAudioProcessor::setCurrentProgram(int index)
{
  if(index != lastParameterSet)
  {
    lastParameterSet = index;
    if(index == 0)
    {
      const char* preset0
          = "<Andromeda><PARAM id=\"distLevel\" value=\"0\" /><PARAM id=\"tone\" value=\"50\" /></Andromeda>";
      juce::XmlDocument doc(preset0);

      auto el = doc.getDocumentElement();
      parameters.state = juce::ValueTree::fromXml(*el);
    }
    else if(index == 1)
    {
      const char* preset1
          = "<Andromeda><PARAM id=\"distLevel\" value=\"100\" /><PARAM id=\"tone\" value=\"50\" /></Andromeda>";
      juce::XmlDocument doc(preset1);

      auto el = doc.getDocumentElement();
      parameters.state = juce::ValueTree::fromXml(*el);
    }
  }
}

const juce::String AndromedaAudioProcessor::getProgramName(int index)
{
  if(index == 0)
  {
    return "Minimum distortion";
  }
  if(index == 0)
  {
    return "Maximum damage";
  }
  return {};
}

void AndromedaAudioProcessor::changeProgramName(int index, const juce::String& newName)
{
}

//==============================================================================
void AndromedaAudioProcessor::prepareToPlay(double dbSampleRate, int samplesPerBlock)
{
  sampleRate = std::lround(dbSampleRate);

  if(sampleRate != inFilter.get_output_sampling_rate())
  {
    inFilter.set_input_sampling_rate(sampleRate);
    inFilter.set_output_sampling_rate(sampleRate);
    bandPassFilter->set_input_sampling_rate(sampleRate);
    bandPassFilter->set_output_sampling_rate(sampleRate);
    oversamplingFilter.set_input_sampling_rate(sampleRate);
    oversamplingFilter.set_output_sampling_rate(sampleRate * OVERSAMPLING);
    distortionFilter->set_input_sampling_rate(sampleRate * OVERSAMPLING);
    distortionFilter->set_output_sampling_rate(sampleRate * OVERSAMPLING);
    toneShapingOverdriveFilter->set_input_sampling_rate(sampleRate * OVERSAMPLING);
    toneShapingOverdriveFilter->set_output_sampling_rate(sampleRate * OVERSAMPLING);
    lowpassFilter.set_input_sampling_rate(sampleRate * OVERSAMPLING);
    lowpassFilter.set_output_sampling_rate(sampleRate * OVERSAMPLING);
    decimationFilter.set_input_sampling_rate(sampleRate * OVERSAMPLING);
    decimationFilter.set_output_sampling_rate(sampleRate);
    lowPass2Filter->set_input_sampling_rate(sampleRate);
    lowPass2Filter->set_output_sampling_rate(sampleRate);
    bandPass2Filter->set_input_sampling_rate(sampleRate);
    bandPass2Filter->set_output_sampling_rate(sampleRate);
    outFilter.set_input_sampling_rate(sampleRate);
    outFilter.set_output_sampling_rate(sampleRate);
  }
  outFilter.dryrun(samplesPerBlock);
}

void AndromedaAudioProcessor::releaseResources()
{
  // When playback stops, you can use this as an opportunity to free up any
  // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool AndromedaAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
#  if JucePlugin_IsMidiEffect
  juce::ignoreUnused(layouts);
  return true;
#  else
  // This is the place where you check if the layout is supported.
  // In this template code we only support mono or stereo.
  // Some plugin hosts, such as certain GarageBand versions, will only
  // load plugins that support stereo bus layouts.
  if(layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
      && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
    return false;

    // This checks if the input layout matches the output layout
#    if !JucePlugin_IsSynth
  if(layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
    return false;
#    endif

  return true;
#  endif
}
#endif

void AndromedaAudioProcessor::processBlock(juce::AudioSampleBuffer& buffer, juce::MidiBuffer& midiMessages)
{
  if(*parameters.getRawParameterValue("distLevel") != old_distLevel)
  {
    old_distLevel = *parameters.getRawParameterValue("distLevel");
    distortionFilter->set_parameter(0, old_distLevel * .99 / 100 + .05);
  }
  if(*parameters.getRawParameterValue("tone") != old_tone)
  {
    old_tone = *parameters.getRawParameterValue("tone");
    toneShapingOverdriveFilter->set_parameter(0, old_tone * .99 / 100 + .05);
  }

  const int totalNumInputChannels = getTotalNumInputChannels();
  const int totalNumOutputChannels = getTotalNumOutputChannels();

  assert(totalNumInputChannels == totalNumOutputChannels);
  assert(totalNumOutputChannels == 1);

  inFilter.set_pointer(buffer.getReadPointer(0), buffer.getNumSamples());
  outFilter.set_pointer(buffer.getWritePointer(0), buffer.getNumSamples());

  outFilter.process(buffer.getNumSamples());
}

//==============================================================================
bool AndromedaAudioProcessor::hasEditor() const
{
  return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AndromedaAudioProcessor::createEditor()
{
  return new AndromedaAudioProcessorEditor(*this, parameters);
}

//==============================================================================
void AndromedaAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
  auto state = parameters.copyState();
  std::unique_ptr<juce::XmlElement> xml(state.createXml());
  xml->setAttribute("version", "0");
  copyXmlToBinary(*xml, destData);
}

void AndromedaAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
  std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

  if(xmlState.get() != nullptr)
  {
    if(xmlState->hasTagName(parameters.state.getType()))
    {
      if(xmlState->getStringAttribute("version") == "0")
      {
        parameters.replaceState(juce::ValueTree::fromXml(*xmlState));
      }
    }
  }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
  return new AndromedaAudioProcessor();
}
