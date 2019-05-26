from speechLibrary import speechAnalysis

analyzer = speechAnalysis.SpeechAnalyzer()

analyzer.lookBackSize = 5
analyzer.stepSize = 1

analyzer.recordingBufferSize = 2048

analyzer.getFeaturesFromLiveInput()
