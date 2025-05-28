
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { SimulationParams, PythonFile, ActiveView, ActualRunUserInput, ParameterSuggestion, AnalyzedIteration } from './types';
import { PYTHON_FILES } from './pythonCode';
import ParameterForm from './components/ParameterForm';
import LocalRunGuidance from './components/LocalRunGuidance';
import SuggestionDisplay from './components/SuggestionDisplay';
import FileViewer from './components/FileViewer';
import IterationHistoryDisplay from './components/IterationHistoryDisplay';
import PromptPreview from './components/PromptPreview';
import { GeminiService, getFullPromptForPreview } from './services/geminiService';
import { CodeIcon } from './components/icons/CodeIcon';
import { ExperimentIcon } from './components/icons/ExperimentIcon'; 
import { ChevronDownIcon } from './components/icons/ChevronDownIcon';
import { ChevronRightIcon } from './components/icons/ChevronRightIcon';
import { HistoryIcon } from './components/icons/HistoryIcon';
import { DocumentMagnifyingGlassIcon } from './components/icons/DocumentMagnifyingGlassIcon';

const MAX_HISTORY_ITEMS = 5;

const App: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<PythonFile | null>(PYTHON_FILES[0] || null);
  const initialParams: SimulationParams = {
    generations: 10, seed: 42, pop_size: 64, max_lookback_data_option: 'common_1200',
    min_common_points: 1200, eval_lag: 1, top_to_backtest: 10, fee: 1.0, hold: 1,
    scale: 'zscore', data_dir: './data', tournament_k: 5, p_mut: 0.4, p_cross: 0.6,
    elite_keep: 4, max_ops: 32, parsimony_penalty: 0.01, corr_penalty_w: 0.15,
    corr_cutoff: 0.20, hof_size: 20, fresh_rate: 0.05,
  };
  const [simulationParams, setSimulationParams] = useState<SimulationParams>(initialParams);
  
  const [localRunCommand, setLocalRunCommand] = useState<string | null>(null);
  const [showRunGuidance, setShowRunGuidance] = useState<boolean>(false);
  const initialActualRunInput: ActualRunUserInput = {
    consoleOutput: '',
    personalOpinion: '',
  };
  const [actualRunInput, setActualRunInput] = useState<ActualRunUserInput>(initialActualRunInput);
  const [parameterSuggestion, setParameterSuggestion] = useState<ParameterSuggestion | null>(null);
  const [iterationHistory, setIterationHistory] = useState<AnalyzedIteration[]>([]);

  const [isLoadingCommand, setIsLoadingCommand] = useState<boolean>(false);
  const [isLoadingAnalysis, setIsLoadingAnalysis] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  const [activeView, setActiveView] = useState<ActiveView>(ActiveView.Code);
  const [geminiService, setGeminiService] = useState<GeminiService | null>(null);
  const [fileExplorerOpen, setFileExplorerOpen] = useState<boolean>(true);

  const [showPromptPreview, setShowPromptPreview] = useState<boolean>(false);
  const [showIterationHistory, setShowIterationHistory] = useState<boolean>(false);

  const guidanceRef = useRef<HTMLDivElement>(null); 
  const suggestionRef = useRef<HTMLDivElement>(null);
  const promptPreviewRef = useRef<HTMLDivElement>(null);
  const iterationHistoryRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (process.env.API_KEY) {
      setGeminiService(new GeminiService(process.env.API_KEY));
    } else {
      setError("API_KEY environment variable not found. Core functionality will be disabled.");
      console.error("API_KEY environment variable not found.");
    }
  }, []);

  useEffect(() => {
    if (parameterSuggestion && !isLoadingAnalysis && suggestionRef.current) {
      setTimeout(() => {
        suggestionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
    }
  }, [parameterSuggestion, isLoadingAnalysis]);

  useEffect(() => {
    if (showPromptPreview && promptPreviewRef.current) {
      setTimeout(() => {
        promptPreviewRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
    }
  }, [showPromptPreview]);

  useEffect(() => {
    if (showIterationHistory && iterationHistoryRef.current) {
      setTimeout(() => {
        iterationHistoryRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 100);
    }
  }, [showIterationHistory]);


  const handleParametersChange = useCallback((params: SimulationParams) => {
    setSimulationParams(params);
    setShowRunGuidance(false); 
    setLocalRunCommand(null);
    setParameterSuggestion(null);
    // Decide if history should be cleared on manual param change, for now, it persists.
  }, []);

  const handleFileSelect = (file: PythonFile) => {
    setSelectedFile(file);
    setActiveView(ActiveView.Code);
  };

  const getPythonCodeContext = useCallback(() => {
    return PYTHON_FILES.reduce((acc, file) => {
      if (file.path === 'src/run_pipeline.py' || file.path === 'src/evolve_alphas.py' || file.path === 'src/alpha_program_core.py') {
        acc[file.path.substring(4)] = file.content; 
      }
      return acc;
    }, {} as { [key: string]: string });
  }, []);

  const handlePrepareLocalRun = useCallback(() => {
    setIsLoadingCommand(true);
    setError(null);
    setParameterSuggestion(null); // Clear previous suggestion
    setActualRunInput(initialActualRunInput); // Reset inputs for new run
    setShowPromptPreview(false); // Collapse preview sections
    setShowIterationHistory(false);


    let command = `uv run run_pipeline.py ${simulationParams.generations}`;
    command += ` --seed ${simulationParams.seed}`;
    command += ` --pop_size ${simulationParams.pop_size}`;
    command += ` --max_lookback_data_option ${simulationParams.max_lookback_data_option}`;
    command += ` --min_common_points ${simulationParams.min_common_points}`;
    command += ` --eval_lag ${simulationParams.eval_lag}`;
    command += ` --top_to_backtest ${simulationParams.top_to_backtest}`;
    command += ` --fee ${simulationParams.fee}`;
    command += ` --hold ${simulationParams.hold}`;
    command += ` --scale ${simulationParams.scale}`;
    command += ` --data_dir "${simulationParams.data_dir}"`;
    command += ` --tournament_k ${simulationParams.tournament_k}`;
    command += ` --p_mut ${simulationParams.p_mut}`;
    command += ` --p_cross ${simulationParams.p_cross}`;
    command += ` --elite_keep ${simulationParams.elite_keep}`;
    command += ` --max_ops ${simulationParams.max_ops}`;
    command += ` --parsimony_penalty ${simulationParams.parsimony_penalty}`;
    command += ` --corr_penalty_w ${simulationParams.corr_penalty_w}`;
    command += ` --corr_cutoff ${simulationParams.corr_cutoff}`;
    command += ` --hof_size ${simulationParams.hof_size}`;
    command += ` --fresh_rate ${simulationParams.fresh_rate}`;
    
    setLocalRunCommand(command);
    setShowRunGuidance(true);
    setIsLoadingCommand(false);

    setTimeout(() => {
      guidanceRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

  }, [simulationParams]);

  const handleActualRunInputChange = useCallback((input: ActualRunUserInput) => {
    setActualRunInput(input);
    setParameterSuggestion(null); 
  }, []);

  const handleAnalyzeActualResults = useCallback(async () => {
    if (!geminiService) {
      setError("Gemini Service not initialized. Cannot analyze results.");
      return;
    }
    if (!actualRunInput.consoleOutput && !actualRunInput.personalOpinion) {
        setError("Please provide Console Output or your Personal Opinion from your local run for analysis.");
        return;
    }

    setIsLoadingAnalysis(true);
    setError(null);
    setParameterSuggestion(null); // Clear previous suggestion before fetching new one
    setShowPromptPreview(false); // Collapse preview sections
    setShowIterationHistory(false);


    const pythonCodeContext = getPythonCodeContext();

    try {
      const suggestion = await geminiService.analyzeActualResultsAndSuggestNextParams(
        simulationParams,
        actualRunInput,
        iterationHistory, // Pass the history
        pythonCodeContext
      );
      setParameterSuggestion(suggestion);

      // Add to history
      const newIterationEntry: AnalyzedIteration = {
        id: new Date().toISOString() + Math.random().toString(),
        timestamp: new Date(),
        paramsUsed: { ...simulationParams }, // Store a copy of params used for this analysis
        userInput: { ...actualRunInput },   // Store a copy of user input for this analysis
        geminiResponse: suggestion,
      };
      setIterationHistory(prevHistory => [newIterationEntry, ...prevHistory].slice(0, MAX_HISTORY_ITEMS));
      
      // Do NOT reset actualRunInput here, user might want to see it with the suggestion.
      // It will be reset when "Prepare Local Run" or "Apply Suggestion" is clicked.

    } catch (e: any) {
      console.error("Analysis error:", e);
      setError(e.message || "An error occurred during analysis.");
    } finally {
      setIsLoadingAnalysis(false);
    }
  }, [geminiService, simulationParams, actualRunInput, iterationHistory, getPythonCodeContext]);
  
  const handleApplySuggestion = useCallback(() => {
    if (parameterSuggestion) {
      setSimulationParams(parameterSuggestion.suggestedParams);
      setShowRunGuidance(false);
      setLocalRunCommand(null);
      setActualRunInput(initialActualRunInput); // Reset for the new iteration
      setParameterSuggestion(null); // Clear current suggestion as it's applied
      setError(null);
      setShowPromptPreview(false); // Collapse preview sections
      setShowIterationHistory(false);
    }
  }, [parameterSuggestion]);


  const renderContent = () => {
    if (activeView === ActiveView.Code) {
      return selectedFile ? <FileViewer content={selectedFile.content} fileName={selectedFile.name} /> : <div className="p-8 text-gray-400">Select a file to view its content.</div>;
    } else if (activeView === ActiveView.ExperimentLoop) {
      const currentPromptForPreview = getFullPromptForPreview(
        simulationParams,
        actualRunInput,
        iterationHistory,
        getPythonCodeContext()
      );
      return (
        <div className="p-4 md:p-6 space-y-6 h-full overflow-y-auto">
          <ParameterForm
            parameters={simulationParams}
            onParametersChange={handleParametersChange}
            onSubmit={handlePrepareLocalRun}
            isLoading={isLoadingCommand}
            isApiKeyMissing={!geminiService}
          />
          
          <div ref={guidanceRef}>
            {showRunGuidance && localRunCommand && (
              <LocalRunGuidance
                command={localRunCommand}
                userInput={actualRunInput}
                onUserInputChange={handleActualRunInputChange}
                onAnalyze={handleAnalyzeActualResults}
                isLoadingAnalysis={isLoadingAnalysis}
                isApiKeyMissing={!geminiService}
              />
            )}
          </div>
          {isLoadingAnalysis && <div className="text-center py-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-sky-400 mx-auto"></div>
            <p className="text-sky-400 mt-2">Gemini is thinking...</p>
            </div>}
          {error && (
            <div className="mt-4 bg-red-800 border border-red-700 p-4 rounded-lg shadow-md">
              <h4 className="text-lg font-semibold text-red-200 mb-2">Error</h4>
              <p className="text-red-100">{error}</p>
            </div>
          )}
          {parameterSuggestion && !isLoadingAnalysis && (
            <div ref={suggestionRef}>
              <SuggestionDisplay
                suggestion={parameterSuggestion}
                onApplySuggestion={handleApplySuggestion}
              />
            </div>
          )}

          {/* Collapsible sections for Prompt Preview and History */}
          {(showRunGuidance || parameterSuggestion) && !isLoadingAnalysis && (
             <div className="mt-6 space-y-4">
                <div>
                    <button
                        onClick={() => setShowPromptPreview(!showPromptPreview)}
                        className="w-full flex justify-between items-center p-3 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-200 font-medium transition-colors"
                        aria-expanded={showPromptPreview}
                    >
                        <span className="flex items-center"><DocumentMagnifyingGlassIcon className="w-5 h-5 mr-2"/> View Full Prompt to be Sent to Gemini</span>
                        {showPromptPreview ? <ChevronDownIcon className="w-5 h-5"/> : <ChevronRightIcon className="w-5 h-5"/>}
                    </button>
                    {showPromptPreview && (
                        <div ref={promptPreviewRef} className="mt-2">
                            <PromptPreview prompt={currentPromptForPreview} />
                        </div>
                    )}
                </div>
                <div>
                    <button
                        onClick={() => setShowIterationHistory(!showIterationHistory)}
                        className="w-full flex justify-between items-center p-3 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-200 font-medium transition-colors"
                        aria-expanded={showIterationHistory}
                    >
                         <span className="flex items-center"><HistoryIcon className="w-5 h-5 mr-2"/> View Experiment Iteration History ({iterationHistory.length} stored)</span>
                        {showIterationHistory ? <ChevronDownIcon className="w-5 h-5"/> : <ChevronRightIcon className="w-5 h-5"/>}
                    </button>
                    {showIterationHistory && (
                        <div ref={iterationHistoryRef} className="mt-2">
                            <IterationHistoryDisplay history={iterationHistory} />
                        </div>
                    )}
                </div>
            </div>
          )}

        </div>
      );
    }
    return null;
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-gray-100">
      <header className="bg-gray-800 p-4 shadow-lg flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-sky-400">AlphaGen Pipeline Explorer</h1>
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveView(ActiveView.Code)}
            className={`p-2 rounded-md flex items-center space-x-2 transition-colors duration-150 ${activeView === ActiveView.Code ? 'bg-sky-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
            title="View Code"
            aria-pressed={activeView === ActiveView.Code}
          >
            <CodeIcon className="w-5 h-5" /> 
            <span className="hidden sm:inline">Code</span>
          </button>
          <button
            onClick={() => setActiveView(ActiveView.ExperimentLoop)}
            className={`p-2 rounded-md flex items-center space-x-2 transition-colors duration-150 ${activeView === ActiveView.ExperimentLoop ? 'bg-sky-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
            title="Experiment Loop"
            aria-pressed={activeView === ActiveView.ExperimentLoop}
          >
            <ExperimentIcon className="w-5 h-5" />
             <span className="hidden sm:inline">Experiment</span>
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <aside className={`bg-gray-800 transition-all duration-300 ease-in-out ${fileExplorerOpen ? 'w-64 md:w-72' : 'w-12'} overflow-y-auto p-2`}>
          <button 
            onClick={() => setFileExplorerOpen(!fileExplorerOpen)} 
            className="p-2 mb-2 text-gray-400 hover:text-sky-400 w-full flex items-center justify-start"
            aria-expanded={fileExplorerOpen}
            aria-controls="file-explorer-nav"
          >
            {fileExplorerOpen ? <ChevronDownIcon className="w-5 h-5 mr-2"/> : <ChevronRightIcon className="w-5 h-5"/>}
            {fileExplorerOpen && <span className="font-semibold">Project Files</span>}
          </button>
          {fileExplorerOpen && (
            <nav id="file-explorer-nav">
              <ul>
                {PYTHON_FILES.map((file) => (
                  <li key={file.path} className="mb-1">
                    <button
                      onClick={() => handleFileSelect(file)}
                      className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors duration-150 truncate
                        ${selectedFile?.path === file.path && activeView === ActiveView.Code ? 'bg-sky-600 text-white' : 'hover:bg-gray-700 text-gray-300'}`}
                      aria-current={selectedFile?.path === file.path && activeView === ActiveView.Code ? "page" : undefined}
                    >
                      {file.name}
                    </button>
                  </li>
                ))}
              </ul>
            </nav>
          )}
        </aside>

        <main className="flex-1 bg-gray-900 overflow-y-auto"> 
          {renderContent()}
        </main>
      </div>
    </div>
  );
};

export default App;
