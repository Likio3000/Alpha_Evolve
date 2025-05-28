
import React from 'react';
import { ActualRunUserInput } from '../types';
import { ClipboardIcon } from './icons/ClipboardIcon'; // New Icon
import { LightBulbIcon } from './icons/LightBulbIcon'; // Re-using or new

interface LocalRunGuidanceProps {
  command: string;
  userInput: ActualRunUserInput;
  onUserInputChange: (input: ActualRunUserInput) => void;
  onAnalyze: () => void;
  isLoadingAnalysis: boolean;
  isApiKeyMissing: boolean;
}

const LocalRunGuidance: React.FC<LocalRunGuidanceProps> = ({
  command,
  userInput,
  onUserInputChange,
  onAnalyze,
  isLoadingAnalysis,
  isApiKeyMissing,
}) => {
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    onUserInputChange({
      ...userInput,
      [name]: value,
    });
  };

  const copyCommandToClipboard = () => {
    navigator.clipboard.writeText(command).then(() => {
      // Optionally, show a "Copied!" message
    }).catch(err => {
      console.error('Failed to copy command: ', err);
    });
  };
  
  const textAreaClass = "w-full p-2.5 bg-gray-700 border border-gray-600 text-gray-200 text-sm rounded-lg focus:ring-sky-500 focus:border-sky-500 min-h-[100px] font-mono";

  return (
    <div className="bg-gray-800 p-6 rounded-xl shadow-2xl mt-6 space-y-6">
      <div>
        <h3 className="text-xl font-semibold text-sky-400 mb-3">1. Run Pipeline Locally</h3>
        <p className="text-sm text-gray-400 mb-2">Copy and paste the following command into your terminal at the root of this project:</p>
        <div className="bg-gray-900 p-3 rounded-md flex items-center justify-between">
          <pre className="text-sky-300 text-sm overflow-x-auto whitespace-pre-wrap"><code>{command}</code></pre>
          <button 
            onClick={copyCommandToClipboard} 
            className="p-2 text-gray-400 hover:text-sky-400"
            title="Copy command"
            aria-label="Copy command to clipboard"
            >
            <ClipboardIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div>
        <h3 className="text-xl font-semibold text-sky-400 mb-3">2. Provide Actual Results</h3>
        <p className="text-sm text-gray-400 mb-3">After the local script finishes, paste the outputs and your thoughts into the text areas below.</p>
        
        <div className="space-y-4">
          <div>
            <label htmlFor="consoleOutput" className="block mb-1 text-sm font-medium text-gray-300">Console Output:</label>
            <textarea
              id="consoleOutput"
              name="consoleOutput"
              value={userInput.consoleOutput}
              onChange={handleInputChange}
              className={textAreaClass}
              rows={6}
              placeholder="Paste relevant console log parts (e.g., errors, progress indicators, final metrics summary printed to console)..."
            />
          </div>
          <div>
            <label htmlFor="personalOpinion" className="block mb-1 text-sm font-medium text-gray-300">Personal Opinion:</label>
            <textarea
              id="personalOpinion"
              name="personalOpinion"
              value={userInput.personalOpinion}
              onChange={handleInputChange}
              className={textAreaClass}
              rows={6}
              placeholder="Share your thoughts on the last run. What did you observe? What worked or didn't? What are your goals for the next iteration or what specific parameters do you want to explore?"
            />
          </div>
        </div>
      </div>
      
      <button
        onClick={onAnalyze}
        disabled={isLoadingAnalysis || isApiKeyMissing || (!userInput.consoleOutput && !userInput.personalOpinion)}
        className="w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-xl shadow-sm text-white bg-teal-600 hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-teal-500 disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors duration-150"
      >
        {isLoadingAnalysis ? (
          <>
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Analyzing with Gemini...
          </>
        ) : (
          <>
            <LightBulbIcon className="w-5 h-5 mr-2" />
            Analyze Results & Get Next Parameter Suggestions
          </>
        )}
      </button>
       {isApiKeyMissing && (
        <p className="text-xs text-red-400 text-center mt-2">Gemini analysis disabled: API Key not configured.</p>
      )}
       {(!userInput.consoleOutput && !userInput.personalOpinion && !isApiKeyMissing) && (
         <p className="text-xs text-yellow-400 text-center mt-2">Please provide Console Output or your Personal Opinion to enable analysis.</p>
       )}
    </div>
  );
};

export default LocalRunGuidance;