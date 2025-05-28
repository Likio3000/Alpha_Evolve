import React from 'react';
import { ParameterSuggestion } from '../types';
import { CheckCircleIcon } from './icons/CheckCircleIcon';
import { marked } from 'marked';

interface SuggestionDisplayProps {
  suggestion: ParameterSuggestion;
  onApplySuggestion: () => void;
}

const SuggestionDisplay: React.FC<SuggestionDisplayProps> = ({ suggestion, onApplySuggestion }) => {
  const htmlJustification = marked.parse(suggestion.justification || '');

  return (
    <div className="bg-gray-800 p-6 rounded-xl shadow-2xl mt-6 border border-teal-500">
      <h3 className="text-xl font-semibold text-teal-400 mb-4">Gemini's Suggestions for Next Run</h3>
      
      <div className="mb-6">
        <h4 className="text-md font-semibold text-gray-200 mb-2">Justification:</h4>
        <div 
          className="markdown-content text-sm text-gray-300 leading-relaxed"
          dangerouslySetInnerHTML={{ __html: htmlJustification }}
        />
      </div>

      <div className="mb-6">
        <h4 className="text-md font-semibold text-gray-200 mb-2">Suggested Parameters:</h4>
        <pre className="bg-gray-900 p-3 rounded-md text-xs text-sky-300 overflow-x-auto whitespace-pre-wrap">
          {JSON.stringify(suggestion.suggestedParams, null, 2)}
        </pre>
      </div>
      
      <button
        onClick={onApplySuggestion}
        className="w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-xl shadow-sm text-white bg-sky-600 hover:bg-sky-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-sky-500 transition-colors duration-150"
      >
        <CheckCircleIcon className="w-5 h-5 mr-2" />
        Apply These Suggestions to Form
      </button>
    </div>
  );
};

export default SuggestionDisplay;