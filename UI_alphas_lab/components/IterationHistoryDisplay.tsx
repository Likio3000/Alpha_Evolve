
import React, { useState } from 'react';
import { AnalyzedIteration } from '../types';
import { ChevronDownIcon } from './icons/ChevronDownIcon';
import { ChevronRightIcon } from './icons/ChevronRightIcon';

interface IterationHistoryDisplayProps {
  history: AnalyzedIteration[];
}

const IterationHistoryDisplay: React.FC<IterationHistoryDisplayProps> = ({ history }) => {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  const toggleExpand = (id: string) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  if (history.length === 0) {
    return (
      <div className="p-4 bg-gray-800 rounded-lg shadow-md border border-gray-700">
        <p className="text-gray-400 text-center">No experiment history recorded yet.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {history.map((iter, index) => {
        const isExpanded = expandedItems.has(iter.id);
        return (
          <div key={iter.id} className="bg-gray-800 rounded-lg shadow-md border border-gray-700 overflow-hidden">
            <button
              onClick={() => toggleExpand(iter.id)}
              className="w-full flex justify-between items-center p-3 bg-gray-750 hover:bg-gray-700 text-gray-200 font-medium transition-colors"
              aria-expanded={isExpanded}
              aria-controls={`history-item-${iter.id}-content`}
            >
              <span className="text-left">
                Iteration {history.length - index} (Run on: {new Date(iter.timestamp).toLocaleString()})
              </span>
              {isExpanded ? <ChevronDownIcon className="w-5 h-5"/> : <ChevronRightIcon className="w-5 h-5"/>}
            </button>
            {isExpanded && (
              <div id={`history-item-${iter.id}-content`} className="p-4 space-y-3 border-t border-gray-700">
                <div>
                  <h5 className="text-sm font-semibold text-sky-300 mb-1">Parameters Used:</h5>
                  <pre className="bg-gray-900 p-2 rounded text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap">
                    {JSON.stringify(iter.paramsUsed, null, 2)}
                  </pre>
                </div>
                <div>
                  <h5 className="text-sm font-semibold text-sky-300 mb-1">User Console Output:</h5>
                  <pre className="bg-gray-900 p-2 rounded text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap max-h-40">
                    {iter.userInput.consoleOutput || "Not provided."}
                  </pre>
                </div>
                <div>
                  <h5 className="text-sm font-semibold text-sky-300 mb-1">User Personal Opinion:</h5>
                  <pre className="bg-gray-900 p-2 rounded text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap max-h-40">
                    {iter.userInput.personalOpinion || "Not provided."}
                  </pre>
                </div>
                <div>
                  <h5 className="text-sm font-semibold text-teal-300 mb-1">Gemini's Suggestion for Next Step:</h5>
                  <p className="text-xs text-gray-400 mb-1"><span className="font-semibold">Justification:</span> {iter.geminiResponse.justification}</p>
                  <pre className="bg-gray-900 p-2 rounded text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap">
                    {JSON.stringify(iter.geminiResponse.suggestedParams, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default IterationHistoryDisplay;
