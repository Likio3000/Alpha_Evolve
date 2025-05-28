
import React from 'react';

interface PromptPreviewProps {
  prompt: string;
}

const PromptPreview: React.FC<PromptPreviewProps> = ({ prompt }) => {
  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-md border border-gray-700">
      <h4 className="text-md font-semibold text-gray-200 mb-2">Full Prompt to be Sent to Gemini:</h4>
      <pre className="bg-gray-900 p-3 rounded text-xs text-gray-300 overflow-auto whitespace-pre-wrap max-h-[500px]">
        {prompt}
      </pre>
      <p className="text-xs text-gray-500 mt-2">
        Note: This is the raw prompt. Python code snippets are included above where indicated.
        The API key is handled separately and not shown here.
      </p>
    </div>
  );
};

export default PromptPreview;
