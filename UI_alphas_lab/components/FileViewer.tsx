
import React from 'react';
import HighlightedCode from './HighlightedCode';

interface FileViewerProps {
  content: string;
  fileName: string;
}

const FileViewer: React.FC<FileViewerProps> = ({ content, fileName }) => {
  return (
    <div className="p-4 md:p-6 h-full bg-gray-900 rounded-lg shadow-inner flex flex-col">
      <h2 className="text-xl font-semibold text-sky-400 mb-4 sticky top-0 bg-gray-900 py-2 z-10">{fileName}</h2>
      <div className="flex-1 overflow-auto">
        <pre className="text-sm text-gray-300 whitespace-pre-wrap h-full">
          <HighlightedCode code={content.trim()} />
        </pre>
      </div>
    </div>
  );
};

export default FileViewer;
