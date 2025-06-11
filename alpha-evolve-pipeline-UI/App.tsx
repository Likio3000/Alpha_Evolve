
import React, { useState, useCallback, useMemo } from 'react';
import { AppParameters, ParameterDefinition } from './types';
import { PARAMETER_DEFINITIONS, SECTIONS, INITIAL_PARAMETERS } from './constants';
import Section from './components/Section';

const App: React.FC = () => {
  const [parameters, setParameters] = useState<AppParameters>(INITIAL_PARAMETERS);
  const [generatedCommand, setGeneratedCommand] = useState<string>('');

  const handleParameterChange = useCallback((id: string, value: string | number | boolean) => {
    setParameters(prevParams => ({
      ...prevParams,
      [id]: value,
    }));
  }, []);

  const handleResetToDefaults = useCallback(() => {
    setParameters(INITIAL_PARAMETERS);
    setGeneratedCommand('');
  }, []);

  const generateCommandString = useCallback(() => {
    // Find the generations parameter definition to get its default value if needed
    const generationsParamDef = PARAMETER_DEFINITIONS.find(p => p.id === 'generations');
    const generationsValue = parameters.generations || (generationsParamDef ? generationsParamDef.defaultValue : '15');
    let command = `uv run run_pipeline.py ${generationsValue}`;

    PARAMETER_DEFINITIONS.forEach(paramDef => {
      if (paramDef.isPositional) return; // Skip positional, already handled

      const value = parameters[paramDef.id];
      
      // For boolean flags, only add them if true
      if (paramDef.type === 'boolean') {
        if (value === true) {
          command += ` ${paramDef.cliFlag}`;
        }
      } 
      // For other types, add flag and value if value is not empty, null, or undefined
      // Special handling for log_file: if empty, don't add.
      else if (paramDef.cliFlag === '--log-file' && (value === null || value === '')) {
        // Do nothing for empty log_file
      }
      else if (value !== '' && value !== null && value !== undefined) {
        let effectiveValue = value;
        // If a number field is cleared, it might become an empty string. Use default in such cases.
        if (value === '' && paramDef.type === 'number') {
          effectiveValue = paramDef.defaultValue;
        }
        command += ` ${paramDef.cliFlag} ${String(effectiveValue)}`;
      }
    });
    setGeneratedCommand(command.trim().replace(/\s\s+/g, ' ')); // Trim and remove multiple spaces
  }, [parameters]);
  
  const handleCopyToClipboard = useCallback(() => {
    if (generatedCommand) {
      navigator.clipboard.writeText(generatedCommand)
        .then(() => alert('Command copied to clipboard!'))
        .catch(err => alert('Failed to copy command: ' + err));
    }
  }, [generatedCommand]);

  const memoizedSections = useMemo(() => {
    return SECTIONS.map(sectionData => (
      <Section
        key={sectionData.title}
        section={sectionData}
        parameters={parameters}
        onParameterChange={handleParameterChange}
        defaultOpen={sectionData.title === 'General & Evolution Core'}
      />
    ));
  }, [parameters, handleParameterChange]);


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-gray-200 p-4 sm:p-6 lg:p-8">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">
          Alpha Evolve Pipeline Configuration
        </h1>
        <p className="mt-2 text-lg text-slate-400">
          Configure parameters and generate the execution command for your Alpha Evolve pipeline.
        </p>
      </header>

      <main className="max-w-5xl mx-auto">
        <div className="bg-slate-800 shadow-2xl rounded-xl p-6 sm:p-8">
          {memoizedSections}

          <div className="mt-8 flex flex-col sm:flex-row justify-end space-y-3 sm:space-y-0 sm:space-x-4">
            <button
              onClick={handleResetToDefaults}
              className="px-6 py-3 bg-slate-600 hover:bg-slate-500 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-opacity-75"
            >
              Reset to Defaults
            </button>
            <button
              onClick={generateCommandString}
              className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold rounded-lg shadow-md transition duration-150 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-opacity-75"
            >
              Generate Command
            </button>
          </div>
        </div>

        {generatedCommand && (
          <div className="mt-10 bg-slate-800 shadow-2xl rounded-xl p-6 sm:p-8">
            <h2 className="text-2xl font-semibold text-slate-100 mb-4">Generated Command:</h2>
            <div className="relative bg-slate-900 p-4 rounded-md overflow-x-auto">
              <pre className="text-sm text-green-300 whitespace-pre-wrap break-all">
                <code>{generatedCommand}</code>
              </pre>
              <button
                onClick={handleCopyToClipboard}
                className="absolute top-2 right-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs font-medium rounded-md transition duration-150 focus:outline-none focus:ring-2 focus:ring-slate-500"
                title="Copy to Clipboard"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 inline mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy
              </button>
            </div>
          </div>
        )}
      </main>
      <footer className="mt-12 text-center text-sm text-slate-500">
        <p>&copy; {new Date().getFullYear()} Alpha Evolve UI.</p>
      </footer>
    </div>
  );
};

export default App;
