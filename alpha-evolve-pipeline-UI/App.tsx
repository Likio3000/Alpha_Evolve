
import React, { useState, useCallback, useMemo } from 'react';
import { AppParameters, ParameterDefinition } from './types';
import { PARAMETER_DEFINITIONS, SECTIONS, INITIAL_PARAMETERS } from './constants';
import { ITER_PARAM_DEFINITIONS, ITER_SECTIONS, ITER_INITIALS } from './iterative';
import Section from './components/Section';
import Dashboard from './components/Dashboard';

const App: React.FC = () => {
  const [mode, setMode] = useState<'pipeline' | 'iterative' | 'dashboard'>('pipeline');
  const [parameters, setParameters] = useState<AppParameters>(INITIAL_PARAMETERS);
  const [iterParams, setIterParams] = useState<AppParameters>(ITER_INITIALS);
  const [generatedCommand, setGeneratedCommand] = useState<string>('');

  const handleParameterChange = useCallback((id: string, value: string | number | boolean) => {
    setParameters(prevParams => ({
      ...prevParams,
      [id]: value,
    }));
  }, []);

  const handleIterParamChange = useCallback((id: string, value: string | number | boolean) => {
    setIterParams(prev => ({
      ...prev,
      [id]: value,
    }));
  }, []);

  const handleResetToDefaults = useCallback(() => {
    if (mode === 'pipeline') {
      setParameters(INITIAL_PARAMETERS);
    } else {
      setIterParams(ITER_INITIALS);
    }
    setGeneratedCommand('');
  }, [mode]);

  const generateCommandString = useCallback(() => {
    if (mode === 'pipeline') {
      // Find the generations parameter definition to get its default value if needed
      const generationsParamDef = PARAMETER_DEFINITIONS.find(p => p.id === 'generations');
      const generationsValue = parameters.generations || (generationsParamDef ? generationsParamDef.defaultValue : '15');
      let command = `uv run run_pipeline.py ${generationsValue}`;

      PARAMETER_DEFINITIONS.forEach(paramDef => {
        if (paramDef.isPositional) return; // Skip positional, already handled
        const value = parameters[paramDef.id];
        if (paramDef.type === 'boolean') {
          if (value === true) {
            command += ` ${paramDef.cliFlag}`;
          }
        } else if (paramDef.cliFlag === '--log-file' && (value === null || value === '')) {
          // skip empty log file
        } else if (value !== '' && value !== null && value !== undefined) {
          let effectiveValue = value;
          if (value === '' && paramDef.type === 'number') {
            effectiveValue = paramDef.defaultValue;
          }
          command += ` ${paramDef.cliFlag} ${String(effectiveValue)}`;
        }
      });
      setGeneratedCommand(command.trim().replace(/\s\s+/g, ' '));
    } else {
      // Iterative mode: build command for scripts/auto_improve.py
      let command = `uv run scripts/auto_improve.py`;
      // Core flags
      ITER_PARAM_DEFINITIONS.forEach(def => {
        const value = iterParams[def.id];
        // Skip passthrough-only knobs here; we will append them after core
        const passthroughOnly = [
          'selection_metric','ramp_fraction','ramp_min_gens','novelty_boost_w','novelty_struct_w','hof_corr_mode',
          'ic_tstat_w','temporal_decay_half_life','rank_softmax_beta_floor','rank_softmax_beta_target','corr_penalty_w'
        ];
        if (passthroughOnly.includes(def.id)) return;
        if (def.type === 'boolean') {
          if (value === true && def.cliFlag) command += ` ${def.cliFlag}`;
        } else if (value !== '' && value !== null && value !== undefined && def.cliFlag) {
          command += ` ${def.cliFlag} ${String(value)}`;
        }
      });
      // Append passthrough flags at the end; these will be forwarded to run_pipeline
      const passthroughDefs = ITER_PARAM_DEFINITIONS.filter(d => [
        'selection_metric','ramp_fraction','ramp_min_gens','novelty_boost_w','novelty_struct_w','hof_corr_mode',
        'ic_tstat_w','temporal_decay_half_life','rank_softmax_beta_floor','rank_softmax_beta_target','corr_penalty_w'
      ].includes(d.id));
      if (passthroughDefs.length > 0) {
        command += ' --'; // separator for parse_known_args passthrough
        passthroughDefs.forEach(def => {
          const value = iterParams[def.id];
          if (def.type === 'boolean') {
            if (value === true && def.cliFlag) command += ` ${def.cliFlag}`;
          } else if (value !== '' && value !== null && value !== undefined && def.cliFlag) {
            command += ` ${def.cliFlag} ${String(value)}`;
          }
        });
      }
      setGeneratedCommand(command.trim().replace(/\s\s+/g, ' '));
    }
  }, [mode, parameters, iterParams]);
  
  const handleCopyToClipboard = useCallback(() => {
    if (generatedCommand) {
      navigator.clipboard.writeText(generatedCommand)
        .then(() => alert('Command copied to clipboard!'))
        .catch(err => alert('Failed to copy command: ' + err));
    }
  }, [generatedCommand]);

  const memoizedSections = useMemo(() => {
    if (mode === 'pipeline') {
      return SECTIONS.map(sectionData => (
        <Section
          key={sectionData.title}
          section={sectionData}
          parameters={parameters}
          onParameterChange={handleParameterChange}
          defaultOpen={sectionData.title === 'General & Evolution Core'}
        />
      ));
    }
    return ITER_SECTIONS.map(sectionData => (
      <Section
        key={sectionData.title}
        section={sectionData}
        parameters={iterParams}
        onParameterChange={handleIterParamChange}
        defaultOpen={sectionData.title === 'Iterative Loop'}
      />
    ));
  }, [mode, parameters, iterParams, handleParameterChange, handleIterParamChange]);


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-gray-200 p-4 sm:p-6 lg:p-8">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">
          Alpha Evolve – Pipeline & Iterative UI
        </h1>
        <p className="mt-2 text-lg text-slate-400">
          Configure parameters and generate commands for the full pipeline or the iterative improver.
        </p>
        <div className="mt-4 inline-flex rounded-md shadow-sm" role="group" aria-label="Mode toggle">
          <button onClick={() => setMode('pipeline')} className={`px-4 py-2 text-sm font-medium border ${mode==='pipeline' ? 'bg-purple-600 text-white border-purple-500' : 'bg-slate-700 text-slate-200 border-slate-600'} rounded-l-lg`}>Pipeline</button>
          <button onClick={() => setMode('iterative')} className={`px-4 py-2 text-sm font-medium border ${mode==='iterative' ? 'bg-purple-600 text-white border-purple-500' : 'bg-slate-700 text-slate-200 border-slate-600'}`}>Iterative</button>
          <button onClick={() => setMode('dashboard')} className={`px-4 py-2 text-sm font-medium border ${mode==='dashboard' ? 'bg-purple-600 text-white border-purple-500' : 'bg-slate-700 text-slate-200 border-slate-600'} rounded-r-lg`}>Dashboard</button>
        </div>
      </header>

      <main className="max-w-5xl mx-auto">
        {mode !== 'dashboard' && (
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
        )}

        {mode !== 'dashboard' && generatedCommand && (
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
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2z" />
                </svg>
                Copy
              </button>
            </div>
          </div>
        )}

        {mode === 'dashboard' && (
          <div className="bg-slate-800 shadow-2xl rounded-xl p-6 sm:p-8">
            <Dashboard />
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
