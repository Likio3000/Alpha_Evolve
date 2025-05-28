
import React, { useState } from 'react';
import { SimulationParams } from '../types';
import { PlayIcon } from './icons/PlayIcon';
import { PARAMETER_DESCRIPTIONS } from '../constants';
import { HelpIcon } from './icons/HelpIcon';

interface ParameterFormProps {
  parameters: SimulationParams;
  onParametersChange: (params: SimulationParams) => void;
  onSubmit: () => void; 
  isLoading: boolean; 
  isApiKeyMissing: boolean;
}

const ParameterForm: React.FC<ParameterFormProps> = ({
  parameters,
  onParametersChange,
  onSubmit,
  isLoading,
  isApiKeyMissing,
}) => {
  const [expandedHelpKey, setExpandedHelpKey] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    
    let processedValue: string | number | boolean = value;
    if (type === 'number') {
      processedValue = parseFloat(value);
      if (isNaN(processedValue)) { 
        if (e.target instanceof HTMLInputElement && e.target.min) processedValue = parseFloat(e.target.min);
        else processedValue = 0; 
      }
    } else if (type === 'checkbox' && e.target instanceof HTMLInputElement) {
      processedValue = e.target.checked;
    }

    onParametersChange({
      ...parameters,
      [name]: processedValue,
    });
  };

  const toggleHelp = (paramKey: string) => {
    setExpandedHelpKey(prevKey => prevKey === paramKey ? null : paramKey);
  };
  
  const gridItemClass = "flex flex-col"; // Removed space-y-1 to manage space with help text
  const labelContainerClass = "flex items-center space-x-1 mb-1";
  const labelClass = "text-sm font-medium text-gray-300";
  const inputClass = "bg-gray-700 border border-gray-600 text-gray-200 text-sm rounded-lg focus:ring-sky-500 focus:border-sky-500 block w-full p-2.5";
  const selectClass = `${inputClass} appearance-none`;
  const helpTextClass = "mt-1 p-2 text-xs text-sky-200 bg-sky-900/50 border border-sky-700 rounded-md";

  const renderInputWithHelp = (paramKey: keyof SimulationParams, defaultLabel: string, inputType: 'text' | 'number' | 'select', options?: {min?: number, max?: number, step?: number, selectOptions?: {value: string, label: string}[] } ) => {
    const description = PARAMETER_DESCRIPTIONS[paramKey];
    const id = paramKey;

    return (
      <div className={gridItemClass}>
        <div className={labelContainerClass}>
          <label htmlFor={id} className={labelClass}>{defaultLabel}</label>
          {description && (
            <button 
              type="button" 
              onClick={() => toggleHelp(paramKey)} 
              className="text-gray-400 hover:text-sky-400"
              aria-label={`Show help for ${defaultLabel}`}
              aria-expanded={expandedHelpKey === paramKey}
            >
              <HelpIcon className="w-4 h-4" />
            </button>
          )}
        </div>
        {inputType === 'select' ? (
          <select 
            id={id} 
            name={paramKey} 
            value={parameters[paramKey] as string} 
            onChange={handleChange} 
            className={selectClass}
          >
            {options?.selectOptions?.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
          </select>
        ) : (
          <input 
            type={inputType} 
            id={id} 
            name={paramKey} 
            value={parameters[paramKey] as string | number} 
            onChange={handleChange} 
            className={inputClass} 
            min={options?.min}
            max={options?.max}
            step={options?.step}
          />
        )}
        {expandedHelpKey === paramKey && description && (
          <div className={helpTextClass} role="status">
            {description}
          </div>
        )}
      </div>
    );
  };


  return (
    <div className="bg-gray-800 p-6 rounded-xl shadow-2xl">
      <h3 className="text-2xl font-semibold text-sky-400 mb-6 border-b border-gray-700 pb-3">Pipeline Parameters</h3>
      
      {isApiKeyMissing && (
        <div className="mb-4 p-3 bg-red-700 text-white rounded-md text-sm">
          Warning: API_KEY is not configured. Gemini analysis features will be disabled.
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-5 mb-6"> {/* Increased gap-y slightly */}
        {renderInputWithHelp('generations', 'Generations', 'number', {min:1})}
        {renderInputWithHelp('seed', 'Seed', 'number')}
        {renderInputWithHelp('pop_size', 'Population Size', 'number', {min:2})}
        {renderInputWithHelp('max_lookback_data_option', 'Data Lookback Option', 'select', {selectOptions: [
            {value: "common_1200", label: "Common 1200"},
            {value: "specific_long_10k", label: "Specific Long 10k"},
            {value: "full_overlap", label: "Full Overlap"}
        ]})}
        {renderInputWithHelp('min_common_points', 'Min Common Points', 'number', {min:100})}
        {renderInputWithHelp('eval_lag', 'Evaluation Lag (bars)', 'number', {min:0})}
        {renderInputWithHelp('top_to_backtest', 'Top N to Backtest', 'number', {min:1})}
        {renderInputWithHelp('fee', 'Fee (bps)', 'number', {min:0, step: 0.1})}
        {renderInputWithHelp('hold', 'Hold Period (bars)', 'number', {min:1})}
        {renderInputWithHelp('scale', 'Signal Scaling', 'select', {selectOptions: [
             {value: "zscore", label: "Z-Score"},
             {value: "rank", label: "Rank"},
             {value: "sign", label: "Sign"}
        ]})}
        {renderInputWithHelp('data_dir', 'Data Directory', 'text')}
        {renderInputWithHelp('tournament_k', 'Tournament K', 'number', {min:1})}
        {renderInputWithHelp('p_mut', 'Mutation Probability', 'number', {min:0, max:1, step:0.01})}
        {renderInputWithHelp('p_cross', 'Crossover Probability', 'number', {min:0, max:1, step:0.01})}
        {renderInputWithHelp('elite_keep', 'Elite Keep Count', 'number', {min:0})}
        {renderInputWithHelp('fresh_rate', 'Fresh Rate (Novelty)', 'number', {min:0, max:1, step:0.01})}
        {renderInputWithHelp('max_ops', 'Max Operations', 'number', {min:1})}
        {renderInputWithHelp('parsimony_penalty', 'Parsimony Penalty', 'number', {min:0, step:0.001})}
        {renderInputWithHelp('corr_penalty_w', 'Correlation Penalty Weight', 'number', {min:0, step:0.01})}
        {renderInputWithHelp('corr_cutoff', 'Correlation Cutoff', 'number', {min:0, max:1, step:0.01})}
        {renderInputWithHelp('hof_size', 'Hall of Fame Size', 'number', {min:1})}
      </div>
      
      <button
        onClick={onSubmit}
        disabled={isLoading} 
        className="w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-xl shadow-sm text-white bg-sky-600 hover:bg-sky-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-sky-500 disabled:bg-gray-500 disabled:cursor-not-allowed transition-colors duration-150"
        aria-label="Prepare Local Run Command"
      >
        {isLoading ? (
          <>
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Preparing...
          </>
        ) : (
          <>
            <PlayIcon className="w-5 h-5 mr-2" />
            Prepare Local Run Command
          </>
        )}
      </button>
    </div>
  );
};

export default ParameterForm;
