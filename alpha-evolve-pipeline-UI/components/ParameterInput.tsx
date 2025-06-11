
import React from 'react';
import { ParameterDefinition } from '../types'; // AppParameters removed as it's not directly used here

interface ParameterInputProps {
  paramDef: ParameterDefinition;
  value: string | number | boolean;
  onChange: (id: string, value: string | number | boolean) => void;
}

const ParameterInput: React.FC<ParameterInputProps> = ({ paramDef, value, onChange }) => {
  const commonInputClass = "mt-1 block w-full px-3 py-2 bg-slate-600 border border-slate-500 text-slate-100 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm disabled:bg-slate-800 disabled:text-slate-400 disabled:border-slate-700 disabled:cursor-not-allowed placeholder-slate-400";
  
  const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    let newValue: string | number | boolean;
    if (paramDef.type === 'number') {
      newValue = event.target.value === '' ? '' : Number(event.target.value);
    } else if (paramDef.type === 'boolean') {
      newValue = (event.target as HTMLInputElement).checked;
    } else {
      newValue = event.target.value;
    }
    onChange(paramDef.id, newValue);
  };

  const renderInput = () => {
    switch (paramDef.type) {
      case 'number':
        return (
          <input
            type="number"
            id={paramDef.id}
            name={paramDef.id}
            value={value as number}
            onChange={handleChange}
            step={paramDef.step}
            min={paramDef.min}
            max={paramDef.max}
            className={commonInputClass}
            aria-describedby={paramDef.description ? `${paramDef.id}-description` : undefined}
          />
        );
      case 'text':
        return (
          <input
            type="text"
            id={paramDef.id}
            name={paramDef.id}
            value={value as string}
            onChange={handleChange}
            className={commonInputClass}
            placeholder={paramDef.id === 'log_file' ? '(optional, e.g., ./run.log)' : undefined}
            aria-describedby={paramDef.description ? `${paramDef.id}-description` : undefined}
          />
        );
      case 'select':
        return (
          <select
            id={paramDef.id}
            name={paramDef.id}
            value={value as string | number}
            onChange={handleChange}
            className={commonInputClass}
            aria-describedby={paramDef.description ? `${paramDef.id}-description` : undefined}
          >
            {paramDef.options?.map(option => (
              <option key={option.value.toString()} value={option.value} className="bg-slate-600 text-slate-100">
                {option.label}
              </option>
            ))}
          </select>
        );
      case 'boolean':
        return (
          <div className="flex items-center h-full mt-1">
            <input
              type="checkbox"
              id={paramDef.id}
              name={paramDef.id}
              checked={value as boolean}
              onChange={handleChange}
              className="h-5 w-5 text-indigo-500 border-slate-500 rounded focus:ring-2 focus:ring-indigo-400 bg-slate-600 focus:ring-offset-2 focus:ring-offset-slate-700"
              aria-describedby={paramDef.description ? `${paramDef.id}-description` : undefined}
            />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-x-4 items-start">
      <label htmlFor={paramDef.id} className="block text-sm font-medium text-slate-300 md:col-span-1 py-2">
        {paramDef.label}
      </label>
      <div className="md:col-span-2">
        {renderInput()}
        {paramDef.description && (
          <p id={`${paramDef.id}-description`} className="mt-1 text-xs text-slate-400">{paramDef.description}</p>
        )}
      </div>
    </div>
  );
};

export default ParameterInput;
