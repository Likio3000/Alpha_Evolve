
import React, { useState } from 'react';
import { ParameterSection as SectionType, AppParameters } from '../types';
import ParameterInput from './ParameterInput';

interface SectionProps {
  section: SectionType;
  parameters: AppParameters;
  onParameterChange: (id: string, value: string | number | boolean) => void;
  defaultOpen?: boolean;
}

const Section: React.FC<SectionProps> = ({ section, parameters, onParameterChange, defaultOpen = true }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="mb-6 bg-slate-700 shadow-xl rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center px-6 py-4 bg-slate-600 hover:bg-slate-500 focus:outline-none transition-colors duration-150"
        aria-expanded={isOpen}
        aria-controls={`section-content-${section.title.replace(/\s+/g, '-')}`}
      >
        <h2 className="text-xl font-semibold text-slate-100">{section.title}</h2>
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          className={`h-6 w-6 text-slate-300 transform transition-transform duration-200 ${isOpen ? 'rotate-180' : 'rotate-0'}`} 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div 
          id={`section-content-${section.title.replace(/\s+/g, '-')}`}
          className="p-6 border-t border-slate-600"
        >
          {section.parameters.map(paramDef => (
            <ParameterInput
              key={paramDef.id}
              paramDef={paramDef}
              value={parameters[paramDef.id]}
              onChange={onParameterChange}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default Section;
