
export type ParameterType = 'number' | 'text' | 'select' | 'boolean';

export interface ParameterOption {
  value: string | number;
  label: string;
}

export interface ParameterDefinition {
  id: string;
  label: string;
  type: ParameterType;
  cliFlag: string;
  defaultValue: string | number | boolean;
  options?: ParameterOption[];
  step?: number;
  min?: number;
  max?: number;
  description?: string;
  isPositional?: boolean; // For 'generations'
}

export interface ParameterSection {
  title: string;
  parameters: ParameterDefinition[];
}

export type AppParameters = Record<string, string | number | boolean>;
