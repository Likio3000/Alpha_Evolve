
import React from 'react';

interface TokenRule {
  name: string;
  regex: RegExp;
  color: string;
  isKeyword?: boolean;
  isFunctionDef?: boolean;
}

// New color palette optimized for dark backgrounds
const vscodeTokenColors = {
  comment: '#8B949E', // GitHub Gray - for # comments and content of docstrings
  docstringDelimiters: '#6B7280', // Tailwind gray-500 for """ and '''
  string: '#9ECBFF', // Light Blue
  keywordImport: '#C981E7', // Light Purple
  functionNameDefinition: '#7EE787', // Light Green (GitHub Green-like)
  parameter: '#FFAB70', // Light Orange/Peach - Note: current param highlighting is limited
  keywordControlFlow: '#F97583', // Pinkish-Red (GitHub Red-like)
  decoratorName: '#79C0FF', // Bright Blue (GitHub Blue-like for entities)
  constantLanguage: '#58A6FF', // Medium Blue
  keywordDefault: '#F97583', // Pinkish-Red (same as control flow for consistency)
  number: '#A5D6A7', // Soft Green (distinct from function names)
};

const tokenRules: TokenRule[] = [
  // Docstrings must come before general comments and strings
  // The 'color' here will be for the delimiters. Content will use 'comment' color.
  { name: 'docstring_triple_double', regex: /"""([\s\S]*?)"""/g, color: vscodeTokenColors.docstringDelimiters },
  { name: 'docstring_triple_single', regex: /'''([\s\S]*?)'''/g, color: vscodeTokenColors.docstringDelimiters },
  // Strings
  { name: 'string_double_quote', regex: /"(?:\\.|[^"\\])*"/g, color: vscodeTokenColors.string },
  { name: 'string_single_quote', regex: /'(?:\\.|[^'\\])*'/g, color: vscodeTokenColors.string },
  // Single-line comments
  { name: 'comment_single_line', regex: /#.*$/gm, color: vscodeTokenColors.comment },

  // Decorators - capture the name after @
  { name: 'decorator', regex: /(@)([a-zA-Z_]\w*)/g, color: vscodeTokenColors.decoratorName }, // Style group 2 (name)

  // Function definitions - capture 'def' and 'name'
  { name: 'function_definition', regex: /\b(def)\s+([a-zA-Z_]\w*)\s*(\()/g, color: vscodeTokenColors.functionNameDefinition, isFunctionDef: true }, // Style group 2 (name)

  // Keywords
  { name: 'keyword_import', regex: /\b(from|import)\b/g, color: vscodeTokenColors.keywordImport, isKeyword: true },
  { name: 'keyword_control_flow', regex: /\b(if|else|elif|for|while|return|try|except|finally|with|as|yield|pass|break|continue)\b/g, color: vscodeTokenColors.keywordControlFlow, isKeyword: true },
  { name: 'keyword_def_class', regex: /\b(class)\b/g, color: vscodeTokenColors.keywordDefault, isKeyword: true }, // 'def' is handled by function_definition
  { name: 'keyword_operators', regex: /\b(and|or|not|in|is)\b/g, color: vscodeTokenColors.keywordDefault, isKeyword: true },
  { name: 'keyword_other', regex: /\b(lambda|global|nonlocal|assert|del|raise|async|await)\b/g, color: vscodeTokenColors.keywordDefault, isKeyword: true },

  // Language Constants
  { name: 'constant_language', regex: /\b(True|False|None|self|cls)\b/g, color: vscodeTokenColors.constantLanguage },
  // Numbers
  { name: 'number', regex: /\b\d+(\.\d+)?([eE][+-]?\d+)?\b/g, color: vscodeTokenColors.number },
];


interface HighlightedCodeProps {
  code: string;
}

const HighlightedCode: React.FC<HighlightedCodeProps> = ({ code }) => {
  
  const highlightTokens = (fullCodeText: string): React.ReactNode[] => {
    let elements: React.ReactNode[] = [fullCodeText];

    for (const rule of tokenRules) {
      const newElements: React.ReactNode[] = [];
      elements.forEach(element => {
        if (typeof element === 'string') {
          const text = element;
          let lastIndex = 0;
          let match;
          // Ensure regex is created new for each use in loop to reset lastIndex if it's global
          const regex = new RegExp(rule.regex); 

          while ((match = regex.exec(text)) !== null) {
            if (match.index > lastIndex) {
              newElements.push(text.substring(lastIndex, match.index));
            }
            
            let tokenToStyle = match[0];
            let fullMatch = match[0]; 
            let prefix = "";
            let suffix = "";

            if (rule.name === 'docstring_triple_double' || rule.name === 'docstring_triple_single') {
              const openQuotes = rule.name === 'docstring_triple_double' ? '"""' : "'''";
              const closeQuotes = openQuotes;
              const content = match[1] || ""; 

              newElements.push(<span key={`${rule.name}-${match.index}-open`} style={{ color: rule.color }}>{openQuotes}</span>);
              newElements.push(<span key={`${rule.name}-${match.index}-content`} style={{ color: vscodeTokenColors.comment }}>{content}</span>);
              newElements.push(<span key={`${rule.name}-${match.index}-close`} style={{ color: rule.color }}>{closeQuotes}</span>);
              
            } else if (rule.name === 'decorator' && match[2]) { 
              prefix = match[1]; 
              tokenToStyle = match[2]; 
              if (prefix) newElements.push(prefix);
              newElements.push(<span key={`${rule.name}-${match.index}-${tokenToStyle}`} style={{ color: rule.color }}>{tokenToStyle}</span>);
            } else if (rule.isFunctionDef && match[2]) { 
              prefix = `${match[1]} `; 
              tokenToStyle = match[2]; 
              suffix = match[3] ? `${match[3]}` : ""; 
              
              if (prefix) newElements.push(prefix); 
              newElements.push(<span key={`${rule.name}-${match.index}-${tokenToStyle}`} style={{ color: rule.color }}>{tokenToStyle}</span>);
              if (suffix) newElements.push(suffix); 
            } else {
              newElements.push(<span key={`${rule.name}-${match.index}-${tokenToStyle}`} style={{ color: rule.color }}>{tokenToStyle}</span>);
            }
            
            lastIndex = regex.lastIndex;
            if (lastIndex === match.index) { // Prevent infinite loops for zero-length or certain global regex matches
                if (match[0].length > 0) { // If it was a non-zero length match but lastIndex didn't advance
                     lastIndex = match.index + match[0].length;
                } else { // Zero-length match, force advance
                    lastIndex++;
                }
            }
          }
          if (lastIndex < text.length) {
            newElements.push(text.substring(lastIndex));
          }
        } else {
          newElements.push(element);
        }
      });
      elements = newElements;
    }
    return elements;
  };

  return <>{highlightTokens(code)}</>;
};

export default HighlightedCode;
