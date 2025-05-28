
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { SimulationParams, ActualRunUserInput, ParameterSuggestion, AnalyzedIteration } from '../types';

const MAX_HISTORY_IN_PROMPT = 4; // Send current + up to 4 past iterations (total 5 items max related to history)

// Helper function to construct the full prompt string (can be used for preview)
export const getFullPromptForPreview = (
  currentParams: SimulationParams,
  currentUserInput: ActualRunUserInput,
  history: AnalyzedIteration[], // Full history, will be sliced internally for prompt
  pythonCodeContext: { [key: string]: string }
): string => {
  let historyPromptSegment = "";
  if (history.length > 0) {
    historyPromptSegment = "\\n\\n## History of Previous Experiment Iterations (most recent first):\\n";
    // Take up to MAX_HISTORY_IN_PROMPT most recent iterations for the prompt
    const historyForPrompt = history.slice(0, MAX_HISTORY_IN_PROMPT);

    historyForPrompt.forEach((iter, index) => {
      historyPromptSegment += `\\n### Iteration ${history.length - index} (Past):\\n`;
      historyPromptSegment += `1.  **Parameters Used for that Run:**\\n\`\`\`json\\n${JSON.stringify(iter.paramsUsed, null, 2)}\\n\`\`\`\\n`;
      historyPromptSegment += `2.  **User's Console Output from that Run:**\\n---\\n${iter.userInput.consoleOutput || "Not provided."}\\n---\\n`;
      historyPromptSegment += `3.  **User's Personal Opinion on that Run:**\\n---\\n${iter.userInput.personalOpinion || "Not provided."}\\n---\\n`;
      historyPromptSegment += `4.  **Your (Gemini's) Suggestion Following that Run:**\\n`;
      historyPromptSegment += `    *   Justification: ${iter.geminiResponse.justification}\\n`;
      historyPromptSegment += `    *   Suggested Params for Next Step:\\n\`\`\`json\\n${JSON.stringify(iter.geminiResponse.suggestedParams, null, 2)}\\n\`\`\`\\n`;
    });
  }

  const prompt = `
You are an expert quantitative finance researcher and Python programmer, assisting a user in an iterative process of optimizing parameters for an algorithmic trading alpha generation pipeline.

The user is providing you with the parameters they used for their most recent local run, the console output from that run, and their personal opinion/observations. They are also providing a history of recent previous iterations if available.

Your goal is to suggest a NEW set of 'SimulationParams' for the user's NEXT local run and provide a justification.
Consider the entire experiment history to understand what has been tried and what the user is aiming for. Avoid repeating suggestions that did not yield desired outcomes unless the user explicitly asks to revisit them or provides new context.

The relevant Python scripts involved are:
<run_pipeline_py_code>
${pythonCodeContext['run_pipeline.py'] || "run_pipeline.py code not available."}
</run_pipeline_py_code>

<evolve_alphas_py_code>
${pythonCodeContext['evolve_alphas.py'] || "evolve_alphas.py code not available."}
</evolve_alphas_py_code>

<alpha_program_core_py_code>
${pythonCodeContext['alpha_program_core.py'] || "alpha_program_core.py code not available."}
</alpha_program_core_py_code>
${historyPromptSegment}

## Current Situation for Analysis:
The user has just completed a local run using the following:
1.  **Current Parameters Used:**
\`\`\`json
${JSON.stringify(currentParams, null, 2)}
\`\`\`
2.  **User's Console Output from This Run:**
---
${currentUserInput.consoleOutput || "Not provided. Assume the run completed without explicit errors unless mentioned in Personal Opinion."}
---
3.  **User's Personal Opinion on This Run:**
---
${currentUserInput.personalOpinion || "Not provided. Focus on general improvements or exploration if opinion is missing."}
---

Your Task:
Based on the "Current Situation for Analysis" (parameters, console output, personal opinion) AND the "History of Previous Experiment Iterations" (if any), suggest a NEW set of 'SimulationParams' for the user's NEXT local run.
Your goal is to guide the user towards parameters that might yield improved performance (e.g., better qualitative results as described by the user, different alpha characteristics), or explore different aspects of the search space based on their 'Personal Opinion' or any issues noted in the 'Console Output'.

Output Format:
Return ONLY a single JSON object with two keys: "suggestedParams" and "justification".
- "suggestedParams": A valid JSON object strictly conforming to the SimulationParams interface. Ensure all fields from SimulationParams are present with valid values.
- "justification": A string explaining your reasoning for the suggested parameter changes, directly addressing the user's inputs (console output and personal opinion from the CURRENT run) and how the new parameters aim to achieve their goals or address observations, considering the historical context.

Example of a valid SimulationParams structure (values will differ):
{
  "generations": 20,
  "seed": 101,
  "pop_size": 128,
  "max_lookback_data_option": "full_overlap",
  "min_common_points": 1000,
  "eval_lag": 0,
  "top_to_backtest": 15,
  "fee": 0.5,
  "hold": 2,
  "scale": "rank",
  "data_dir": "./data",
  "tournament_k": 7,
  "p_mut": 0.35,
  "p_cross": 0.65,
  "elite_keep": 5,
  "max_ops": 25,
  "parsimony_penalty": 0.005,
  "corr_penalty_w": 0.1,
  "corr_cutoff": 0.25,
  "hof_size": 25,
  "fresh_rate": 0.05
}

Focus on suggesting incremental, logical changes. If the 'Console Output' is minimal, base your suggestions more heavily on the user's 'Personal Opinion' and their stated goals, in light of the history. Ensure your 'suggestedParams' output is a complete and valid SimulationParams object.
Critically consider if the 'Personal Opinion' (current and historical) implies a desire for exploration or refinement.
`;
  return prompt;
};


export class GeminiService {
  private ai: GoogleGenAI;
  private modelName = 'gemini-2.5-flash-preview-04-17';

  constructor(apiKey: string) {
    this.ai = new GoogleGenAI({ apiKey });
  }

  private cleanJsonString(jsonStr: string): string {
    let cleaned = jsonStr.trim();
    const fenceRegex = /^```(\w*json)?\\s*\\n?(.*?)\\n?\\s*```$/s; 
    const match = cleaned.match(fenceRegex);
    if (match && match[2]) {
      cleaned = match[2].trim();
    }
    return cleaned;
  }

  async analyzeActualResultsAndSuggestNextParams(
    currentParams: SimulationParams,
    currentUserInput: ActualRunUserInput,
    history: AnalyzedIteration[], // New parameter for iteration history
    pythonCodeContext: { [key: string]: string }
  ): Promise<ParameterSuggestion> {
    
    const fullPrompt = getFullPromptForPreview(currentParams, currentUserInput, history, pythonCodeContext);

    try {
      const response: GenerateContentResponse = await this.ai.models.generateContent({
        model: this.modelName,
        contents: fullPrompt,
        config: { responseMimeType: "application/json" }
      });
      
      const jsonStr = this.cleanJsonString(response.text);
      const parsedData = JSON.parse(jsonStr) as ParameterSuggestion;

      if (parsedData && typeof parsedData.suggestedParams === 'object' && typeof parsedData.justification === 'string') {
        const requiredKeys: (keyof SimulationParams)[] = [
            "generations", "seed", "pop_size", "max_lookback_data_option", "min_common_points", 
            "eval_lag", "top_to_backtest", "fee", "hold", "scale", "data_dir", "tournament_k", 
            "p_mut", "p_cross", "elite_keep", "max_ops", "parsimony_penalty", "corr_penalty_w", 
            "corr_cutoff", "hof_size", "fresh_rate"
        ];
        const missingKeys = requiredKeys.filter(key => !(key in parsedData.suggestedParams));
        if (missingKeys.length > 0) {
            console.error("Parsed JSON for parameter suggestion is missing keys:", missingKeys, parsedData.suggestedParams);
            throw new Error(`Received malformed parameter suggestion: missing keys ${missingKeys.join(', ')}.`);
        }
        // Type cast for specific enum-like fields
        const lookbackOptions: SimulationParams['max_lookback_data_option'][] = ['common_1200', 'specific_long_10k', 'full_overlap'];
        if (!lookbackOptions.includes(parsedData.suggestedParams.max_lookback_data_option)) {
          parsedData.suggestedParams.max_lookback_data_option = 'common_1200'; // Default if invalid
        }
        const scaleOptions: SimulationParams['scale'][] = ['zscore', 'rank', 'sign'];
        if (!scaleOptions.includes(parsedData.suggestedParams.scale)) {
          parsedData.suggestedParams.scale = 'zscore'; // Default if invalid
        }
        return parsedData;
      } else {
        console.error("Parsed JSON for parameter suggestion has incorrect structure:", parsedData);
        throw new Error("Received malformed parameter suggestion data from API.");
      }

    } catch (error) {
      console.error('Gemini API error in analyzeActualResultsAndSuggestNextParams or JSON parsing:', error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      // Include part of the prompt for debugging, but be careful with length
      const promptSnippet = fullPrompt.substring(0, 500) + (fullPrompt.length > 500 ? "..." : "");
      throw new Error(`Gemini API/JSON error: ${errorMessage}. Prompt snippet: ${promptSnippet}`);
    }
  }
}
