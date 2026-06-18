*This project has been created as part of the 42 curriculum by wabdi.*
 
# Call Me, Maybe?
 
## Description
 
**Call Me Maybe** is a function-calling framework that translates natural-language prompts into structured, schema-compliant function calls using a small language model (Qwen/Qwen3-0.6B by default). Given a prompt like *"What is the sum of 40 and 2?"*, the system does not return `42` — it returns:
 
```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```
 
The core idea is to decompose the task into two semantic inference problems handled by the LLM under constrained decoding, and one deterministic assembly step handled entirely by Python. This separation guarantees 100% valid JSON output regardless of model quality, while keeping the LLM responsible only for the decisions that require linguistic understanding.
 
---
 
## Instructions
 
### Requirements
 
- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) package manager
### Installation
 
Clone the repository, then run:
 
```bash
uv sync
```
 
This installs all dependencies including `pydantic`, `numpy`, and the local `llm_sdk` package.
 
### Running the program
 
```bash
make run
```
 
Or explicitly with custom paths:
 
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --model Qwen/Qwen3-0.6B
```
 
All arguments are optional and fall back to the defaults above.
 
### Other Makefile targets
 
```bash
make install      # install dependencies via uv sync
make debug        # run with Python debugger (pdb)
make clean        # remove __pycache__, .mypy_cache, *.pyc
make lint         # run flake8 and mypy with standard flags
make lint-strict  # run flake8 and mypy --strict
```
 
---
 
## Example Usage
 
Given `data/input/functions_definition.json`:
 
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"}
  }
]
```
 
And `data/input/function_calling_tests.json`:
 
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"}
]
```
 
Running `make run` produces `data/output/function_calling_results.json`:
 
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```
 
---
 
## Challenges Faced
 
**Token ID magic numbers** — The stopping conditions for generation (token 198 for newline, tokens 1 and 698 for quote characters) are hardcoded values specific to Qwen's vocabulary. Identifying these required inspecting the vocab file directly and running test generations to observe where the model naturally stopped.
 
**Type annotation with a third-party SDK** — The `llm_sdk` package ships without type stubs, causing mypy to reject all references to `Small_LLM_Model`. The solution was to define a `LLMModel` Protocol in `src/models.py` that describes the interface the application needs, and use that throughout the codebase. The concrete class is only referenced at instantiation in `__main__.py`, isolated with a single `# type: ignore[attr-defined]`.
 
**Flake8 crawling into `.venv`** — Running `flake8 .` caused a recursion error inside sympy's source tree within the virtual environment. Fixed by explicitly excluding `.venv` and `llm_sdk` in the flake8 invocation.
 
**String argument extraction** — Unlike numbers and booleans, strings have no natural stopping condition from a type perspective. The current approach stops on newline tokens and quote tokens (1, 698), which works well in practice but is vocabulary-dependent.
 
---
 
## Testing Strategy
 
The implementation was validated by:
 
- Running the pipeline on the provided sample input files and manually verifying the output JSON matches expected function names and argument values
- Testing edge cases: prompts with large numbers, prompts with ambiguous wording, functions with multiple parameters of different types
- Verifying that `json.loads(output)` succeeds on every produced file (structural validity)
- Confirming argument types match the schema — numbers as floats, booleans as Python bools, strings as str
- Running `make lint` and `make lint-strict` to confirm zero type and style violations
---
 
## Algorithm Explanation
 
### Abstract
 
This project studies the problem of function calling in large language models (LLMs) from a computational and algorithmic perspective. Given a natural-language prompt and a finite set of candidate function definitions, the objective is to infer a function identifier and its corresponding arguments. Rather than delegating the full structured output to the model, the system decomposes the task into semantic inference and deterministic structural assembly. This separation allows the LLM to operate only on subproblems that require linguistic interpretation, while Python guarantees syntactic validity, schema conformity, and output reproducibility.
 
The central technical ideas are: (i) prefix-constrained generation for function names and booleans, (ii) asymptotically improved token filtering using set membership, (iii) iterative context accumulation for argument extraction, and (iv) host-language construction of JSON output.
 
### 1. Introduction
 
Let $\mathcal{F} = \{f_1, f_2, \dots, f_m\}$ denote a finite set of available functions, and let $p$ be a user prompt drawn from a natural-language distribution. The task is to estimate a mapping
 
$$\Phi : p \mapsto (f, \theta),$$
 
where $f \in \mathcal{F}$ is the selected function and $\theta$ is a parameter assignment consistent with the function's schema.
 
This problem is difficult because the target object is not merely a sequence of tokens but a structured semantic object. A direct autoregressive formulation would require the model to generate both content and syntax, introducing failure modes related to invalid punctuation, malformed JSON, and schema inconsistency. The present system instead factorizes the problem into:
 
$$\Phi(p) = A(S(p), E(p)),$$
 
where $S(p)$ denotes function selection, $E(p)$ denotes argument extraction, and $A$ is deterministic Python-side assembly.
 
### 2. Formal Problem Setup
 
Assume each function $f_i \in \mathcal{F}$ is equipped with:
 
- a name $n_i$,
- a description $d_i$,
- a parameter set $\Pi_i = \{\pi_{i1}, \dots, \pi_{ik_i}\}$,
- and a return type $r_i$.
The goal is to infer:
 
$$f^* = \arg\max_{f_i \in \mathcal{F}} \; \Pr(f_i \mid p),$$
 
followed by parameter estimation
 
$$\theta^{\ast} = \arg\max_{\theta \in \Theta(f^{\ast})} \; \Pr(\theta \mid p, f^{\ast})$$

where $\Theta(f^{\ast})$ is the admissible parameter space determined by the schema of $f^*$.
 
### 3. Methodological Contributions
 
#### 3.1 Prefix-Constrained Decoding as a Trie-Like Search
 
The most important algorithmic component is the constrained generation of function names and boolean values. Let $\Sigma$ denote the token alphabet, and let $\mathcal{N} = \{n_1, \dots, n_m\}$ be the set of function names. During decoding, the system maintains a partial prefix $x_{1:t}$, and at each step restricts the next token to preserve consistency with at least one candidate name.
 
Formally, define the prefix-consistent candidate set
 
$$\mathcal{N}_t = \{n \in \mathcal{N} : x_{1:t} \prec n\},$$
 
where $x_{1:t} \prec n$ means $x_{1:t}$ is a prefix of $n$. The admissible next-character set is then
 
$$C_t = \{c \in \Sigma : \exists n \in \mathcal{N}_t \text{ such that } n_{t+1} = c\}.$$
 
The decoder masks all tokens whose surface form does not begin with an element of $C_t$. This creates a trie-like search process over the space of function names. If the branching factor at depth $t$ is $b_t = |C_t|$, the search space shrinks from the unconstrained vocabulary size $|\mathcal{V}|$ to a much smaller admissible subset.
 
For boolean values, the admissible language is
 
$$\mathcal{B} = \{\texttt{true}, \texttt{false}\},$$
 
and the same prefix-based procedure is applied. This ensures that every generated boolean token sequence remains in the regular language defined by $\mathcal{B}$.
 
#### 3.2 Complexity Reduction via Set Membership
 
The implementation uses set membership to filter valid token identifiers. Let $\mathcal{V}$ denote the vocabulary, $|\mathcal{V}| = n$, and let $K_t \subseteq \mathcal{V}$ denote the admissible token set at decoding step $t$, with $|K_t| = k_t$.
 
If admissible tokens are stored as a list, then each membership query has worst-case cost $O(k_t)$, yielding a total filtering cost of
 
$$O(n \cdot k_t)$$
 
per decoding step, since each of the $n$ logits must be checked against the admissible set.
 
If admissible tokens are stored as a hash-based set, membership is $O(1)$ on average, giving total filtering complexity
 
$$O(n).$$
 
This is the correct asymptotic improvement: from $O(n \cdot k_t)$ to $O(n)$. Given a model such as Qwen with a vocabulary on the order of $n \approx 1.5 \times 10^5$, the reduction in multiplicative overhead is practically significant.
 
#### 3.3 Python-Owned JSON Assembly
 
A major design decision is to let Python construct the JSON structure entirely. Let $J$ denote the output JSON object. Instead of asking the model to emit the full tree $J$, the system computes
 
$$J = \mathcal{A}(p, f^{\ast}, \theta^{\ast}),$$
 
where $\mathcal{A}$ is a deterministic assembler implemented in Python.
 
This decomposition has three consequences:
 
1. **Structural correctness becomes invariant** under the assembly function $\mathcal{A}$.
2. **The language model is removed from the syntax-generation burden**, reducing the number of LLM inference calls — the most expensive operation in the pipeline. Every `{`, `"key"`, `:`, `,`, `}` not generated by the model is a forward pass saved.
3. **Type compliance is enforced by the host runtime**, rather than approximated through token constraints.
#### 3.4 Sequential Argument Extraction as Context Accumulation
 
Let $\theta = (\theta_1, \dots, \theta_k)$ be the parameter vector of a function. The extraction process is sequential: after estimating $\theta_1, \dots, \theta_{j-1}$, the system conditions the next prompt on the partial assignment
 
$$\Theta_{j-1} = \{(\pi_1, \theta_1), \dots, (\pi_{j-1}, \theta_{j-1})\}.$$
 
This is realized textually by concatenating previously extracted assignments:
 
$$\texttt{already} = \bigoplus_{i=1}^{j-1} \left(\pi_i = \theta_i\right).$$
 
This acts as a lightweight conditioning operator. It reduces the entropy of subsequent predictions by supplying the model with a compact summary of the current state, lowering the risk of redundant or inconsistent estimates across parameters.
 
#### 3.5 Decoupled Model Interface via Protocol
 
The project uses a `LLMModel` Protocol as an abstraction layer between application logic and the underlying inference backend. The application depends on the interface $\mathcal{I}$ rather than a particular implementation $\mathcal{M}$:
 
$$\mathcal{M} \models \mathcal{I}$$
 
Any backend satisfying the Protocol can be substituted without changing surrounding code. Combined with the `--model` CLI argument, this makes the pipeline model-agnostic: it works with any HuggingFace causal LM supported by the SDK, not just Qwen/Qwen3-0.6B.
 
### 4. Computational Interpretation
 
The pipeline can be understood as a composition of mappings:
 
$$p \xrightarrow{S} f^* \xrightarrow{E} \theta^* \xrightarrow{\mathcal{A}} J.$$
 
Each stage is constrained differently:
 
- $S$: constrained lexical generation over function names,
- $E$: type-aware extraction over parameter values,
- $\mathcal{A}$: deterministic JSON construction.
This layered formulation yields a system with substantially lower failure probability than unconstrained end-to-end generation, because the most error-prone structural part is removed from the stochastic model.
 
---
 
## Design Decisions
 
- **Protocol over concrete class** — `LLMModel` in `src/models.py` decouples all application logic from `llm_sdk`, making the codebase testable and model-agnostic.
- **Set-based token filtering** — valid token sets are converted to Python `set` before the masking loop, reducing per-step complexity from $O(n \cdot k_t)$ to $O(n)$.
- **Python JSON assembly** — `json.dump()` owns the output structure, guaranteeing schema and syntax validity unconditionally.
- **Pydantic for schema validation** — all function definitions are parsed and validated through Pydantic models, catching malformed input before it reaches generation.
- **Argparse defaults matching the spec** — all default paths mirror the subject specification exactly so the program runs correctly with zero arguments.
---
 
## Performance Analysis
 
- **JSON validity**: 100% — structure is Python-assembled, not model-generated.
- **Function selection accuracy**: depends on model quality and prompt clarity; the constrained prefix search ensures the output is always a valid function name from the provided set.
- **Speed**: the main bottleneck is LLM forward passes. By keeping the model responsible only for semantic tokens (function name characters, argument values), the number of inference calls is minimized relative to full JSON generation approaches.
- **Vocabulary size**: Qwen/Qwen3-0.6B uses $|\mathcal{V}| \approx 1.5 \times 10^5$ tokens. The set-based filtering processes this in $O(n)$ per step regardless of how many valid tokens exist.
---
 
## Resources
 
### References
 
- Ziegler et al., *Typesafe LLM Decoding with Outlines*, 2023
- HuggingFace Transformers documentation: https://huggingface.co/docs/transformers
- Pydantic documentation: https://docs.pydantic.dev
- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- BPE tokenization — Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016
### AI Usage
 
AI assistance (Claude, Anthropic) was used during this project for the following:
 
- Identifying and fixing mypy type errors related to the `LLMModel` Protocol design
- Reviewing flake8 violations and suggesting fixes for line length and spacing
- Discussing the correct asymptotic complexity of the set-membership optimization
- Reviewing the README for mathematical notation correctness
All generated suggestions were reviewed, understood, and adapted before inclusion. No code was copied without being read and validated against the project's logic.
