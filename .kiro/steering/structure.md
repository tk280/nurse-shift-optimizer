## Directory Patterns

### Solver Core
**Location**: `/my-first-vscode-repo/optimizer/`

**Purpose**: Houses the optimization solver, explicitly separated by responsibility.

The solver core is internally organized into the following conceptual modules:

- **model/**  
  Owns solver initialization, variable definitions, and shared expressions.
  This layer defines *what can be decided* but does not encode constraints.

- **constraints/**  
  Owns all constraint definitions.
  Hard and soft constraints are defined independently and registered into the model.
  Constraint logic is never written directly in model files.

- **objectives/**  
  Owns objective and penalty aggregation logic.
  Soft constraints expose penalty variables that are composed here.

- **schemas.py**  
  Dataclasses representing problem definitions and normalized inputs.

- **config.py**  
  Policy and configuration loading helpers.

Solver orchestration composes these modules but does not define constraints inline.

**Non-goals**:
- Model files do not contain constraint expressions.
- Constraint modules do not access filesystem or CLI concerns.
