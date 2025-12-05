# AI-Driven Emergency Response System under Real-Time Traffic and Resource Constraints

This project models an AI-assisted emergency response system for a **rainy Saturday evening in Bhubaneswar** just after a large football match. The system has to react to multiple road accidents, heavy and unpredictable traffic, limited ambulances, and dynamic hospital availability.

The codebase is organised into **five main modules**:

1. **Module 1 – Bayesian Accident Severity & Future Risk Estimation**
2. **Module 2 – Predictive Search-Based Ambulance Route Optimisation**
3. **Module 3 – Adaptive Planning with GraphPlan & Partial-Order Planning (POP)**
4. **Module 4 – Reinforcement Learning for Proactive Resource Allocation**
5. **Module 5 – LLM-Based Control Room Summary & Decision Justification**

A small **common scenario layer** (hospital/ambulance/accident definitions) is shared by all modules so that they operate on the same realistic setup.

---

## 1. Project Structure

```text
AI_EMERGENCY_RESPONSE/
├── common/
│   ├── __init__.py
│   ├── constants.py        # Enums for weather, etc.
│   ├── entities.py         # Accident, Hospital, Ambulance dataclasses
│   └── scenario.py         # create_base_scenario() used by multiple modules
│
├── module1_bayes/
│   ├── __init__.py
│   ├── bayes_model.py          # Bayesian Network structure + CPDs
│   ├── inference_engine.py     # Wrapper around pgmpy VariableElimination
│   ├── inference_demo.py       # Demo: run BN on ACC1
│   └── data_analysis_optional.py  # Optional: analyse CSV to calibrate priors
│
├── module2_search/
|   ├── __init__.py
|   ├── search_algorithms.py   # UCS and A* implementations
|   └── scenario_routes.py     # builds graphs, runs both, compares routes
│
├── module3_planning/
|   ├── __init__.py
|   ├── planning_domain.py   # Fluents, actions, initial state, goals
|   ├── graphplan.py         # Simple GraphPlan-style planner (returns strict sequence)
|   ├── pop_planner.py       # Partial-order planner structure with contingencies
|   └── planning_demo.py     # Runs both planners and prints plans
│
├── module4_rl/
|   ├── __init__.py
|   ├── env.py               # MDP environment
|   ├── q_learning.py        # Q-learning agent
|   ├── baseline_policy.py   # hand-crafted dispatch-immediately policy
|   └── run_experiments.py   # training + evaluation script
│
├── module5_llm/
│   ├── __init__.py
│   ├── prompt_templates.py     # DecisionContext + system/user prompts
│   ├── summarizer.py           # Generate briefing (LLM or deterministic fallback)
│   └── demo_llm_summary.py     # Non-trivial control room summary demo
│
├── data/
│   └── india_road_accident_severity.csv  # Public accident dataset (Module 1)
│
├── requirements.txt  (recommended – see Dependencies section)
└── README.md
```

---

## 2. Scenario Overview (Common Setup)

All modules share the same basic scenario:

- **City:** Bhubaneswar, Odisha  
- **Time:** 2025-07-12, around **20:15 hrs**  
- **Weather:** Heavy **rain**, reduced visibility  
- **Context:** Large football match has just ended near the city stadium  
- **Traffic:** Unpredictable congestion and diversions near the stadium, Jaydev Vihar, Rupali Square, and Airport corridors  

**Hospitals**

- **H1 – H1 Trauma Hospital**  
  - ~11 km from stadium cluster  
  - Level-1 trauma centre, CT scanner available, neurosurgery on call  

- **H2 – H2 General Hospital**  
  - ~6 km from stadium cluster  
  - General emergency care, **CT scanner offline** in this scenario  

**Ambulances**

- **A1** – Based at H1, can be pre-deployed near Nayapalli / stadium corridor  
- **A2** – Based at H2, naturally covers Rupali / central areas  

**Accidents**

- **ACC1:** High-speed bike collision at **Jaydev Vihar flyover**  
- **ACC2:** Car–auto crash at **Rupali Square**  
- **ACC3:** SUV rollover with airbags deployed at **Airport approach road**

---

## 3. Installation and Setup

### 3.1. Python Version

- Recommended: **Python 3.10+**

### 3.2. Create and Activate Virtual Environment (optional but recommended)

```bash
cd AI_EMERGENCY_RESPONSE

python3 -m venv .venv
source .venv/bin/activate     # On Linux/macOS
# .venv\Scripts\activate      # On Windows (PowerShell/cmd)
```

### 3.3. Install Dependencies

Create a `requirements.txt` similar to:

```text
pgmpy
numpy
pandas
networkx
```

Then install:

```bash
pip install -r requirements.txt
```

## 4. How to Run Each Module

All commands below assume you are in the project root (`AI_EMERGENCY_RESPONSE`) with the virtual environment activated.

---

### 4.1. Module 1 – Bayesian Accident Severity & Future Risk

**What it does**

- Builds a Bayesian Network over:
  - ImpactSpeed, VehicleType, Weather, CallerUrgency  
  - TrafficDelayProb, CurrentSeverity, FutureRisk  
- Uses real dataset (`india_road_accident_severity.csv`) to calibrate the baseline severity distribution.  
- Runs inference for a **non-trivial case** (e.g., ACC1 – high-speed bike crash at Jaydev Vihar in rain with panicked caller).  
- Outputs posterior distributions for:
  - `CurrentSeverity` ∈ {minor, moderate, critical}  
  - `FutureRisk` ∈ {low, medium, high}  

#### 4.1.1. Optional: Check Dataset-Based Priors

```bash
python3 -m module1_bayes.data_analysis_optional
```

This prints the empirical distribution of accident severity and maps it to the BN states:

- `minor` (Slight Injury)  
- `moderate` (Serious Injury)  
- `critical` (Fatal injury)  

#### 4.1.2. Run Bayesian Inference Demo

```bash
python3 -m module1_bayes.inference_demo
```

**Expected behaviour (summary):**

- Loads the base scenario (including accidents).  
- Picks **ACC1** (high-speed bike collision near Jaydev Vihar flyover).  
- Converts ACC1 attributes + traffic state into Bayesian evidence.  
- Computes posteriors:

  - Posterior over `CurrentSeverity` (e.g., probabilities for minor, moderate, critical)  
  - Posterior over `FutureRisk` (low/medium/high, influenced by both severity and traffic delay)

The console output will show the evidence used and posterior probabilities, and print the **most probable severity** and **future risk** categories.

---

### 4.2. Module 2 – Predictive Search-Based Ambulance Route Optimisation

**What it does**

- Models the city as a **dynamic graph** with travel times affected by rain and event congestion.  
- Implements:
  - **Uniform-Cost Search (UCS)** on the “current” graph  
  - **A\*** search on a “future” graph that anticipates stadium diversions and heavier congestion  
- Compares:
  - A UCS path that looks fast now but may get blocked later  
  - An A\* path that is slightly longer at present but more reliable over time  

#### 4.2.1. Run Route Planning Demo

```bash
python3 -m module2_search.scenario_routes
```

**Expected behaviour (summary):**

- Start node: `hospital_h1`  
- Goal node: `airport_approach` (ACC3 location)  

The script prints:

- A **UCS route** under current conditions and its estimated travel time (e.g., via stadium).  
- An **A\*** route under predicted conditions (heavy stadium congestion and diversions).  
- A re-evaluation of the UCS route under future conditions (can become effectively **blocked** with infinite delay).  

The summary at the end explains why the A\* route is safer and more reliable even if it's longer at time zero.

---

### 4.3. Module 3 – Adaptive Planning with GraphPlan & POP

**What it does**

- Defines high-level actions such as:
  - `PreDeployAmbulanceNearHotspot`
  - `NotifyHospital`
  - `TriggerDroneRecon`
  - `RequestTrafficDiversion`
  - `StartTransportToAccident`
  - `DeliverPatientToH1`
  - `RerouteMidJourneyToH2`
- Produces:
  - A **strict GraphPlan sequence** (fixed order, single path) for the ACC3 scenario.  
  - A **Partial-Order Plan (POP)** that:
    - Keeps some actions unordered (can run in parallel), and  
    - Explicitly encodes contingency branches (e.g., reroute to H2 if CT scanner comes online).  

#### 4.3.1. Run Planning Demo

```bash
python3 -m module3_planning.planning_demo
```

**Expected behaviour (summary):**

- Prints a title banner for Module 3.  
- Shows the strict GraphPlan linear sequence, e.g.:

  1. NotifyHospital(H1)  
  2. PreDeployAmbulanceNearHotspot(A1, nayapalli_chowk)  
  3. TriggerDroneRecon(ACC3)  
  4. RequestTrafficDiversion(stadium_corridor)  
  5. StartTransportToAccident(A1, ACC3)  
  6. DeliverPatientToH1(A1, ACC3, H1)  

- Then prints the POP plan:
  - List of actions with preconditions and effects.  
  - Ordering constraints (partial order).  
  - Causal links.  
  - Contingency branches like:

    - **IF** `CTAvailable(H2)` → execute `RerouteMidJourneyToH2`  
    - **ELSE** (CTOffline(H2)) → execute `DeliverPatientToH1`

This demonstrates an **adaptive plan** that can adjust mid-journey based on hospital status.

---

### 4.4. Module 4 – Reinforcement Learning for Proactive Resource Allocation

**What it does**

- Treats emergency response as a **Markov Decision Process (MDP)** where:
  - **State** summarises:
    - Active accidents and their severities/hotspots
    - Ambulance statuses (free, en route, location zone)
    - Road condition level (e.g., congestion)
  - **Actions** include decisions like:
    - Immediate dispatch to a given incident  
    - Pre-positioning an ambulance in a hotspot  
    - Possibly delaying dispatch briefly  
  - **Reward** encourages:
    - Low average response time  
    - Serving critical cases within a time threshold  
    - Avoiding unnecessary waste or idle travel  

- Implements **Q-learning** and compares it to a simple baseline policy:
  - **Baseline:** “Always dispatch immediately to the nearest active accident.”  

#### 4.4.1. Run RL Training and Evaluation

```bash
python3 -m module4_rl.run_experiments
```

**Expected behaviour (summary):**

- Trains a Q-learning agent over multiple episodes (simulated days/nights).  
- Prints the last few episode rewards to show learning behaviour.  
- Evaluates and prints:

  - Baseline metrics:
    - Average reward  
    - Average response time  
    - Average number of accidents served  
    - Fraction of critical cases served within time threshold  

  - RL policy metrics (same set).  

- Finally, prints a comparison summary, e.g.:

  - RL achieves **higher average reward**  
  - RL achieves **lower response time** than baseline  
  - Served accidents and critical-case fraction are compared to show trade-offs  

This module shows that a learned policy can outperform a naive “always-dispatch-immediately” strategy in terms of overall efficiency.

---

### 4.5. Module 5 – LLM-Based Control Room Summary & Decision Justification

**What it does**

- Takes a structured **DecisionContext** object that aggregates outputs from Modules 1–4:
  - Severity estimates and risk forecast  
  - Route and hospital choices  
  - Planning/POP highlights  
  - RL policy behaviour  
  - Key operational risks and safety notes  
- Builds a carefully engineered **system prompt and user prompt** for an LLM.  
- Generates a **professional, operator-ready briefing** explaining:
  - Current situation (weather, incidents, congestion)  
  - Which ambulance went where and why  
  - Why a particular hospital was chosen even if farther  
  - What main risks and contingencies are  
  - Reminder that recommendations are advisory  


#### 4.5.1. Run LLM Summary Demo

```bash
python3 -m module5_llm.demo_llm_summary
```

**Expected behaviour (summary):**

- Builds a non-trivial scenario where:
  - ACC3 (SUV rollover at Airport approach) is critical.  
  - A1 is pre-deployed and dispatched to ACC3 via a stadium diversion route.  
  - H1 is chosen over closer H2 due to CT and trauma capability.  
- Calls `generate_briefing` with `use_llm=False` by default (template fallback).  
- Prints an operator-style briefing text to the console.  

If you set `use_llm=True` inside `demo_llm_summary.py` and have an API key, it will instead use a real OpenAI LLM to generate the briefing with the same structured prompt.

---

## 5. Dependencies Summary

Main Python dependencies (recommended to put in `requirements.txt`):

- `pgmpy` – for Bayesian Network modelling and inference (Module 1)  
- `numpy` – numerical computations  
- `pandas` – dataset reading and basic analysis (Module 1)  
- `networkx` – graph representation for routing and planning (Module 2, 3)  

Standard library modules used include `dataclasses`, `typing`, `heapq`, etc.

Install everything via:

```bash
pip install -r requirements.txt
```

