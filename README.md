# FrozenLake Dynamic Programming Study (Value Iteration & Policy Iteration)

This repository is a focused study of **dynamic programming methods** for solving finite Markov Decision Processes (MDPs) applied to the classic **FrozenLake-v1** environment (custom maps) from Gym / Gymnasium. It centers on two core algorithms:

- `frosen_VALUE_ITERATION.ipynb` – iterative Bellman optimality updates to converge directly to the optimal value function and greedy policy.
- `Frosen_POLICY_ITERATION.ipynb` – classic policy iteration loop alternating policy evaluation and policy improvement until convergence.

A supporting written report (`deep_RL_compte_rendu1.pdf`) summarizes theoretical background, convergence behavior, and compares both algorithms empirically.

---
## Contexte académique (FR)
**Implémentation et Analyse des Algorithmes de Value Iteration et Policy Iteration sur l’Environnement FrozenLake**

- Cours : Reinforcement Learning (Apprentissage par Renforcement)
- Encadré par : Pr. Jamal Riffi
- Réalisé par : Mohamed Zaim
- Master : Machine Learning Avancé et Intelligence Multimédia (MLAIM)

_Introduction._ Dans le cadre du cours de Reinforcement Learning, ce travail a pour objectif d’implémenter et d’analyser deux algorithmes fondamentaux des méthodes de planification basées sur modèle :
- Value Iteration (Itération de Valeur)
- Policy Iteration (Itération de Politique)

Ces deux approches permettent de trouver la politique optimale dans un environnement de type Markov Decision Process (MDP).

> NOTE: Some filenames intentionally retain the original "Frosen" spelling to match coursework submission artifacts.

---
## Resources / Ressources
- Report / Compte rendu (PDF): [deep_RL_compte_rendu1.pdf](./deep_RL_compte_rendu1.pdf)
- Notebooks:
  - [frosen_VALUE_ITERATION.ipynb](./frosen_VALUE_ITERATION.ipynb)
  - [Frosen_POLICY_ITERATION.ipynb](./Frosen_POLICY_ITERATION.ipynb)
- Requirements: [requirements.txt](./requirements.txt) (primarily `gym` or `gymnasium`, `numpy`, `matplotlib`)

## 🎥 FrozenLake RL Demo

[![FrozenLake RL Demo](https://img.youtube.com/vi/57r_puXAYhg/maxresdefault.jpg)](https://youtu.be/57r_puXAYhg)
**▶ Watch Video on YouTube**




---
## 1. Environment & Problem Setting

We use a deterministic or stochastic (slippery) gridworld (FrozenLake):
- States: discrete tiles (Start S, Frozen F, Hole H, Goal G)
- Actions: Left, Down, Right, Up (indices 0..3)
- Transitions: Either deterministic (`is_slippery=False`) or stochastic slippage (default Gym behavior)
- Reward: +1 on reaching Goal, 0 otherwise; episode ends on Hole or Goal

Custom maps are embedded in the notebooks (modifiable list of strings). You can adjust:
- Map layout (shape, holes placement)
- Discount factor `gamma`
- Convergence threshold `theta`
- Slipperiness flag (if you recreate env differently)

---
## 2. Algorithms Overview

### 2.1 Value Iteration (Notebook: `frosen_VALUE_ITERATION.ipynb`)
Applies Bellman optimality backup:
```
V_{k+1}(s) = max_a Σ_{s',r} P(s'|s,a)[ r + γ V_k(s') ]
```
Stops when the maximum absolute state-value change < `theta`.
Then derives the greedy policy:
```
π*(s) = argmax_a Σ_{s',r} P(s'|s,a)[ r + γ V(s') ]
```
**Pros:** Often fewer full sweeps than policy iteration in large spaces.  
**Cons:** Each sweep computes a full action maximization (more expensive per iteration).

### 2.2 Policy Iteration (Notebook: `Frosen_POLICY_ITERATION.ipynb`)
Alternates:
1. **Policy Evaluation:** Iteratively solve `V^π` until value change < `theta`.
2. **Policy Improvement:** Make policy greedy w.r.t. `V^π`.
Terminates when policy is stable.
**Pros:** Rapid convergence in small/medium MDPs; evaluation reuses structure.  
**Cons:** Each evaluation phase may require many sweeps if `theta` is very small.

### 2.3 Practical Notes
| Aspect | Value Iteration | Policy Iteration |
|--------|-----------------|------------------|
| Work per iteration | Higher (action maximization) | Lower (eval uses fixed π) |
| Iterations needed | Often fewer | More (policy eval + improve cycles) |
| Memory | Same (store V and π) | Same |
| Convergence detection | Max ΔV < θ | Policy unchanged |

---
## 3. Notebook Structure & Key Cells
Both notebooks follow a similar pattern:
1. Imports & environment construction (custom map + `gym.make`).
2. Extraction of transition model `P = env.unwrapped.P`, and sizes `nS`, `nA`.
3. Definition of hyperparameters: `gamma`, `theta`.
4. Core algorithm functions (value iteration OR policy eval/improvement + loop).
5. Execution & convergence reporting (iterations, runtime if added).
6. Rendering / one policy rollout (human render window if `render_mode='human'`).
7. Pretty printing of:
   - Value function grid (reshaped to map size)
   - Policy arrows using ← ↓ → ↑ mapping (handles non-square by block formatting)

---
## 4. Running the Notebooks
### 4.1 Requirements Installation
```cmd
pip install -r requirements.txt
```
If you prefer Gymnasium explicitly:
```cmd
pip install gymnasium
```
(Otherwise classic `gym` fallback is attempted.)

### 4.2 Launch Jupyter / VS Code
```cmd
python -m pip install notebook
python -m notebook
```
Open:
- `Frosen_POLICY_ITERATION.ipynb`
- `frosen_VALUE_ITERATION.ipynb`

### 4.3 Adjust Parameters
Inside the first parameter / hyperparameter cell, tweak:
- `gamma = 0.99` (try 0.9, 0.95)
- `theta = 1e-8` (relax to `1e-6` for faster but approximate convergence)
- Map layout list (ensure exactly one `S` and one `G`)
- (Optional) Use deterministic environment by recreating env with `is_slippery=False`.

### 4.4 Execution Flow
Run cells top to bottom. A small window (Pygame/Gym render) appears during rollout. If it becomes unresponsive, ensure you allow event pumping (already included) or close it after viewing.

---
## 5. Interpreting Outputs
**Value Grid:** Each cell shows approximate `V*(s)`. Higher values cluster near paths leading to the goal. Holes and dead-ends propagate lower upstream values.  
**Policy Arrows:** Show the greedy action in each non-terminal state. Terminal (Goal / Hole) states typically omitted or printed implicitly by their value (often 0 or 1 for goal).  
**Convergence Report:**
- Policy Iteration prints iteration count (number of policy improvement cycles).
- Value Iteration prints number of sweeps until max |ΔV| < θ (if included; add timing easily if needed).

A typical small-grid deterministic run yields:  
- Goal state value ≈ 1.0  
- Values decreasing outward along optimal paths  
- Arrows forming a “funnel” toward the goal

---
## 6. Extending Experiments
Suggested variations for the report or further study:
1. **Gamma Sensitivity:** Compare optimal paths and value magnitudes for γ ∈ {0.8, 0.9, 0.95, 0.99}.
2. **Slipperiness Impact:** Toggle `is_slippery` and measure increased iteration counts / altered policy robustness.
3. **Theta Trade-off:** Plot convergence iterations vs. θ (10⁻³ to 10⁻⁹).
4. **Random Hole Density:** Generate random maps with fixed start/goal and observe failure rates on rollout.
5. **Hybrid (Modified Policy Iteration):** Limit evaluation sweeps per cycle (e.g., 1–3) and compare speed vs. stability.

---
## 7. Reproducing Results from the PDF (`deep_RL_compte_rendu1.pdf`)
While the PDF is not parsed here, its conceptual sections typically map to:
| Report Section (Indicative) | Notebook Cell Range | How to Reproduce |
|-----------------------------|---------------------|------------------|
| MDP Formalization | Top import + env setup | Re-run env creation, print `nS`, `nA` |
| Bellman Equations | Algorithm function cells | Inspect function definitions |
| Convergence Evidence | Execution + prints | Adjust θ and log iterations |
| Policy Visualization | Final printing cells | Rerun with alternate maps/slippery |
| Comparative Analysis | Run both notebooks sequentially | Record iteration counts & runtime |

If you add timing:
```python
import time
start = time.time()
pi_opt, V_opt, iters = policy_iteration(gamma=gamma, theta=theta)
print('Elapsed:', time.time()-start)
```
Likewise for value iteration.

---
## 8. Minimal Reference Implementations (Pseudocode)
**Value Iteration**
```
initialize V(s)=0
repeat:
  Δ = 0
  for each state s:
    v = V(s)
    V(s) = max_a Σ_{s'} P(s'|s,a)[r + γ V(s')]
    Δ = max(Δ, |v - V(s)|)
until Δ < θ
π(s) = argmax_a Σ_{s'} P(s'|s,a)[r + γ V(s')]
```
**Policy Iteration**
```
initialize π randomly
repeat:
  (Policy Evaluation)  solve V^π via iterative backups until stable
  (Policy Improvement) π'(s)=argmax_a Σ_{s'} P(s'|s,a)[r + γ V^π(s')]
  if π' == π: stop else π = π'
```

---
## 9. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| No render window | Headless environment | Remove `render_mode='human'` or run locally |
| Window freezes | Event loop not pumped | Keep provided `pygame.event.pump()` loop |
| Slow convergence | Very small θ (e.g. 1e-10) | Relax θ (e.g. 1e-6) |
| Policy looks odd | Slippery stochasticity | Try `is_slippery=False` |
| Value uniform | Map unreachable goal / all holes | Fix map layout |

---
## 10. License / Academic Use
Use freely for educational and research purposes. Cite the Gym / Gymnasium project if publishing results.

---
## 11. Quick Start (One-Liner)
```cmd
pip install -r requirements.txt && python -m pip install notebook && python -m notebook
```
Open the two notebooks, run top to bottom, compare outputs.

---
*Happy studying — explore, tweak, and observe how dynamic programming converges to optimal behavior!*
