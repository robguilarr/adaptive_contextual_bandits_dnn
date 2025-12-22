# Synthetic Data Generation Specification: Contextual Bandits for IAP Optimization

## 1. Overview
This document outlines the requirements and logic for generating synthetic training data for a Contextual Multi-Armed Bandit model. The generator acts as the "World Model" (Environment), defining the **Ground Truth** logic that links User Context (State) to IAP Conversions (Reward).

## 2. Theoretical Framework

### 2.1 The Bandit Problem
We define the tuple $(S, A, R)$:
*   **$S$ (State/Context):** Observable features about the user and session.
*   **$A$ (Action):** The specific IAP power-up presented to the user.
*   **$R$ (Reward):** The value generated. $R = \text{Price} \times \mathbb{I}(\text{Purchase})$.

### 2.2 Data Collection Policy (Epsilon-Greedy)
The generator must support simulating different stages of the model lifecycle:

1.  **Cold Start ($\epsilon = 1.0$):** Pure exploration. Actions are chosen uniformly at random. This corresponds to the initial "historical data" used to train the first model.
2.  **Production/Epsilon-Greedy ($\epsilon < 1.0$):** Simulates a live model.
    *   With probability $1 - \epsilon$: The system "Exploits" (chooses the optimal action based on Ground Truth).
    *   With probability $\epsilon$: The system "Explores" (chooses a random action).
    *   *Note: The text specifies a 30% exploration rate for the production system.*

## 3. Schema Definition

The output CSV files will contain the following columns:

### 3.1 Final Dataset Columns
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `event_id` | Integer | Unique identifier for the event (renamed from `id`). |
| `event_timestamp` | Datetime | Random timestamp between 2024 and 2025. |
| `distance_avg` | Float | User Feature: Average distance run. |
| `coins_spent` | Integer | User Feature: Total coins spent. |
| `game_day` | Integer | User Feature: Days since install. |
| `geo_country` | String | User Feature: Country code. |
| `device_os` | String | User Feature: iOS or Android. |
| `last_run_end_reason`| String | Context Feature: How the previous run ended (`laser`, `wall`). |
| `presented_powerup` | String | Action: The item shown to the user. |
| `is_powerup_clicked` | Integer | Outcome: Target variable (1/0). |

*Note: `price` and `reward` are implicit/calculated during training if needed, but not in the raw log file.*

### 3.2 Action Space (Power-ups)
| Power-up Name | Price ($) | Utility / Affinity Logic |
| :--- | :--- | :--- |
| `time_machine` | 1.99 | High `distance_avg` |
| `coin_magnet` | 0.99 | High `coins_spent` |
| `coin_multiplier`| 0.99 | High `coins_spent` |
| `sparky_armor` | 2.99 | `last_run_end_reason` == 'laser' |
| `extra_life` | 4.99 | Universal High Value |
| `head_start` | 1.49 | Low `distance_avg` |
| `parachute` | 1.99 | `last_run_end_reason` == 'wall' |
| `nuclear_missle` | 3.99 | High `game_day` |

### 3.3 Reward Function (Ground Truth)
$$ P(buy) = \text{BaseProb} \times \text{AffinityMultiplier} \times \text{PriceSensitivity} $$

1.  **Contextual Triggers:** Laser $\to$ Armor ($5x$), Wall $\to$ Parachute ($5x$).
2.  **User Segments:** Whales $\to$ Coin items ($3x$), Veterans $\to$ Nuke/Time ($3x$), Strugglers $\to$ Help items ($4x$).
3.  **Price Sensitivity:** $1 / \log(1 + \text{Price})$.

## 4. Execution Logic
1.  Generate User State (`event_id`, `event_timestamp`, features).
2.  **Calculate Ground Truth Probabilities** for *all* potential actions for this user.
3.  **Select Action** based on $\epsilon$:
    *   If Exploit: Pick action with max(Probability * Price).
    *   If Explore: Pick random action.
4.  **Simulate Outcome:** Sample Bernoulli using the probability of the *selected* action to set `is_powerup_clicked`.
5.  Drop auxiliary columns (`price`, `reward`).
6.  Export to CSV.

### 3.4. Why Choose 100% during first training

To train the first version of a model, you need unbiased data covering all possible actions. If we simulated an Epsilon-greedy policy (e.g., picking the "best" 70% of the time) inside the generator, we would introduce Selection Bias. The model would have lots of data for the "good" actions and very little for the "bad" ones, making it hard to learn why the bad ones are bad.

**Adding the Epsilon-Greedy Feature**

A robust generator should be able to simulate production logs as well. You might want to generate data that looks like it came from a live system (where the model is already making decisions 70% of the time) to test how well you can re-train a model using biased data (Off-Policy Learning).

1.  `epsilon = 1.0 (Default): Pure Exploration (Cold Start Data)`
2.  `epsilon = 0.3: Simulates the production scenario (Re-training)`

**How to use it**

You can now generate the specific "production logs" scenario if you wish (2):

```
python3 src/utilities/datasets/data_generator.py --train_size 150000 --epsilon 0.3
```

Or the standard unbiased training data (1):

```
python3 src/utilities/datasets/data_generator.py --train_size 150000 --epsilon 1.0
```
