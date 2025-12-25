"""Synthetic data generation for contextual bandit training and evaluation."""

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, timedelta

class DataGenerator:
    """
    Generates synthetic training data for Contextual Bandits.
    Simulates IAP interactions in the 'Flappy Sparky' game environment.
    Supports Epsilon-Greedy policies for generating historical or production logs.
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # --- Configuration ---
        self.countries = ['US', 'Canada', 'China', 'Japan', 'Germany', 'India', 'France', 'UK', 'Italy', 'Russia', 'South Korea']
        self.country_weights = [0.2, 0.05, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1]
        
        self.os_types = ['iOS', 'Android']
        self.os_weights = [0.4, 0.6]
        
        self.end_reasons = ['laser', 'wall']
        
        # Action Space
        self.powerups = [
            {'name': 'time_machine',    'price': 1.99},
            {'name': 'coin_magnet',     'price': 0.99},
            {'name': 'coin_multiplier', 'price': 0.99},
            {'name': 'sparky_armor',    'price': 2.99},
            {'name': 'extra_life',      'price': 4.99},
            {'name': 'head_start',      'price': 1.49},
            {'name': 'parachute',       'price': 1.99},
            {'name': 'nuclear_missle',  'price': 3.99},
        ]
        self.powerup_names = np.array([p['name'] for p in self.powerups])
        self.powerup_prices = np.array([p['price'] for p in self.powerups])
        self.price_map = {p['name']: p['price'] for p in self.powerups}

    def _generate_timestamps(self, n, start_year=2024, end_year=2025):
        """Generates random timestamps between start_year and end_year."""
        start_date = pd.Timestamp(f'{start_year}-01-01')
        end_date = pd.Timestamp(f'{end_year}-12-31')
        delta_seconds = int((end_date - start_date).total_seconds())
        
        # Generate random seconds offsets
        random_seconds = np.random.randint(0, delta_seconds, size=n)
        return start_date + pd.to_timedelta(random_seconds, unit='s')

    def generate_users(self, n):
        """Generates the State Space (User Context)."""
        print(f"Generating {n} user profiles...")
        data = {
            'event_id': np.arange(n),
            'event_timestamp': self._generate_timestamps(n),
            'geo_country': np.random.choice(self.countries, size=n, p=self.country_weights),
            'device_os': np.random.choice(self.os_types, size=n, p=self.os_weights),
            'game_day': np.maximum(0, np.random.lognormal(mean=3, sigma=1, size=n)).astype(int),
            'distance_avg': np.maximum(0, np.random.normal(loc=100, scale=40, size=n)).astype(int),
            'coins_spent': np.maximum(0, np.random.normal(loc=1000, scale=600, size=n)).astype(int),
            'last_run_end_reason': np.random.choice(self.end_reasons, size=n)
        }
        return pd.DataFrame(data)

    def _calculate_ground_truth_probs(self, df):
        """
        Calculates the purchase probability for EVERY action for EVERY user.
        Returns a matrix of shape (N_users, N_actions).
        
        The calculation starts with a uniform base_prob (baseline purchase probability)
        applied to all user-action pairs, which is then modified by contextual factors:
        - Contextual triggers (e.g., death reason → relevant powerup)
        - User segments (veterans, whales, strugglers)
        - Price sensitivity
        
        base_prob represents the minimum conversion rate before any contextual adjustments,
        serving as the foundation for probability modeling. Higher values create more
        balanced datasets with better signal for training. Note: keep it  below 0.15 to avoid a "Logging Policy" bias.
        """
        n = len(df)
        n_actions = len(self.powerups)
        
        # 1. Base Probability
        base_prob = 0.10
        # Shape: (N_users, N_actions)
        probs = np.full((n, n_actions), base_prob)
        
        # 2. Vectorized Feature Extraction
        reasons = df['last_run_end_reason'].values
        game_days = df['game_day'].values
        coins = df['coins_spent'].values
        distances = df['distance_avg'].values
        
        # 3. Apply Affinities per Action
        # We iterate actions to apply specific logic columns
        
        for i, powerup in enumerate(self.powerup_names):
            # Contextual Triggers
            if powerup == 'sparky_armor':
                # Boost if died by laser
                probs[:, i] = np.where(reasons == 'laser', probs[:, i] * 5.0, probs[:, i])
            
            elif powerup == 'parachute':
                # Boost if died by wall
                probs[:, i] = np.where(reasons == 'wall', probs[:, i] * 5.0, probs[:, i])
                
            # Veteran Affinity
            if powerup in ['nuclear_missle', 'time_machine']:
                mask_vet = game_days > 100
                probs[:, i] = np.where(mask_vet, probs[:, i] * 3.0, probs[:, i])
                
            # Whale Affinity
            if powerup in ['coin_magnet', 'coin_multiplier']:
                mask_whale = coins > 1500
                probs[:, i] = np.where(mask_whale, probs[:, i] * 3.0, probs[:, i])
                
            # Struggler Affinity
            if powerup in ['head_start', 'extra_life']:
                mask_struggle = distances < 50
                probs[:, i] = np.where(mask_struggle, probs[:, i] * 4.0, probs[:, i])
        
        # 4. Price Sensitivity
        # Multiplier = 1 / log(1 + price)
        # prices shape (N_actions,) broadcast to (N_users, N_actions)
        price_sensitivity = 1.0 / np.log1p(self.powerup_prices)
        probs *= price_sensitivity
        
        return np.minimum(probs, 0.8)

    def simulate_interactions(self, df, epsilon=1.0):
        """
        Simulates the Policy (Epsilon-Greedy) and the Outcome.
        epsilon: Probability of Random Action (Exploration). 
                 1.0 = Pure Random (Cold Start).
                 0.3 = 30% Random, 70% Exploit (Production).
        """
        print(f"Simulating interactions with Epsilon={epsilon}...")
        n = len(df)
        
        # 1. Calculate Ground Truth Probabilities for ALL actions
        # This is needed to know which action is "Optimal" for the Exploit phase
        all_probs = self._calculate_ground_truth_probs(df) # Shape (N, 8)
        
        # Calculate Expected Value (Prob * Price) to find the "Best" action
        expected_values = all_probs * self.powerup_prices
        best_actions_indices = np.argmax(expected_values, axis=1)
        best_actions = self.powerup_names[best_actions_indices]
        
        # 2. Select Actions (Epsilon-Greedy)
        # Create a mask for Exploration (True = Explore/Random, False = Exploit/Best)
        explore_mask = np.random.random(size=n) < epsilon
        
        # Random actions for everyone (we will only use these where explore_mask is True)
        random_actions = np.random.choice(self.powerup_names, size=n)
        
        # Final Action Selection
        final_actions = np.where(explore_mask, random_actions, best_actions)
        df['presented_powerup'] = final_actions
        df['price'] = df['presented_powerup'].map(self.price_map)
        
        # 3. Simulate Outcome (Conversion)
        # We need to extract the probability corresponding to the *chosen* action
        # Create an indexer [0..N-1, action_indices]
        action_indices = np.searchsorted(self.powerup_names, final_actions)
        
        selected_probs = np.zeros(n)
        for i, name in enumerate(self.powerup_names):
            mask = (final_actions == name)
            selected_probs[mask] = all_probs[mask, i]
            
        # Bernoulli Trial
        random_draws = np.random.random(size=n)
        df['is_powerup_clicked'] = (random_draws < selected_probs).astype(int)
        df['reward'] = df['is_powerup_clicked'] * df['price']
        
        return df

    def _finalize_dataframe(self, df):
        """
        Formats the DataFrame to match the strict schema.
        Drops 'price', 'reward' and reorders columns.
        """
        cols_to_keep = [
            'event_id', 'event_timestamp',
            'distance_avg', 'coins_spent', 'game_day', 'geo_country', 
            'device_os', 'last_run_end_reason', 'presented_powerup', 'is_powerup_clicked'
        ]
        return df[cols_to_keep]

    def run(self, train_size=7000000, val_size=2000000, test_size=1000000, epsilon=1.0):
        
        print(f"--- Configuration: Epsilon = {epsilon} ---")
        if epsilon == 1.0:
            print("(Generating Pure Random Historical Data)")
        else:
            print(f"(Generating Production Logs with {epsilon*100}% Exploration)")

        # Generate Training Data
        print("--- Creating Training Set ---")
        train_df = self.generate_users(train_size)
        train_df = self.simulate_interactions(train_df, epsilon=epsilon)
        train_df = self._finalize_dataframe(train_df)
        train_path = os.path.join('data', 'raw', 'training.csv')
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_df.to_csv(train_path, index=False)
        print(f"Saved {train_path} ({len(train_df)} rows)")

        # Generate Validation Data
        print("--- Creating Validation Set ---")
        val_df = self.generate_users(val_size)
        val_df = self.simulate_interactions(val_df, epsilon=epsilon)
        val_df = self._finalize_dataframe(val_df)
        val_path = os.path.join('data', 'raw', 'validation.csv')
        val_df.to_csv(val_path, index=False)
        print(f"Saved {val_path} ({len(val_df)} rows)")
        
        # Generate Test Data
        print("--- Creating Test Set ---")
        test_df = self.generate_users(test_size)
        test_df = self.simulate_interactions(test_df, epsilon=epsilon)
        test_df = self._finalize_dataframe(test_df)
        test_path = os.path.join('data', 'raw', 'test.csv')
        test_df.to_csv(test_path, index=False)
        print(f"Saved {test_path} ({len(test_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic data for IAP Bandits.')
    parser.add_argument('--train_size', type=int, default=700000, help='Number of training examples')
    parser.add_argument('--val_size', type=int, default=200000, help='Number of validation examples')
    parser.add_argument('--test_size', type=int, default=100000, help='Number of test examples')
    parser.add_argument('--epsilon', type=float, default=1.0, 
                        help='Exploration rate (0.0 - 1.0). 1.0 = Random Data (Default), 0.3 = Production Logs.')
    
    args = parser.parse_args()
    
    generator = DataGenerator()
    generator.run(args.train_size, args.val_size, args.test_size, args.epsilon)
