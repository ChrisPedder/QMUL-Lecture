"""
Free Energy Principle Agent in a 3D Environment
================================================

This simulation demonstrates core FEP concepts:
- Generative model: Agent's beliefs about world dynamics
- Variational free energy: Quantity the agent minimizes
- Active inference: Actions selected to minimize expected free energy
- Perception: Belief updates to explain sensory input

The agent navigates toward a goal location by:
1. Receiving noisy sensory observations
2. Updating beliefs about its true position (perception)
3. Selecting actions that minimize expected free energy (action)

Key equations implemented:
- Free Energy F ≈ -log P(o|s) - log P(s) + log Q(s)
  (prediction error + complexity, simplified as Gaussian surprisal)
- Belief update: gradient descent on F w.r.t. beliefs
- Action selection: choose action minimizing expected F

Author: Educational demonstration of FEP concepts
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration parameters for the FEP simulation."""

    # Environment
    world_bounds: Tuple[float, float] = (-10.0, 10.0)
    goal_position: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0]))

    # Agent starting position
    start_position: np.ndarray = field(default_factory=lambda: np.array([-5.0, -5.0, -5.0]))

    # Noise parameters
    sensory_noise_std: float = 0.5      # Observation noise
    process_noise_std: float = 0.1      # Movement noise

    # FEP parameters
    belief_learning_rate: float = 0.3   # How quickly beliefs update
    prior_precision: float = 0.5        # Confidence in prior (goal) preference
    sensory_precision: float = 2.0      # Confidence in sensory data

    # Action parameters
    action_magnitude: float = 1.0       # Step size for movement
    num_action_samples: int = 50        # Actions to evaluate
    planning_horizon: int = 1           # Steps to look ahead (1 = greedy)

    # Simulation
    max_steps: int = 100
    goal_threshold: float = 1.0         # Distance to consider goal reached

    # Visualization
    plot_interval: int = 10             # Steps between plot updates


# =============================================================================
# GENERATIVE MODEL
# =============================================================================

class GenerativeModel:
    """
    The agent's internal model of how the world works.

    This encodes:
    - P(o|s): How hidden states cause observations (likelihood)
    - P(s): Prior beliefs about states (includes goal preference)
    - State transition dynamics
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.goal = config.goal_position.copy()

        # Precision matrices (inverse variance)
        # Higher precision = more confident/precise
        self.sensory_precision = config.sensory_precision
        self.prior_precision = config.prior_precision

    def likelihood(self, observation: np.ndarray, state: np.ndarray) -> float:
        """
        P(o|s): Probability of observation given hidden state.

        Models the belief that observations are noisy versions of true state.
        Returns log probability (negative surprisal from this term).
        """
        prediction_error = observation - state
        # Gaussian log-likelihood (up to constant)
        log_prob = -0.5 * self.sensory_precision * np.sum(prediction_error ** 2)
        return log_prob

    def prior(self, state: np.ndarray) -> float:
        """
        P(s): Prior probability of state.

        Encodes preference for being near the goal.
        This is what makes the agent goal-directed!
        """
        distance_to_goal = state - self.goal
        # Gaussian prior centered on goal
        log_prob = -0.5 * self.prior_precision * np.sum(distance_to_goal ** 2)
        return log_prob

    def predict_observation(self, state: np.ndarray) -> np.ndarray:
        """Predict expected observation given state (identity mapping here)."""
        return state.copy()

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict state after taking action."""
        return state + action


# =============================================================================
# VARIATIONAL BELIEFS
# =============================================================================

class VariationalBeliefs:
    """
    Q(s): The agent's approximate posterior beliefs about hidden states.

    We use a simple Gaussian approximation:
    - Mean: best estimate of state
    - Precision: confidence in estimate
    """

    def __init__(self, initial_state: np.ndarray, initial_precision: float = 1.0):
        self.mean = initial_state.copy()
        self.precision = initial_precision  # Scalar precision for simplicity

    def entropy(self) -> float:
        """
        Entropy of the belief distribution.

        Higher entropy = more uncertainty.
        """
        # Entropy of 3D Gaussian (up to constant)
        return -1.5 * np.log(self.precision)

    def sample(self) -> np.ndarray:
        """Sample a state from current beliefs."""
        std = 1.0 / np.sqrt(self.precision)
        return self.mean + np.random.randn(3) * std


# =============================================================================
# FREE ENERGY COMPUTATION
# =============================================================================

class FreeEnergyCalculator:
    """
    Computes variational free energy and its gradients.

    Free Energy F = E_Q[-log P(o,s)] + H[Q]
                  ≈ -log P(o|s) - log P(s) + entropy(Q)

    For point estimate (mean of Q):
    F ≈ prediction_error + prior_cost

    Lower F means:
    - Better predictions (lower surprise)
    - States closer to preferred states
    """

    def __init__(self, generative_model: GenerativeModel):
        self.model = generative_model

    def compute_free_energy(self,
                           observation: np.ndarray,
                           beliefs: VariationalBeliefs) -> float:
        """
        Compute variational free energy given observation and beliefs.

        F = -log P(o|μ) - log P(μ) + entropy(Q)

        where μ is the belief mean.
        """
        state_estimate = beliefs.mean

        # Likelihood term: how well do beliefs explain observations?
        log_likelihood = self.model.likelihood(observation, state_estimate)

        # Prior term: how consistent are beliefs with preferences?
        log_prior = self.model.prior(state_estimate)

        # Entropy term: uncertainty in beliefs
        entropy = beliefs.entropy()

        # Free energy (we want to minimize this)
        # F = -log P(o|s) - log P(s) + H[Q]
        # Note: negative log terms become positive (costs)
        free_energy = -log_likelihood - log_prior + entropy

        return free_energy

    def compute_expected_free_energy(self,
                                    beliefs: VariationalBeliefs,
                                    action: np.ndarray) -> float:
        """
        Compute expected free energy for action selection.

        G = E_Q[F after action]

        This includes:
        - Expected prediction error (pragmatic value)
        - Expected information gain (epistemic value)

        For simplicity, we use the prior cost of the expected next state.
        """
        # Predict next state
        predicted_state = self.model.predict_next_state(beliefs.mean, action)

        # Expected free energy ≈ -log P(s')
        # This makes agent move toward high-prior (goal) regions
        expected_free_energy = -self.model.prior(predicted_state)

        return expected_free_energy

    def belief_gradient(self,
                       observation: np.ndarray,
                       beliefs: VariationalBeliefs) -> np.ndarray:
        """
        Compute gradient of free energy w.r.t. belief mean.

        This tells us how to update beliefs to reduce F.

        ∂F/∂μ = precision_o * (μ - o) + precision_p * (μ - goal)
        """
        state_estimate = beliefs.mean

        # Gradient from likelihood (move toward observation)
        sensory_gradient = self.model.sensory_precision * (state_estimate - observation)

        # Gradient from prior (move toward goal)
        prior_gradient = self.model.prior_precision * (state_estimate - self.model.goal)

        return sensory_gradient + prior_gradient


# =============================================================================
# ENVIRONMENT
# =============================================================================

class Environment:
    """
    The actual 3D world the agent inhabits.

    The agent doesn't have direct access to this - it only
    receives noisy observations.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.agent_position = config.start_position.copy()
        self.goal_position = config.goal_position.copy()
        self.bounds = config.world_bounds

    def get_observation(self) -> np.ndarray:
        """
        Generate noisy observation of agent's true position.

        This simulates imperfect sensory information.
        """
        noise = np.random.randn(3) * self.config.sensory_noise_std
        return self.agent_position + noise

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        """
        Apply action to move agent, with process noise.

        Returns the new position.
        """
        # Add process noise (imperfect motor control)
        noise = np.random.randn(3) * self.config.process_noise_std
        new_position = self.agent_position + action + noise

        # Clip to world bounds
        new_position = np.clip(new_position, self.bounds[0], self.bounds[1])

        self.agent_position = new_position
        return new_position

    def distance_to_goal(self) -> float:
        """Compute Euclidean distance to goal."""
        return np.linalg.norm(self.agent_position - self.goal_position)

    def goal_reached(self) -> bool:
        """Check if agent has reached the goal."""
        return self.distance_to_goal() < self.config.goal_threshold


# =============================================================================
# FEP AGENT
# =============================================================================

class FEPAgent:
    """
    An agent that uses the Free Energy Principle for perception and action.

    The agent:
    1. Receives observations from the environment
    2. Updates beliefs to minimize free energy (perception)
    3. Selects actions to minimize expected free energy (action)
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        # Initialize generative model (agent's world model)
        self.generative_model = GenerativeModel(config)

        # Initialize beliefs (start uncertain about position)
        self.beliefs = VariationalBeliefs(
            initial_state=config.start_position.copy(),
            initial_precision=1.0
        )

        # Free energy calculator
        self.fe_calculator = FreeEnergyCalculator(self.generative_model)

        # History for analysis
        self.belief_history: List[np.ndarray] = []
        self.free_energy_history: List[float] = []
        self.action_history: List[np.ndarray] = []

    def perceive(self, observation: np.ndarray) -> float:
        """
        Update beliefs based on new observation (perceptual inference).

        Uses gradient descent on free energy w.r.t. beliefs.
        This is the key "perception as inference" idea in FEP.

        Returns the current free energy.
        """
        # Perform multiple gradient descent steps for better convergence
        num_inference_steps = 10

        for _ in range(num_inference_steps):
            # Compute gradient of free energy w.r.t. belief mean
            gradient = self.fe_calculator.belief_gradient(observation, self.beliefs)

            # Clip gradient to prevent numerical instability
            gradient_norm = np.linalg.norm(gradient)
            max_gradient = 10.0
            if gradient_norm > max_gradient:
                gradient = gradient * (max_gradient / gradient_norm)

            # Update beliefs (gradient descent)
            self.beliefs.mean -= self.config.belief_learning_rate * gradient

        # Update precision based on prediction error
        prediction_error = np.sum((observation - self.beliefs.mean) ** 2)
        # Adaptive precision: more confident when predictions are good
        # Bounded to prevent extreme values
        self.beliefs.precision = np.clip(1.0 / (0.1 + prediction_error), 0.01, 100.0)

        # Compute and store free energy
        free_energy = self.fe_calculator.compute_free_energy(observation, self.beliefs)
        self.free_energy_history.append(free_energy)
        self.belief_history.append(self.beliefs.mean.copy())

        return free_energy

    def select_action(self) -> np.ndarray:
        """
        Select action that minimizes expected free energy.

        This implements active inference: acting to make the world
        conform to the agent's preferred states.
        """
        best_action = np.zeros(3)
        best_efe = float('inf')

        # Sample candidate actions
        for _ in range(self.config.num_action_samples):
            # Generate random action direction
            action = np.random.randn(3)
            action = action / (np.linalg.norm(action) + 1e-8)  # Normalize
            action = action * self.config.action_magnitude

            # Compute expected free energy
            efe = self.fe_calculator.compute_expected_free_energy(
                self.beliefs, action
            )

            if efe < best_efe:
                best_efe = efe
                best_action = action

        self.action_history.append(best_action.copy())
        return best_action

    def get_belief_state(self) -> np.ndarray:
        """Return current belief about position."""
        return self.beliefs.mean.copy()


# =============================================================================
# SIMULATION
# =============================================================================

class Simulation:
    """
    Runs the FEP agent simulation and collects results.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.environment = Environment(self.config)
        self.agent = FEPAgent(self.config)

        # History
        self.true_position_history: List[np.ndarray] = []
        self.observation_history: List[np.ndarray] = []

    def run(self, verbose: bool = True) -> dict:
        """
        Run the simulation.

        Returns a dictionary with results and history.
        """
        if verbose:
            print("=" * 60)
            print("Free Energy Principle Agent Simulation")
            print("=" * 60)
            print(f"Start position: {self.config.start_position}")
            print(f"Goal position:  {self.config.goal_position}")
            print(f"Initial distance: {self.environment.distance_to_goal():.2f}")
            print("-" * 60)

        # Initial observation
        self.true_position_history.append(self.environment.agent_position.copy())

        for step in range(self.config.max_steps):
            # 1. Get sensory observation
            observation = self.environment.get_observation()
            self.observation_history.append(observation.copy())

            # 2. Perceptual inference (update beliefs)
            free_energy = self.agent.perceive(observation)

            # 3. Action selection (active inference)
            action = self.agent.select_action()

            # 4. Execute action in environment
            self.environment.apply_action(action)
            self.true_position_history.append(self.environment.agent_position.copy())

            # Progress report
            if verbose and (step + 1) % self.config.plot_interval == 0:
                distance = self.environment.distance_to_goal()
                belief = self.agent.get_belief_state()
                print(f"Step {step + 1:3d} | "
                      f"F: {free_energy:7.2f} | "
                      f"Dist: {distance:5.2f} | "
                      f"Belief: [{belief[0]:5.2f}, {belief[1]:5.2f}, {belief[2]:5.2f}]")

            # Check if goal reached
            if self.environment.goal_reached():
                if verbose:
                    print("-" * 60)
                    print(f"Goal reached at step {step + 1}!")
                    print(f"Final distance: {self.environment.distance_to_goal():.2f}")
                break
        else:
            if verbose:
                print("-" * 60)
                print(f"Max steps reached. Final distance: {self.environment.distance_to_goal():.2f}")

        # Compile results
        results = {
            'true_positions': np.array(self.true_position_history),
            'observations': np.array(self.observation_history),
            'beliefs': np.array(self.agent.belief_history),
            'free_energy': np.array(self.agent.free_energy_history),
            'actions': np.array(self.agent.action_history),
            'goal_reached': self.environment.goal_reached(),
            'final_distance': self.environment.distance_to_goal(),
            'steps_taken': len(self.agent.free_energy_history)
        }

        return results

    def plot_results(self, results: dict, save_path: Optional[str] = None):
        """
        Create visualization of simulation results.
        """
        fig = plt.figure(figsize=(16, 10))

        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')

        positions = results['true_positions']
        beliefs = results['beliefs']
        goal = self.config.goal_position

        # Plot true trajectory
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'b-', linewidth=2, label='True path', alpha=0.7)
        ax1.scatter(*positions[0], c='green', s=100, marker='o', label='Start')
        ax1.scatter(*positions[-1], c='blue', s=100, marker='s', label='End')

        # Plot belief trajectory
        ax1.plot(beliefs[:, 0], beliefs[:, 1], beliefs[:, 2],
                'r--', linewidth=1.5, label='Believed path', alpha=0.7)

        # Plot goal
        ax1.scatter(*goal, c='gold', s=200, marker='*', label='Goal')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Agent Trajectory in 3D Space')
        ax1.legend()

        # Free energy over time
        ax2 = fig.add_subplot(2, 2, 2)
        steps = range(len(results['free_energy']))
        ax2.plot(steps, results['free_energy'], 'purple', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Free Energy')
        ax2.set_title('Free Energy Minimization Over Time')
        ax2.grid(True, alpha=0.3)

        # Distance to goal over time
        ax3 = fig.add_subplot(2, 2, 3)
        distances = [np.linalg.norm(pos - goal) for pos in positions]
        ax3.plot(range(len(distances)), distances, 'green', linewidth=2)
        ax3.axhline(y=self.config.goal_threshold, color='r', linestyle='--',
                   label=f'Goal threshold ({self.config.goal_threshold})')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Distance to Goal')
        ax3.set_title('Distance to Goal Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Belief error (difference between belief and true position)
        ax4 = fig.add_subplot(2, 2, 4)
        # Align arrays (beliefs has one less entry than positions)
        min_len = min(len(positions) - 1, len(beliefs))
        belief_errors = [np.linalg.norm(beliefs[i] - positions[i+1])
                        for i in range(min_len)]
        ax4.plot(range(len(belief_errors)), belief_errors, 'orange', linewidth=2)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Belief Error')
        ax4.set_title('Perceptual Accuracy (Belief - True Position)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

        return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run the FEP simulation with default parameters."""

    # Create configuration
    config = SimulationConfig(
        start_position=np.array([-5.0, -5.0, -5.0]),
        goal_position=np.array([5.0, 5.0, 5.0]),
        sensory_noise_std=0.5,
        sensory_precision=2.0,
        prior_precision=0.5,
        belief_learning_rate=0.3,
        max_steps=100,
        action_magnitude=1.0,
        num_action_samples=50
    )

    # Run simulation
    sim = Simulation(config)
    results = sim.run(verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Goal reached: {results['goal_reached']}")
    print(f"Steps taken: {results['steps_taken']}")
    print(f"Final distance to goal: {results['final_distance']:.3f}")
    print(f"Initial free energy: {results['free_energy'][0]:.2f}")
    print(f"Final free energy: {results['free_energy'][-1]:.2f}")
    print(f"Free energy reduction: {results['free_energy'][0] - results['free_energy'][-1]:.2f}")

    # Plot results
    sim.plot_results(results, save_path='fep_simulation_results.png')

    return results


def run_parameter_exploration():
    """
    Explore how different parameters affect agent behavior.

    This helps understand the role of each FEP parameter.
    """
    print("\n" + "=" * 60)
    print("PARAMETER EXPLORATION")
    print("=" * 60)

    base_config = SimulationConfig()

    # Test different sensory precisions
    print("\n--- Effect of Sensory Precision ---")
    for precision in [0.5, 2.0, 8.0]:
        config = SimulationConfig(
            sensory_precision=precision,
            max_steps=100
        )
        sim = Simulation(config)
        results = sim.run(verbose=False)
        print(f"Precision {precision:4.1f}: "
              f"Steps={results['steps_taken']:3d}, "
              f"Final dist={results['final_distance']:.2f}")

    # Test different prior precisions (goal attraction strength)
    print("\n--- Effect of Prior Precision (Goal Attraction) ---")
    for precision in [0.1, 1.0, 5.0]:
        config = SimulationConfig(
            prior_precision=precision,
            max_steps=100
        )
        sim = Simulation(config)
        results = sim.run(verbose=False)
        print(f"Precision {precision:4.1f}: "
              f"Steps={results['steps_taken']:3d}, "
              f"Final dist={results['final_distance']:.2f}")

    # Test different noise levels
    print("\n--- Effect of Sensory Noise ---")
    for noise in [0.1, 0.5, 2.0]:
        config = SimulationConfig(
            sensory_noise_std=noise,
            max_steps=100
        )
        sim = Simulation(config)
        results = sim.run(verbose=False)
        print(f"Noise {noise:4.1f}: "
              f"Steps={results['steps_taken']:3d}, "
              f"Final dist={results['final_distance']:.2f}")


if __name__ == "__main__":
    # Run main simulation
    results = main()

    # Optionally run parameter exploration
    # run_parameter_exploration()
