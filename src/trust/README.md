# Trust-Based Intervention System

This module implements a dynamic trust model between an autonomous driving agent and a simulated human supervisor, enabling context-aware interventions during vehicle operation.

## Trust Model Overview

The trust system operates on a continuous feedback loop:

1. **Driving Metrics Calculation**: Vehicle state data is directly used to compute metrics
2. **Trust Level Calculation**: Trust level is directly computed from driving metrics
3. **Intervention Probability**: Calculated from trust level and driving context
4. **Behavior Parameters**: Updated based on trust level and driving metrics
5. **Intervention Selection**: The type of intervention is based on driving metrics

## Driving Metrics Computation

The system directly calculates several key metrics from the vehicle state:

```python
def update_driving_metrics(self, vehicle):
    # Get current vehicle state
    velocity = vehicle.get_velocity()
    current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)  # km/h
    current_steering = vehicle.get_control().steer
    
    # Calculate acceleration
    acceleration = (current_speed - self.last_speed) / dt if dt > 0 else 0
    
    # Update histories
    self.acceleration_history.append(acceleration)
    self.steering_history.append(current_steering)
    
    # Calculate steering stability based on variance in steering inputs
    steering_variance = np.var(self.steering_history)
    self.driving_metrics['steering_stability'] = max(0.0, min(1.0, 1.0 - steering_variance * 10.0))
    
    # Calculate acceleration smoothness
    if acceleration > self.abrupt_acceleration_threshold:
        self.driving_metrics['acceleration_smoothness'] -= 0.1  # Penalize abrupt acceleration
    else:
        self.driving_metrics['acceleration_smoothness'] += 0.02  # Gradually recover
    
    # Calculate braking smoothness
    if acceleration < self.abrupt_braking_threshold:
        self.driving_metrics['braking_smoothness'] -= 0.1  # Penalize abrupt braking
    else:
        self.driving_metrics['braking_smoothness'] += 0.02  # Gradually recover
    
    # Detect hesitation (low speed near decision points)
    if self.near_decision_point and current_speed < 5.0:
        # Track time spent hesitating
        if current_time - self.hesitation_start_time > self.hesitation_threshold:
            self.driving_metrics['hesitation_level'] += 0.1
    else:
        self.driving_metrics['hesitation_level'] -= 0.01  # Gradually decrease
```

### Key Metrics:

- **Steering Stability (0-1)**: Based on variance in steering inputs over time. Lower variance means higher stability.
- **Acceleration Smoothness (0-1)**: Measures how smoothly the vehicle accelerates. Abrupt acceleration reduces this value.
- **Braking Smoothness (0-1)**: Measures how smoothly the vehicle brakes. Sudden braking reduces this value.
- **Hesitation Level (0-1)**: Increases when vehicle moves slowly near decision points, indicating driver uncertainty.

## Direct Trust Level Calculation

Unlike traditional systems that gradually adjust trust over time, this system calculates trust directly from driving metrics:

```python
def update_trust(self, intervention=False, intervention_type=None, dt=0.0):
    # Direct calculation from current driving metrics
    smoothness_factor = (self.driving_metrics['acceleration_smoothness'] + 
                          self.driving_metrics['braking_smoothness']) / 2.0
    stability_factor = self.driving_metrics['steering_stability']
    confidence_factor = 1.0 - self.driving_metrics['hesitation_level']
    
    # Trust level is a weighted average of driving performance factors
    self.trust_level = 0.4 * smoothness_factor + 0.4 * stability_factor + 0.2 * confidence_factor
```

This means trust level is always a direct reflection of current driving quality, with:
- 40% weight on driving smoothness (acceleration and braking)
- 40% weight on steering stability
- 20% weight on driver confidence (lack of hesitation)

## Behavior Parameters

The system maintains behavior parameters directly derived from trust and driving metrics:

```python
def update_behavior_adjustment(self):
    # Get current trust level and metrics
    trust_level = self.trust_level
    stability_factor = self.driving_metrics['steering_stability']
    smoothness_factor = (self.driving_metrics['acceleration_smoothness'] + 
                         self.driving_metrics['braking_smoothness']) / 2.0
    hesitation_factor = 1.0 - self.driving_metrics['hesitation_level']
    
    # Store parameters for use in action modification
    self.behavior_adjustment = {
        'trust_level': trust_level,
        'stability_factor': stability_factor,
        'smoothness_factor': smoothness_factor,
        'hesitation_factor': hesitation_factor
    }
```

These parameters are used directly to determine intervention probability and type.

## Intervention Probability Calculation

The probability of intervention is calculated from trust level and context:

```python
# Base probability inversely related to trust
base_intervention_prob = (1.0 - trust_level) * 0.2  # Range: 0.0 to 0.2

# Higher probability near decision points
decision_point_factor = 0.04 if self.is_near_decision_point else 0.0

# Combined probability (capped at 0.5)
intervention_prob = min(0.5, base_intervention_prob + decision_point_factor)
```

When an intervention occurs, the type is selected based on the three behavior factors:

```python
# Intervention type probabilities based on driving metrics deficiencies
intervention_probabilities = [
    1 - stability_factor,       # Steering intervention
    1 - smoothness_factor,      # Throttle/brake intervention
    hesitation_factor           # Hesitation
]

# Normalize probabilities and choose intervention type
intervention_type = np.random.choice(
    ['steer', 'throttle_or_brake', 'hesitation'], 
    p=intervention_probabilities
)
```

This means the system intelligently chooses the most appropriate intervention based on the specific driving deficiency.

## Intervention Types and Application

The system applies three different types of interventions:

1. **Steering Intervention**:
   ```python
   # Adjust steering based on stability and trust
   steering_adjustment = 0.3 + 0.7 * (trust_level * stability_factor)
   adjusted_action[0] *= steering_adjustment
   ```

2. **Throttle/Brake Intervention**:
   ```python
   if adjusted_action[1] > 0:  # Throttle
       throttle_adjustment = 0.3 + 0.7 * (trust_level * smoothness_factor)
       adjusted_action[1] *= throttle_adjustment
   else:  # Brake
       brake_adjustment = 1.0 + (1.0 - trust_level) * 0.5
       adjusted_action[1] *= brake_adjustment
   ```

3. **Hesitation Intervention**:
   ```python
   hesitation_reduction = 1.0 - (hesitation_factor * 0.5)
   adjusted_action *= hesitation_reduction
   ```

Each intervention is recorded and affects future behavior:

```python
def record_intervention(self, intervention_type='brake'):
    self.intervention_active = True
    self.intervention_type = intervention_type
    timestamp = self.world.get_snapshot().timestamp.elapsed_seconds
    
    # Record in general and type-specific intervention lists
    self.manual_interventions.append({
        'timestamp': timestamp,
        'trust_level': self.trust_level,
        'type': intervention_type
    })
    self.intervention_types[intervention_type].append(timestamp)
```

## Integration with Environment

The trust interface integrates with the environment through:

1. **Driving Metrics Updates**: At each step, the vehicle state is used to update metrics
   ```python
   self.trust_interface.update_driving_metrics(self.vehicle)
   ```

2. **Intervention Detection**: The system can detect and record interventions
   ```python
   self.trust_interface.detect_interventions_and_update_trust(
       control, self.prev_control, self.world.get_snapshot(), dt)
   ```

3. **Trust-Based Action Adjustment**: Actions are modified based on trust and driving metrics
   ```python
   adjusted_action = self._adjust_action_based_on_trust(action)
   ```

4. **Behavior Parameter Updates**: Behavior parameters are regularly updated
   ```python
   self.trust_interface.update_behavior_adjustment()
   ```

## Visualization Features

The system provides comprehensive visualization:

- **Trust Level**: Current value and history graph
- **Intervention Probability**: Color-coded from green (low) to red (high)
- **Active Intervention**: Clear indication when an intervention is in progress
- **Behavior Parameters**: Current stability, smoothness, and hesitation factors
- **Intervention Statistics**: Counts of different intervention types

## Conclusion

The trust-based intervention system creates a dynamic interaction mechanism that:

1. Directly computes driving metrics from vehicle state data
2. Calculates trust level as a weighted combination of these metrics 
3. Determines intervention probability based on trust level and context
4. Selects intervention types based on specific driving deficiencies
5. Applies proportional interventions to improve driving behavior

This creates a responsive and intelligent system that adapts to the specific needs of the driving situation, providing targeted assistance where it's most needed.
