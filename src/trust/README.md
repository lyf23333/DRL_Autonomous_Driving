# Trust Feedback Interface

## Overview

The `TrustInterface` class is designed to measure and update the trust level between a human operator and an autonomous vehicle in the CARLA simulator. This trust level is influenced by the operator's interventions and the vehicle's smooth operation. The interface provides visual feedback and logs data for further analysis.

## Example

The trust level is a value between 0 and 1. The trust level is increased by 0.01 per second if the vehicle is driving smoothly. If the vehicle is driving with interventions, the trust level is decreased by 0.05. In this way, the driving is encouraged to be smooth and to avoid interventions. As a starting point, I currently just have human intervention as a sign for low trust.

## Trust Measurement

### Trust Level

- **Initial Trust Level**: The trust level starts at 0.5, on a scale from 0.0 to 1.0.
- **Trust Increase**: During smooth operation without interventions, the trust level gradually increases at a rate of 0.01 per second.
- **Trust Decrease**: When a manual intervention occurs, the trust level decreases by 0.05.

### Manual Interventions

- **Definition**: A manual intervention is an action taken by the operator to override the vehicle's autonomous behavior.
- **Recording**: Each intervention is recorded with a timestamp and the current trust level.
- **Cooldown**: A minimum of 5 seconds must pass between interventions to be considered separate events.

- **Recent Interventions**:
    - **Window**: Interventions are considered "recent" if they occur within the last 5 seconds.
    - **Observation**: The system can provide a binary observation indicating whether any recent interventions have occurred.

## Trust Data Collection

The system logs the following data during a session:

- **Session ID**: A unique identifier for each session, based on the current date and time.
- **Manual Interventions**: A list of all interventions, including timestamps and trust levels.
- **Final Trust Level**: The trust level at the end of the session.
- **Intervention Count**: The total number of interventions during the session.

## Trust Feedback Display

The interface provides a visual representation of the trust level:

- **Trust Level Bar**: A green bar that fills according to the current trust level.
- **Textual Feedback**: Displays the current trust level and the number of recent interventions.

## Integration with CARLA

The trust level influences the vehicle's behavior in the CARLA simulator:

- **Intervention Probability**: The probability of requiring an intervention increases as the trust level decreases.
- **Behavior Adjustment**: The system can adjust the vehicle's behavior based on the trust level and recent interventions, although specific behavior adjustments are not detailed in the current implementation. And this adjustment is meant to be learnt by the RL agent. We will provide rewards which are not directly linked to trust level e.g. encouraging smooth driving and penalties for interventions.

## Data Logging and Cleanup

At the end of each session, the system saves the collected data to a JSON file for analysis. The `cleanup` method ensures that all resources are properly released.

## Conclusion

The `TrustInterface` provides a comprehensive mechanism for measuring and updating trust in autonomous vehicle systems. By logging interventions and adjusting trust levels, it offers valuable insights into the human-vehicle interaction dynamics.
