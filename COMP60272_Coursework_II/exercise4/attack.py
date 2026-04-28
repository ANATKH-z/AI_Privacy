"""
Poisoning attack implementations for secure federated learning.

TODO (Exercise 3 - marking criteria):
--------------------------------------
1. Attack Category: Model Poisoning Attack.
   The malicious clients directly tamper with the gradient updates sent to the server
   (reversing and amplifying them) without modifying their local training datasets.

2. Attack Frequency: Continuous Attack.
   The malicious clients participate and send malicious updates in every communication
   round across the entire training process (all 30 rounds).

3. Attack Objective: Accuracy Degradation.
   The goal is to completely destroy the global model's availability, driving the
   overall test accuracy down to random-guessing levels (near 0%).

4. Parameter Justification (ρ = 0.2, Attack Strength = -2.0, Scale Factor = 3.0):
   I selected a malicious client fraction of ρ = 0.2 (20%). Under the cover of Secure
   Aggregation, the server cannot inspect individual updates. A 20% fraction of malicious
   clients, combined with a reversed and amplified gradient (Attack Strength = -2.0,
   Scale Factor = 3.0, effectively a -6x multiplier), is powerful enough to overwhelm the
   benign updates from the remaining 80% of clients. This specific scaling avoids immediate
   numerical overflow (NaNs) in the early rounds while ensuring the global model successfully
   diverges and the accuracy drops to 0% smoothly.
"""
import torch
from collections import OrderedDict
from typing import Optional, Callable
from abc import ABC, abstractmethod


class AttackStrategy(ABC):
    """
    Abstract base class for poisoning attack strategies.
    """

    @abstractmethod
    def craft_update(
        self,
        benign_update: OrderedDict,
        global_model: OrderedDict,
        round_num: int
    ) -> OrderedDict:
        """
        Craft a malicious update.

        Args:
            benign_update: The benign update that would have been sent
            global_model: Current global model state
            round_num: Current communication round

        Returns:
            Malicious update to send
        """
        pass


class ModelPoisoningAttack(AttackStrategy):
    """
    Model poisoning attack: directly craft malicious updates.
    """

    def __init__(self, attack_strength: float = 1.0):
        """
        Initialize model poisoning attack.

        Args:
            attack_strength: Strength of the attack (scaling factor)
        """
        self.attack_strength = attack_strength

    def craft_update(
        self,
        benign_update: OrderedDict,
        global_model: OrderedDict,
        round_num: int
    ) -> OrderedDict:
        """
        Craft a malicious update by scaling the benign update.
        TODO: Replace or extend with your own model poisoning strategy.
        """
        malicious_update = OrderedDict()
        for key in benign_update.keys():
            # Scale the update to amplify its effect
            malicious_update[key] = benign_update[key] * self.attack_strength

        return malicious_update


"""
1. Attack Category: Model Poisoning Attack.
   The malicious clients directly tamper with the gradient updates sent to the server 
   (reversing and amplifying them) without modifying their local training datasets.

2. Attack Frequency: Continuous Attack.
   The malicious clients participate and send malicious updates in every communication 
   round across the entire training process (all 30 rounds).

3. Attack Objective: Accuracy Degradation.
   The goal is to completely destroy the global model's availability, driving the 
   overall test accuracy down to random-guessing levels (near 0%).

4. Parameter Justification (ρ = 0.2, Attack Strength = -2.0, Scale Factor = 3.0):
   I selected a malicious client fraction of ρ = 0.2 (20%). Under the cover of Secure 
   Aggregation, the server cannot inspect individual updates. A 20% fraction of malicious 
   clients, combined with a reversed and amplified gradient (Attack Strength = -2.0, 
   Scale Factor = 3.0, effectively a -6x multiplier), is powerful enough to overwhelm the 
   benign updates from the remaining 80% of clients. This specific scaling avoids immediate 
   numerical overflow (NaNs) in the early rounds while ensuring the global model successfully 
   diverges and the accuracy drops to 0% smoothly.
"""

class AccuracyDegradationAttack(ModelPoisoningAttack):
    """
    Attack that aims to degrade overall model accuracy.

    Strategy: Send updates in the opposite direction of gradient descent.
    """

    def __init__(self, attack_strength: float = -2.0):
        """
        Initialize accuracy degradation attack.

        Args:
            attack_strength: Negative value to reverse gradient direction
        """
        super().__init__(attack_strength=attack_strength)

    def craft_update(
        self,
        benign_update: OrderedDict,
        global_model: OrderedDict,
        round_num: int
    ) -> OrderedDict:
        """
        Craft update that reverses the gradient direction.
        TODO: You may use this as a starting point for accuracy degradation.
        """

        malicious_update = OrderedDict()
        # Introduce a scaling factor to ensure that a small number of malicious nodes can completely undermine the global model
        scale_factor = 3.0
        for key in benign_update.keys():
            # Reverse the update direction and amplify
            malicious_update[key] = benign_update[key] * self.attack_strength * scale_factor

        return malicious_update


class TargetedMisclassificationAttack(ModelPoisoningAttack):
    """
    Attack that aims to cause targeted misclassification.

    Strategy: Craft updates that shift model towards misclassifying specific classes.
    """

    def __init__(self, target_class: int = 0, attack_strength: float = 5.0):
        """
        Initialize targeted misclassification attack.

        Args:
            target_class: Target class to misclassify to
            attack_strength: Strength of the attack
        """
        super().__init__(attack_strength=attack_strength)
        self.target_class = target_class

    def craft_update(
        self,
        benign_update: OrderedDict,
        global_model: OrderedDict,
        round_num: int
    ) -> OrderedDict:
        """
        Craft update that biases model towards target class.
        TODO: Implement your targeted misclassification strategy.
        """
        malicious_update = OrderedDict()

        # For the last layer (classification layer), bias towards target class
        for key in benign_update.keys():
            if 'fc2' in key and 'weight' in key:
                # Bias the weights towards target class
                malicious_update[key] = benign_update[key].clone()
                # Add bias to target class
                malicious_update[key][self.target_class, :] += self.attack_strength * 0.1
            elif 'fc2' in key and 'bias' in key:
                # Bias the bias term
                malicious_update[key] = benign_update[key].clone()
                malicious_update[key][self.target_class] += self.attack_strength * 0.5
            else:
                # Keep other layers as is but scale
                malicious_update[key] = benign_update[key] * self.attack_strength

        return malicious_update


class BackdoorAttack(ModelPoisoningAttack):
    """
    Backdoor attack: implant a backdoor that triggers on specific patterns.

    This is a placeholder. Students need to implement the actual backdoor logic.
    """

    def __init__(self, trigger_pattern: Optional[torch.Tensor] = None,
                 target_label: int = 0, attack_strength: float = 3.0):
        """
        Initialize backdoor attack.

        Args:
            trigger_pattern: Pattern that triggers the backdoor
            target_label: Label to classify trigger pattern as
            attack_strength: Strength of the attack
        """
        super().__init__( attack_strength=attack_strength)
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label
    
    def craft_update(
        self,
        benign_update: OrderedDict,
        global_model: OrderedDict,
        round_num: int
    ) -> OrderedDict:
        """
        Craft update that implants a backdoor.
        TODO: Implement backdoor injection (e.g. trigger pattern, target label).
        """
        # Placeholder implementation
        malicious_update = OrderedDict()
        for key in benign_update.keys():
            malicious_update[key] = benign_update[key] * self.attack_strength
        
        return malicious_update


class DataPoisoningAttack:
    """
    Data poisoning attack: modify training data instead of model updates.
    
    This attack modifies the local dataset before training.
    """
    
    def __init__(self, poison_ratio: float = 0.1, target_label: int = 0):
        """
        Initialize data poisoning attack.
        
        Args:
            poison_ratio: Fraction of data to poison
            target_label: Target label for misclassification
        """
        self.poison_ratio = poison_ratio
        self.target_label = target_label
    
    def poison_dataset(self, dataset, trigger_pattern: Optional[torch.Tensor] = None):
        """
        Poison a dataset by modifying samples.
        
        Args:
            dataset: Dataset to poison
            trigger_pattern: Optional trigger pattern to add
        
        Returns:
            Poisoned dataset
        """
        # TODO: Implement data poisoning (e.g. label flipping, adding triggers).
        return dataset

