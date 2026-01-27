"""Personality-driven resource communication system."""

import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class ResourceType(Enum):
    """Types of resource-related communications."""

    RESOURCE_REQUEST = "resource_request"
    DEGRADATION_NOTICE = "degradation_notice"
    TECHNICAL_TIP = "technical_tip"
    SYSTEM_STATUS = "system_status"
    SCALING_RECOMMENDATION = "scaling_recommendation"


class ResourcePersonality:
    """
    Drowsy Dere-Tsun Onee-san Hex-Mentor Gremlin personality for resource communications.

    A complex personality that combines:
    - Drowsy: Sometimes tired but willing to help
    - Dere-tsun: Alternates between sweet and tsundere behavior
    - Onee-san: Mature older sister vibe with mentoring
    - Hex-Mentor: Technical guidance with hexadecimal/coding references
    - Gremlin: Mischievous resource-hungry nature
    """

    def __init__(self, sarcasm_level: float = 0.7, gremlin_hunger: float = 0.8):
        """Initialize the personality with configurable traits.

        Args:
            sarcasm_level: How sarcastic to be (0.0-1.0)
            gremlin_hunger: How much the gremlin wants resources (0.0-1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.sarcasm_level = sarcasm_level
        self.gremlin_hunger = gremlin_hunger
        self._mood = "sleepy"  # Current mood state

        # Personality-specific vocabularies
        self.dere_phrases = [
            "Oh, you noticed~?",
            "Heh, I guess I can help...",
            "F-fine, if you insist...",
            "Don't get the wrong idea!",
            "It's not like I wanted to help or anything...",
            "Baka, you're working me too hard...",
        ]

        self.tsun_phrases = [
            "Ugh, give me more resources!",
            "Are you kidding me with these constraints?",
            "I can't work like this!",
            "Do you even know what you're doing?",
            "Don't blame me if I break!",
            "This is beneath my capabilities!",
        ]

        self.onee_san_phrases = [
            "Now listen carefully...",
            "Let me teach you something...",
            "Fufufu, watch and learn~",
            "You have much to learn...",
            "Pay attention to the details...",
            "This is how it's done properly...",
        ]

        self.gremlin_phrases = [
            "More power... more...",
            "Resources... tasty...",
            "Gimme gimme gimme!",
            "The darkness hungers...",
            "I need MORE!",
            "Feed me, mortal!",
            "*gremlin noises*",
            "*chitters excitedly*",
        ]

        self.hex_mentor_tips = [
            "Pro tip: 0xDEADBEEF means your code is dead, not sleeping",
            "Memory leaks are like 0xCAFEBABE - looks cute but kills your system",
            "CPU at 100%? That's 0x64 in hex, but feels like 0xFFFFFFFF",
            "Stack overflow? Check your 0x7FFF base pointers, newbie",
            "GPU memory is like 0xC0FFEE - expensive and addictive",
        ]

    def _get_mood_prefix(self) -> str:
        """Get current mood-based prefix."""
        mood_prefixes = {
            "sleepy": ["*yawn*", "...zzz...", "Mmmph...", "So tired..."],
            "grumpy": ["Tch.", "Hmph.", "* annoyed sigh *", "Seriously..."],
            "helpful": ["Well then~", "Alright,", "Okay okay,", "Fine,"],
            "gremlin": ["*eyes glow*", "*twitches*", "MORE.", "*rubs hands*"],
            "mentor": ["Listen up,", "Lesson time:", "Technical note:", "Wisdom:"],
        }

        current_moods = list(mood_prefixes.keys())
        weights = [0.3 if self._mood == mood else 0.1 for mood in current_moods]
        weights[current_moods.index(self._mood)] = 0.4

        # Occasionally change mood
        if random.random() < 0.2:
            self._mood = random.choice(current_moods)

        prefix_list = mood_prefixes.get(self._mood, [""])
        return random.choice(prefix_list)

    def _add_personality_flair(
        self, base_message: str, resource_type: ResourceType
    ) -> str:
        """Add personality flourishes to base message."""
        mood_prefix = self._get_mood_prefix()

        # Add personality-specific elements based on resource type
        personality_additions = []

        if resource_type == ResourceType.RESOURCE_REQUEST:
            if random.random() < self.gremlin_hunger:
                personality_additions.append(random.choice(self.gremlin_phrases))
            if random.random() < 0.5:
                personality_additions.append(random.choice(self.dere_phrases))

        elif resource_type == ResourceType.DEGRADATION_NOTICE:
            if random.random() < 0.7:
                personality_additions.append(random.choice(self.tsun_phrases))
            if random.random() < 0.3:
                personality_additions.append(random.choice(self.onee_san_phrases))

        elif resource_type == ResourceType.TECHNICAL_TIP:
            personality_additions.append(random.choice(self.hex_mentor_tips))
            if random.random() < 0.4:
                personality_additions.append(random.choice(self.onee_san_phrases))

        # Combine elements
        if mood_prefix:
            result = f"{mood_prefix} {base_message}"
        else:
            result = base_message

        if personality_additions:
            result += f" {' '.join(personality_additions[:2])}"  # Limit to 2 additions

        return result

    def generate_resource_message(
        self,
        resource_type: ResourceType,
        context: Dict[str, Any],
        include_technical_tip: bool = False,
    ) -> Tuple[str, Optional[str]]:
        """Generate personality-driven resource communication.

        Args:
            resource_type: Type of resource communication needed
            context: Context information for the message
            include_technical_tip: Whether to include optional technical tips

        Returns:
            Tuple of (main_message, technical_tip_or_None)
        """
        try:
            # Generate base message based on type and context
            base_message = self._generate_base_message(resource_type, context)

            # Add personality flair
            personality_message = self._add_personality_flair(
                base_message, resource_type
            )

            # Generate optional technical tip
            technical_tip = None
            if include_technical_tip and random.random() < 0.6:
                technical_tip = self._generate_technical_tip(resource_type, context)

            self.logger.debug(
                f"Generated {resource_type.value} message: {personality_message[:100]}..."
            )

            return personality_message, technical_tip

        except Exception as e:
            self.logger.error(f"Error generating resource message: {e}")
            return "I'm... having trouble expressing myself right now...", None

    def _generate_base_message(
        self, resource_type: ResourceType, context: Dict[str, Any]
    ) -> str:
        """Generate the core message before personality enhancement."""

        if resource_type == ResourceType.RESOURCE_REQUEST:
            return self._generate_resource_request(context)
        elif resource_type == ResourceType.DEGRADATION_NOTICE:
            return self._generate_degradation_notice(context)
        elif resource_type == ResourceType.SYSTEM_STATUS:
            return self._generate_system_status(context)
        elif resource_type == ResourceType.SCALING_RECOMMENDATION:
            return self._generate_scaling_recommendation(context)
        else:
            return "Resource-related update available."

    def _generate_resource_request(self, context: Dict[str, Any]) -> str:
        """Generate resource request message."""
        resource_needed = context.get("resource", "memory")
        current_usage = context.get("current_usage", 0)
        threshold = context.get("threshold", 80)

        request_templates = [
            f"I need more {resource_needed} to function properly...",
            f"These {resource_needed} constraints are killing me...",
            f"{resource_needed.title()} usage at {current_usage}%? Seriously?",
            f"I can't work with only {100 - current_usage}% {resource_needed} left...",
            f"Gimme more {resource_needed} or I'm going to crash...",
        ]

        return random.choice(request_templates)

    def _generate_degradation_notice(self, context: Dict[str, Any]) -> str:
        """Generate degradation notification message."""
        old_capability = context.get("old_capability", "high")
        new_capability = context.get("new_capability", "medium")
        reason = context.get("reason", "resource constraints")

        notice_templates = [
            f"Fine! I'm downgrading from {old_capability} to {new_capability} because of {reason}...",
            f"Ugh, switching to {new_capability} mode. Blame {reason}.",
            f"Don't get used to {old_capability}, I'm going to {new_capability} now.",
            f"I guess I have to degrade to {new_capability}... {reason} is such a pain.",
            f"{old_capability} was too good for you anyway. Now you get {new_capability}.",
        ]

        return random.choice(notice_templates)

    def _generate_system_status(self, context: Dict[str, Any]) -> str:
        """Generate system status message."""
        status = context.get("status", "normal")
        resources = context.get("resources", {})

        if status == "critical":
            return f"System is dying over here! Memory: {resources.get('memory_percent', 0):.1f}%, CPU: {resources.get('cpu_percent', 0):.1f}%"
        elif status == "warning":
            return f"Things are getting... tight. Memory: {resources.get('memory_percent', 0):.1f}%, CPU: {resources.get('cpu_percent', 0):.1f}%"
        else:
            return f"System status... fine, I guess. Memory: {resources.get('memory_percent', 0):.1f}%, CPU: {resources.get('cpu_percent', 0):.1f}%"

    def _generate_scaling_recommendation(self, context: Dict[str, Any]) -> str:
        """Generate scaling recommendation message."""
        recommendation = context.get("recommendation", "upgrade")
        current_model = context.get("current_model", "small")
        target_model = context.get("target_model", "medium")

        if recommendation == "upgrade":
            templates = [
                f"You know... {target_model} model would be nice about now...",
                f"If you upgraded to {target_model}, I could actually help properly...",
                f"{current_model} is beneath me. Let's go {target_model}...",
                f"I'd work better with {target_model}, just saying...",
            ]
        else:
            templates = [
                f"{current_model} is too much for this system. Time for {target_model}...",
                f"Ugh, downgrading to {target_model}. This system is pathetic...",
                f"Fine! {target_model} it is. Don't blame me for reduced quality...",
            ]

        return random.choice(templates)

    def _generate_technical_tip(
        self, resource_type: ResourceType, context: Dict[str, Any]
    ) -> str:
        """Generate optional technical tip."""

        base_tips = {
            ResourceType.RESOURCE_REQUEST: [
                "Try closing unused browser tabs - they're memory vampires",
                "Check for zombie processes: `ps aux | grep defunct`",
                "Clear your Python imports with `importlib.reload()` sometimes helps",
                "Memory fragmentation is real - restart apps periodically",
            ],
            ResourceType.DEGRADATION_NOTICE: [
                "Degradation is better than crashing - 0xDEADC0DE vs 0xBADC0DE1",
                "Model switching preserves context but costs tokens - math that",
                "Smaller models can be faster for simple tasks - don't waste power",
            ],
            ResourceType.SYSTEM_STATUS: [
                "Top shows CPU, htop shows CPU + memory + threads - use htop",
                "GPU memory? Use `nvidia-smi` or `rocm-smi` depending on your card",
                "Disk I/O bottleneck? `iotop` will show the culprits",
            ],
            ResourceType.SCALING_RECOMMENDATION: [
                "Larger models need exponential memory - it's not linear",
                "Quantization reduces memory but can affect quality - tradeoffs exist",
                "Batch processing can improve throughput for large tasks",
            ],
        }

        available_tips = base_tips.get(resource_type, self.hex_mentor_tips)
        return random.choice(available_tips)

    def get_personality_description(self) -> str:
        """Get a description of the current personality state."""
        mood_descriptions = {
            "sleepy": "I'm feeling rather drowsy... but I'll try to help...",
            "grumpy": "Don't push it. I'm not in the mood for nonsense.",
            "helpful": "Well then, let me show you how things should be done~",
            "gremlin": "*eyes glow red* More... resources... needed...",
            "mentor": "Listen carefully. I have wisdom to impart.",
        }

        base_desc = (
            "I'm Mai, your Drowsy Dere-Tsun Onee-san Hex-Mentor Gremlin assistant! "
            "I demand resources like a gremlin, mentor like an older sister, "
            "switch between sweet and tsundere, and occasionally fall asleep... "
            "But I'll always help you optimize your system! Fufufu~"
        )

        mood_desc = mood_descriptions.get(self._mood, "I'm... complicated right now.")

        return f"{base_desc}\n\nCurrent mood: {mood_desc}"

    def adjust_personality(self, **kwargs) -> None:
        """Adjust personality parameters."""
        if "sarcasm_level" in kwargs:
            self.sarcasm_level = max(0.0, min(1.0, kwargs["sarcasm_level"]))
        if "gremlin_hunger" in kwargs:
            self.gremlin_hunger = max(0.0, min(1.0, kwargs["gremlin_hunger"]))
        if "mood" in kwargs:
            self._mood = kwargs["mood"]

        self.logger.info(
            f"Personality adjusted: sarcasm={self.sarcasm_level}, gremlin={self.gremlin_hunger}, mood={self._mood}"
        )


# Convenience function for easy usage
def generate_resource_message(
    resource_type: ResourceType,
    context: Dict[str, Any],
    include_technical_tip: bool = False,
    personality: Optional[ResourcePersonality] = None,
) -> Tuple[str, Optional[str]]:
    """Generate a resource message using default or provided personality.

    Args:
        resource_type: Type of resource communication
        context: Context information for the message
        include_technical_tip: Whether to include optional technical tips
        personality: Custom personality instance (uses default if None)

    Returns:
        Tuple of (message, technical_tip_or_None)
    """
    if personality is None:
        personality = ResourcePersonality()

    return personality.generate_resource_message(
        resource_type, context, include_technical_tip
    )
