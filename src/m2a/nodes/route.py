"""Routing node for the Manhwa-to-Anime pipeline."""

import logging
import time

# TODO: Import actual state types
# from m2a.state import RunState, PipelineConfig

logger = logging.getLogger(__name__)

def route(state: 'RunState', config: 'PipelineConfig') -> 'RunState':
    """Decide generation routing (i2v, parallax, kenburns) for each panel.
    
    Args:
        state: Current run state.
        config: Pipeline configuration.
        
    Returns:
        Updated run state.
    """
    start_time = time.time()
    logger.info("Starting route node...")
    
    try:
        # GPU budget defaults to 100 if not provided
        # TODO: Implement robust GPU budget tracking on the state object
        gpu_budget = getattr(config, "gpu_budget", 100)
        
        for panel in state.panels:
            understanding = panel.get("understanding", {})
            motion_intensity = understanding.get("motion_intensity", "low")
            shot_type = understanding.get("shot_type", "medium")
            is_action = understanding.get("is_action", False)
            
            # Hero panel logic
            is_hero_panel = (shot_type == "close_up" and motion_intensity == "high") or is_action
            
            if is_hero_panel and gpu_budget >= 10:
                route_decision = "i2v"
                gpu_budget -= 10
            elif shot_type == "medium" and motion_intensity == "moderate" and gpu_budget >= 5:
                route_decision = "parallax"
                gpu_budget -= 5
            else:
                route_decision = "kenburns"
                gpu_budget -= 1
                
            panel["route"] = route_decision
            
    except Exception as e:
        logger.exception("Error in route node")
        state.errors.append(str(e))
        
    logger.info(f"Route node completed in {time.time() - start_time:.2f}s")
    return state
