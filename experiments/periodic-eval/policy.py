"""Pure milestone decision shared by trainer and tests."""
def due(hours,step,total_steps,policy):
 return bool(policy.get('enabled') and (hours>=policy['next_hours'] or (step==total_steps and policy.get('evaluate_final',True))))
