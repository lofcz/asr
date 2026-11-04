"""Bound pathological generation without requiring period to divide 64 tokens."""
def repetition_period(tokens):
    if len(tokens)<256:return None
    for period in range(1,129):
        span=max(256,period*8)
        if len(tokens)<span:continue
        tail=tokens[-span:]
        if all(tail[i]==tail[i-period] for i in range(period,len(tail))):return period
    return None
