"""Reorder only unconsumed windows; keep stage order and randomize matched pairs."""
import random,itertools,collections

def cost(row,lengths):
 x=lengths[row['window_id']];n=x['tokens']
 return n+n*n/65536+x['target_tokens']*.5

def balance(plan,completed_steps,lengths,seed=20261022):
 prefix=plan[:completed_steps*2];suffix=plan[completed_steps*2:];result=list(prefix);rng=random.Random(seed+completed_steps)
 for bucket,group in itertools.groupby(suffix,key=lambda r:r['bucket']):
  rows=sorted(group,key=lambda r:cost(r,lengths))
  if len(result)%2:result.append(rows.pop(0))
  pairs=[rows[i:i+2] for i in range(0,len(rows)-1,2)];rng.shuffle(pairs)
  for pair in pairs:
   if rng.random()<.5:pair.reverse()
   result.extend(pair)
  if len(rows)%2:result.append(rows[-1])
 assert result[:completed_steps*2]==prefix
 assert collections.Counter(r['window_id'] for r in result)==collections.Counter(r['window_id'] for r in plan)
 assert [r['bucket'] for r in result]==[r['bucket'] for r in plan]
 assert {r['window_id']:r for r in result}=={r['window_id']:r for r in plan}
 return result

def pair_cost(rows,lengths):return sum(max(cost(r,lengths) for r in rows[i:i+2]) for i in range(0,len(rows),2))
