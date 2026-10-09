"""Audit Train own-state context visibility; no Dev/Test labels or model fitting."""
import copy, json, sys
from pathlib import Path
from collections import defaultdict, Counter
import torch
from transformers import AutoTokenizer
from newsvendor.io import read,lines,write,digest
from newsvendor.structured_rollout import ResearchRouter
from newsvendor.structured_inputs import query_tokens
config=read('configs/l40s-research-replay-v1.json')['encoder']
tok=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],cache_dir='.cache/torch-models',local_files_only=True)
router=ResearchRouter(None,tok,config)
source=Path('results/l40s-research-replay-v1/base/42/value-targets-0.jsonl')
rows=lines(source)
counts=defaultdict(Counter); details=[]
for i,r in enumerate(rows):
 assert r['split']=='train'
 linked='forecast' in r['input']['task']
 group='retail' if linked else 'generated'
 v=router.view(r['input'],r['state'])
 counts[group]['states']+=1; counts[group]['truncated']+=int(v['queryTruncated'])
 for field in ['q','gamma','types','values']:
  s=copy.deepcopy(r['state'])
  if field=='q':
   if s['q'] is None: continue
   s['q']+=1
  elif field=='gamma':
   if s['gamma'] is None:continue
   s['gamma']+=1
  elif field=='types':
   s['types']['b']='fact' if s['types'].get('b')!='fact' else 'preference'
  else:
   if not isinstance(s['values'].get('p'),(int,float)):continue
   s['values']['p']*=1.01
  other=router.view(r['input'],s)
  same=(torch.equal(v['batch']['input_ids'],other['batch']['input_ids']) and torch.equal(v['batch']['attention_mask'],other['batch']['attention_mask']) and v['actions']==other['actions'] and v['allowedActions']==other['allowedActions'])
  counts[group][field+'_tested']+=1;counts[group][field+'_same']+=int(same)
  if same and len([x for x in details if x['group']==group and x['field']==field])<2:
   details.append({'id':r['id'],'group':group,'field':field,'stateHash':digest(r['state']),'prefix':tok.decode(v['batch']['input_ids'][0,1:129])})
 if (i+1)%200==0: print(i+1,flush=True)
result={'scope':'Train-only model-input aliasing audit. State perturbations are visibility diagnostics, not alternative ground truth or forecasts.','testUsed':False,'devUsed':False,'source':str(source),'hash':digest(source.read_bytes()),'counts':dict(counts),'examples':details,'scriptHash':digest(Path(__file__).read_bytes())}
write('results/research-checks/policy-context-audit.json',result)
print(dict(counts),flush=True)
