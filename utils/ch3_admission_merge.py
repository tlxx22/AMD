"""Measured UrbanEV confirmation + explicit, immutable 87-group carry-forward.

No model execution, CSV parsing, old-result rewriting or self-granted review.
"""
import copy
from pathlib import Path
from utils.ch3_contract import digest,profile,task_by_id
from utils.ch3_extension_probe import PACKAGE,PRODUCTION,read,sha,specification as extension_spec
D=PACKAGE/'urban-numeric-confirmation-v1'
PROOF=D/'carry-forward.json'
PURPOSE='m6_merged_resource_admission_v1'

def bound(ref):
 p=Path(ref['path'])
 if p.is_symlink()or not p.is_file()or sha(p)!=ref['sha256']:raise ValueError('source path/SHA mismatch: '+str(p))
 d=read(p)
 if sha(p)!=ref['sha256']:raise ValueError('source changed while reading')
 return d

def inherited_sources(c):
 """Validate accepted JSON sources and reviewed code-delta declaration, not rerun models."""
 proof=read(PROOF)
 if proof['protocol_sha']!=digest(c)or proof['profile_shas']!={t['id']:digest(profile(c,t))for t in c['tasks']}:raise ValueError('carry-forward scientific profile changed')
 from ch3_runner import code_binding
 if proof['after_code']!=code_binding():raise ValueError('carry-forward code delta no longer matches implementation')
 source={k:bound(v)for k,v in proof['sources'].items()}
 partial=source['partial'];seven=source['seven'];parent=source['parent'];revision=source['revision'];plan=source['extension_plan']
 spec=extension_spec(c)
 if revision.get('reviewed')is not True or set(seven['accepted_decisions'])!=set(spec['new_measurements'])-{'TimeMixer-UrbanEV-F4'}:raise ValueError('accepted review coverage')
 if proof['sources']['partial']not in seven['sources']:raise ValueError('seven-group receipt did not bind this partial report')
 if partial['protocol_sha']!=digest(c)or partial['scope']['plan_sha256']!=proof['sources']['extension_plan']['sha256']:raise ValueError('parent scope/plan')
 if partial['scope']['parent_report_sha']!=proof['sources']['parent']['sha256']:raise ValueError('parent report identity')
 source_by_path={proof['sources'][k]['path']:v for k,v in source.items()}
 decisions={};lineage={}
 def put(gid,value,kind,ref,extra=None):
  if gid in decisions:raise ValueError('duplicate carry-forward group')
  decisions[gid]=copy.deepcopy(value);lineage[gid]=dict(kind=kind,source=ref,actual_commit=source_by_path[ref['path']].get('commit'),actual_protocol_sha=source_by_path[ref['path']].get('protocol_sha'),actual_code=source_by_path[ref['path']].get('code'),proof=extra)
 for gid in spec['original_groups']:put(gid,parent['decisions'][gid],'original_unchanged_51',proof['sources']['parent'])
 for gid in spec['proposed_inheritance']:
  item=next(v for v in plan['extension_groups']if v['group']==gid);old=parent['decisions'][item['parent_group']]
  if old['status']!='Passed'or old['concurrency']!=1 or set(item['profile_differences'])-{'dataset','features'}:raise ValueError('invalid q1 transfer proof')
  value=copy.deepcopy(old);value.update(representatives=spec['representatives'][gid],scope='reviewed computational q1 transfer; new data independently bound')
  put(gid,value,'reviewed_transfer_26',proof['sources']['parent'],item)
 for gid in spec['excluded_original_groups']:put(gid,revision['decisions'][gid],'reviewed_revised_lr_3',proof['sources']['revision'])
 for gid,value in seven['accepted_decisions'].items():put(gid,value,'reviewed_q1_7',proof['sources']['partial'],proof['sources']['seven'])
 if len(decisions)!=87 or decisions!=partial['decisions']:raise ValueError('accepted 87 decisions do not reconstruct exactly')
 from ch3_runner import verified_waves
 groups={g['id']:g for g in c['groups']}
 for gid,value in decisions.items():
  if value['status']!='Passed':raise ValueError('unaccepted inherited decision')
  verified_waves(groups[gid],value)
 # Hardware/environment remain actual old report fields, never rewritten.
 if partial['environment']!=revision['environment']or partial['hardware']!=revision['hardware']:raise ValueError('incompatible inherited environment/hardware')
 return dict(decisions=decisions,lineage=lineage,environment=partial['environment'],hardware=partial['hardware'],proof_sha=sha(PROOF))

def build_merged(c,confirmation_ref):
 from utils import ch3_urban_confirmation as scope
 from ch3_runner import code_binding,verified_waves
 if confirmation_ref['path']!=str(scope.ROOT/'complete.json'):raise ValueError('confirmation output identity')
 raw=bound(confirmation_ref);scope.validate_completion(c,raw)
 inherited=inherited_sources(c)
 if any(raw[k]!=inherited[k]for k in ('environment','hardware')):raise ValueError('confirmation differs from accepted resource environment')
 decisions=dict(inherited['decisions']);gid='TimeMixer-UrbanEV-F4'
 if gid in decisions or set(raw['decisions'])!={gid}:raise ValueError('88th identity')
 decisions[gid]=raw['decisions'][gid]
 if set(decisions)!={g['id']for g in c['groups']}or len(decisions)!=88:raise ValueError('88-group partition')
 for g in c['groups']:verified_waves(g,decisions[g['id']])
 lineage=dict(inherited['lineage']);lineage[gid]=dict(kind='new_urban_confirmation_1',source=confirmation_ref,actual_commit=raw['commit'],actual_protocol_sha=raw['protocol_sha'],actual_code=raw['code'],policy_sha=digest(scope.POLICY))
 return dict(purpose=PURPOSE,protocol_sha=digest(c),commit=raw['commit'],code=code_binding(),environment=raw['environment'],hardware=raw['hardware'],Q=sum(g['q']for g in c['groups']),decisions=decisions,lineage=lineage,partition=dict(original=51,transfer=26,accepted_q1=7,revised_lr=3,new_urban=1),carry_forward=dict(path=str(PROOF),sha256=inherited['proof_sha']),confirmation=confirmation_ref,execution_complete=True,reviewed=False,admission_granted=False,synthetic_fixture=False)

def validate_merged(c,report,require_review=False):
 if report.get('purpose')!=PURPOSE or report.get('synthetic_fixture')is not False:raise ValueError('not an actual merged report')
 expected=build_merged(c,report['confirmation'])
 # Only explicit external review metadata may differ from the generated report.
 content={k:v for k,v in report.items()if k not in ('reviewed','review')}
 reference={k:v for k,v in expected.items()if k!='reviewed'}
 if content!=reference:raise ValueError('merged report is not reconstructible from measured/accepted sources')
 if require_review:
  if report.get('reviewed')is not True:raise ValueError('new confirmation/merge result review required')
  review=report.get('review',{});ref=review.get('original_report',{})
  if ref.get('path')!=str(D/'merged-admission.json')or not review.get('source'):raise ValueError('explicit result review source required')
  if bound(ref)!=expected:raise ValueError('review is not bound to original unreviewed merged report')
 return report

def formal_report_reasons(c,report,approval):
 try:
  validate_merged(c,report,require_review=True)
  if not approval or any(report[k]!=approval.get(k)for k in ('code','environment','hardware','commit')):raise ValueError('formal authorization source/environment binding')
  for domain,q in [('ETTh1',4),('Weather',4),('ECL',2),('Exchange',4)]:
   d=report['decisions']['TimeMixer-'+domain+'-MS']
   if d['status']!='Passed'or d['concurrency']!=q:raise ValueError('revision waves lack admission')
  return []
 except (OSError,KeyError,ValueError,TypeError)as exc:return ['merged admission: '+str(exc)]
