"""Fixed M launcher capability; never issues an experiment permission."""
import hashlib,json,os,secrets,stat,subprocess
from pathlib import Path
from utils import ch3_m_tasks as s
from utils.ch3_contract import digest
from m6_remaining_entry import identity,same
TOKEN_ENV='CH3_M_LAUNCH_TOKEN'
def spec(kind):
 from utils import ch3_m_execution as e
 if kind=='probe':return dict(scope=s.PROBE,log=e.launcher_log(True),root=e.PROBE_ROOT,session='ch3-m-baselines-probe-v1',wrapper=s.ROOT/'scripts/ch3/start_m_probe.sh')
 if kind=='formal':return dict(scope=s.ID,log=e.launcher_log(False),root=e.CONTROL,session='ch3-m-baselines-v1',wrapper=s.ROOT/'scripts/ch3/start_m_baselines.sh')
 if kind=='handoff':return dict(scope=s.ID+'-handoff',log=s.PACKAGE/'m-handoff-launcher.log',root=s.PACKAGE/'handoff-execution-v1',session='ch3-m-baselines-handoff-v1',wrapper=s.ROOT/'scripts/ch3/start_m_handoff.sh')
 raise ValueError('unknown fixed M launch scope')
def metadata_path(kind):return Path(str(spec(kind)['log'])+'.launch.json')
def claim_path(kind):return Path(str(spec(kind)['log'])+'.claimed.json')
def file_identity(path):
 path=Path(path)
 if path.resolve()!=path or path.is_symlink():raise ValueError('M launcher exact path required')
 st=path.lstat()
 if not stat.S_ISREG(st.st_mode)or st.st_nlink!=1 or st.st_uid!=os.getuid():raise ValueError('M launcher regular owned unshared file required')
 return dict(device=st.st_dev,inode=st.st_ino,owner=st.st_uid)
def exclusive_json(path,value):
 fd=os.open(str(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w')as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def ancestors(pid):
 rows=[]
 for _ in range(32):
  v=identity(pid)
  if not v:break
  rows.append(v);fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split();parent=int(fields[1])
  if parent<=1 or parent==pid:break
  pid=parent
 return rows
def command_has(pid,path):
 proc=Path('/proc')/str(pid);args=(proc/'cmdline').read_bytes().decode().split('\0');cwd=(proc/'cwd').resolve()
 return 'start'in args and any(a and(Path(a)if Path(a).is_absolute()else cwd/a)==path for a in args)
def verify_creator(kind,driver,owner):
 if not same(owner):raise ValueError('M launch creator identity is not live')
 if not any(v['pid']==owner['pid']and v['start_ticks']==owner['start_ticks']for v in ancestors(os.getpid())):raise ValueError('M creator is not an ancestor')
 path=spec(kind)['wrapper']if driver=='tmux'else s.ROOT/'m6_m_handoff_entry.py'
 if driver not in ('tmux','handoff')or not command_has(owner['pid'],path):raise ValueError('fixed M wrapper/owned handoff required')
 if driver=='handoff'and kind=='handoff':raise ValueError('recursive handoff forbidden')
def tmux_view(session):
 text=subprocess.check_output(['tmux','display-message','-p','-t',session+':0.0','#{session_name}\t#{session_id}\t#{pane_pid}\t#{socket_path}'],text=True).strip()
 name,sid,pid,socket=text.split('\t');pane=identity(int(pid))
 if name!=session or not pane or not any(v['pid']==pane['pid']and v['start_ticks']==pane['start_ticks']for v in ancestors(os.getpid())):raise ValueError('M start is outside its exact tmux pane')
 return dict(session=name,session_id=sid,pane=pane,socket=socket)
def prepare(c,kind,owner_pid,*,driver='tmux',approval=None):
 from ch3_runner import git
 x=spec(kind);owner=identity(owner_pid);verify_creator(kind,driver,owner)
 if x['root'].exists()or x['root'].is_symlink():raise FileExistsError('M retained controller')
 if x['log'].stat().st_size!=0:raise ValueError('M launcher must be newly created empty')
 if claim_path(kind).exists():raise FileExistsError('M launch already claimed')
 token=secrets.token_hex(32)
 value=dict(kind=kind,scope=x['scope'],driver=driver,owner=owner,log=str(x['log']),log_identity=file_identity(x['log']),token_sha256=hashlib.sha256(token.encode()).hexdigest(),commit=git('rev-parse','HEAD'),protocol_sha=digest(c),approval=s.ref(approval)if approval else None,wrapper=str(x['wrapper']))
 exclusive_json(metadata_path(kind),value);return token
def verify(c,kind,token,*,approval=None):
 from ch3_runner import git
 if not isinstance(token,str)or len(token)!=64:raise PermissionError('matching wrapper launch identity required')
 x=spec(kind);p=metadata_path(kind);file_identity(p);value=json.loads(p.read_text())
 for k,v in dict(kind=kind,scope=x['scope'],log=str(x['log']),wrapper=str(x['wrapper']),commit=git('rev-parse','HEAD'),protocol_sha=digest(c),approval=s.ref(approval)if approval else None).items():
  if value.get(k)!=v:raise ValueError('M launch binding mismatch')
 if not secrets.compare_digest(value.get('token_sha256',''),hashlib.sha256(token.encode()).hexdigest()):raise PermissionError('M launch token mismatch')
 if value['log_identity']!=file_identity(x['log']):raise ValueError('M launch log inode/owner changed')
 if claim_path(kind).exists():raise FileExistsError('M launch single-use claim already exists')
 if value['driver']=='tmux':value['actual_tmux']=tmux_view(x['session'])
 elif value['driver']=='handoff':
  owner=value['owner']
  if os.getppid()!=owner['pid']or not same(owner)or not command_has(owner['pid'],s.ROOT/'m6_m_handoff_entry.py'):raise PermissionError('M child not owned by this live handoff')
  value['actual_tmux']=tmux_view(spec('handoff')['session'])
 else:raise ValueError('M unknown launcher driver')
 return value
def claim(c,kind,token,*,approval=None):
 value=verify(c,kind,token,approval=approval);exclusive_json(claim_path(kind),dict(launch=s.ref(metadata_path(kind)),consumer=identity(os.getpid()),actual_tmux=value['actual_tmux']));return value
