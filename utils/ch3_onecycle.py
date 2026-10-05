"""Explicit TimeMixer ETTh1-derived OneCycle recipe; no model imports."""
import copy
from utils.ch3_contract import digest

RECIPE = dict(name='OneCycleLR', max_lr=0.01, pct_start=0.2,
              anneal_strategy='cos', cycle_momentum=True,
              base_momentum=0.85, max_momentum=0.95,
              div_factor=25.0, final_div_factor=10000.0,
              three_phase=False, last_epoch=-1, verbose=False)


def configuration(epochs, train_batches):
    if type(epochs) is not int or type(train_batches) is not int or min(epochs, train_batches) < 1:
        raise ValueError('positive frozen epoch/batch arithmetic required')
    return dict(RECIPE, total_steps=None, epochs=epochs, steps_per_epoch=train_batches)


def validate_config(config):
    expected=configuration(config.get('epochs'), config.get('steps_per_epoch'))
    if config != expected:
        raise ValueError('exact frozen OneCycle configuration required')
    return expected


class Schedule:
    def __init__(self, optimizer, config):
        import torch
        if torch.__version__ != '2.0.1':
            raise ValueError('frozen PyTorch 2.0.1 required')
        self.config=copy.deepcopy(validate_config(config))
        self.optimizer=optimizer
        self.scheduler=torch.optim.lr_scheduler.OneCycleLR(
            optimizer, **{k:v for k,v in self.config.items() if k!='name'})
        self.updates=0
        self.frozen={k:copy.deepcopy(v) for k,v in self.raw_state().items()
                     if k not in ('last_epoch','_step_count','_last_lr','_get_lr_called_within_step')}
        self.beta2=[g['betas'][1] for g in optimizer.param_groups]
        optimizer._ch3_onecycle=self

    def after_successful_update(self):
        if self.updates >= self.scheduler.total_steps:
            raise ValueError('scheduler update exceeds scientific epoch cap')
        self.scheduler.step()
        self.updates += 1
        self.validate()

    def validate(self):
        s=self.raw_state()
        if self.scheduler.anneal_func.__func__.__name__!='_annealing_cos':raise ValueError('frozen cosine anneal function required')
        if {k:v for k,v in s.items() if k in self.frozen} != self.frozen:
            raise ValueError('scheduler recipe/state changed')
        if s['total_steps'] != self.config['epochs']*self.config['steps_per_epoch'] or s['last_epoch'] != self.updates or s['_step_count'] != self.updates+1:
            raise ValueError('scheduler step accounting')
        if len(self.optimizer.param_groups)!=len(s['_last_lr']):raise ValueError('scheduler group count')
        for n,(group,lr) in enumerate(zip(self.optimizer.param_groups,s['_last_lr'])):
            if group['lr'] != lr or len(group['betas']) != 2:
                raise ValueError('optimizer LR/scheduler identity')
            start=0.0
            for i,phase in enumerate(s['_schedule_phases']):
                if self.updates <= phase['end_step'] or i==len(s['_schedule_phases'])-1:
                    pct=(self.updates-start)/(phase['end_step']-start)
                    expected=self.scheduler.anneal_func(group[phase['start_momentum']],group[phase['end_momentum']],pct)
                    if group['betas']!=(expected,self.beta2[n]):raise ValueError('scheduled beta1/beta2 mismatch')
                    break
                start=phase['end_step']
        return self

    def raw_state(self):
        # PyTorch 2.0.1 exposes a bound method in state_dict. Its frozen name is
        # config-bound; never pickle an optimizer-bearing scheduler object or
        # serialize that callable into probe JSON.
        return {k:copy.deepcopy(v) for k,v in self.scheduler.state_dict().items() if k!='anneal_func'}

    def state_dict(self):
        self.validate()
        return dict(schema='ch3-onecycle-state-v1', config=copy.deepcopy(self.config),
                    config_sha=digest(self.config), updates=self.updates,
                    scheduler=self.raw_state(),
                    groups=[dict(lr=g['lr'],betas=list(g['betas'])) for g in self.optimizer.param_groups])

    def load_state_dict(self, value, steps):
        if value.get('schema')!='ch3-onecycle-state-v1' or value.get('config')!=self.config or value.get('config_sha')!=digest(self.config) or value.get('updates')!=steps:
            raise ValueError('scheduler checkpoint/profile mismatch')
        if set(value['scheduler'])!=set(self.raw_state()):raise ValueError('scheduler checkpoint schema mismatch')
        self.scheduler.load_state_dict(value['scheduler'])
        self.updates=steps
        if value['groups'] != [dict(lr=g['lr'],betas=list(g['betas'])) for g in self.optimizer.param_groups]:
            raise ValueError('checkpoint optimizer LR/beta mismatch')
        self.validate()


def build(optimizer, profile):
    config=profile['training'].get('scheduler')
    if isinstance(config,dict) and config.get('name')=='TimeXerType1':
        from utils.ch3_type1 import Schedule as Type1Schedule
        return Type1Schedule(optimizer,config)
    return Schedule(optimizer, config) if isinstance(config,dict) else None


def optimizer_groups_static(groups):
    """Only declared OneCycle LR and beta1 are dynamic; beta2 stays exact."""
    result=copy.deepcopy(groups)
    for g in result:
        g['lr']='scheduled-exact-per-step'
        g['betas'][0]='scheduled-beta1-exact-per-step'
    return result
