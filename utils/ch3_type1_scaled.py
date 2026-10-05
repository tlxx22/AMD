"""Project-authorized horizon-scaled type1; pure Python, fixed Adam betas."""
import copy
from utils.ch3_contract import digest
from utils.ch3_type1 import Schedule as LegacySchedule

NAME='type1_horizon_scaled_v1'
STATE_SCHEMA='ch3-type1-horizon-scaled-state-v1'
def coefficient(epochs):
    if type(epochs)is not int or epochs<1:raise ValueError('positive fixed maximum epochs')
    return 1. if epochs<=10 else 8./(epochs-2)
def lr_used(epoch,epochs):
    if type(epoch)is not int or not 1<=epoch<=epochs:raise ValueError('epoch within fixed horizon')
    return 1e-4*0.5**(max(epoch-2,0)*coefficient(epochs))
def configuration(epochs,train_batches):
    if type(train_batches)is not int or train_batches<1:raise ValueError('positive train batches')
    return dict(name=NAME,formula_version='max(e-2,0)*c(E); c=1 if E<=10 else 8/(E-2)',initial_lr=1e-4,factor=0.5,
        epochs=epochs,steps_per_epoch=train_batches,coefficient=coefficient(epochs),
        adjustment='after_validation_best_update_before_checkpoint_if_continuing',first_decay_training_epoch=3)
def validate_config(c):
    if c!=configuration(c.get('epochs'),c.get('steps_per_epoch')):raise ValueError('exact frozen horizon-scaled type1')
    return c
class Schedule(LegacySchedule):
    checkpoint_schema='ch3-state-v4-type1-horizon-scaled'
    def __init__(self,optimizer,config):
        self.optimizer=optimizer;self.config=copy.deepcopy(validate_config(config));self.updates=0;self.completed_epoch=0;self.continuing=True
        self.betas=[list(g['betas'])for g in optimizer.param_groups];self.lr_epoch=1
        for g in optimizer.param_groups:g['lr']=lr_used(1,self.config['epochs'])
        optimizer._ch3_type1=self
    def after_epoch(self,epoch,continuing):
        if epoch!=self.completed_epoch+1 or self.updates!=epoch*self.config['steps_per_epoch'] or not self.continuing:raise ValueError('complete epoch boundary required')
        if continuing and epoch>=self.config['epochs']:raise ValueError('fixed horizon exhausted')
        self.completed_epoch=epoch;self.continuing=bool(continuing);self.lr_epoch=epoch+1 if continuing else epoch
        for g in self.optimizer.param_groups:g['lr']=lr_used(self.lr_epoch,self.config['epochs'])
        self.validate()
    def validate(self):
        if self.updates<self.completed_epoch*self.config['steps_per_epoch'] or self.updates>min(self.config['epochs'],self.completed_epoch+1)*self.config['steps_per_epoch']:raise ValueError('scaled type1 progress')
        if len(self.betas)!=len(self.optimizer.param_groups):raise ValueError('scaled type1 groups')
        for g,b in zip(self.optimizer.param_groups,self.betas):
            if list(g['betas'])!=b or g['lr']!=lr_used(self.lr_epoch,self.config['epochs']):raise ValueError('fixed betas/exact scaled LR')
    def state_dict(self):
        self.validate();E=self.config['epochs']
        return dict(schema=STATE_SCHEMA,config=copy.deepcopy(self.config),config_sha=digest(self.config),updates=self.updates,
            completed_epoch=self.completed_epoch,lr_epoch=self.lr_epoch,continuing=self.continuing,
            current_lr=lr_used(self.lr_epoch,E),next_epoch_lr=lr_used(min(E,self.completed_epoch+1),E),
            fixed_betas=copy.deepcopy(self.betas),adjustment_position=self.config['adjustment'],
            groups=[dict(lr=g['lr'],betas=list(g['betas']))for g in self.optimizer.param_groups])
    def load_state_dict(self,value,steps):
        if value.get('schema')!=STATE_SCHEMA or value.get('config')!=self.config or value.get('config_sha')!=digest(self.config) or value.get('updates')!=steps:raise ValueError('scaled checkpoint/profile; old scheduler prohibited')
        if value.get('fixed_betas')!=self.betas or steps!=value.get('completed_epoch',-1)*self.config['steps_per_epoch']:raise ValueError('scaled checkpoint committed epoch/steps/betas')
        self.updates=steps;self.completed_epoch=value['completed_epoch'];self.lr_epoch=value['lr_epoch'];self.continuing=value['continuing']
        if self.lr_epoch!=(self.completed_epoch+1 if self.continuing else self.completed_epoch) or value!=self.state_dict():raise ValueError('scaled checkpoint LR adjustment position')
def validate_probe_trace(config,traces,betas):
    validate_config(config)
    if len(traces)!=6:raise ValueError('six short update snapshots')
    for i,value in enumerate(traces,1):
        schedule=Schedule(type('SyntheticGroups',(),{'param_groups':[dict(lr=1e-4,betas=tuple(betas))]})(),config)
        schedule.updates=i
        if value!=schedule.state_dict():raise ValueError('exact scaled scheduler/LR/betas; short trajectory is not an epoch')
