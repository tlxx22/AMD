"""Locked TimeXer epoch-end type1 schedule; pure Python, no model/torch imports."""
import copy
from utils.ch3_contract import digest

NAME='TimeXerType1'
def configuration(epochs,train_batches):
    if type(epochs)is not int or type(train_batches)is not int or min(epochs,train_batches)<1:raise ValueError('positive epoch/batch arithmetic')
    return dict(name=NAME,initial_lr=1e-4,factor=0.5,epochs=epochs,steps_per_epoch=train_batches,
                adjustment='after_validation_best_update_before_checkpoint_if_continuing',first_decay_training_epoch=3)
def validate_config(c):
    if c!=configuration(c.get('epochs'),c.get('steps_per_epoch')):raise ValueError('exact frozen type1 recipe')
    return c
def lr_used(epoch):
    if type(epoch)is not int or epoch<1:raise ValueError('1-based epoch')
    return 1e-4*0.5**max(epoch-2,0)
class Schedule:
    checkpoint_schema='ch3-state-v3-type1'
    def __init__(self,optimizer,config):
        self.optimizer=optimizer;self.config=copy.deepcopy(validate_config(config));self.updates=0;self.completed_epoch=0;self.continuing=True
        self.betas=[list(g['betas']) for g in optimizer.param_groups]
        for g in optimizer.param_groups:g['lr']=1e-4
        self.lr_epoch=1;optimizer._ch3_type1=self
    def after_successful_update(self):
        if self.updates>=self.config['epochs']*self.config['steps_per_epoch']:raise ValueError('type1 update cap')
        self.updates+=1;self.validate()
    def after_epoch(self,epoch,continuing):
        if epoch!=self.completed_epoch+1 or self.updates!=epoch*self.config['steps_per_epoch'] or not self.continuing:raise ValueError('complete epoch boundary required')
        self.completed_epoch=epoch;self.continuing=bool(continuing)
        self.lr_epoch=epoch+1 if continuing else epoch
        for g in self.optimizer.param_groups:g['lr']=lr_used(self.lr_epoch)
        self.validate()
    def validate(self):
        if self.updates<self.completed_epoch*self.config['steps_per_epoch'] or self.updates>min(self.config['epochs'],self.completed_epoch+1)*self.config['steps_per_epoch']:raise ValueError('type1 progress')
        if len(self.betas)!=len(self.optimizer.param_groups):raise ValueError('type1 groups')
        for g,b in zip(self.optimizer.param_groups,self.betas):
            if list(g['betas'])!=b or g['lr']!=lr_used(self.lr_epoch):raise ValueError('fixed betas/LR')
    def state_dict(self):
        self.validate()
        return dict(schema='ch3-type1-state-v1',config=copy.deepcopy(self.config),config_sha=digest(self.config),updates=self.updates,
            completed_epoch=self.completed_epoch,lr_epoch=self.lr_epoch,continuing=self.continuing,
            current_lr=lr_used(self.lr_epoch),next_epoch_lr=lr_used(self.completed_epoch+1),fixed_betas=copy.deepcopy(self.betas),
            adjustment_position=self.config['adjustment'],groups=[dict(lr=g['lr'],betas=list(g['betas'])) for g in self.optimizer.param_groups])
    def load_state_dict(self,value,steps):
        if value.get('schema')!='ch3-type1-state-v1' or value.get('config')!=self.config or value.get('config_sha')!=digest(self.config) or value.get('updates')!=steps:raise ValueError('type1 checkpoint/profile')
        if value.get('fixed_betas')!=self.betas or steps!=value.get('completed_epoch',-1)*self.config['steps_per_epoch']:raise ValueError('type1 checkpoint committed epoch/steps/betas')
        self.updates=steps;self.completed_epoch=value['completed_epoch'];self.lr_epoch=value['lr_epoch'];self.continuing=value['continuing']
        if self.lr_epoch!=(self.completed_epoch+1 if self.continuing else self.completed_epoch) or value!=self.state_dict():raise ValueError('type1 checkpoint LR adjustment position')
def validate_probe_trace(config,traces,betas):
    validate_config(config)
    if len(traces)!=6:raise ValueError('six short update snapshots')
    for i,s in enumerate(traces,1):
        if s.get('config')!=config or s.get('config_sha')!=digest(config) or s.get('schema')!='ch3-type1-state-v1' or s.get('updates')!=i or s.get('completed_epoch')!=0 or s.get('lr_epoch')!=1 or s.get('continuing')is not True or s.get('current_lr')!=1e-4 or s.get('next_epoch_lr')!=1e-4 or s.get('fixed_betas')!=[betas] or s.get('groups')!=[dict(lr=1e-4,betas=betas)] or s.get('adjustment_position')!=config['adjustment']:raise ValueError('type1 exact scheduler/LR/betas; no complete epoch claimed')
