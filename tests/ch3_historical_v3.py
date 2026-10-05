"""Version-pinned v3 metadata fixture in a new worktree, without running old code."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
from utils.ch3_native_recovery_records import sha

CONFIG_SHAS={'MS':'6bdfc55357f8fad0d102efec83d453ef77418cc567de40158ff3ed974124aa09','M':'f627260225b971d13d1e881d8123a0426c63304e64d8e8afa56fa0ab73a3cdb2'}
def configurations(test_class,tasks,chain):
    refs={}
    for stage,expected in CONFIG_SHAS.items():
        path=tasks.file(stage)
        if sha(path)!=expected:raise ValueError('frozen historical v3 configuration changed')
        c=json.loads(path.read_text());refs[stage]=c['baseline_unified']['direct_parent_ref']
    # The source paths inside frozen configs are the original W, with fixed SHA.
    # Only the test fixture aligns those refs; it does not change any runtime module.
    source_package=tasks.PACKAGE
    temp=tempfile.TemporaryDirectory(dir=source_package.with_name('baseline-type1-followup-v1'),prefix='historical-v3-fixture-')
    test_class.addClassCleanup(temp.cleanup);package=Path(temp.name)/tasks.PROTOCOL;package.mkdir()
    for stage in ('MS','M'):
        name=stage.lower()+'-plan.json';(package/name).write_bytes((source_package/name).read_bytes())
    for name in ('protected-before.json','ms-profile-diff.json','m-profile-diff.json'):
        (package/name).write_bytes((source_package/name).read_bytes())
    template=json.loads((source_package/'start-approval.template.json').read_text())
    for stage in ('MS','M'):
        template['config_refs'][stage]['path']=str(tasks.file(stage))
        template['plan_refs'][stage]['path']=str(package/(stage.lower()+'-plan.json'))
    (package/'start-approval.template.json').write_text(json.dumps(template))
    fixture=patch.multiple(tasks,DIRECT_PARENT_REFS=refs,PACKAGE=package,RESULT=package/'unstarted-result')
    fixture.start();test_class.addClassCleanup(fixture.stop)
    for name,value in (('LOG',package/'unified-launcher.log'),('CONTROL',package/'unstarted-result/queue/controller')):
        context=patch.object(chain,name,value);context.start();test_class.addClassCleanup(context.stop)
    return chain.configs()
