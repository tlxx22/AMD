"""Fresh empty CPU process using the formal worker's shared thread setup."""
import json
import sys
from pathlib import Path
from ch4.resources import configure_threads

if __name__=='__main__':
    plan=json.loads(Path(sys.argv[1]).read_text())
    task=plan['tasks'][0]
    result=configure_threads(task,plan['resource_plan'])
    import ch4
    assert 'ch4.worker' not in sys.modules and 'ch4.model' not in sys.modules
    assert Path(ch4.__file__).resolve()==Path(__file__).resolve().parents[1]/'ch4/__init__.py'
    result.update(phase=sys.argv[2],models_imported=False,package_file=ch4.__file__)
    print(json.dumps(result))
