"""Exact B proof checks against real source, Git parent and both CPU parsers."""
import ast,contextlib,copy,hashlib,io,json,os,runpy,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from utils import ch3_third_round_after_selection as b,ch3_type1_chain as q
from utils.ch3_native_recovery_records import bound,ref
from utils.ch3_contract import profile
b.activate()


def fixture():
    root=Path(tempfile.mkdtemp(prefix='synthetic-AST-',dir=b.PACKAGE/'fixtures'))
    (root/'SYNTHETIC_ONLY.json').write_text('{"GPU":0,"permission":false,"training":0}\n')
    return root


def nodes(raw):return {n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


class BASTProof(unittest.TestCase):
    def test_all66_real_source_and_Git_parent_match(self):
        proof=bound(ref(b.PACKAGE/'producer-delta-proof.v11.json'));old=bound(b.SUPERSEDED_STARTUP_PROOF_REF);count=0
        self.assertEqual(proof['AST_algorithm'],b.AST_PROOF_ALGORITHM)
        self.assertEqual({f:set(v)for f,v in proof['protected_AST'].items()},{f:set(v)for f,v in old['protected_AST'].items()})
        for name,registered in proof['protected_AST'].items():
            now=nodes((b.ROOT/name).read_bytes());parent=nodes(subprocess.check_output(['git','show',proof['parent_commit']+':'+name],cwd=b.ROOT))
            for symbol,want in registered.items():
                self.assertEqual(b.protected_ast_fingerprint(now[symbol]),want,(name,symbol));self.assertEqual(b.protected_ast_fingerprint(parent[symbol]),proof['parent_AST'][name][symbol],(name,symbol));count+=1
        self.assertEqual(count,66);self.assertEqual({k:set(v)for k,v in proof['approved_AST_changes'].items()},b.APPROVED_AST_CHANGES)

    def test_true_production_verifiers_return_in_real_repository(self):
        self.assertEqual(Path.cwd(),b.ROOT)
        self.assertEqual(b.verify_production_inheritance()['protected_function_count'],66)
        self.assertTrue(b.verify_source())

    def test_both_Python_versions_run_true_verifiers_and_match66(self):
        root=fixture()
        code="""import ast,json;from utils import ch3_third_round_after_selection as b;b.activate();p=b.verify_production_inheritance();s=b.verify_source();rows={f:{n.name:b.protected_ast_fingerprint(n)for n in ast.parse((b.ROOT/f).read_text()).body if getattr(n,'name',None)in v}for f,v in p['protected_AST'].items()};assert rows==p['protected_AST'];print(json.dumps(dict(rows=rows,source=bool(s),production=True)))"""
        outputs=[]
        for label,python in [('formal',sys.executable),('server-default','/public/home/yueweiting/miniconda/bin/python')]:
            result=subprocess.run([python,'-B','-c',code],cwd=b.ROOT,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=60)
            (root/(label+'.stdout')).write_text(result.stdout);(root/(label+'.stderr')).write_text(result.stderr)
            self.assertEqual(result.returncode,0,result.stderr);outputs.append(json.loads(result.stdout))
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(sum(map(len,outputs[0]['rows'].values())),66)

    def test_semantic_empty_fields_and_nonempty_type_parameters_are_protected(self):
        original=ast.parse('def f(x=1):\n    return x\n').body[0];base=b.protected_ast_fingerprint(original)
        for change in ('decorator','default','body','type_params'):
            node=copy.deepcopy(original)
            if change=='decorator':node.decorator_list.append(ast.Name(id='decorator',ctx=ast.Load()))
            elif change=='default':node.args.defaults[0].value=2
            elif change=='body':node.body.append(ast.Pass())
            else:
                if 'type_params'not in node._fields:node._fields=(*node._fields,'type_params')
                node.type_params=[ast.Name(id='T',ctx=ast.Load())]
            self.assertNotEqual(b.protected_ast_fingerprint(node),base,change)
        compat=copy.deepcopy(original)
        if 'type_params'not in compat._fields:compat._fields=(*compat._fields,'type_params')
        compat.type_params=[];self.assertEqual(b.protected_ast_fingerprint(compat),base)

    def test_proof_fingerprint_tamper_rejected_by_real_verifier(self):
        root=fixture();proof=bound(ref(b.PACKAGE/'producer-delta-proof.v11.json'));file=next(k for k,v in proof['protected_AST'].items()if v);symbol=next(iter(proof['protected_AST'][file]));proof['protected_AST'][file][symbol]='0'*64
        (root/'producer-delta-proof.v11.json').write_text(json.dumps(proof)+'\n')
        with patch.object(b,'PACKAGE',root):self.assertRaises(ValueError,b.verify_production_inheritance)

    def test_dropping_one_protected_symbol_rejected_by_real_verifier(self):
        root=fixture();proof=bound(ref(b.PACKAGE/'producer-delta-proof.v11.json'));file=next(k for k,v in proof['protected_AST'].items()if v);proof['protected_AST'][file].pop(next(iter(proof['protected_AST'][file])))
        (root/'producer-delta-proof.v11.json').write_text(json.dumps(proof)+'\n')
        with patch.object(b,'PACKAGE',root):self.assertRaises(ValueError,b.verify_production_inheritance)

    def test_protected_function_tamper_hits_real_AST_gate_after_file_binding(self):
        root=fixture();source=root/'source';source.mkdir();proof=bound(ref(b.PACKAGE/'producer-delta-proof.v11.json'))
        for name in set(proof['protected_exact'])|set(proof['after_code']):
            path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((b.ROOT/name).read_bytes())
        name='utils/ch3_event_resources.py';path=source/name;tree=ast.parse(path.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='enabled');node.body.append(ast.Pass());path.write_text(ast.unparse(tree)+'\n')
        # Intentionally authorize this synthetic file's changed bytes so the
        # independent protected_AST gate must reject its semantic change.
        proof['after_code'][name]=hashlib.sha256(path.read_bytes()).hexdigest();(root/'producer-delta-proof.v11.json').write_text(json.dumps(proof)+'\n')
        real_query=subprocess.check_output
        def parent_query(args,**kwargs):
            if args[:2]==['git','show']:kwargs['cwd']=Path(__file__).resolve().parents[1]
            return real_query(args,**kwargs)
        with patch.object(b,'PACKAGE',root),patch.object(b,'ROOT',source),patch.object(subprocess,'check_output',side_effect=parent_query):self.assertRaisesRegex(ValueError,'protected mathematics/runtime function changed: '+name,b.verify_production_inheritance)

    def test_prepare_script_complete_source_path_without_GPU_or_permission(self):
        script=b.PACKAGE/'prepare-permission.v7.py';real_output=subprocess.check_output;real_run=subprocess.run
        def query(args,**kwargs):
            if 'nvidia-smi'in str(args)or 'nvml'in str(args).lower():raise AssertionError('no GPU source-check')
            return real_output(args,**kwargs)
        def run(args,**kwargs):
            if 'nvidia-smi'in str(args)or 'nvml'in str(args).lower():raise AssertionError('no GPU source-check')
            return real_run(args,**kwargs)
        output=io.StringIO()
        with patch.object(sys,'argv',[str(script),'source-check']),patch.object(subprocess,'check_output',side_effect=query),patch.object(subprocess,'run',side_effect=run),contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit)as done:runpy.run_path(str(script),run_name='__main__')
        self.assertEqual(done.exception.code,0);value=json.loads(output.getvalue());self.assertEqual(value['source_check'],'Passed');self.assertFalse(value['permission_generated']);self.assertFalse(value['READY_TO_ARM_HANDOFF']);self.assertEqual(value['GPU_queries'],0)
        self.assertFalse((b.PACKAGE/'start-review.json').exists());self.assertFalse(b.RESULT.exists())

    def test_future_template_uses_new_proof_and_main_rejects_unclosed_candidate(self):
        review=bound(ref(b.PACKAGE/'start-approval.template.v7.json'));self.assertEqual(review,q.start_template());self.assertFalse(review['execution_permitted'])
        self.assertEqual(review['upstream_anchors']['producer_delta_ref'],ref(b.PACKAGE/'producer-delta-proof.v11.json'));self.assertNotIn(str(b.STARTUP_PACKAGE/'start-review.json'),(b.PACKAGE/'operations.sh').read_text())
        script=b.PACKAGE/'prepare-permission.v7.py';result=subprocess.run([sys.executable,'-B',str(script),'main','--reviewed-closure','8de112a2f1ff2117e94c22e5119d86ccaf0bdec7'],cwd=b.ROOT,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0);self.assertFalse((b.PACKAGE/'start-review.json').exists())

    def test_same_attempt_science_Weather_and_all_historical_SHA(self):
        self.assertEqual(b.ATTEMPT,'THIRD-PatchTST-dm4-Urban6-four-H-reviewed-q-r1');self.assertEqual(b.CURRENT_STAGE_RUNS,dict(URBAN_SUBSET=168,EPF_ALL=35,M_ALL=168));self.assertEqual(b.selected_main()['retained_cells'],351)
        c=q.configs()['M_ALL'];parent=bound(b.contract()['parent_M_ALL_ref'])
        for t in c['tasks']:
            if t['model']=='PatchTST'and t['dataset']=='Weather':self.assertEqual(profile(c,t),profile(parent,t));self.assertEqual(profile(c,t)['structure']['d_model'],128)
        v=b.verify_stopped_B();inventory=bound(v['retained_artifact_inventory_ref'])
        for row in inventory['files']:
            if row['sha256']is not None:self.assertEqual(hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),row['sha256'],row['path'])
            else:self.assertEqual(Path(row['path']).stat().st_size,row['size'])
        self.assertEqual(b.historical_startup_failure()['formal_completed'],0)

    def test_generator_and_verifier_share_actual_fingerprint_function(self):
        path=b.PACKAGE/'generate-producer-proof-v11.py';tree=ast.parse(path.read_bytes())
        calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='protected_ast_fingerprint']
        self.assertEqual(len(calls),2);proof=bound(ref(b.PACKAGE/'producer-delta-proof.v11.json'));self.assertEqual(proof['generator_ref'],ref(path));self.assertEqual(proof['AST_algorithm'],b.AST_PROOF_ALGORITHM);self.assertEqual(set(proof['approved_AST_changes']['utils/ch3_third_round_after_selection.py']),b.APPROVED_AST_CHANGES['utils/ch3_third_round_after_selection.py'])


if __name__=='__main__':unittest.main()
