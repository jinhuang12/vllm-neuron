import importlib.metadata,importlib.util,json,os,sys
from pathlib import Path
from benchmarks.glm53_moe.full_model.ordered_trace import source_tree
source=Path.cwd()
expected='c366203b7643192407527bec6ff5866cfdf8773ab8bed80a101638aa9abf6c93'
files,actual=source_tree(source)
assert actual==expected,(actual,expected)
assert not list(Path('/dev').glob('neuron*'))
contract=json.loads((source/'benchmarks/glm53_moe/full_model/preservation/ordered-trace-contract.json').read_text())
versions={name:importlib.metadata.version(name) for name in contract['runtime_versions']}
assert versions==contract['runtime_versions'],versions
origin=Path(importlib.util.find_spec('vllm_neuron').origin).resolve()
assert source in origin.parents,origin
mounts=['/home/ubuntu/glm53-moe-fullmodel-20260917/pr-snapshot-source', '/home/ubuntu/glm53-moe-fullmodel-20260917/deps', '/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0', '/opt/aws/neuron']
readonly={p:bool(os.statvfs(p).f_flag & os.ST_RDONLY) for p in mounts}
assert all(readonly.values()),readonly
print(json.dumps({'status':'PASS','stage':'CPU-only SDK preflight','production_tree_sha256':actual,'production_file_count':len(files),'build_metadata_sha256':{k:files[k] for k in ('pyproject.toml','setup.py')},'runtime_versions':versions,'source_module_origin':str(origin),'devices_exposed':[],'read_only_mounts':readonly,'test_count_expected':63}),flush=True)
os.execv(sys.executable,[sys.executable,'-m','pytest','-p','no:cacheprovider','-q',*['test/vllm_neuron/model/glm5_next/test_prepared_weight_release.py', 'test/vllm_neuron/model/glm5_next/test_weights_free_load_path.py', 'test/vllm_neuron/model/glm5_next/test_load_weights.py::test_blocked_the_shared_expert_prep_completes_a_load_and_the_publish_ran', 'test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_forward.py::test_tiny_routed_experts_forward_matches_the_reference', 'test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_forward.py::test_tiny_moe_block_forward_matches_the_reference', 'test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_one_graph.py', 'experiments/glm53_moe_nki/test_model_integration.py', 'experiments/glm53_moe_nki/test_decode_rows.py'],'--disable-warnings'])
