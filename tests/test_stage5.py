"""Stage 5 data/regression checks, independent of publication claims."""
from pathlib import Path
import json,glob
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/'E5F'

def read(p):return json.loads((R/p).read_text())

def test_exact_seed4_initial_information_and_nested_rates():
    a=read('exact16/bethe_seed4_p80.json')
    assert abs(a['information'][0][0])<1e-10
    assert abs(a['information'][0][1])<1e-10
    assert np.min(a['rate'])>=-1e-9
    assert a['coarse_first_passage']['hitting_fraction']>=a['fine_first_passage']['hitting_fraction']

def test_exact_candidate_is_nontrivial():
    a=read('exact16/bethe_seed4_p80.json')
    assert a['accepted_candidate_tree']
    assert a['heldout_ARI'][1]>.9
    assert a['within_minus_sibling']>.2

def test_failures_recorded_without_optimizing():
    a=read('exact16/bethe_seed6_p96.json')
    assert not a['accepted_candidate_tree']
    assert min(a['prior_leaf_masses'])<.03
    b=read('exact16/bethe_seed12_p96.json')
    assert not b['accepted_candidate_tree']

def test_nulls_fail_structural_criteria():
    for f in glob.glob(str(R/'exact16'/'two_state*.json'))+glob.glob(str(R/'exact16'/'product*.json')):
        assert not json.loads(Path(f).read_text())['accepted_candidate_tree']

def test_larger_ESS_restriction_and_training_splits():
    for f in glob.glob(str(R/'importance'/'*.json')):
        d=json.loads(Path(f).read_text());assert d['training_samples_per_independent_run']==800
        assert d['posterior_reference_particles']==800
        assert d['last_all_fields_ESS_qualified_gamma']<=.06
        assert d['independent_prior_runs']==2
        assert min(d['prior_roundtrips_min'])>=5

def test_large_nulls_do_not_pass():
    for f in glob.glob(str(R/'nulls'/'*.json')):
        assert not json.loads(Path(f).read_text())['structural_candidate_pass']
