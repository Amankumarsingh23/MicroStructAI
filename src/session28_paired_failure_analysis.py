"""Session 28 standalone artificial paired comparisons and failure analysis."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path, PureWindowsPath
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

METHODS = ('dark_120', 'dark_165')
CONDITIONS = ('bright', 'nominal', 'dim')


def seal(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'protocol': payload, 'sha256': hashlib.sha256(canonical).hexdigest()}


def verify(sealed):
    if sealed != seal(sealed['protocol']):
        raise ValueError('Protocol seal mismatch')


def binary_mask(mask):
    if mask.ndim != 2 or mask.dtype.kind not in 'iu' or not mask.size or np.any(mask < 0) or np.any(mask > 9):
        raise ValueError('Original integer mask IDs 0..9 required')
    return (mask != 9).astype(np.uint8)


def preflight(rows, root):
    root = Path(root).resolve()
    checked, seen, sources = [], set(), {}
    def protected(path):
        return any(re.search(r'(^|[^a-z])(test|testing|holdout)([^a-z]|$)', s.lower()) for s in path.parts)
    if not rows or protected(root):
        raise ValueError('Development-only root and manifest required')
    for row in rows:
        if row['split'] not in ('train', 'val'):
            raise ValueError('Protected partition rejected before image access')
        if row['image_id'] in seen:
            raise ValueError('Duplicate image ID')
        seen.add(row['image_id'])
        sources.setdefault(row['source_id'], set()).add(row['split'])
        paths = []
        for key in ('image_rel', 'mask_rel'):
            rel = Path(row[key]); win = PureWindowsPath(row[key])
            if rel.is_absolute() or win.drive or '..' in rel.parts or '..' in win.parts or protected(rel):
                raise ValueError('Protected or escaping path')
            path = (root/rel).resolve()
            if not path.is_relative_to(root) or protected(path) or not path.is_file():
                raise ValueError('Invalid resolved development path')
            paths.append(path)
        checked.append((row, *paths))
    if any(len(splits)>1 for splits in sources.values()):
        raise ValueError('Cross-partition source lineage overlap')
    return checked


def generate(root, seed=28):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rr, cc = np.indices((64,64))
    rows = []
    for group in range(18):
        mask = np.full((64,64),9,np.uint8)
        center = rng.integers(19,44,2)
        radius = int(rng.integers(9,15))
        mask[(rr-center[0])**2+(cc-center[1])**2<=radius**2] = group%8
        mask[12:48,54] = 8
        noise = rng.normal(0,5,mask.shape)
        for condition, foreground, background, thin_intensity in (
                ('bright',145,225,165),('nominal',95,185,145),('dim',55,130,100)):
            gray = np.where(mask==9,background,foreground).astype(float)
            gray[mask==8] = thin_intensity
            gray = np.clip(np.rint(gray+noise),0,255).astype(np.uint8)
            name = f'source_{group:02d}_{condition}'
            Image.fromarray(gray).save(root/f'{name}.png')
            Image.fromarray(mask).save(root/f'{name}_mask.png')
            rows.append({'image_id':name,'source_id':f'artificial_source_{group:02d}',
                         'split':'train' if group<6 else 'val','condition':condition,
                         'image_rel':f'{name}.png','mask_rel':f'{name}_mask.png'})
    return rows


def metrics(original, prediction):
    truth = binary_mask(original).astype(bool)
    if prediction.shape != truth.shape or not np.isin(prediction,(0,1)).all():
        raise ValueError('Matching binary prediction required')
    pred = prediction.astype(bool)
    tp=int((truth&pred).sum()); fp=int((~truth&pred).sum()); fn=int((truth&~pred).sum())
    tn=int((~truth&~pred).sum()); union=tp+fp+fn
    boundaries = [m & ~ndi.binary_erosion(m,structure=np.ones((3,3)),border_value=0) for m in (truth,pred)]
    actual, predicted = boundaries
    na,npred=int(actual.sum()),int(predicted.sum())
    bp=float((predicted & (ndi.distance_transform_edt(~actual)<=1)).sum()/npred) if na and npred else 0.
    br=float((actual & (ndi.distance_transform_edt(~predicted)<=1)).sum()/na) if na and npred else 0.
    f1=2*bp*br/(bp+br) if bp+br else (0. if na or npred else None)
    failure = 'no_pixel_errors' if not fp+fn else ('oversegmentation' if fp>=2*fn else ('undersegmentation' if fn>=2*fp else 'mixed'))
    thin = original==8
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'iou':tp/union if union else None,
            'dice':2*tp/(2*tp+fp+fn) if union else None,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'boundary_f1':f1,'signed_area_error_pp':100*(fp-fn)/truth.size,
            'thin_recall':float(pred[thin].mean()) if thin.any() else None,'failure':failure}


def paired_sources(rows):
    grouped = {}
    for row in rows:
        key = (row['source_id'],row['condition'])
        methods = grouped.setdefault(key,{})
        if row['method'] in methods or row['iou'] is None:
            raise ValueError('Unique paired methods with defined IoU required')
        methods[row['method']] = row['iou']
    differences = {}
    for (source,condition),scores in grouped.items():
        if set(scores)!=set(METHODS):
            raise ValueError('Missing matched method')
        differences.setdefault(source,{})[condition] = scores[METHODS[0]]-scores[METHODS[1]]
    result=[]
    for source,conditions in sorted(differences.items()):
        if set(conditions)!=set(CONDITIONS):
            raise ValueError('All three correlated source views required')
        result.append({'source_id':source,'paired_iou_difference':float(np.mean(list(conditions.values())))})
    return result


def bootstrap(differences, repetitions=500, seed=28):
    values=np.asarray(differences,float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all() or type(repetitions) is not int or repetitions<1:
        raise ValueError('Finite source differences and positive repetitions required')
    rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),size=(repetitions,len(values)))].mean(axis=1)
    return draws


def sign_flip(differences):
    values=np.asarray(differences,float)
    if values.ndim!=1 or not 1<=len(values)<=20 or not np.isfinite(values).all():
        raise ValueError('One to twenty finite paired source differences required')
    observed=float(values.mean())
    means=np.array([np.mean(values*np.array(signs)) for signs in itertools.product((-1,1),repeat=len(values))])
    return {'observed_mean':observed,'configurations':len(means),
            'two_sided_p':float(np.mean(np.abs(means)>=abs(observed)-1e-12)),
            'assumptions':'Independent source differences with sign symmetry under an appropriate null; exploratory synthetic diagnostic'}


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(output,seed=28,repetitions=500):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    if type(seed) is not int or seed<0 or type(repetitions) is not int or repetitions<1:
        raise ValueError('Nonnegative seed and positive repetitions required')
    protocol={'seed':seed,'repetitions':repetitions,'sources':18,'training_sources':6,'validation_sources':12,
              'views':list(CONDITIONS),'methods':list(METHODS),'comparison':'120 minus 165, average views within source then equal-source mean',
              'boundary_tolerance_px':1,'failure_rule':'FP>=2FN over; FN>=2FP under; otherwise mixed; zero errors separate',
              'thin_code':8,'official_test_policy':'never_access'}
    sealed=seal(protocol)
    output.mkdir(parents=True,exist_ok=False)
    protocol_path=output/'session28_frozen_protocol.json'
    protocol_path.write_text(json.dumps(sealed,indent=2)+'\n',encoding='utf-8')
    root=output/'artificial_development'
    manifest=generate(root,seed)
    write_csv(output/'session28_source_manifest.csv',manifest)
    verify(json.loads(protocol_path.read_text()))
    checked=preflight(manifest,root)
    measured=[]
    input_hashes={}
    for row,image_path,mask_path in checked:
        with Image.open(image_path) as opened:
            image=np.array(opened.convert('L'))
        with Image.open(mask_path) as opened:
            original=np.array(opened)
        for path in (image_path,mask_path):
            input_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for method,threshold in zip(METHODS,(120,165)):
            measured.append({**{k:row[k] for k in ('image_id','source_id','split','condition')},
                             'method':method,**metrics(original,(image<threshold).astype(np.uint8))})
    validation=[r for r in measured if r['split']=='val']
    paired=paired_sources(validation)
    differences=[r['paired_iou_difference'] for r in paired]
    draws=bootstrap(differences,repetitions,seed)
    conditions=[]
    for condition in CONDITIONS:
        for method in METHODS:
            selected=[r for r in validation if r['condition']==condition and r['method']==method]
            conditions.append({'condition':condition,'method':method,'images':len(selected),
                               **{key:float(np.mean([r[key] for r in selected])) for key in
                                  ('iou','boundary_f1','thin_recall','signed_area_error_pp')},
                               **{name:sum(r['failure']==name for r in selected) for name in
                                  ('oversegmentation','undersegmentation','mixed','no_pixel_errors')}})
    report={'implementation':'Reconstructed from Session 28 handout; supplied package unavailable locally',
            'protocol_sha256':sealed['sha256'],'manifest_sha256':hashlib.sha256((output/'session28_source_manifest.csv').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'input_hashes':input_hashes,
            'source_groups':18,'validation_source_groups':12,'images':54,'validation_images':36,
            'mean_source_paired_difference':float(np.mean(differences)),
            'bootstrap_percentile_95':np.quantile(draws,[.025,.975]).tolist(),
            'sign_flip':sign_flip(differences),'condition_diagnostics':conditions,
            'scientific_status':'NO-GO for real-material claims; generated fixtures and unverifiable physical source independence',
            'limits':['Three views per source are dependent; resample source means',
                      'Equal lighting weights are a designed estimand, not an observed deployment distribution',
                      'Synthetic ID 8 marks a diagnostic line, not a verified material feature',
                      'Bootstrap and sign flipping cannot establish physical provenance or causality']}
    write_csv(output/'session28_pair_metrics.csv',measured)
    write_csv(output/'session28_source_comparison.csv',paired)
    write_csv(output/'session28_condition_diagnostics.csv',conditions)
    write_csv(output/'session28_bootstrap_draws.csv',[{'repeat':i,'mean_difference':float(d)} for i,d in enumerate(draws)])
    (output/'session28_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    try:
        x=np.arange(3)
        for method,offset in zip(METHODS,(-.18,.18)):
            rows=[r for r in conditions if r['method']==method]
            axes[0,0].bar(x+offset,[r['iou'] for r in rows],width=.36,label=method)
            axes[1,0].bar(x+offset,[r['thin_recall'] for r in rows],width=.36,label=method)
        axes[0,0].set(title='Validation IoU by lighting'); axes[1,0].set(title='Marked thin-line recall')
        for ax in (axes[0,0],axes[1,0]):
            ax.set_xticks(x,CONDITIONS); ax.legend()
        axes[0,1].bar(range(len(differences)),differences)
        axes[0,1].axhline(0,color='black',linewidth=.8)
        axes[0,1].set(title='Source mean IoU difference',xlabel='Validation source',ylabel='120 minus 165')
        axes[1,1].hist(draws,bins=20)
        axes[1,1].axvline(float(np.mean(differences)),color='red')
        axes[1,1].set(title='Descriptive whole-source bootstrap',xlabel='Mean paired difference')
        fig.suptitle('Session 28 synthetic paired comparison and failure analysis')
        fig.savefig(output/'session28_failure_analysis.png',dpi=135)
    finally:
        plt.close(fig)
    print('Mean paired source difference',report['mean_source_paired_difference'])
    print('Bootstrap span',report['bootstrap_percentile_95'],'sign flip',report['sign_flip']['two_sided_p'])
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/session28_v1'))
    parser.add_argument('--seed',type=int,default=28)
    parser.add_argument('--repetitions',type=int,default=500)
    args=parser.parse_args()
    run(args.output_dir,args.seed,args.repetitions)
