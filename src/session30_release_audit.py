"""Session 30 independent read-only audit of saved Session 29 records."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def safe_path(root,relative):
    root=Path(root).resolve(); rel=Path(relative); win=PureWindowsPath(relative)
    def protected(path):
        return any(re.search(r'(^|[^a-z])(test|testing|holdout)([^a-z]|$)',part.lower()) for part in path.parts)
    if (rel.is_absolute() or win.drive or '..' in rel.parts or '..' in win.parts
            or protected(root) or protected(rel)):
        raise ValueError('Protected or escaping audit path')
    target=(root/rel).resolve()
    if not target.is_relative_to(root) or protected(target) or not target.is_file():
        raise ValueError('Invalid resolved audit path')
    return target


def read_csv(root,name):
    with safe_path(root,name).open(newline='',encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def indexed(rows,key):
    if len({r[key] for r in rows})!=len(rows):
        raise ValueError(f'Duplicate {key}')
    return {r[key]:r for r in rows}


def equal(actual,expected,label):
    if actual is None or expected is None:
        if actual != expected:
            raise ValueError(f'Mismatch for {label}')
    elif not np.isfinite(float(actual)) or not np.isclose(float(actual),float(expected),atol=1e-9,rtol=1e-9):
        raise ValueError(f'Mismatch for {label}')


def audit(session29_dir):
    root=Path(session29_dir).resolve()
    report=json.loads(safe_path(root,'session29_report.json').read_text())
    frozen=json.loads(safe_path(root,'session29_frozen_protocol.json').read_text())
    if frozen['sha256']!=seal(frozen['protocol']) or report['protocol_sha256']!=frozen['sha256']:
        raise ValueError('Protocol seal mismatch')
    protocol=frozen['protocol']
    for key,value in [('specimens',36),('images',108)]:
        equal(report[key],value,'report '+key)
    if protocol['specimens']!=36 or protocol['fields_each']!=3 or protocol['role_counts']!=[18,9,9]:
        raise ValueError('Unsupported cohort specification')
    if report['readiness']['scientific_status']!='NO-GO for real-material claims':
        raise ValueError('Unsupported scientific readiness claim')
    if protocol['official_test_policy']!='never_access' or protocol['primary_segmentation']!='dark_120':
        raise ValueError('Unsupported protocol')
    if report['source_sha256']!=digest(Path(__file__).with_name('session29_end_to_end_validation.py')):
        raise ValueError('Session 29 source checksum mismatch')
    manifest=read_csv(root,'session29_source_manifest.csv')
    properties=read_csv(root,'session29_properties.csv')
    fields=read_csv(root,'session29_field_features.csv')
    specimens=read_csv(root,'session29_specimen_features.csv')
    scores=read_csv(root,'session29_segmentation_metrics.csv')
    predictions=read_csv(root,'session29_property_predictions.csv')
    registry=json.loads(safe_path(root,'session29_input_checksums.json').read_text())
    normalized_registry = {PureWindowsPath(k).as_posix(): v for k,v in registry.items()}
    if len(normalized_registry) != len(registry):
        raise ValueError('Duplicate normalized checksum paths')
    registry = normalized_registry
    if [len(x) for x in (manifest,properties,fields,specimens,scores,predictions)]!=[108,36,108,36,216,9] or len(registry)!=540:
        raise ValueError('Unexpected ledger counts')
    m=indexed(manifest,'field_id'); f=indexed(fields,'field_id')
    p=indexed(properties,'specimen_id'); s=indexed(specimens,'specimen_id'); pred=indexed(predictions,'specimen_id')
    if set(m)!=set(f) or set(p)!=set(s) or {r['specimen_id'] for r in manifest}!=set(p):
        raise ValueError('Broken field/specimen/property joins')
    roles={key:{} for key in ('specimen_id','source_id','texture_id','template_id')}
    input_paths={}; output_paths={}; expected_registry={}
    # Resolve every referenced path and validate roles before any image content reads.
    for row in manifest:
        role=row['role']; sid=row['specimen_id']
        if role not in ('fit','calibration','evaluation') or row['split']!=('train' if role=='fit' else 'val'):
            raise ValueError('Protected partition or invalid role')
        if row['schema']!='original10' or p[sid]['role']!=role or p[sid]['batch_id']!=row['batch_id']:
            raise ValueError('Inconsistent specimen metadata')
        for key,mapping in roles.items():
            if not row[key]:
                raise ValueError('Missing lineage')
            mapping.setdefault(row[key],set()).add(role)
        for key,kind in (('image_rel','image'),('mask_rel','reference_mask'),('roi_rel','roi')):
            relative='artificial_development/'+row[key]
            path=safe_path(root,relative)
            if path in input_paths:
                raise ValueError('Reused input path')
            input_paths[path]=(row,kind)
            expected_registry[row[key]]=(path,row['field_id'],kind)
        field=f[row['field_id']]
        if any(field[key]!=row[key] for key in ('specimen_id','batch_id','role')):
            raise ValueError('Field provenance mismatch')
        for method in ('dark_120','dark_165'):
            relative=f"segmentation_masks/{row['field_id']}_{method}.png"
            path=safe_path(root,relative)
            output_paths[(row['field_id'],method)]=path
            expected_registry[relative]=(path,row['field_id'],method)
        if PureWindowsPath(field['prediction_rel']).as_posix()!=f"segmentation_masks/{row['field_id']}_dark_120.png":
            raise ValueError('Broken predicted-mask trace')
    if any(len(values)>1 for mapping in roles.values() for values in mapping.values()):
        raise ValueError('Cross-role source or specimen lineage')
    if set(registry)!=set(expected_registry):
        raise ValueError('Checksum registry links differ from manifest')
    for name,(path,fid,kind) in expected_registry.items():
        record=registry[name]
        if record['field_id']!=fid or record['kind']!=kind or record['sha256']!=digest(path):
            raise ValueError('Input/output checksum mismatch')
    if report['manifest_sha256']!=digest(root/'session29_source_manifest.csv'):
        raise ValueError('Manifest checksum mismatch')
    score_index={}
    for row in scores:
        key=(row['field_id'],row['method'])
        if key in score_index or key not in output_paths:
            raise ValueError('Duplicate or unknown segmentation record')
        score_index[key]=row
    if set(score_index)!=set(output_paths):
        raise ValueError('Missing segmentation record')
    totals={method:{'tp':0,'fp':0,'fn':0} for method in ('dark_120','dark_165')}
    observed_pixels=set()
    for row in manifest:
        fid=row['field_id']
        with Image.open(root/'artificial_development'/row['image_rel']) as img:
            gray=np.array(img.convert('L'))
        with Image.open(root/'artificial_development'/row['mask_rel']) as img:
            original=np.array(img)
        with Image.open(root/'artificial_development'/row['roi_rel']) as img:
            raw_roi=np.array(img)
        if (original.ndim!=2 or original.dtype.kind not in 'iu' or not np.isin(original,range(10)).all()
                or gray.shape!=original.shape or raw_roi.shape!=original.shape
                or not np.isin(raw_roi,(0,255)).all() or not raw_roi.any()):
            raise ValueError('Invalid image/mask/ROI')
        pix=hashlib.sha256(str(gray.shape).encode()+gray.tobytes()).hexdigest()
        if pix in observed_pixels:
            raise ValueError('Duplicate decoded images')
        observed_pixels.add(pix)
        roi=raw_roi.astype(bool); truth=original!=9
        for method,threshold in (('dark_120',120),('dark_165',165)):
            with Image.open(output_paths[(fid,method)]) as img:
                predicted=np.array(img)
            if predicted.shape!=gray.shape or not np.array_equal(predicted,(gray<threshold).astype(np.uint8)):
                raise ValueError('Prediction mask violates fixed method')
            score=score_index[(fid,method)]
            if score['specimen_id']!=row['specimen_id'] or score['role']!=row['role']:
                raise ValueError('Segmentation identity mismatch')
            t=truth[roi]; q=predicted[roi].astype(bool)
            counts={'tp':int((t&q).sum()),'fp':int((~t&q).sum()),'fn':int((t&~q).sum())}
            for key,value in counts.items():
                equal(score[key],value,'segmentation '+key)
                if row['role']=='evaluation':
                    totals[method][key]+=value
            union=sum(counts.values())
            equal(None if score['iou']=='' else score['iou'],counts['tp']/union if union else None,'IoU')
            equal(score['signed_area_error_pp'],100*(counts['fp']-counts['fn'])/roi.sum(),'area error')
            if method=='dark_120':
                field=f[fid]
                _,components=ndi.label(predicted.astype(bool)&roi,structure=np.ones((3,3)))
                for key,value in [('valid_pixels',int(roi.sum())),('foreground_pixels',int(predicted[roi].sum())),('components',components)]:
                    equal(field[key],value,'field '+key)
    for sid,row in s.items():
        group=[r for r in fields if r['specimen_id']==sid]
        if len(group)!=3 or row['role']!=p[sid]['role'] or row['batch_id']!=p[sid]['batch_id']:
            raise ValueError('Specimen grouping mismatch')
        valid=sum(int(r['valid_pixels']) for r in group)
        positive=sum(int(r['foreground_pixels']) for r in group)
        for key,value in [('n_fields',3),('valid_pixels',valid),('foreground_pixels',positive),
                          ('area_fraction',positive/valid),('component_density_px2',sum(int(r['components']) for r in group)/valid),
                          ('target_au',p[sid]['target_au'])]:
            equal(row[key],value,'specimen '+key)
        if row['target_provenance']!=p[sid]['target_provenance']:
            raise ValueError('Target provenance mismatch')
    groups={role:[r for r in specimens if r['role']==role] for role in ('fit','calibration','evaluation')}
    if [len(groups[k]) for k in groups]!=[18,9,9] or report['role_counts']!={k:len(v) for k,v in groups.items()}:
        raise ValueError('Role counts mismatch')
    model=report['model']; fit=groups['fit']
    x=np.array([float(r['area_fraction']) for r in fit]); y=np.array([float(r['target_au']) for r in fit])
    mu=float(x.mean()); sd=float(x.std()); sd=sd if sd>1e-12 else 1.
    weight=float(np.linalg.lstsq(((x-mu)/sd)[:,None],y-y.mean(),rcond=None)[0][0])
    for key,value in [('mean',mu),('std',sd),('target_mean',y.mean()),('weight',weight)]:
        equal(model[key],value,'model '+key)
    if model['fitting_specimens']!=[r['specimen_id'] for r in fit]:
        raise ValueError('Fitting identity mismatch')
    def predict(rows):
        return float(y.mean())+weight*(np.array([float(r['area_fraction']) for r in rows])-mu)/sd
    cal=groups['calibration']; alpha=protocol['alpha']
    if not 0<alpha<1:
        raise ValueError('Invalid alpha')
    rank=math.ceil((len(cal)+1)*(1-alpha))
    if rank>len(cal):
        raise ValueError('Unavailable finite calibration rank')
    radius=float(np.sort(abs(np.array([float(r['target_au']) for r in cal])-predict(cal)))[rank-1])
    equal(report['calibration']['rank'],rank,'calibration rank')
    equal(report['calibration']['radius_au'],radius,'calibration radius')
    equal(report['calibration']['alpha'],alpha,'calibration alpha')
    if report['calibration']['calibration_specimens']!=[r['specimen_id'] for r in cal]:
        raise ValueError('Calibration identity mismatch')
    evaluation=groups['evaluation']; eval_ids={r['specimen_id'] for r in evaluation}
    trace=indexed(report['trace'],'specimen_id')
    if set(pred)!=eval_ids or set(trace)!=eval_ids:
        raise ValueError('Broken evaluation trace links')
    errors=[]; covered=[]
    for row,value in zip(evaluation,predict(evaluation)):
        sid=row['specimen_id']; saved=pred[sid]; actual=float(row['target_au'])
        if saved['batch_id']!=row['batch_id']:
            raise ValueError('Prediction batch mismatch')
        for key,target in [('actual_au',actual),('prediction_au',value),('lower_au',value-radius),('upper_au',value+radius),('absolute_error_au',abs(actual-value))]:
            equal(saved[key],target,'property '+key)
        contains=bool(value-radius<=actual<=value+radius)
        if saved['covered']!=str(contains):
            raise ValueError('Incorrect covered flag')
        errors.append(abs(actual-value)); covered.append(contains)
        link=trace[sid]
        expected_fields={r['field_id'] for r in fields if r['specimen_id']==sid}
        if len(link['field_ids'])!=3 or set(link['field_ids'])!=expected_fields or link['protocol_sha256']!=frozen['sha256']:
            raise ValueError('Broken field/protocol trace')
        for key,expected in [('model','session29_report.json#model'),('manifest','session29_source_manifest.csv'),('checksums','session29_input_checksums.json')]:
            if link[key]!=expected:
                raise ValueError('Broken artifact trace')
        equal(link['pooled_fraction'],row['area_fraction'],'trace fraction')
        equal(link['prediction_au'],value,'trace prediction')
    mae=float(np.mean(errors)); coverage=float(np.mean(covered))
    equal(report['evaluation_mae_au'],mae,'reported MAE')
    equal(report['coverage'],coverage,'reported coverage')
    equal(report['covered'],sum(covered),'covered count'); equal(report['evaluation_n'],9,'evaluation n')
    segmentation={}
    if len(report['segmentation'])!=2:
        raise ValueError('Expected two segmentation summaries')
    for saved in report['segmentation']:
        method=saved['method']; counts=totals[method]
        iou=counts['tp']/sum(counts.values())
        equal(saved['pooled_iou'],iou,'reported segmentation IoU')
        equal(saved['evaluation_images'],27,'segmentation images')
        segmentation[method]=iou
    if set(segmentation)!=set(totals):
        raise ValueError('Missing segmentation summaries')
    return {'verification':'passed','specimens':36,'fields':108,'role_counts':{k:len(v) for k,v in groups.items()},
            'checksum_entries_verified':540,'input_checksums':324,'prediction_checksums':216,
            'segmentation_records':216,'evaluation_predictions':9,'recomputed_mae_au':mae,
            'recomputed_coverage':coverage,'recomputed_radius_au':radius,'segmentation_iou':segmentation,
            'protocol_sha256':frozen['sha256'],'scientific_status':'NO-GO for real-material claims',
            'limits':['Software consistency and hashes do not validate real specimen origins or material measurements',
                      'This independent auditor was reconstructed from the written Session 30 requirements']}


def verify_release(output):
    output=Path(output)
    release=json.loads((output/'session30_release_manifest.json').read_text())
    payload={k:v for k,v in release.items() if k!='seal_sha256'}
    if release['seal_sha256']!=seal(payload):
        raise ValueError('Release seal mismatch')
    expected={'session30_final_audit.json','session30_claims_matrix.csv','session30_project_summary.md','session30_readiness.png'}
    if set(release['artifact_sha256'])!=expected:
        raise ValueError('Incomplete release artifact manifest')
    for name,value in release['artifact_sha256'].items():
        if digest(safe_path(output,name))!=value:
            raise ValueError('Release artifact checksum mismatch')
    return True


def run(session29_dir,output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    source=Path(session29_dir).resolve()
    if output.resolve().is_relative_to(source):
        raise ValueError('Audit output must be outside Session 29 input tree')
    result=audit(source)
    claims=[{'claim':'Synthetic pipeline consistency','status':'verified','evidence':'Independent saved-record and pixel recomputation'},
            {'claim':'Input/output integrity','status':'verified','evidence':'540 SHA-256 entries recomputed'},
            {'claim':'Real source independence','status':'unresolved','evidence':'Requires independently audited source lineage'},
            {'claim':'Physical calibration and labels','status':'unavailable','evidence':'Requires calibrated microscopy and expert annotations'},
            {'claim':'Measured steel property prediction','status':'unvalidated','evidence':'Only simulated targets available'},
            {'claim':'Real-material generalization','status':'NO-GO','evidence':'Requires representative independent material evaluation'}]
    summary=("# MicroStructAI project summary\n\n"
             "Implemented a Python/PyTorch computational-metallography learning pipeline covering classical segmentation, educational U-Net, DeepLab-style and transformer architectures, morphology, stereology, spatial descriptors, specimen regression and reliability diagnostics.\n\n"
             f"The final independent synthetic audit verified 36 specimens, 108 fields and 540 checksums. Recomputed property MAE is {result['recomputed_mae_au']:.6f} artificial units with observed coverage {result['recomputed_coverage']:.1%} on nine generated evaluation specimens.\n\n"
             "Real-material readiness remains NO-GO pending verified source lineage, independently annotated microscopy, physical calibration, specimen-matched measured properties and independent validation.\n\n"
             "The next research phase should establish versioned development data and licensing, source-aware partitions, expert labels and calibration, fair classical/neural tuning, a frozen evaluation protocol and independent physical validation.\n")
    output.mkdir(parents=True,exist_ok=False)
    (output/'session30_final_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'session30_claims_matrix.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(claims[0])); writer.writeheader(); writer.writerows(claims)
    (output/'session30_project_summary.md').write_text(summary,encoding='utf-8')
    fig,axis=plt.subplots(figsize=(10,4),layout='constrained')
    try:
        axis.barh(['Software consistency','File integrity','Real source validation','Physical calibration','Measured-property validation'],[1,1,0,0,0])
        axis.set(xlim=(0,1.1),xlabel='Verified evidence available (1=yes, 0=unresolved)',title='MicroStructAI final readiness evidence')
        fig.savefig(output/'session30_readiness.png',dpi=135)
    finally:
        plt.close(fig)
    artifacts={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    release={'artifact_sha256':artifacts,'session29_report_sha256':digest(source/'session29_report.json'),
             'auditor_source_sha256':digest(Path(__file__)),'seal_type':'Canonical SHA-256 integrity seal; no cryptographic signer'}
    release['seal_sha256']=seal(release)
    (output/'session30_release_manifest.json').write_text(json.dumps(release,indent=2)+'\n',encoding='utf-8')
    verify_release(output)
    print('Release audit passed, checksums',result['checksum_entries_verified'],'MAE',result['recomputed_mae_au'],'coverage',result['recomputed_coverage'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session29-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/session30_final'))
    args=parser.parse_args()
    run(args.session29_dir,args.output_dir)
