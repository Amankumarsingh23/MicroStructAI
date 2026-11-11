"""Session 29 standalone traceable synthetic pipeline validation."""
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

ROLES = ('fit', 'calibration', 'evaluation')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(protocol):
    encoded=json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return {'protocol':protocol,'sha256':hashlib.sha256(encoded).hexdigest()}


def verify(sealed):
    if sealed != seal(sealed['protocol']):
        raise ValueError('Protocol seal mismatch')


def original_to_binary(mask):
    if mask.ndim!=2 or mask.dtype.kind not in 'iu' or not mask.size or np.any(mask<0) or np.any(mask>9):
        raise ValueError('Require original integer categorical IDs 0..9')
    return (mask!=9).astype(np.uint8)


def preflight(rows, root):
    root=Path(root).resolve()
    def protected(path):
        return any(re.search(r'(^|[^a-z])(test|testing|holdout)([^a-z]|$)',p.lower()) for p in path.parts)
    if not rows or protected(root):
        raise ValueError('Authorized development root required')
    seen=set(); roles={key:{} for key in ('specimen_id','source_id','texture_id','template_id')}
    checked=[]
    for row in rows:
        if row['split'] not in ('train','val') or row['role'] not in ROLES:
            raise ValueError('Protected partition or invalid role')
        if row['split'] != ('train' if row['role']=='fit' else 'val'):
            raise ValueError('Partition and role disagree')
        if row['field_id'] in seen:
            raise ValueError('Duplicate field ID')
        seen.add(row['field_id'])
        for key, mapping in roles.items():
            value=row[key]
            if not value or value.lower()=='unknown':
                raise ValueError('Declared synthetic lineage required')
            mapping.setdefault(value,set()).add(row['role'])
        paths=[]
        for key in ('image_rel','mask_rel','roi_rel'):
            rel=Path(row[key]); windows=PureWindowsPath(row[key])
            if rel.is_absolute() or windows.drive or '..' in rel.parts or '..' in windows.parts or protected(rel):
                raise ValueError('Protected or escaping path')
            path=(root/rel).resolve()
            if not path.is_relative_to(root) or protected(path) or not path.is_file():
                raise ValueError('Resolved path outside authorized development root')
            paths.append(path)
        checked.append((row,*paths))
    if any(len(values)>1 for mapping in roles.values() for values in mapping.values()):
        raise ValueError('Cross-role specimen/source/texture/template overlap')
    return checked


def generate(root,seed):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed)
    rows=[]; targets=[]
    for batch in range(3):
        order=rng.permutation(12)
        assignments={int(index):('fit' if position<6 else ('calibration' if position<9 else 'evaluation'))
                     for position,index in enumerate(order)}
        for index in range(12):
            sid=f'B{batch+1}_S{index+1:02d}'; role=assignments[index]
            latent=float(np.clip(.18+.045*batch+rng.normal(0,.035),.06,.5))
            target=float(103+130*latent+8*batch+rng.normal(0,3.5))
            targets.append({'specimen_id':sid,'batch_id':f'B{batch+1}','role':role,
                            'target_au':target,'target_provenance':'simulated_from_latent_and_batch'})
            for field in range(3):
                roi=np.ones((64,64),bool); roi[:8,49:]=False
                if field:
                    roi[-3*field:,:]=False
                texture=ndi.gaussian_filter(rng.normal(size=roi.shape),1.75)
                fraction=float(np.clip(latent+rng.normal(0,.02),.03,.65))
                positive=(texture>=np.quantile(texture[roi],1-fraction))&roi
                original=np.full(roi.shape,9,np.uint8); original[positive]=(index+field)%9
                background=200-14*batch
                image=np.clip(np.rint(np.where(positive,55,background)+rng.normal(0,5,roi.shape)),0,255).astype(np.uint8)
                fid=f'{sid}_F{field+1}'
                names=[f'{fid}.png',f'{fid}_mask.png',f'{fid}_roi.png']
                for name,array in zip(names,(image,original,roi.astype(np.uint8)*255)):
                    Image.fromarray(array).save(root/name)
                rows.append({'field_id':fid,'specimen_id':sid,'batch_id':f'B{batch+1}',
                             'role':role,'split':'train' if role=='fit' else 'val',
                             'source_id':f'source_{sid}','texture_id':f'texture_{sid}','template_id':f'template_{sid}',
                             'schema':'original10','image_rel':names[0],'mask_rel':names[1],'roi_rel':names[2]})
    return rows,targets


def audit(rows,root):
    checked=preflight(rows,root)
    decoded=[]; hashes={}; image_pixels={}
    for row,image_path,mask_path,roi_path in checked:
        if row['schema']!='original10':
            raise ValueError('Original schema required')
        with Image.open(image_path) as opened:
            gray=np.array(opened.convert('L'))
        with Image.open(mask_path) as opened:
            original=np.array(opened)
        with Image.open(roi_path) as opened:
            raw_roi=np.array(opened)
        truth=original_to_binary(original)
        if gray.shape!=truth.shape or raw_roi.shape!=truth.shape or not np.isin(raw_roi,(0,255)).all() or not raw_roi.any():
            raise ValueError('Invalid image/mask/ROI pairing')
        pixel_hash=hashlib.sha256(str(gray.shape).encode()+gray.tobytes()).hexdigest()
        # Any duplicate image is an audit blocker, including duplicate fields within a role.
        if pixel_hash in image_pixels:
            raise ValueError('Duplicate decoded image pixels')
        image_pixels[pixel_hash]=row['field_id']
        for kind,path in zip(('image','reference_mask','roi'),(image_path,mask_path,roi_path)):
            hashes[path.name]={'field_id':row['field_id'],'kind':kind,'sha256':checksum(path)}
        decoded.append((row,gray,truth,raw_roi.astype(bool)))
    return decoded,hashes


def segmentation_metrics(truth,prediction,roi):
    t=truth[roi].astype(bool); p=prediction[roi].astype(bool)
    tp=int((t&p).sum()); fp=int((~t&p).sum()); fn=int((t&~p).sum())
    union=tp+fp+fn
    return {'tp':tp,'fp':fp,'fn':fn,'iou':tp/union if union else None,
            'signed_area_error_pp':100*(fp-fn)/len(t)}


def aggregate(fields,targets):
    if len({t['specimen_id'] for t in targets})!=len(targets):
        raise ValueError('Duplicate property targets')
    if len({r['field_id'] for r in fields})!=len(fields):
        raise ValueError('Duplicate measured field')
    grouped={t['specimen_id']:[] for t in targets}
    for row in fields:
        if row['specimen_id'] not in grouped:
            raise ValueError('Unmatched measured specimen')
        grouped[row['specimen_id']].append(row)
    result=[]
    for target in targets:
        selected=grouped[target['specimen_id']]
        if len(selected)!=3 or any(r['role']!=target['role'] or r['batch_id']!=target['batch_id'] for r in selected):
            raise ValueError('Expected three consistent fields per property target')
        total=sum(r['valid_pixels'] for r in selected)
        foreground=sum(r['foreground_pixels'] for r in selected)
        result.append({**target,'n_fields':3,'valid_pixels':total,'foreground_pixels':foreground,
                       'area_fraction':foreground/total,
                       'component_density_px2':sum(r['components'] for r in selected)/total})
    return result


def fit_model(fitting):
    x=np.array([r['area_fraction'] for r in fitting]); y=np.array([r['target_au'] for r in fitting])
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite fitting data required')
    mean=float(x.mean()); scale=float(x.std())
    scale=scale if scale>1e-12 else 1.
    z=(x-mean)/scale
    weight=float(np.linalg.lstsq(z[:,None],y-y.mean(),rcond=None)[0][0])
    return {'mean':mean,'std':scale,'weight':weight,'target_mean':float(y.mean()),
            'fitting_specimens':[r['specimen_id'] for r in fitting]}


def predict(model,rows):
    return model['target_mean']+model['weight']*(np.array([r['area_fraction'] for r in rows])-model['mean'])/model['std']


def calibrate(model,rows,alpha):
    if not 0<alpha<1 or not np.isfinite(alpha):
        raise ValueError('Alpha must be in (0,1)')
    errors=np.sort(abs(np.array([r['target_au'] for r in rows])-predict(model,rows)))
    rank=math.ceil((len(rows)+1)*(1-alpha))
    if not len(errors) or rank>len(errors):
        raise ValueError('Insufficient finite calibration rank')
    return {'rank':rank,'radius_au':float(errors[rank-1]),'alpha':alpha,
            'calibration_specimens':[r['specimen_id'] for r in rows]}


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(output,seed=29,alpha=.1):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    if type(seed) is not int or seed<0 or isinstance(alpha,bool) or not np.isfinite(alpha) or not 0<alpha<1:
        raise ValueError('Invalid seed/alpha')
    protocol={'seed':seed,'alpha':alpha,'specimens':36,'fields_each':3,'role_counts':[18,9,9],
              'primary_segmentation':'dark_120','comparison':'dark_165',
              'original_mapping':'0..8 foreground; 9 background','model':'fitting-only standardized area-only least squares',
              'aggregation':'pool valid foreground counts per specimen','official_test_policy':'never_access'}
    sealed=seal(protocol)
    output.mkdir(parents=True,exist_ok=False)
    protocol_path=output/'session29_frozen_protocol.json'
    protocol_path.write_text(json.dumps(sealed,indent=2)+'\n',encoding='utf-8')
    root=output/'artificial_development'
    manifest,targets=generate(root,seed)
    write_csv(output/'session29_source_manifest.csv',manifest)
    write_csv(output/'session29_properties.csv',targets)
    verify(json.loads(protocol_path.read_text()))
    decoded,hashes=audit(manifest,root)
    predictions_dir=output/'segmentation_masks'; predictions_dir.mkdir()
    fields=[]; scores=[]; trace=[]
    for row,gray,truth,roi in decoded:
        for method,threshold in (('dark_120',120),('dark_165',165)):
            mask=(gray<threshold).astype(np.uint8)
            path=predictions_dir/f"{row['field_id']}_{method}.png"
            Image.fromarray(mask).save(path)
            hashes[str(path.relative_to(output))]={'field_id':row['field_id'],'kind':method,'sha256':checksum(path)}
            scores.append({'field_id':row['field_id'],'specimen_id':row['specimen_id'],'role':row['role'],
                           'method':method,**segmentation_metrics(truth,mask,roi)})
            if method=='dark_120':
                _,components=ndi.label(mask.astype(bool)&roi,structure=np.ones((3,3)))
                fields.append({'field_id':row['field_id'],'specimen_id':row['specimen_id'],'batch_id':row['batch_id'],
                               'role':row['role'],'valid_pixels':int(roi.sum()),'foreground_pixels':int(mask[roi].sum()),
                               'components':int(components),'prediction_rel':str(path.relative_to(output))})
    specimens=aggregate(fields,targets)
    fitting=[r for r in specimens if r['role']=='fit']
    calibration=[r for r in specimens if r['role']=='calibration']
    evaluation=[r for r in specimens if r['role']=='evaluation']
    model=fit_model(fitting); radius=calibrate(model,calibration,alpha)
    predicted=predict(model,evaluation)
    property_predictions=[]
    for row,value in zip(evaluation,predicted):
        actual=row['target_au']; low=float(value-radius['radius_au']); high=float(value+radius['radius_au'])
        property_predictions.append({'specimen_id':row['specimen_id'],'batch_id':row['batch_id'],
                                     'actual_au':actual,'prediction_au':float(value),'lower_au':low,'upper_au':high,
                                     'covered':bool(low<=actual<=high),'absolute_error_au':float(abs(actual-value))})
        trace.append({'specimen_id':row['specimen_id'],'field_ids':[r['field_id'] for r in fields if r['specimen_id']==row['specimen_id']],
                      'pooled_fraction':row['area_fraction'],'prediction_au':float(value),'model':'session29_report.json#model',
                      'manifest':'session29_source_manifest.csv','checksums':'session29_input_checksums.json',
                      'protocol_sha256':sealed['sha256']})
    segmentation=[]
    for method in ('dark_120','dark_165'):
        selected=[r for r in scores if r['role']=='evaluation' and r['method']==method]
        tp=sum(r['tp'] for r in selected); fp=sum(r['fp'] for r in selected); fn=sum(r['fn'] for r in selected)
        segmentation.append({'method':method,'evaluation_images':len(selected),'pooled_iou':tp/(tp+fp+fn)})
    report={'implementation':'Reconstructed from Session 29 handout; package unavailable locally',
            'protocol_sha256':sealed['sha256'],'source_sha256':checksum(Path(__file__)),
            'manifest_sha256':checksum(output/'session29_source_manifest.csv'),
            'specimens':36,'images':108,'role_counts':{role:sum(r['role']==role for r in specimens) for role in ROLES},
            'model':model,'calibration':radius,'evaluation_mae_au':float(np.mean([r['absolute_error_au'] for r in property_predictions])),
            'covered':sum(r['covered'] for r in property_predictions),'evaluation_n':len(evaluation),
            'coverage':float(np.mean([r['covered'] for r in property_predictions])),
            'segmentation':segmentation,'trace':trace,
            'readiness':{'synthetic_pipeline':'verified','physical_calibration':'unavailable','source_independence':'unresolved',
                         'real_annotations':'unavailable','measured_properties':'unavailable','scientific_status':'NO-GO for real-material claims'},
            'limits':['Generated masks and responses use artificial semantics and arbitrary units',
                      'Nine calibration/evaluation specimens provide coarse diagnostics',
                      'Conformal validity requires exchangeability; fixed batch composition is illustrative',
                      'Hashes provide identities without independently verifying provenance or scientific correctness']}
    write_csv(output/'session29_field_features.csv',fields)
    write_csv(output/'session29_specimen_features.csv',specimens)
    write_csv(output/'session29_segmentation_metrics.csv',scores)
    write_csv(output/'session29_property_predictions.csv',property_predictions)
    (output/'session29_input_checksums.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    (output/'session29_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    try:
        axes[0,0].bar(list(report['role_counts']),list(report['role_counts'].values()))
        axes[0,0].set(title='Disjoint specimen roles',ylabel='Specimens')
        axes[0,1].bar([r['method'] for r in segmentation],[r['pooled_iou'] for r in segmentation])
        axes[0,1].set(title='Artificial evaluation segmentation',ylabel='Pooled foreground IoU',ylim=(0,1.05))
        for batch in ('B1','B2','B3'):
            rows=[r for r in specimens if r['batch_id']==batch]
            axes[1,0].scatter([r['area_fraction'] for r in rows],[r['target_au'] for r in rows],label=batch)
        axes[1,0].set(title='Specimen feature and simulated property',xlabel='Pooled predicted fraction',ylabel='Target (a.u.)')
        axes[1,0].legend()
        axes[1,1].errorbar(range(9),predicted,yerr=radius['radius_au'],fmt='o',label='Prediction and interval')
        axes[1,1].scatter(range(9),[r['target_au'] for r in evaluation],marker='x',color='red',label='Simulated target')
        axes[1,1].set(title='Nine evaluation specimens',ylabel='Response (a.u.)'); axes[1,1].legend()
        fig.suptitle('Session 29 synthetic pipeline validation')
        fig.savefig(output/'session29_pipeline_validation.png',dpi=135)
    finally:
        plt.close(fig)
    print('Property MAE',report['evaluation_mae_au'],'coverage',report['covered'],'/',len(evaluation),'radius',radius['radius_au'])
    print('Segmentation',segmentation)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/session29_v1'))
    parser.add_argument('--seed',type=int,default=29)
    parser.add_argument('--alpha',type=float,default=.1)
    args=parser.parse_args()
    run(args.output_dir,args.seed,args.alpha)
