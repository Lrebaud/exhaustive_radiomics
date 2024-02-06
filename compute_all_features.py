from glob import glob
from radiomics import featureextractor, getTestCase
from scipy.ndimage import binary_erosion
from scipy.ndimage import label, binary_dilation
from sklearn.metrics import pairwise_distances_chunked
from totalsegmentator.map_to_binary import class_map
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import sys
import json

organ_groups = {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "brain",
        7: "stomach",
        8: "esophagus",
        9: "small_bowel",
        10: "duodenum",
        11: "colon",
        12: "trachea",
        13: "pancreas",
        14: "adrenal_gland_right",
        15: "adrenal_gland_left",
        16: "urinary_bladder",
        17: {"name": "cardiovascular",
             "organs":["aorta",
                       "inferior_vena_cava",
                       "portal_vein_and_splenic_vein",
                       "pulmonary_artery",
                       "iliac_artery_left",
                       "iliac_artery_right",
                       "iliac_vena_left",
                       "iliac_vena_right"]},
        18: {"name": "lung_left",
             "organs":["lung_upper_lobe_left",
                       "lung_lower_lobe_left"]},
        19: {"name": "lung_right",
             "organs":["lung_upper_lobe_right",
                       "lung_middle_lobe_right",
                       "lung_lower_lobe_right"]},

        20: {"name": "skeleton",
             "organs":["vertebrae_L5",
                       "vertebrae_L4",
                       "vertebrae_L3",
                       "vertebrae_L2",
                       "vertebrae_L1",
                       "vertebrae_T12",
                       "vertebrae_T11",
                       "vertebrae_T10",
                       "vertebrae_T9",
                       "vertebrae_T8",
                       "vertebrae_T7",
                       "vertebrae_T6",
                       "vertebrae_T5",
                       "vertebrae_T4",
                       "vertebrae_T3",
                       "vertebrae_T2",
                       "vertebrae_T1",
                       "vertebrae_C7",
                       "vertebrae_C6",
                       "vertebrae_C5",
                       "vertebrae_C4",
                       "vertebrae_C3",
                       "vertebrae_C2",
                       "vertebrae_C1",
                       "rib_left_1",
                       "rib_left_2",
                       "rib_left_3",
                       "rib_left_4",
                       "rib_left_5",
                       "rib_left_6",
                       "rib_left_7",
                       "rib_left_8",
                       "rib_left_9",
                       "rib_left_10",
                       "rib_left_11",
                       "rib_left_12",
                       "rib_right_1",
                       "rib_right_2",
                       "rib_right_3",
                       "rib_right_4",
                       "rib_right_5",
                       "rib_right_6",
                       "rib_right_7",
                       "rib_right_8",
                       "rib_right_9",
                       "rib_right_10",
                       "rib_right_11",
                       "rib_right_12",
                       "humerus_left",
                       "humerus_right",
                       "scapula_left",
                       "scapula_right",
                       "clavicula_left",
                       "clavicula_right",
                       "femur_left",
                       "femur_right",
                       "hip_left",
                       "hip_right",
                       "sacrum"]},

        21: {"name": "heart",
             "organs":["heart_myocardium",
            "heart_atrium_left",
            "heart_ventricle_left",
            "heart_atrium_right",
            "heart_ventricle_right"]},

        22: {"name": "insidemuscle",
            "organs": ["gluteus_maximus_left",
            "gluteus_maximus_right",
            "gluteus_medius_left",
            "gluteus_medius_right",
            "gluteus_minimus_left",
            "gluteus_minimus_right",
            "autochthon_left",
            "autochthon_right",
            "iliopsoas_left",
            "iliopsoas_right"]},

}


def agg_range(v):
    return np.max(v)-np.min(v)

aggregation_method = {
    'mini': np.nanmin,
    'mean': np.nanmean,
    'media': np.nanmedian,
    'maxi': np.nanmax,
    'std': np.nanstd,
    'range': agg_range,
}


def read_nifti(path):
    """Read a NIfTI image. Return a SimpleITK Image."""
    nifti = sitk.ReadImage(str(path))
    return nifti


def write_nifti(sitk_img, path):
    """Save a SimpleITK Image to disk in NIfTI format."""
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(str(path))
    writer.Execute(sitk_img)


def get_attributes(sitk_image):
    """Get physical space attributes (meta-data) of the image."""
    attributes = {}
    attributes['orig_pixelid'] = sitk_image.GetPixelIDValue()
    attributes['orig_origin'] = sitk_image.GetOrigin()
    attributes['orig_direction'] = sitk_image.GetDirection()
    attributes['orig_spacing'] = np.array(sitk_image.GetSpacing())
    attributes['orig_size'] = np.array(sitk_image.GetSize(), dtype=int)
    return attributes


def resample_sitk_image(sitk_image,
                        new_spacing=[1, 1, 1],
                        new_size=None,
                        attributes=None,
                        interpolator=sitk.sitkLinear,
                        fill_value=0):
    """
    Resample a SimpleITK Image.

    Parameters
    ----------
    sitk_image : sitk.Image
        An input image.
    new_spacing : list of int
        A distance between adjacent voxels in each dimension given in physical units (mm) for the output image.
    new_size : list of int or None
        A number of pixels per dimension of the output image. If None, `new_size` is computed based on the original
        input size, original spacing and new spacing.
    attributes : dict or None
        The desired output image's spatial domain (its meta-data). If None, the original image's meta-data is used.
    interpolator
        Available interpolators:
            - sitk.sitkNearestNeighbor : nearest
            - sitk.sitkLinear : linear
            - sitk.sitkGaussian : gaussian
            - sitk.sitkLabelGaussian : label_gaussian
            - sitk.sitkBSpline : bspline
            - sitk.sitkHammingWindowedSinc : hamming_sinc
            - sitk.sitkCosineWindowedSinc : cosine_windowed_sinc
            - sitk.sitkWelchWindowedSinc : welch_windowed_sinc
            - sitk.sitkLanczosWindowedSinc : lanczos_windowed_sinc
    fill_value : int or float
        A value used for padding, if the output image size is less than `new_size`.

    Returns
    -------
    sitk.Image
        The resampled image.

    Notes
    -----
    This implementation is based on https://github.com/deepmedic/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """
    sitk_interpolator = interpolator

    # provided attributes:
    if attributes:
        orig_pixelid = attributes['orig_pixelid']
        orig_origin = attributes['orig_origin']
        orig_direction = attributes['orig_direction']
        orig_spacing = attributes['orig_spacing']
        orig_size = attributes['orig_size']

    else:
        # use original attributes:
        orig_pixelid = sitk_image.GetPixelIDValue()
        orig_origin = sitk_image.GetOrigin()
        orig_direction = sitk_image.GetDirection()
        orig_spacing = np.array(sitk_image.GetSpacing())
        orig_size = np.array(sitk_image.GetSize(), dtype=int)

    # new image size:
    if not new_size:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetDefaultPixelValue(fill_value)
    resample_filter.SetOutputPixelType(orig_pixelid)
    
    resampled_sitk_image = resample_filter.Execute(sitk_image)

    return resampled_sitk_image


def get_pad_amount(arr, target_shape):
    half_missing = (target_shape-arr.shape)/2
    half_missing = half_missing.astype('int32')
    first_half = half_missing
    second_half = target_shape - (arr.shape+half_missing)
    pad_amount = tuple([(first_half[i], second_half[i]) for i in range(3)])
    return pad_amount



def get_shortest_and_longest_distance(maskA, maskB, max_n_points=10000):
    shellA = maskA - binary_erosion(maskA, iterations=1).astype(maskA.dtype)
    shellB = maskB - binary_erosion(maskB, iterations=1).astype(maskB.dtype)
    pointsA = np.stack(np.where(shellA==1)).T
    pointsB = np.stack(np.where(shellB==1)).T
    if pointsA.shape[0] > max_n_points:
        pointsA = pointsA[np.random.choice(np.arange(pointsA.shape[0]), max_n_points)]
    if pointsB.shape[0] > max_n_points:
        pointsB = pointsB[np.random.choice(np.arange(pointsB.shape[0]), max_n_points)]
        
    pointsA = list([tuple(x) for x in pointsA])
    pointsB = list([tuple(x) for x in pointsB])

    all_max, all_min = [], []
    for n in pairwise_distances_chunked(pointsA, pointsB, n_jobs=1):
        all_max.append(np.max(n))
        all_min.append(np.min(n))
    return np.min(all_min), np.max(all_max)

def resample_and_match_sitk(img_itk, ref_attributes, interpolator, voxel_size):
    img_itk = resample_sitk_image(img_itk,
                             new_spacing=[voxel_size]*3,
                             attributes=ref_attributes,
                             interpolator=interpolator)
    img_arr = sitk.GetArrayFromImage(img_itk)
    img_arr = np.rot90(img_arr, 2)
    img_itk = sitk.GetImageFromArray(img_arr)
    img_itk.SetSpacing([voxel_size]*3)
    return img_itk

def group_organs(organ_itk, fatmuscle_itk, organ_groups):
    organ_arr = sitk.GetArrayFromImage(organ_itk)
    fatmuscle_arr = sitk.GetArrayFromImage(fatmuscle_itk)
    corrected_organ_arr = np.full(organ_arr.shape, 0, dtype=organ_arr.dtype)
    corrected_organ_arr[fatmuscle_arr==33] = 33
    corrected_organ_arr[fatmuscle_arr==34] = 34
    for oid in organ_groups:
        if isinstance(organ_groups[oid], dict):
            organ_name = organ_groups[oid]['name']
            for suborgan in organ_groups[oid]['organs']:
                total_segmentator_id = [x for x in range(1,105) if class_map['total'][x]==suborgan][0]
                corrected_organ_arr[organ_arr==total_segmentator_id] = oid
        else:
            organ_name = organ_groups[oid]
            total_segmentator_id = [x for x in range(1,105) if class_map['total'][x]==organ_name][0]
            corrected_organ_arr[organ_arr==total_segmentator_id] = oid
    corrected_organ_itk = sitk.GetImageFromArray(corrected_organ_arr)
    corrected_organ_itk.CopyInformation(organ_itk)
    return corrected_organ_itk

def compute_lesion_features(pt_itk, ct_itk, lesion_itk, voxel_size):
    results = []
    
    attributes_pt = get_attributes(pt_itk)
    pt = pt_itk
    pt = sitk.GetArrayFromImage(pt).astype(np.float32)
    pt = np.rot90(pt, 2)

    ct = ct_itk
    ct = sitk.GetArrayFromImage(ct).astype(np.float32)
    ct = np.rot90(ct, 2)

    mask = lesion_itk
    mask = sitk.GetArrayFromImage(mask).astype(np.float32)
    mask = np.rot90(mask, 2)
    mask_itk = sitk.GetImageFromArray(mask)
    mask_itk.SetSpacing([voxel_size]*3)
    
    imgs = {
        'CT': ct,
        'PT': pt,
    }
       
    roi_ids = np.unique(sitk.GetArrayFromImage(mask_itk))
    roi_ids = roi_ids[roi_ids!=0]
    nb_roi = len(roi_ids)    
    roi_ids = [-1.]+list(roi_ids)
    
    for idx in roi_ids:
        if idx == -1:
            mask_roi = (mask >= 1).astype('int32')
        else:
            mask_roi = (mask == idx).astype('int32')
            
        mask_shell_roi = binary_dilation(mask_roi, iterations=8) - mask_roi
        
        x = np.any(mask_shell_roi, axis=(1, 2))
        y = np.any(mask_shell_roi, axis=(0, 2))
        z = np.any(mask_shell_roi, axis=(0, 1))
        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        pad = 2
        xmin, xmax = xmin-pad, xmax+pad
        ymin, ymax = ymin-pad, ymax+pad
        zmin, zmax = zmin-pad, zmax+pad
        
        mask_roi = mask_roi[xmin:xmax, ymin:ymax, zmin:zmax]
        mask_shell_roi = mask_shell_roi[xmin:xmax, ymin:ymax, zmin:zmax]
        
        
        mask_roi_itk = sitk.GetImageFromArray(mask_roi)
        mask_roi_itk.SetSpacing([voxel_size]*3)
        
        mask_shell_roi_itk = sitk.GetImageFromArray(mask_shell_roi)
        mask_shell_roi_itk.SetSpacing([voxel_size]*3)

        imgs_cropped = {}
        for modality in ['PT', 'CT']:
            img_arr = imgs[modality]
            img_arr = img_arr[xmin:xmax, ymin:ymax, zmin:zmax]
            img_itk = sitk.GetImageFromArray(img_arr)
            img_itk.SetSpacing([voxel_size]*3)
            imgs_cropped[modality] = img_itk
        
            

        for mask_type in ['lesion', 'shell']:
            if mask_type == 'lesion':
                cmask = mask_roi_itk
                cmask_arr = mask_roi
                modalities = ['PT', 'CT']
                
            elif mask_type == 'shell':
                cmask = mask_shell_roi_itk    
                cmask_arr = mask_shell_roi
                modalities = ['PT', 'CT']

            row = {
                'X_size': pt.shape[0],
                'Y_size': pt.shape[1],
                'Z_size': pt.shape[2],
            }
            row['n_roi'] = nb_roi
            row['mask_type'] = mask_type
            row['roi_id'] = idx
        
            max_var, besti = None, None
            for i in range(cmask_arr.shape[1]):
                var = np.var(cmask_arr[:,i,:])
                if max_var is None or var > max_var:
                    max_var,besti = var, i            
            
            for modality in modalities:
                if modality == 'PT':
                    extractor = featureextractor.RadiomicsFeatureExtractor('pyradiomics_params_PT.yaml')
                else:
                    extractor = featureextractor.RadiomicsFeatureExtractor('pyradiomics_params_CT.yaml')
                
                features = extractor.execute(imgs_cropped[modality], cmask)
                for k in features:
                    if k[:12] != 'diagnostics_':
                        name = k.replace('original', modality)
                        if 'shape' in name and 'PT' in name:
                            name = name.replace('PT_', '')
                        #name = mask_type + '_' + name
                        row[name] = features[k][()]

            results.append(row)
    results = pd.DataFrame(results)
    return results


def compute_organ_features(pt_itk, ct_itk, lesion_itk, organ_itk, voxel_size):
    
    row = {}
                
    lesion_itk_bin = lesion_itk > 0
    lesion_arr_bin = sitk.GetArrayFromImage(lesion_itk_bin)
    tmtv = np.sum(lesion_arr_bin)
    nb_organ_with_tumor = 0
    for oid in organ_groups:
        # try:
        if isinstance(organ_groups[oid], dict):
            organ_name = organ_groups[oid]['name']
        else:
            organ_name = organ_groups[oid]
        for modality in ['PT', 'CT']:
            if modality == 'PT':
                img = pt_itk
            elif modality == 'CT':
                img = ct_itk

            extractor = featureextractor.RadiomicsFeatureExtractor('pyradiomics_params_'+modality+'.yaml')
            features = extractor.execute(img, organ_itk==oid)

            for k in features:
                if k[:12] != 'diagnostics_':
                    name = k.replace('original', modality)
                    if 'shape' in name and 'PT' in name:
                        name = name.replace('PT_', '')
                    name = organ_name + '_' + name
                    row[name] = features[k][()]

        oid_mask = organ_itk==oid
        interesct = oid_mask & lesion_itk_bin
        oid_mask = sitk.GetArrayFromImage(oid_mask)
        interesct = sitk.GetArrayFromImage(interesct)
        vol_organ = np.sum(oid_mask)
        vol_intersect = np.sum(interesct)
        row[organ_name+'_volTumorInside'] = vol_intersect
        if vol_intersect == 0:
            row[organ_name+'_gotTumor'] = False
        else:
            row[organ_name+'_gotTumor'] = True
            nb_organ_with_tumor += 1
        row[organ_name+'_volTumorInside/vol_organ'] = vol_intersect / vol_organ
        row[organ_name+'_volTumorInside/tmtv'] = vol_intersect / tmtv
        distance_to_closest_tumor, distance_to_farthest_tumor = get_shortest_and_longest_distance(oid_mask, lesion_arr_bin)
        row[organ_name+'_shortestDistanceToTumor'] = distance_to_closest_tumor
        row[organ_name+'_longestDistanceToTumor'] = distance_to_farthest_tumor
        # except:
        #     pass


    row['nb_organ_with_tumor'] = nb_organ_with_tumor
    muscle_mask = sitk.GetArrayFromImage(organ_itk==33)
    fat_mask = sitk.GetArrayFromImage(organ_itk==34)
    vol_muscle = np.sum(muscle_mask)
    vol_fat = np.sum(fat_mask)
    row['vol_muscle / (vol_muscle+vol_fat)'] = vol_muscle / (vol_muscle+vol_fat)
    row['vol_fat / (vol_muscle+vol_fat)'] = vol_fat / (vol_muscle+vol_fat)
    row['vol_fat / vol_muscle'] = vol_fat / vol_muscle
    row['tmtv / vol_muscle'] = tmtv / vol_muscle
    row['tmtv / vol_fat'] = tmtv / vol_fat
    row['tmtv / (vol_muscle+vol_fat)'] = tmtv / (vol_muscle+vol_fat)
    
    return row



def compute_surrogate_biomarkers(df, lesion_features):
    df['Low ratio of volume of subcutaneous fat / volume of muscle'] = df['vol_fat / vol_muscle']
    df['High number of lesions'] = df['n_lesions']
    df['High number of invaded organs'] = df['nb_organ_with_tumor']
    df['High Tumor Energy from the PET image'] = df['oneroi_PT_firstorder_Energy']
    df['Large liver'] = df['liver_shape_MajorAxisLength']
    df['Presence of small lesions'] = df['lesion_shape_MeshVolume_mini']
    df['Presence of homogeneous density lesion'] = df['lesion_CT_firstorder_Kurtosis_maxi']
    df['Presence of lesion in a region of homogeneous density'] = df['shell_CT_gldm_DependenceEntropy_mini']
    df['High tumoral activity in the pancreas'] = df['pancreas_PT_firstorder_Energy']*df['pancreas_gotTumor']
    df['Low elongation of the pancreas'] = -df['pancreas_shape_Elongation']
    df['High roundness of tumor burden'] = -df['oneroi_shape_Flatness']
    df['Trachea involvement'] = df['trachea_gotTumor']
    df['Lungs involvement'] = float((df['lung_right_gotTumor']==1) | (df['lung_left_gotTumor']==1))
    df['Activity of liver involvement'] = df['liver_gotTumor'] * df['liver_PT_firstorder_Maximum']    
    df['Large lungs'] = df['lung_right_shape_MeshVolume']+df['lung_left_shape_MeshVolume']   
    df['High bronchus density'] = df['lung_right_CT_glcm_ClusterProminence']
    df['No air in stomach'] = df['stomach_CT_firstorder_RobustMeanAbsoluteDeviation']
    df['Right kidney involvement'] = df['kidney_right_gotTumor']
    df['Esophagus involvement'] = df['esophagus_gotTumor']
    df['Colon involvement'] = df['colon_gotTumor']
    df['Lesions near the bladder'] = df['urinary_bladder_gotTumor']

    n_occult = ((lesion_features['lesion_shape_MeshVolume'] < 20000) & (lesion_features['lesion_PT_firstorder_Maximum'] < 5)).sum()
    if n_occult > 0:
        df['Presence of occulte lesion'] = 1
    else:
        df['Presence of occulte lesion'] = 0

    return df
    


def compute_biomarkers(pt_path,
                       ct_path,
                       lesion_segmentation_path,
                       total_segmentator_segmentation_path,
                       moose_muscle_fat_segmentation,
                       voxel_size=1):

    pt_itk = sitk.ReadImage(pt_path)
    ct_itk = sitk.ReadImage(ct_path)
    lesion_itk = sitk.ReadImage(lesion_segmentation_path)
    organ_itk = sitk.ReadImage(total_segmentator_segmentation_path)
    fatmuscle_itk = sitk.ReadImage(moose_muscle_fat_segmentation)

    organ_itk = group_organs(organ_itk, fatmuscle_itk, organ_groups)
    
    attributes_pt = get_attributes(pt_itk)
    pt_itk = resample_and_match_sitk(pt_itk, attributes_pt, sitk.sitkBSpline, voxel_size)
    ct_itk = resample_and_match_sitk(ct_itk, attributes_pt, sitk.sitkBSpline, voxel_size)
    lesion_itk = resample_and_match_sitk(lesion_itk, attributes_pt, sitk.sitkNearestNeighbor, voxel_size)
    organ_itk = resample_and_match_sitk(organ_itk, attributes_pt, sitk.sitkNearestNeighbor, voxel_size)
    
    lesion_features = compute_lesion_features(pt_itk, ct_itk, lesion_itk, voxel_size)
    organ_features = compute_organ_features(pt_itk, ct_itk, lesion_itk, organ_itk, voxel_size)

    n_lesions = len(lesion_features['roi_id'].unique())-1
    
    not_features = ['roi_id', 'X_size', 'Y_size', 'Z_size', 'nb_roi', 'mask_type']
    features = [x for x in lesion_features.columns if x not in not_features and 'norm_shape_' not in x]
    one_roi = lesion_features[lesion_features['roi_id']==-1]
    one_roi_lesions = one_roi[one_roi['mask_type']=='lesion'][features]
    one_roi_shell = one_roi[one_roi['mask_type']=='shell'][features]
    one_roi_shell = one_roi_shell.dropna(how='all', axis=1)
    one_roi_lesions = one_roi_lesions.add_prefix('oneroi_')
    one_roi_shell = one_roi_shell.add_prefix('shell_oneroi_')
    record = {}
    lesion_features = lesion_features[lesion_features['roi_id']!=-1]
    for roi_type in ['shell', 'lesion']:
        rows_type = lesion_features[lesion_features['mask_type']==roi_type]
        rows_type = rows_type.dropna(how='all', axis=1)
        sub_features = list(set(features) & set(rows_type.columns))
        for f in sub_features:
            values = rows_type[f].values
            for method in aggregation_method:
                record[roi_type+'_'+f+'_'+method] = aggregation_method[method](values)
    all_features = record | one_roi_lesions.iloc[0].to_dict() | one_roi_shell.iloc[0].to_dict() | organ_features

    all_features['n_lesions'] = n_lesions
    
    for k in all_features:
        try:
            all_features[k] = all_features[k][()]
        except:
            pass
        all_features[k] = float(all_features[k])

    shell_features = lesion_features[lesion_features['mask_type']=='shell'].add_prefix('shell_').drop(columns=['shell_mask_type'])
    lesion_features = lesion_features[lesion_features['mask_type']=='lesion'].add_prefix('lesion_').drop(columns=['lesion_mask_type'])
    lesion_features = pd.concat([lesion_features, shell_features, one_roi_lesions, one_roi_shell], axis=1)

    all_features = compute_surrogate_biomarkers(all_features, lesion_features) 

    
    return all_features


if __name__ == '__main__':
    pt_path = sys.argv[1]
    ct_path = sys.argv[2]
    lesion_segmentation_path = sys.argv[3]
    total_segmentator_segmentation_path = sys.argv[4]
    moose_muscle_fat_segmentation = sys.argv[5]
    output_path = sys.argv[6]

    if len(sys.argv) == 8:
        voxel_size = int(sys.argv[7])
    else:
        voxel_size = 1

    res = compute_biomarkers(pt_path,
                             ct_path,
                             lesion_segmentation_path,
                             total_segmentator_segmentation_path,
                             moose_muscle_fat_segmentation,
                             voxel_size)

    
    with open(output_path, "w") as outfile: 
        json.dump(res, outfile)
    