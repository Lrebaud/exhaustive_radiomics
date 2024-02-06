# exhaustive_radiomics

This repository contains the code to compute an exhaustive list of radiomics features. These features are computed from the PET and CT images, and using the segmentation of the lesions, the organs, and fat and muscle.

The files required are the following:

- pet.nii.gz: the PET scan in the NIFTI format. Voxels values must be SUV.
- ct.nii.gz: the CT scan in the NIFTI format.
- lesions.nii.gz: the segmented lesions in the NIFTI format. Each voxel of each spatially disconnected lesion must be identifed with a unique value, unique to the lesion
- total_segmentator.nii.gz: The segmentation file produced by Total Segmentator version 1.5.3
- moose_fat_muscle.nii.gz: The Fat-Muscle_*.dcm.nii.gz file produced by MOOSE version 0.1.4 that segment muscle and fats

To compute the features, run the following command:

`python compute_all_features.py /path/to/pet.nii.gz /path/to/ct.nii.gz /path/to/lesions.nii.gz /path/to/total_segmentator.nii.gz /path/to/moose_fat_muscle.nii.gz output.json`

where output.json is the file in which features values will be stored.

To quickly test if the program is working, you can run:  
`python compute_all_features.py /path/to/pet.nii.gz /path/to/ct.nii.gz /path/to/lesions.nii.gz /path/to/total_segmentator.nii.gz /path/to/moose_fat_muscle.nii.gz output.json 4`


where the last digit is the size in mm of the resampled voxels. 1mm should be used, but to quickly test the program, putting a higher value such as 4 or 8 allow for quick execution.