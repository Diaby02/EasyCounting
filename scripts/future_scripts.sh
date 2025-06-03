# degrees of libarty padding: nb_epoch ->100, ope, trainable encoder (w/ or w/o Deformable), images 512, avec TinyViT
# do the same by modifying the code
# 32 ope
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope -ps 32 -s train -d cuda:0 -a MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 3 -at bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32 -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 0 -at bool bool int

# 32 se
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_see -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 0 3 -at bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32 -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 0 -at bool bool int

# some orin scriiiiipt 32 ope rot
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_ope_rot -w checkpoints/AdaptedLocaExp/adapted_loca_32_bis.pt -ps 32 -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 0 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_bis_32 -end adapted_loca_bis_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_bis_32' | xargs rm -r &&
cd ../../scripts/

# script final analysis 32x32 ope rot padding MobileNetV3 (choose the best one between the two)
# make sure testExp2 is with test_FSC_indu initialy
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_p_rot_ope_mob -w checkpoints/AdaptedLocaExp/adapted_loca_32_p_rot_ope_mobbest.pt -ps 32 -d cuda:1 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av MobileNetV3 1 1 3 -at str bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -exp testExp -v -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p_rot_ope_mob -w checkpoints/AdaptedLocaExp/adapted_loca_32_p_rot_ope_mobbest.pt -ps 32 -d cuda:1 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av MobileNetV3 1 1 3 -at str bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_rot_ope_mob_32 -end adapted_loca_p_rot_ope_mob_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_rot_ope_mob_32' | xargs rm -r &&
cd ../../scripts/

#tinyViT + ca
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_tiny_ca -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__DECODER_LAYERS -av TinyViT 3 -at str int &&

