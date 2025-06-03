# degrees of libarty padding: nb_epoch ->100, ope, trainable encoder (w/ or w/o Deformable), images 512, avec TinyViT
# do the same by modifying the code

# script final analysis 64x64 padding DefaultExp (choose the best one between the two)
# make sure testExp2 is with test_FSC_indu initialy
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_p_100 -w checkpoints/AdaptedLocaExp/adapted_loca_64_p_100.pt -ps 64 -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av DefaultExp 0 1 0 -at str bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -exp testExp -hm

