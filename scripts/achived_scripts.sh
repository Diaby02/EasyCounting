/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_uni_48 -ps 48 -s train_ar_uniform -d cuda:0 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_uni_64 -ps 64 -s train_ar_uniform -d cuda:0 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_uni_32 -ps 32 -s train_ar_uniform -d cuda:0 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -exp DefaultExp -m adapted_loca -sn adapted_loca_32_finetuned -ps 32 -s train_finetune -d cuda:0 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py
# run uniform 48
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_u -w checkpoints/AdaptedLocaExp/adapted_loca_uni_48.pt -ps 48 -d cuda:1 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/script_orin.py -hm &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_u_48 -end adapted_loca_u_48 -ht &&
cd adapted_loca/Results/ &&
find . -name '*adapted_loca_u_48' | xargs rm -r &&
cd ../.. &&
#run uniform 64
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_u -w checkpoints/AdaptedLocaExp/adapted_loca_uni_64.pt -ps 64 -d cuda:1 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/script_orin.py -hm &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_u_64 -end adapted_loca_u_64 -ht &&
cd adapted_loca/Results/ &&
find . -name '*adapted_loca_u_64' | xargs rm -r &&
cd ../..
# run uniform 32
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_u -w checkpoints/AdaptedLocaExp/adapted_loca_uni_32.pt -ps 32 -d cuda:0 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/script_orin2.py -hm &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_u_32 -end adapted_loca_u_32 &&
cd adapted_loca/Results/ &&
find . -name '*adapted_loca_u_32' | xargs rm -r &&
cd ../.. &&
# run of finetuned dataset
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_f -w checkpoints/AdaptedLocaExp/adapted_loca_32_finetuned.pt -ps 32 -d cuda:0 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/script_orin2.py -hm &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_f_32 -end adapted_loca_f_32 -ht &&
cd adapted_loca/Results/ &&
find . -name '*adapted_loca_f_32' | xargs rm -r &&
cd ../.. &&
# run of loca on orin dataset
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -e -exp testExp -m loca -mn loca_o -w checkpoints/loca_full.pt -d cuda:0 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/script_orin2.py -exp testExp -m loca -hm &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s loca -end loca_o -ht &&
cd loca/Results/ &&
find . -name '*loca_o' | xargs rm -r
#run cacvit
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -e -exp testExpLinux -m cacvit -mn cacvit_o -w checkpoints/best-model.pth -d cuda:1 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/script_orin.py -exp testExpLinux -m cacvit -hm &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s cacvit -end cacvit_o -ht &&
cd cacvit/Results/ &&
find . -name '*cacvit_o' | xargs rm -r &&
cd ../..
# countgd
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/modify_yaml.py -e -exp testExp -m countgd -mn countgd_o -d cuda:1 &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/script_orin.py -exp testExp -m countgd -hm &&
/mnt/grodisk-nvme/Nicolas_student/env/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s countgd -end countgd_o -r countgd/Results -ht &&
cd countgd/Results/ &&
find . -name '*countgd_o' | xargs rm -r &&
cd ../..

/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f3_24 -w checkpoints/DefaultExp2/loca_50_384_24_fr.pt -ps 24 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f3_32 -w checkpoints/DefaultExp2/loca_50_384_32_fr.pt -ps 32 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f3_48 -w checkpoints/DefaultExp2/loca_50_384_48_fr.pt -ps 48 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f3_64 -w checkpoints/DefaultExp2/loca_50_384_64_fr.pt -ps 64 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f5_24 -w checkpoints/DefaultExp2/loca_50_384_24_5_fr.pt -ps 24 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f5_32 -w checkpoints/DefaultExp2/loca_50_384_32_5_fr.pt -ps 32 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f5_48 -w checkpoints/DefaultExp2/loca_50_384_48_5_fr.pt -ps 48 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f5_64 -w checkpoints/DefaultExp2/loca_50_384_64_5_fr.pt -ps 64 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f7_24 -w checkpoints/DefaultExp2/loca_50_384_24_7_fr.pt -ps 24 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f7_32 -w checkpoints/DefaultExp2/loca_50_384_32_7_fr.pt -ps 32 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f7_48 -w checkpoints/DefaultExp2/loca_50_384_48_7_fr.pt -ps 48 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_f7_64 -w checkpoints/DefaultExp2/loca_50_384_64_7_fr.pt -ps 64 -d cuda:1 &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd
# test backbone
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_Light -w checkpoints/BasicAndLightResNet/loca_100_frozen_2.pt -ps 32 -d cuda:1 -a MODEL__LAST_LAYER -av layer3 -at str &&
/mnt/grodisk-nvme/miniconda3/envs/countgd/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_Mob -w checkpoints/MobileNetV3/loca_100_frozen_2.pt -ps 32 -d cuda:1 -a MODEL__BACKBONE_MODEL -av MobileNetV3 -at str &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_TinyViT -w checkpoints/TinyViT/loca_100_frozen_2.pt -ps 32 -d cuda:1 -a MODEL__BACKBONE_MODEL -av TinyViT -at str&&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_Eff -w checkpoints/EfficientNet_frozen/loca_50_retrained.pt -ps 32 -d cuda:1 -a MODEL__BACKBONE_MODEL -av EfficientNet -at str &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -sd
#training cross-attention with different number of blocks
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp CrossAttention -m adapted_loca -sn adapted_loca_100_3_ca -s train -ps 32 -d cuda:0 -a MODEL__NUM_DECODER_LAYERS TRAINING__EPOCH -av 3 100 -at int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp CrossAttention &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp CrossAttention -m adapted_loca -sn adapted_loca_100_4_ca -s train -ps 32 -d cuda:0 -a MODEL__NUM_DECODER_LAYERS TRAINING__EPOCH -av 4 100 -at int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp CrossAttention &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp CrossAttention -m adapted_loca -sn adapted_loca_100_5_ca -s train -ps 32 -d cuda:0 -a MODEL__NUM_DECODER_LAYERS TRAINING__EPOCH -av 5 100 -at int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp CrossAttention
#training cross attention with padding
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp CrossAttention -m adapted_loca -sn adapted_loca_100_3_ca -s train -ps 32 -d cuda:0 -a MODEL__NUM_DECODER_LAYERS MODEL__PADDING TRAINING__EPOCH -av 3 1 100 -at int int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp CrossAttention &&
#training ada_loca with padding todo
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p -ps 32 -s train -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p -ps 48 -s train -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p -ps 64 -s train -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p -ps 32 -s train -d cuda:1 -a MODEL__PADDING -av 0 -at int &&
#test
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p -w checkpoints/AdaptedLocaExp/adapted_loca_32_p.pt -ps 32 -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_32 -end adapted_loca_p_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_32' | xargs rm -r &&
cd ../../scripts/ &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p -w checkpoints/AdaptedLocaExp/adapted_loca_48_p.pt -ps 48 -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_48 -end adapted_loca_p_48 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_48' | xargs rm -r &&
cd ../../scripts/ &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p -w checkpoints/AdaptedLocaExp/adapted_loca_64_p.pt -ps 64 -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_64 -end adapted_loca_p_64 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_64' | xargs rm -r &&
cd ../../scripts/
# add ope
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope300 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope310 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 0 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope301 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope311 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope311 -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 0 -at bool bool int
#train r21
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_100 -ps 32 -s train -d cuda:1 -a MODEL__KERNEL_DIM TRAINING__EPOCH DATASET__PATCH_SIZE_RATIO -av 3 100 1 -at int int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_100 -ps 48 -s train -d cuda:1 -a MODEL__KERNEL_DIM TRAINING__EPOCH DATASET__PATCH_SIZE_RATIO -av 5 100 1 -at int int int&&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp
# reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_100 -ps 32 -s train -d cuda:1 -a MODEL__KERNEL_DIM TRAINING__EPOCH -av 3 50 -at int int &&
#training
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_r12 -ps 32 -s train -d cuda:0 -a DATASET__PATCH_SIZE_RATIO -av 2 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#test
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_r12 -w checkpoints/AdaptedLocaExp/adapted_loca_32_r12.pt -ps 32 -d cuda:1 -a DATASET__PATCH_SIZE_RATIO -av 2 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_r12 -end adapted_loca_r12_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_r12_32' | xargs rm -r &&
cd ../../scripts/
# ope with 100 epochs, padding and 32
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope300 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope310 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 0  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope301 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 1  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#!/bin/bash
# training adapted loca with trainable transformer for the reference features
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_100_tr -ps 32 -s train -d cuda:1 -a MODEL__KERNEL_DIM MODEL__TRAINABLE_REFERENCES TRAINING__EPOCH -av 3 1 100 -at int bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp
#reset to init default
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_100_tr -ps 32 -s train -d cuda:1 -a MODEL__KERNEL_DIM MODEL__TRAINABLE_REFERENCES TRAINING__EPOCH -av 3 0 100 -at int bool int &&
# ope with 100 epochs
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope300 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 0 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope310 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 0 0 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope301 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 1 0 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope311 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 1 0 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope311 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 0 100 -at bool bool int int &&
# script orin for ope_100
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_ope300 -w checkpoints/AdaptedLocaExp/adapted_loca_32_ope300.pt -ps 32 -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 0 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_32_ope300 -end adapted_loca_ope300_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_ope300_32' | xargs rm -r &&
cd ../../scripts/ &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_ope310 -w checkpoints/AdaptedLocaExp/adapted_loca_32_ope310.pt -ps 32 -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 0 0 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_32_ope310 -end adapted_loca_ope310_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_ope310_32' | xargs rm -r &&
cd ../../scripts/ &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_ope301 -w checkpoints/AdaptedLocaExp/adapted_loca_32_ope301.pt -ps 32 -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 1 0 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_32_ope301 -end adapted_loca_ope301_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_ope301_32' | xargs rm -r &&
cd ../../scripts/ &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_ope311 -w checkpoints/AdaptedLocaExp/adapted_loca_32_ope311.pt -ps 32 -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 1 0 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_32_ope311 -end adapted_loca_ope311_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_ope311_32' | xargs rm -r &&
cd ../../scripts/ &&
# reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_ope310 -w checkpoints/AdaptedLocaExp/adapted_loca_32_ope310.pt -ps 32 -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 0 100 -at bool bool int int &&
# training padding with 100 epochs
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p -ps 32 -s train -d cuda:1 -a MODEL__PADDING TRAINING__EPOCH -av 1 100 -at int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p -ps 48 -s train -d cuda:1 -a MODEL__PADDING TRAINING__EPOCH -av 1 100 -at int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_100 -ps 64 -s train -d cuda:1 -a MODEL__PADDING TRAINING__EPOCH -av 1 100 -at int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p -ps 32 -s train -d cuda:1 -a MODEL__PADDING TRAINING__EPOCH -av 0 100 -at int int &&
# script_orin for paddding_100
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p -w checkpoints/AdaptedLocaExp/adapted_loca_32_p.pt -ps 32 -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_32 -end adapted_loca_p_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_32' | xargs rm -r &&
cd ../../scripts/ &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p -w checkpoints/AdaptedLocaExp/adapted_loca_48_p.pt -ps 48 -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_48 -end adapted_loca_p_48 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_48' | xargs rm -r &&
cd ../../scripts/ &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p_100 -w checkpoints/AdaptedLocaExp/adapted_loca_64_p_100.pt -ps 64 -d cuda:1 -a MODEL__PADDING -av 1 -at int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_100_64 -end adapted_loca_p_100_64 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_100_64' | xargs rm -r &&
cd ../../scripts/
# modify data loader and do the same for 12
# ope with 100 epochs, padding and 32
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope311 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 1  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope311 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 0 0 100 -at bool bool bool int int &&
# ope with 100 epochs, padding and 48
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p_ope300 -ps 48 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p_ope310 -ps 48 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 0  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p_ope301 -ps 48 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 1  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p_ope311 -ps 48 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 1  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope311 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 0 0 100 -at bool bool bool int int &&
# ope with 100 epochs, padding and 64
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_ope300 -ps 64 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_ope310 -ps 64 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 0  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_ope301 -ps 64 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 1  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_ope311 -ps 64 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 1 1  1 3 100 -at bool bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope311 -ps 32 -s train -d cuda:1 -a MODEL__SCALE_ONLY MODEL__SCALE_AS_KEY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 0 0 0 0 100 -at bool bool bool int int

# padding,64 and trainable references
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_tr -ps 64 -s train -d cuda:0 -a MODEL__TRAINABLE_REFERENCES MODEL__PADDING TRAINING__EPOCH -av 1 1 100 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with tiny
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_tr_tiny -ps 64 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING TRAINING__EPOCH -av TinyViT 1 1 100 -at str bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with tiny, without trainable ref
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_tiny -ps 64 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING TRAINING__EPOCH -av TinyViT 0 1 100 -at str bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64 -ps 64 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING TRAINING__EPOCH -av resNet50 0 0 100 -at str bool bool int
# ope with 100 epochs, padding and 32 and trainable reference
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp 1 1 3 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with tiny
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope_tiny -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av TinyViT 1 1 3 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with tiny, without ope
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_tiny -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av TinyViT 1 1 0 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope_tiny -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp 0 0 0 100 -at str bool bool int int
#with tiny, without trainable ref
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope_tiny -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av TinyViT 0 1 3 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope_tiny -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp 0 0 0 100 -at str bool bool int int
# padding 48 and kernel 5
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p_5 -ps 48 -s train -d cuda:0 -a MODEL__KERNEL_DIM MODEL__PADDING TRAINING__EPOCH -av 5 1 100 -at int bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
# padding 48, 5 and trainable references
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p_5_tr -ps 48 -s train -d cuda:0 -a MODEL__KERNEL_DIM MODEL__TRAINABLE_REFERENCES MODEL__PADDING TRAINING__EPOCH -av 5 1 1 100 -at int bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
# padding 48, 5 and trainable references and ope
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_p_5_tr_ope -ps 48 -s train -d cuda:0 -a MODEL__KERNEL_DIM MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av 5 1 1 3 100 -at int bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope_tiny -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp 0 0 0 100 -at str bool bool int int
#with MobileNet, with ope, with trainable ref
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope_tr_mob -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av MobileNetV3 1 1 3 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with MobileNet, with ope, without trainable ref
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope_mob -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av MobileNetV3 0 1 3 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with MobileNet, without ope
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_tr_mob -ps 64 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av MobileNetV3 1 1 0 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with MobileNet, wit ope, with trainable ref
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_tr_ope_mob -ps 64 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av MobileNetV3 1 1 3 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with MobileNet, wit ope, without trainable ref
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_ope_mob -ps 64 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av MobileNetV3 0 1 3 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope_mobnet -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp 0 0 0 100 -at str bool bool int int
#with MobileNet
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_mob -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av MobileNetV3 0 0 0 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with TinyViT
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_tiny -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av TinyViT 0 0 0 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with EffNet
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_eff -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av EfficientNet 0 0 0 100 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with LightResNet
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_light -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__LAST_LAYER MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp layer3 0 0 0 100 -at str str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope_mobnet -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__LAST_LAYER MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp layer4 0 0 0 100 -at str str bool bool int int
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_rotation -ps 32 -s train -d cuda:0 -a MODEL__ROTATION -av 1 -at bool &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_rotation -ps 32 -s train -d cuda:0 -a MODEL__ROTATION -av 0 -at bool &&
#with trainable rotation
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trainable_rotation -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION -av 1 -at bool &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trainable_rotation -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION -av 0 -at bool
#with rotation
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_bis -ps 32 -s train -d cuda:0 &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with trainable rotation and padding
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trot_p -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING -av 1 1 -at bool bool &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with trainable rotation and padding and ope
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trot_p_ope -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trainable_rotation -ps 32 -s train -d cuda:0 -a MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 -at int
#with trainable rotation and padding
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_trot_p -ps 64 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING -av 1 1 -at bool bool &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trainable_rotation -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING -av 0 0 -at bool bool
#retrain ope, padding 32
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope -ps 32 -s train -d cuda:0 -a MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 3 -at bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trainable_rotation -ps 32 -s train -d cuda:0 -a MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 -at bool int
#with rotation 48 5 => need to modify to get kernel = 5
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_48_5_rot -ps 48 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__KERNEL_DIM -av 1 5 -at bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trainable_rotation -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__KERNEL_DIM -av 0 3 -at bool int

#with padding, ope and rot
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope_rot -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with ope and rot
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope_rot -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 0 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#with rot and 3 conv blocks instead of 1
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_rot_3_blocks -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS MODEL__TRAINABLE_ROT_NB_BLOCKS -av 1 0 0 3 -at bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trainable_rotation -ps 32 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL_PADDING MODEL__NUM_OPE_ITERATIVE_STEPS MODEL__TRAINABLE_ROT_NB_BLOCKS -av 0 0 0 1 -at bool int

# some orin scriiiiipt
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p -w checkpoints/AdaptedLocaExp/adapted_loca_32_trot.pt -ps 32 -d cuda:0 -a MODEL__TRAINABLE_ROTATION -av 1 -at bool &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_trot_32 -end adapted_loca_trot_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_trot_32' | xargs rm -r &&
cd ../../scripts/

#retrain ope, padding 32 
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_ope_bis -ps 32 -s train -d cuda:0 -a MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 3 -at bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_trainable_rotation -ps 32 -s train -d cuda:0 -a MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 -at bool int
#64 with MobileNet, without ope, bis
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_tr_mob_200 -ps 64 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av MobileNetV3 1 1 0 200 -at str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp 0 0 0 100 -at str bool bool int int

#64 with LightResNet, without ope, bis
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_tr_lightresnet_200 -ps 64 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__LAST_LAYER MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp layer3 1 1 0 200 -at str str bool bool int int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_tr_ope_mobnet -ps 32 -s train -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__LAST_LAYER MODEL__TRAINABLE_REFERENCES MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS TRAINING__EPOCH -av DefaultExp layer4 0 0 0 100 -at str str bool bool int int

# padding rot 64
#with padding, no ope and rot
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_rot -ps 64 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 0 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&

# padding rot 64
#with padding, no ope and rot
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_rot_ope -ps 64 -s train -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&

# some orin scriiiiipt 64
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p_rot -w checkpoints/AdaptedLocaExp/adapted_loca_64_p_rot.pt -ps 64 -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING -av 1 1 -at bool bool &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_rot_64 -end adapted_loca_p_rot_64 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_rot_64' | xargs rm -r &&
cd ../../scripts/

# some orin scriiiiipt 32
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_trot -w checkpoints/AdaptedLocaExp/adapted_loca_32_rot.pt -ps 32 -d cuda:0 -a MODEL__TRAINABLE_ROTATION -av 1 -at bool &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_trot_32 -end adapted_loca_trot_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_trot_32' | xargs rm -r &&
cd ../../scripts/

#retrain ope, padding 32 and se
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_p_se -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32 -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 0 -at bool bool int
# 64 padding se
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_64_p_se -ps 64 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32 -ps 64 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 0 -at bool bool int

# some orin scriiiiipt 32
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p_se -w checkpoints/AdaptedLocaExp/adapted_loca_32_p_se.pt -ps 32 -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_se_32 -end adapted_loca_p_se_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_se_32' | xargs rm -r &&
cd ../../scripts/

# some orin scriiiiipt 64
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p_se -w checkpoints/AdaptedLocaExp/adapted_loca_64_p_se.pt -ps 64 -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 1 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_se_64 -end adapted_loca_p_se_64 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_se_64' | xargs rm -r &&
cd ../../scripts/

# 32 ope
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope -ps 32 -s train -d cuda:0 -a MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 3 -at bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32 -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 0 -at bool bool int

# 32 ope se
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32_ope_se -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 0 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/main.py -exp AdaptedLocaExp &&
#reset
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -exp AdaptedLocaExp -m adapted_loca -sn adapted_loca_32 -ps 32 -s train -d cuda:0 -a MODEL__SCALE_ONLY MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 0 0 0 -at bool bool int

# some orin scriiiiipt 32 ope rot
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_ope_rot -w checkpoints/AdaptedLocaExp/adapted_loca_32_ope_rot.pt -ps 32 -d cuda:0 -a MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av 1 0 3 -at bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_ope_rot_32 -end adapted_loca_ope_rot_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_ope_rot_32' | xargs rm -r &&
cd ../../scripts/

# script final analysis 32x32 ope rot padding DefaultExp (choose the best one between the two)
# make sure testExp2 is with test_FSC_indu initialy
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp -m adapted_loca -mn adapted_loca_p_ope_rot_1 -w checkpoints/AdaptedLocaExp/adapted_loca_32_p_ope_rot_1.pt -ps 32 -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av DefaultExp 1 1 3 -at str bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/adapted_loca/test.py -exp testExp -v -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/modify_yaml.py -e -exp testExp2 -m adapted_loca -mn adapted_loca_p_ope_rot_1 -w checkpoints/AdaptedLocaExp/adapted_loca_32_p_ope_rot_1.pt -ps 32 -d cuda:0 -a MODEL__BACKBONE_MODEL MODEL__TRAINABLE_ROTATION MODEL__PADDING MODEL__NUM_OPE_ITERATIVE_STEPS -av DefaultExp 1 1 3 -at str bool bool int &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/scripts/script_orin2.py -hm &&
/mnt/grodisk-nvme/miniconda3/envs/adaloca/bin/python /mnt/grodisk-nvme/Nicolas_student/deepcounting/plots_and_csv_statistics/results_analysis.py -f hm.csv -s adapted_loca_p_ope_rot_1_32 -end adapted_loca_p_ope_rot_1_32 -ht &&
cd ../adapted_loca/Results/ &&
find . -name '*adapted_loca_p_ope_rot_1_32' | xargs rm -r &&
cd ../../scripts/


