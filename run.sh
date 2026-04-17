# python optimized_attack_result.py --model densenet121
# python optimized_attack_result.py --model resnext50
# python optimized_attack_result.py --model vgg19bn
# python optimized_attack_result.py --model incres_v2
# python optimized_attack_result.py --model inc_v3
# python optimized_attack_result.py --model inc_v4
# python optimized_attack_result.py --model resnet101
# python optimized_attack_result.py --model resnet152

# python optimized_attack_result_incremental.py --attack mifgsm --regularization CE
# python optimized_attack_result_incremental.py --attack mifgsm --regularization KL

# python attack_main.py --attack mifgsm --regularization CE
# python attack_main.py --attack mifgsm --regularization MSE
# python attack_main.py --attack mifgsm --regularization KL --weight 0.01
# python attack_main.py --attack mifgsm --regularization KL --weight 1000
# python attack_main.py --attack mifgsm --regularization KL --weight 100
# python attack_main.py --attack mifgsm --regularization KL --weight 10
# python attack_main.py --attack mifgsm --regularization KL --weight 1
# python attack_main.py --attack mifgsm --regularization KL --weight 0.1
# python attack_main.py --attack mifgsm --regularization KL --weight 0.001

python attack_main.py --model resnet50
python attack_main.py --model densenet121
python attack_main.py --model resnext50
python attack_main.py --model vgg19bn
python attack_main.py --model incres_v2
python attack_main.py --model inc_v3
python attack_main.py --model inc_v4
python attack_main.py --model resnet101
python attack_main.py --model resnet152
python attack_main.py --model adv_inception_v3
python attack_main.py --model ens_adv_inception_resnet_v2
python attack_main.py --model visformer_small
python attack_main.py --model vit_b
python attack_main.py --model swin_b
python attack_main.py --model pit_b
python attack_main.py --model mobilenet
